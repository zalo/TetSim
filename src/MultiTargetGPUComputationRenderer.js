import {
	Camera,
	ClampToEdgeWrapping,
	DataTexture,
	FloatType,
	Mesh,
	NearestFilter,
	PlaneGeometry,
	RGBAFormat,
	Scene,
	RawShaderMaterial,
	WebGLRenderTarget,
    WebGLMultipleRenderTargets,
    GLSL3
} from '../node_modules/three/build/three.module.js';

/**
 * GPUComputationRenderer, based on SimulationRenderer by zz85
 *
 * The GPUComputationRenderer uses the concept of variables. These variables are RGBA float textures that hold 4 floats
 * for each compute element (texel)
 *
 * Each variable has a fragment shader that defines the computation made to obtain the variable in question.
 * You can use as many variables you need, and make dependencies so you can use textures of other variables in the shader
 * (the sampler uniforms are added automatically) Most of the variables will need themselves as dependency.
 *
 * The renderer has actually two render targets per variable, to make ping-pong. Textures from the current frame are used
 * as inputs to render the textures of the next frame.
 *
 * The render targets of the variables can be used as input textures for your visualization shaders.
 *
 * Variable names should be valid identifiers and should not collide with THREE GLSL used identifiers.
 * a common approach could be to use 'texture' prefixing the variable name; i.e texturePosition, textureVelocity...
 *
 * The size of the computation (sizeX * sizeY) is defined as 'resolution' automatically in the shader. For example:
 * #DEFINE resolution vec2( 1024.0, 1024.0 )
 *
 * -------------
 *
 * Basic use:
 *
 * // Initialization...
 *
 * // Create computation renderer
 * const gpuCompute = new GPUComputationRenderer( 1024, 1024, renderer );
 *
 * // Create initial state float textures
 * const pos0 = gpuCompute.createTexture();
 * const vel0 = gpuCompute.createTexture();
 * // and fill in here the texture data...
 *
 * // Add texture variables
 * const velVar = gpuCompute.addVariable( "textureVelocity", fragmentShaderVel, pos0 );
 * const posVar = gpuCompute.addVariable( "texturePosition", fragmentShaderPos, vel0 );
 *
 * // Add variable dependencies
 * gpuCompute.setVariableDependencies( velVar, [ velVar, posVar ] );
 * gpuCompute.setVariableDependencies( posVar, [ velVar, posVar ] );
 *
 * // Add custom uniforms
 * velVar.material.uniforms.time = { value: 0.0 };
 *
 * // Check for completeness
 * const error = gpuCompute.init();
 * if ( error !== null ) {
 *		console.error( error );
 * }
 *
 *
 * // In each frame...
 *
 * // Compute!
 * gpuCompute.compute();
 *
 * // Update texture uniforms in your visualization materials with the gpu renderer output
 * myMaterial.uniforms.myTexture.value = gpuCompute.getCurrentRenderTarget( posVar ).texture;
 *
 * // Do your rendering
 * renderer.render( myScene, myCamera );
 *
 * -------------
 *
 * Also, you can use utility functions to create ShaderMaterial and perform computations (rendering between textures)
 * Note that the shaders can have multiple input textures.
 *
 * const myFilter1 = gpuCompute.createShaderMaterial( myFilterFragmentShader1, { theTexture: { value: null } } );
 * const myFilter2 = gpuCompute.createShaderMaterial( myFilterFragmentShader2, { theTexture: { value: null } } );
 *
 * const inputTexture = gpuCompute.createTexture();
 *
 * // Fill in here inputTexture...
 *
 * myFilter1.uniforms.theTexture.value = inputTexture;
 *
 * const myRenderTarget = gpuCompute.createRenderTarget();
 * myFilter2.uniforms.theTexture.value = myRenderTarget.texture;
 *
 * const outputRenderTarget = gpuCompute.createRenderTarget();
 *
 * // Now use the output texture where you want:
 * myMaterial.uniforms.map.value = outputRenderTarget.texture;
 *
 * // And compute each frame, before rendering to screen:
 * gpuCompute.doRenderTarget( myFilter1, myRenderTarget );
 * gpuCompute.doRenderTarget( myFilter2, outputRenderTarget );
 *
 *
 *
 * @param {int} sizeX Computation problem size is always 2d: sizeX * sizeY elements.
 * @param {int} sizeY Computation problem size is always 2d: sizeX * sizeY elements.
 * @param {WebGLRenderer} renderer The renderer
  */

class MultiTargetGPUComputationRenderer {

    constructor(sizeX, sizeY, renderer) {

        this.variables = [];

        let dataType = FloatType;

        this.passes = [];

        const scene = new Scene();

        const camera = new Camera();
        camera.position.z = 1;

        const passThruUniforms = {
            passThruTexture: { value: null }
        };

        const passThruShader = createShaderMaterial(getPassThroughFragmentShader(), passThruUniforms);

        const mesh = new Mesh(new PlaneGeometry(2, 2), passThruShader);
        scene.add(mesh);


        this.setDataType = function (type) {

            dataType = type;
            return this;

        };

        this.addVariable = function (variableName, initialValueTexture, count) {

            const variable = {
                name: variableName,
                initialValueTexture: initialValueTexture,
                renderTargets: [],
                wrapS: null,
                wrapT: null,
                minFilter: NearestFilter,
                magFilter: NearestFilter,
                count: count,
                currentTextureIndex: 0
            };

            this.variables.push(variable);

            return variable;

        };

        this.addPass = function (variable, dependencies, computeFragmentShader) {

            let pass = {
                variable: variable,
                material: this.createShaderMaterial(computeFragmentShader),
                dependencies: dependencies
            };

            this.passes.push(pass);

            return pass;

        }

        this.init = function () {

            if (renderer.capabilities.isWebGL2 === false && renderer.extensions.has('OES_texture_float') === false) {

                return 'No OES_texture_float support for float textures.';

            }

            if (renderer.capabilities.maxVertexTextures === 0) {

                return 'No support for vertex shader textures.';

            }

            for (let i = 0; i < this.variables.length; i++) {

                const variable = this.variables[i];

                // Creates rendertargets and initialize them with input texture
                variable.renderTargets[0] = this.createRenderTarget(sizeX, sizeY, variable.wrapS, variable.wrapT, variable.minFilter, variable.magFilter, variable.count);
                variable.renderTargets[1] = this.createRenderTarget(sizeX, sizeY, variable.wrapS, variable.wrapT, variable.minFilter, variable.magFilter, variable.count);
                this.renderTexture(variable.initialValueTexture, variable.renderTargets[0]);
                this.renderTexture(variable.initialValueTexture, variable.renderTargets[1]);

            }

            for (let i = 0; i < this.passes.length; i++) {

                // Adds dependencies uniforms to the ShaderMaterial
                const pass = this.passes[i];
                const variable = pass.variable;
                const material = pass.material;
                const uniforms = material.uniforms;

                if (pass.dependencies !== null) {

                    for (let d = 0; d < pass.dependencies.length; d++) {

                        const depVar = pass.dependencies[d];

                        if (depVar.name !== variable.name) {

                            // Checks if variable exists
                            let found = false;

                            for (let j = 0; j < this.variables.length; j++) {

                                if (depVar.name === this.variables[j].name) {

                                    found = true;
                                    break;

                                }

                            }

                            if (!found) {

                                return 'Variable dependency not found. Variable=' + variable.name + ', dependency=' + depVar.name;

                            }

                        }

                        if (!depVar.count || depVar.count < 2) {
                            uniforms[depVar.name] = { value: null };
                            material.fragmentShader = '\nuniform sampler2D ' + depVar.name + ';\n' + material.fragmentShader;
                        } else {
                            uniforms[depVar.name] = { value: [] };
                            material.fragmentShader = '\nuniform sampler2D ' + depVar.name + '[ ' + depVar.count + ' ];\n' + material.fragmentShader;
                        }

                        // Add the previous frame's dependencies as well
                        // (only if the dependency is not the current variable)
                        if (depVar.name !== variable.name) {
                            if (!depVar.count || depVar.count < 2) {
                                uniforms["prev_"+depVar.name] = { value: null };
                                material.fragmentShader = '\nuniform sampler2D prev_' + depVar.name + ';\n' + material.fragmentShader;
                            } else {
                                uniforms["prev_"+depVar.name] = { value: [] };
                                material.fragmentShader = '\nuniform sampler2D prev_' + depVar.name + '[ ' + depVar.count + ' ];\n' + material.fragmentShader;
                            }
                        }

                    }

                }

            }

            return null;

        };

        this.compute = function () {

            for (let i = 0, il = this.passes.length; i < il; i++) {

                const pass = this.passes[i];
                const currentTextureIndex = pass.variable.currentTextureIndex;
                const nextTextureIndex = currentTextureIndex === 0 ? 1 : 0;

                // Sets texture dependencies uniforms
                if (pass.dependencies !== null) {

                    const uniforms = pass.material.uniforms;

                    for (let d = 0, dl = pass.dependencies.length; d < dl; d++) {

                        const depVar = pass.dependencies[d];

                        uniforms[depVar.name].value = depVar.renderTargets[depVar.currentTextureIndex].texture;

                        if (depVar.name !== pass.variable.name) {
                            uniforms["prev_"+depVar.name].value = depVar.renderTargets[depVar.currentTextureIndex === 0 ? 1 : 0].texture;
                        }

                    }

                }

                // Performs the computation for this variable
                this.doRenderTarget(pass.material, pass.variable.renderTargets[nextTextureIndex]);

                pass.variable.currentTextureIndex = nextTextureIndex;

            }

        };

        this.getCurrentRenderTarget = function (variable) {

            return variable.renderTargets[variable.currentTextureIndex];

        };

        this.getAlternateRenderTarget = function (variable) {

            return variable.renderTargets[variable.currentTextureIndex === 0 ? 1 : 0];

        };

        function addResolutionDefine(materialShader) {

            materialShader.defines.resolution = 'vec2( ' + sizeX.toFixed(1) + ', ' + sizeY.toFixed(1) + ' )';
            //materialShader.defines.gl_FragColor = 'pc_fragColor';
            materialShader.defines.texture2D = 'texture';
            materialShader.defines.derp = '0.0;\nprecision highp float;';

        }

        this.addResolutionDefine = addResolutionDefine;


        // The following functions can be used to compute things manually

        function createShaderMaterial(computeFragmentShader, uniforms) {

            uniforms = uniforms || {};

            const material = new RawShaderMaterial({
                uniforms: uniforms,
                vertexShader: getPassThroughVertexShader(),
                fragmentShader: computeFragmentShader,
                glslVersion: GLSL3
            });

            addResolutionDefine(material);

            return material;

        }

        this.createShaderMaterial = createShaderMaterial;

        this.createRenderTarget = function (sizeXTexture, sizeYTexture, wrapS, wrapT, minFilter, magFilter, count) {

            sizeXTexture = sizeXTexture || sizeX;
            sizeYTexture = sizeYTexture || sizeY;

            wrapS = wrapS || ClampToEdgeWrapping;
            wrapT = wrapT || ClampToEdgeWrapping;

            minFilter = minFilter || NearestFilter;
            magFilter = magFilter || NearestFilter;

            if (!count || count == 1) {
                const renderTarget = new WebGLRenderTarget(sizeXTexture, sizeYTexture, {
                    wrapS: wrapS,
                    wrapT: wrapT,
                    minFilter: minFilter,
                    magFilter: magFilter,
                    format: RGBAFormat,
                    type: dataType,
                    depthBuffer: false
                });

                return renderTarget;
            } else {

                const renderTarget = new WebGLMultipleRenderTargets(
                    sizeXTexture, sizeYTexture, count, {
                    wrapS: wrapS,
                    wrapT: wrapT,
                    minFilter: minFilter,
                    magFilter: magFilter,
                    format: RGBAFormat,
                    type: dataType,
                    depthBuffer: false
                });

                // For PC users?
                for (let i = 0, il = renderTarget.texture.length; i < il; i++) {
                    renderTarget.texture[i].wrapS = wrapS;
                    renderTarget.texture[i].wrapT = wrapT;
                    renderTarget.texture[i].minFilter = minFilter;
                    renderTarget.texture[i].magFilter = magFilter;
                    renderTarget.texture[i].format = RGBAFormat;
                    renderTarget.texture[i].type = dataType;
                    renderTarget.texture[i].depthBuffer = false;
                    renderTarget.texture[i].name = "gbuffer" + i;
                }

                return renderTarget;
            }

        };

        this.createTexture = function () {

            const data = new Float32Array(sizeX * sizeY * 4);
            return new DataTexture(data, sizeX, sizeY, RGBAFormat, FloatType);

        };

        this.renderTexture = function (input, output) {

            // Takes a texture, and render out in rendertarget
            // input = Texture
            // output = RenderTarget

            passThruUniforms.passThruTexture.value = input;

            if (output.isWebGLMultipleRenderTargets) {

                let multiPassthroughShader = createShaderMaterial(`
                layout(location = 0) out highp vec4 tex0;
                layout(location = 1) out highp vec4 tex1;
                layout(location = 2) out highp vec4 tex2;
                layout(location = 3) out highp vec4 tex3;
                uniform sampler2D[4] passThruTexture;
                void main() {
                    vec2 uv = gl_FragCoord.xy / resolution.xy;
                    tex0 = texture2D( passThruTexture[0], uv );
                    tex1 = texture2D( passThruTexture[1], uv );
                    tex2 = texture2D( passThruTexture[2], uv );
                    tex3 = texture2D( passThruTexture[3], uv );
                }`, passThruUniforms)
                this.doRenderTarget(multiPassthroughShader, output);
            } else {
                this.doRenderTarget(passThruShader, output);
            }
            

            passThruUniforms.passThruTexture.value = null;

        };

        this.doRenderTarget = function (material, output) {

            const currentRenderTarget = renderer.getRenderTarget();

            mesh.material = material;
            renderer.setRenderTarget(output);
            renderer.render(scene, camera);
            mesh.material = passThruShader;

            renderer.setRenderTarget(currentRenderTarget);

        };

        // Shaders

        function getPassThroughVertexShader() {
            return `
			in vec3 position;

			uniform mat4 modelViewMatrix;
			uniform mat4 projectionMatrix;

			void main() {

				vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );

				gl_Position = projectionMatrix * mvPosition;

			}`
        }

        function getPassThroughFragmentShader() {

            return `
            out highp vec4 pc_fragColor;
            uniform sampler2D passThruTexture;
                void main() {
                	vec2 uv = gl_FragCoord.xy / resolution.xy;
                	pc_fragColor = texture2D( passThruTexture, uv );
                }`;

        }

    }

}

export { MultiTargetGPUComputationRenderer };
