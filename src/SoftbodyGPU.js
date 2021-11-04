import * as THREE from '../node_modules/three/build/three.module.js';
import { MultiTargetGPUComputationRenderer } from './MultiTargetGPUComputationRenderer.js';

export class SoftBodyGPU {
    constructor(vertices, tetIds, tetEdgeIds, physicsParams,
        visVerts, visTriIds, visMaterial, world) {
        this.physicsParams = physicsParams; // Set the Uniforms using these later
        /** @type {THREE.WebGLRenderer} */ 
        this.renderer = world.renderer;

        this.numParticles       = vertices.length / 3;
        this.numElems           = tetIds.length / 4;
        this.texDim             = Math.ceil(Math.sqrt(this.numElems));
        this.tetPositionsArray  = new Float32Array(this.texDim * this.texDim * 4); // Used for GPU Readback
        this.elemPositionsArray = new Float32Array(this.texDim * this.texDim * 4); // Used for GPU Readback
        this.inputPos           = vertices.slice(0);
        this.grabPos            = new Float32Array(3);
        this.grabId             = -1;

        // Initialize the General Purpose GPU Computation Renderer
        this.gpuCompute = new MultiTargetGPUComputationRenderer(this.texDim, this.texDim, this.renderer)

        // Allocate static textures that are used to initialize the simulation
        this.pos0                  = this.gpuCompute.createTexture(); // Set to vertices
        this.vel0                  = this.gpuCompute.createTexture(); // Leave as 0s for zero velocity
        this.invMass               = this.gpuCompute.createTexture(); // Inverse Mass Per Particle
        this.invRestVolumeAndColor = this.gpuCompute.createTexture(); // Inverse Volume and Graph Color Per Element
        this.elemToParticlesTable  = this.gpuCompute.createTexture(); // Maps from elems to the 4 tet vertex positions for the gather step
        this.elems0               = [this.gpuCompute.createTexture(), // Stores the original vertex positions per tet
                                     this.gpuCompute.createTexture(),
                                     this.gpuCompute.createTexture(),
                                     this.gpuCompute.createTexture()];
        this.quats0                = this.gpuCompute.createTexture();
        // Maps from vertices back to the elems gbuffer for the scatter step
        // There is more than one because a particle may have a bunch of elems sharing it
        this.numTables = 11; this.particleToElemVertsTable = [];
        for (let table = 0; table < this.numTables; table++) {
            this.particleToElemVertsTable.push(this.gpuCompute.createTexture());
        }

        // Fill in the above textures with the appropriate data
        this.tetIds = tetIds;
        this.initPhysics(this.physicsParams.density);

        // Allocate the variables that are computed at runtime
        this.pos        = this.gpuCompute.addVariable("texturePos"    , this.pos0);
        this.prevPos    = this.gpuCompute.addVariable("texturePrevPos", this.pos0);
        this.vel        = this.gpuCompute.addVariable("textureVel"    , this.vel0);
        // Create a multi target element texture; this temporarily stores the 4 vertex results of solveElem
        // (before the gather step where they are accumulated back into pos via the particleToElemVertsTable)
        this.elems      = this.gpuCompute.addVariable("textureElem"  , this.elems0, 4);
        this.quats      = this.gpuCompute.addVariable("textureQuat"  , this.quats0);

        // Set up the 7 GPGPU Passes for each substep of the FEM Simulation
        // 1. Copy prevPos to Pos 
        this.copyPrevPosPass = this.gpuCompute.addPass(this.prevPos, [this.pos], `
            out highp vec4 pc_fragColor;
            void main()	{
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                pc_fragColor = vec4( texture2D( texturePos, uv ).xyz, 0.0 );
            }`);

        // 2. XPBD Prediction/Integration
        this.xpbdIntegratePass = this.gpuCompute.addPass(this.pos, [this.vel, this.pos], `
            out highp vec4 pc_fragColor;
            uniform float dt;
            void main()	{
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                pc_fragColor = vec4( texture2D( texturePos, uv ).xyz +
                                   ( texture2D( textureVel, uv ).xyz * dt), 0.0 );
            }`);
        this.xpbdIntegratePass.material.uniforms['dt'] = { value: this.physicsParams.dt };
        this.xpbdIntegratePass.material.uniformsNeedUpdate = true;
        this.xpbdIntegratePass.material.needsUpdate = true;

        // 3. Gather into the Elements and Enforce Element Shape Constraint
        this.solveElemPass = this.gpuCompute.addPass(this.quats, [this.pos, this.elems, this.quats], `
            uniform sampler2D elemToParticlesTable;
            vec3[4] currentTets, lastRestTets;
            out vec4 quat;

            vec2 uvFromIndex(int index) {
                return vec2(  index % int(resolution.x),
                             (index / int(resolution.x))) / (resolution - 1.0); }

            // Begin Polar Decomposition Routine ---------------------------------------------------
            mat3 TransposeMult(vec3[4] tet1, vec3[4] tet2) {
                mat3 covariance = mat3(0.0);
                for (int k = 0; k < 4; k++) { //k is the column in this matrix
                    vec3 left  = tet1[k]; vec3 right = tet2[k];
                    covariance[0][0] += left[0] * right[0];
                    covariance[1][0] += left[1] * right[0];
                    covariance[2][0] += left[2] * right[0];
                    covariance[0][1] += left[0] * right[1];
                    covariance[1][1] += left[1] * right[1];
                    covariance[2][1] += left[2] * right[1];
                    covariance[0][2] += left[0] * right[2];
                    covariance[1][2] += left[1] * right[2];
                    covariance[2][2] += left[2] * right[2];
                }
                return covariance;
            }
            vec4 RotationToQuaternion(vec3 axis, float angle) {
                float half_angle = angle * 0.5; //angle * halfpi / 180.0;
                vec2 s = sin(vec2(half_angle, half_angle + 1.57));
                return vec4(axis * s.x, s.y);
            }
            vec3 Rotate(vec3 pos, vec4 quat) {
                return pos + 2.0 * cross(quat.xyz, cross(quat.xyz, pos) + quat.w * pos);
            }
            vec4 quat_mult(vec4 q1, vec4 q2) { 
                vec4 qr;
                qr.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
                qr.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
                qr.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
                qr.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
                return qr;
            }
            vec4 extractRotation(mat3 A, vec4 q) {
                for (int iter = 0; iter < 9; iter++) {
                    vec3 X = Rotate(vec3(1.0, 0.0, 0.0), q);
                    vec3 Y = Rotate(vec3(0.0, 1.0, 0.0), q);
                    vec3 Z = Rotate(vec3(0.0, 0.0, 1.0), q);

                    vec3 omega =  (cross(X, A[0]) +
                                   cross(Y, A[1]) +
                                   cross(Z, A[2])) *
                          (1.0 / abs(dot(X, A[0]) +
                                     dot(Y, A[1]) +
                                     dot(Z, A[2]) + 0.000000001));
                    float w = length(omega);
                    if (w < 0.000000001) { break; }
                    q = quat_mult(RotationToQuaternion(omega / w, w), q);
                }
                return q;
            }
            // End Polar Decomposition Routine ---------------------------------------------------

            void main()	{
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                // Grab the Relevant Element Variables

                // Gather this tetrahedron's 4 vertex positions
                vec4 tetIndices = texture2D( elemToParticlesTable, uv );
                currentTets[0]  = texture2D( texturePos, uvFromIndex(int(tetIndices.x))).xyz;
                currentTets[1]  = texture2D( texturePos, uvFromIndex(int(tetIndices.y))).xyz;
                currentTets[2]  = texture2D( texturePos, uvFromIndex(int(tetIndices.z))).xyz;
                currentTets[3]  = texture2D( texturePos, uvFromIndex(int(tetIndices.w))).xyz;

                // The Reference Rest Pose Positions (the the last output of this texture)
                // These are the same as the resting pose, but they're already pre-rotated
                // to a good approximation of the current pose
                lastRestTets[0] = texture2D( textureElem[0], uv ).xyz;
                lastRestTets[1] = texture2D( textureElem[1], uv ).xyz;
                lastRestTets[2] = texture2D( textureElem[2], uv ).xyz;
                lastRestTets[3] = texture2D( textureElem[3], uv ).xyz;

                // Get the centroids for the 
                vec3 curCentroid      = ( currentTets[0] +  currentTets[1] +  currentTets[2] +  currentTets[3]) * 0.25;
                vec3 lastRestCentroid = (lastRestTets[0] + lastRestTets[1] + lastRestTets[2] + lastRestTets[3]) * 0.25;

                // Center the Deformed Tetrahedron
                currentTets [0] -= curCentroid;
                currentTets [1] -= curCentroid;
                currentTets [2] -= curCentroid;
                currentTets [3] -= curCentroid;

                // Center the Undeformed Tetrahedron
                lastRestTets[0] -= lastRestCentroid;
                lastRestTets[1] -= lastRestCentroid;
                lastRestTets[2] -= lastRestCentroid;
                lastRestTets[3] -= lastRestCentroid;

                // Find the rotational offset between the two and rotate the undeformed tetrahedron by it
                vec4 rotation = extractRotation(TransposeMult(lastRestTets, currentTets), vec4(0.0, 0.0, 0.0, 1.0));

                // Write out the undeformed tetrahedron
                quat = normalize(quat_mult(rotation, texture2D( textureQuat, uv ))); // Keep track of the current Quaternion for normals
            }`);
        this.solveElemPass.material.uniforms['elemToParticlesTable'] = { value: this.elemToParticlesTable };
        this.solveElemPass.material.uniformsNeedUpdate = true;
        this.solveElemPass.material.needsUpdate = true;

        // 4. Gather into the Elements and use the solved rotations to rotate the rest pose
        this.gatherElemPass = this.gpuCompute.addPass(this.elems, [this.pos, this.elems, this.quats], `
            uniform float dt;
            uniform sampler2D elemToParticlesTable, invRestVolume, invMassTex;
            vec3[4] currentTets, lastRestTets, restTets;
            float[4] invMass;

            layout(location = 0) out vec4 vert1;
            layout(location = 1) out vec4 vert2;
            layout(location = 2) out vec4 vert3;
            layout(location = 3) out vec4 vert4;
            //layout(location = 4) out vec4 quat;

            vec2 uvFromIndex(int index) {
                return vec2(  index % int(resolution.x),
                             (index / int(resolution.x))) / (resolution - 1.0); }

            vec3 Rotate(vec3 pos, vec4 quat) {
                return pos + 2.0 * cross(quat.xyz, cross(quat.xyz, pos) + quat.w * pos);
            }
            vec4 quat_conj(vec4 q) { return normalize(vec4(-q.x, -q.y, -q.z, q.w)); }
            vec4 quat_mult(vec4 q1, vec4 q2) { 
                vec4 qr;
                qr.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
                qr.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
                qr.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
                qr.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
                return qr;
            }

            void main()	{
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                // Grab the Relevant Element Variables
                float invVolume  = 1.0/texture2D( invRestVolume, uv ).x;

                // Gather this tetrahedron's 4 vertex positions
                vec4 tetIndices = texture2D( elemToParticlesTable, uv );
                currentTets[0] = texture2D( texturePos, uvFromIndex(int(tetIndices.x))).xyz;
                currentTets[1] = texture2D( texturePos, uvFromIndex(int(tetIndices.y))).xyz;
                currentTets[2] = texture2D( texturePos, uvFromIndex(int(tetIndices.z))).xyz;
                currentTets[3] = texture2D( texturePos, uvFromIndex(int(tetIndices.w))).xyz;

                // The Reference Rest Pose Positions (the the last output of this texture)
                // These are the same as the resting pose, but they're already pre-rotated
                // to a good approximation of the current pose
                lastRestTets[0]    = texture2D( textureElem[0], uv ).xyz;
                lastRestTets[1]    = texture2D( textureElem[1], uv ).xyz;
                lastRestTets[2]    = texture2D( textureElem[2], uv ).xyz;
                lastRestTets[3]    = texture2D( textureElem[3], uv ).xyz;

                vec4 latestQuat   = texture2D( textureQuat, uv );
                vec4 previousQuat = texture2D( prev_textureQuat, uv );
                vec4 relativeQuat = normalize(quat_mult(latestQuat, quat_conj(previousQuat)));

                // Unused: The inverse mass of each vertex; I'm weighting positional
                // updates by the the inverse volume instead because it looks better(?)
                //invMass[0] = texture2D( invMassTex, uvFromIndex(int(tetIndices.x))).x;
                //invMass[1] = texture2D( invMassTex, uvFromIndex(int(tetIndices.y))).x;
                //invMass[2] = texture2D( invMassTex, uvFromIndex(int(tetIndices.z))).x;
                //invMass[3] = texture2D( invMassTex, uvFromIndex(int(tetIndices.w))).x;

                // Get the default vert0 centered resting pose and the centroids
                vec3 curCentroid      = ( currentTets[0] +  currentTets[1] +  currentTets[2] +  currentTets[3]) * 0.25;
                vec3 lastRestCentroid = (lastRestTets[0] + lastRestTets[1] + lastRestTets[2] + lastRestTets[3]) * 0.25;

                // Rotate the undeformed tetrahedron by the deformed's rotation
                lastRestTets[0] = Rotate(lastRestTets[0] - lastRestCentroid, relativeQuat) + curCentroid;
                lastRestTets[1] = Rotate(lastRestTets[1] - lastRestCentroid, relativeQuat) + curCentroid;
                lastRestTets[2] = Rotate(lastRestTets[2] - lastRestCentroid, relativeQuat) + curCentroid;
                lastRestTets[3] = Rotate(lastRestTets[3] - lastRestCentroid, relativeQuat) + curCentroid;

                // Write out the rotated undeformed tetrahedron
                vert1 = vec4(lastRestTets[0], invVolume);
                vert2 = vec4(lastRestTets[1], invVolume);
                vert3 = vec4(lastRestTets[2], invVolume);
                vert4 = vec4(lastRestTets[3], invVolume);
            }`);
        this.gatherElemPass.material.uniforms['dt'                  ] = { value: this.physicsParams.dt };
        this.gatherElemPass.material.uniforms['elemToParticlesTable'] = { value: this.elemToParticlesTable };
        this.gatherElemPass.material.uniforms['invRestVolume'       ] = { value: this.invRestVolumeAndColor };
        this.gatherElemPass.material.uniforms['invMassTex'          ] = { value: this.invMass      };
        this.gatherElemPass.material.uniformsNeedUpdate = true;
        this.gatherElemPass.material.needsUpdate = true;

        // 5. Gather the particles back from the elements
        let unrollLoop = '';
        for (let t = 2; t < this.numTables; t++) {
            unrollLoop += '} else if (index == '+t+') { result  = texture2D( particleToElemVertsTable['+t+'], uv );\n';
        }
        this.applyElemPass = this.gpuCompute.addPass(this.pos, [this.elems, this.pos],
        `
            out highp vec4 pc_fragColor;
            uniform float dt;
            uniform sampler2D particleToElemVertsTable[`+this.numTables+`];

            vec2 uvFromIndex(int index) {
                return vec2(  index % int(resolution.x),
                             (index / int(resolution.x))) / (resolution - 1.0); }

            // Need to use if-else blocks because GLSL doesn't support dynamic indexing into sampler2d Arrays
            vec4 textureElemSample(int index, vec2 uv) {
                 vec4 result = texture2D( textureElem[0], uv );
                        if (index == 1) { result  = texture2D( textureElem[1], uv );
                 } else if (index == 2) { result  = texture2D( textureElem[2], uv );
                 } else if (index == 3) { result  = texture2D( textureElem[3], uv ); }
                 return result;
            }
            vec4 particleToElemVertsTableSample(int index, vec2 uv) {
                vec4 result = texture2D( particleToElemVertsTable[0], uv );
                       if (index == 1) { result  = texture2D( particleToElemVertsTable[1], uv );\n`
                       + unrollLoop + `}
                return result;
            }
            void main()	{
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec3 pos = texture2D( texturePos, uv ).xyz;
                vec3 sumVertex = vec3(0.0); int breaker = 0; float sum = 0.0;
                for(int tableId = 0; tableId < `+this.numTables+`; tableId++) {
                    vec4 vertIndices = particleToElemVertsTableSample(tableId, uv);
                    for(int component = 0; component < 4; component++) {
                        if(vertIndices[component] > -1.0) {
                            int elemId = int(vertIndices[component]) / 4;
                            int vertId = int(vertIndices[component]) % 4;
                            vec4 elemVerts = textureElemSample(vertId, uvFromIndex(elemId));
                            sumVertex += elemVerts.xyz * elemVerts.w;
                            sum += elemVerts.w; // Weight by the volume of each contributing element
                        } else { breaker = 1; break; }
                    }
                    if(breaker == 1) { break; }
                }
                pc_fragColor = vec4(sumVertex / sum, 0.0); // Output the average vertex position
            }`);
        this.applyElemPass.material.uniforms['particleToElemVertsTable'] = { value: this.particleToElemVertsTable };
        this.applyElemPass.material.uniformsNeedUpdate = true;
        this.applyElemPass.material.needsUpdate = true;
  
        // 5. Enforce Collisions and Grab Forces
        this.collisionPass = this.gpuCompute.addPass(this.pos, [this.pos, this.prevPos],  `
            out highp vec4 pc_fragColor;
            uniform float dt, friction, grabId;
            uniform vec3 grabPos;

            vec2 uvFromIndex(int index) {
                return vec2(  index % int(resolution.x),
                             (index / int(resolution.x))) / (resolution - 1.0); }

            // This isn't quite correct, but it's close enough for now
            int indexFromUV(vec2 uv) {
                return int(uv.x * (resolution.x-1.0)) +
                       int(uv.y * (resolution.x-1.0) * (resolution.y)); }

            void main()	{
                vec2 uv  = gl_FragCoord.xy / resolution.xy;
                vec3 pos = texture2D( texturePos    , uv ).xyz;

                // Execute a possible Grab
                if(float(indexFromUV(uv)) == grabId) { pos = grabPos; }
                // Clamp the Domain
                pos = clamp(pos, vec3(-2.5, -1.0, -2.5), vec3(2.5, 10.0, 2.5));
                // Collide with the floor using "simple friction"
                if(pos.y < 0.0) {
                    pos.y = 0.0;
                    vec3 F = texture2D( texturePrevPos, uv ).xyz - pos;
                    pos.xz += F.xz * min(1.0, dt * friction);
                }
                pc_fragColor = vec4(pos, 0.0 );
            }`);
        this.collisionPass.material.uniforms['dt'      ] = { value: this.physicsParams.dt };
        this.collisionPass.material.uniforms['friction'] = { value: this.physicsParams.friction };
        this.collisionPass.material.uniforms['grabId'  ] = { value: -1 };
        this.collisionPass.material.uniforms['grabPos' ] = { value: new THREE.Vector3(0,0,0) }
        this.collisionPass.material.uniformsNeedUpdate = true;
        this.collisionPass.material.needsUpdate = true;

        //// 6. XPBD Velocity + Gravity Update
        this.xpbdVelocityPass = this.gpuCompute.addPass(this.vel, [this.pos, this.prevPos], `
            out highp vec4 pc_fragColor;
            uniform float dt, gravity;
            void main()	{
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                pc_fragColor = vec4((( texture2D( texturePos    , uv ).xyz -
                                       texture2D( texturePrevPos, uv ).xyz) / dt )
                                    + (vec3(0, gravity, 0) * dt ), 0.0 );
            }`);
        this.xpbdVelocityPass.material.uniforms['dt'     ] = { value: this.physicsParams.dt };
        this.xpbdVelocityPass.material.uniforms['gravity'] = { value: this.physicsParams.gravity };
        this.xpbdVelocityPass.material.uniformsNeedUpdate = true;
        this.xpbdVelocityPass.material.needsUpdate = true;

        // Initialize the whole pipeline
        const error = this.gpuCompute.init();
        if ( error !== null ) { console.error( error ); }

        // Show debug texture
        if (!this.labelMesh) {
            this.labelMaterial = new THREE.MeshBasicMaterial( 
                { map: this.gpuCompute.getCurrentRenderTarget(this.elems).texture[0], side: THREE.DoubleSide });
            this.labelPlane = new THREE.PlaneGeometry(1, 1);
            this.labelMesh = new THREE.Mesh(this.labelPlane, this.labelMaterial);
            this.labelMesh.position.set(0, 2.5, 0);
            world.scene.add(this.labelMesh);
        }


        // visual edge mesh
        this.edgeMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
        this.edgeMaterial.onBeforeCompile = (shader) => {
            // Vertex Shader: Set Vertex Positions to the texture positions
            let bodyStart = shader.vertexShader.indexOf( 'void main() {' );
            shader.vertexShader =
                 shader.vertexShader.slice(0, bodyStart) +
                 `uniform sampler2D texturePos;
                 vec4 getValueByIndexFromTexture(sampler2D tex, int index) {
                     ivec2 texSize = textureSize(tex, 0);
                     return texelFetch(tex, ivec2(
                         index % texSize.x,
                         index / texSize.x), 0);
                 }\n` +
                 shader.vertexShader.slice(bodyStart - 1, - 1) +
                 `mvPosition = getValueByIndexFromTexture(texturePos, gl_VertexID);
                 mvPosition = modelViewMatrix * vec4( mvPosition.xyz, 1.0 );
                 gl_Position = projectionMatrix * mvPosition;
            }`;
            
            shader.uniforms.texturePos = { value: this.gpuCompute.getCurrentRenderTarget(this.pos) } //this.pos.texture };
        };
        this.geometry = new THREE.BufferGeometry();
        this.geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        this.geometry.setIndex(tetEdgeIds);
        this.edgeMesh = new THREE.LineSegments(this.geometry, this.edgeMaterial);
        this.edgeMesh.userData = this;    // for raycasting
        this.edgeMesh.layers.enable(1);
        this.edgeMesh.visible = true;

        // Vertex Shader: Set Vertex Positions to the texture positions
        let vertShaderInit = `attribute vec4 tetWeights;
        uniform sampler2D texturePos, elemToParticlesTable, textureQuat;
        vec4 getValueByIndexFromTexture(sampler2D tex, int index) {
            ivec2 texSize = textureSize(tex, 0); return texelFetch(tex, ivec2( index % texSize.x, index / texSize.x), 0); }
        vec3 Rotate(vec3 pos, vec4 quat) { return pos + 2.0 * cross(quat.xyz, cross(quat.xyz, pos) + quat.w * pos); }\n`;
        let vertShaderMain = `vec4 tetQuaternion = getValueByIndexFromTexture(textureQuat, int(tetWeights.x));
            vec4 vertIndices = getValueByIndexFromTexture(elemToParticlesTable, int(tetWeights.x));
            float lastTetWeight = 1.0 - (tetWeights.y + tetWeights.z + tetWeights.w);
            vec4 vertPosition = vec4(((getValueByIndexFromTexture(texturePos, int(vertIndices.x)) * tetWeights.y) + 
                                       (getValueByIndexFromTexture(texturePos, int(vertIndices.y)) * tetWeights.z) + 
                                       (getValueByIndexFromTexture(texturePos, int(vertIndices.z)) * tetWeights.w) + 
                                       (getValueByIndexFromTexture(texturePos, int(vertIndices.w)) * lastTetWeight)).xyz, 1.0);
            mvPosition = modelViewMatrix * vertPosition;
            gl_Position = projectionMatrix * mvPosition;
        }`;
        let vertShaderMainColor = vertShaderMain.slice(0, -1) + `
            transformedNormal = Rotate(objectNormal, tetQuaternion);
            vNormal = normalMatrix * transformedNormal;
            worldPosition = modelMatrix * vertPosition; // This line and below might not be needed...
            shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
            shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ 0 ].shadowNormalBias, 0 );
            vDirectionalShadowCoord[ 0 ] = directionalShadowMatrix[ 0 ] * shadowWorldPosition;
            shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * spotLightShadows[ 0 ].shadowNormalBias, 0 );
            vSpotShadowCoord[ 0 ] = spotShadowMatrix[ 0 ] * shadowWorldPosition;
        }`;

        // visual embedded mesh
        //this.visVerts = visVerts; // <- These are barycentric weights inside each tetrahedra
        //this.numVisVerts = visVerts.length / 4;
        //this.geometry = new THREE.BufferGeometry();
        //this.geometry.setAttribute('position', new THREE.BufferAttribute(
        //    new Float32Array(3 * this.numVisVerts), 3));
        //this.geometry.setAttribute('tetWeights', new THREE.BufferAttribute(this.visVerts, 4));
        //this.geometry.setIndex(visTriIds);
        //this.visMesh = new THREE.Mesh(this.geometry, visMaterial);
        //this.visMesh.castShadow = true;
        //this.visMesh.userData = this;    // for raycasting
        //this.visMesh.layers.enable(1);

        //this.visMesh.material.onBeforeCompile = (shader) => {
        //    let bodyStart = shader.vertexShader.indexOf( 'void main() {' );
        //    shader.vertexShader =
        //         shader.vertexShader.slice(0, bodyStart) + vertShaderInit +
        //         shader.vertexShader.slice(bodyStart - 1, - 1) + vertShaderMainColor;
        //    shader.uniforms.texturePos = { value: this.gpuCompute.getCurrentRenderTarget(this.pos) }
        //    shader.uniforms.textureQuat = { value: this.gpuCompute.getCurrentRenderTarget(this.quats) }
        //    shader.uniforms.elemToParticlesTable = { value: this.elemToParticlesTable }
        //};
        //this.visMesh.customDepthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
        //this.visMesh.customDepthMaterial.onBeforeCompile = (shader) => {
        //    let bodyStart = shader.vertexShader.indexOf( 'void main() {' );
        //    shader.vertexShader =
        //         shader.vertexShader.slice(0, bodyStart) + vertShaderInit +
        //         shader.vertexShader.slice(bodyStart - 1, - 1) + vertShaderMain.slice(0, -1) + "vHighPrecisionZW = gl_Position.zw;}";
        //    shader.uniforms.texturePos  = { value: this.gpuCompute.getCurrentRenderTarget(this.pos) }
        //    shader.uniforms.textureQuat = { value: this.gpuCompute.getCurrentRenderTarget(this.quats) }
        //    shader.uniforms.elemToParticlesTable = { value: this.elemToParticlesTable }
        //};

        //this.geometry.computeVertexNormals();
        //this.updateVisMesh();
    }

    initPhysics(density) {
        // and fill in here the texture data from vertices

        // Initialize velocities and masses to 0
        for (let i = 0; i < this.vel0.image.data.length; i++){
            this.vel0                       .image.data[i] = 0.0;
            this.invMass                    .image.data[i] = 0.0;
            for (let table = 0; table < this.numTables; table++) {
                this.particleToElemVertsTable[table].image.data[i] = -1.0;
            }
        }

        // Initialize the positions of the vertices
        //this.pos0                     // Set to vertices
        let posIndex = 0;
        for (let i = 0; i < this.pos0.image.data.length; i += 4){
            this.pos0.image.data[i  ] = this.inputPos[posIndex++];
            this.pos0.image.data[i+1] = this.inputPos[posIndex++];
            this.pos0.image.data[i+2] = this.inputPos[posIndex++];
        }

        this.oldrestingPose = new Float32Array(9 * this.numElems);
        let biggestT = 0;
        for (let i = 0; i < this.numElems; i++) {
            let id0 = this.tetIds[(4 * i)    ];
            let id1 = this.tetIds[(4 * i) + 1];
            let id2 = this.tetIds[(4 * i) + 2];
            let id3 = this.tetIds[(4 * i) + 3];

            // Initialize elems to the resting position of each tet
            this.elems0[0].image.data[(4 * i)    ] = this.inputPos[(id0 * 3) + 0];
            this.elems0[0].image.data[(4 * i) + 1] = this.inputPos[(id0 * 3) + 1];
            this.elems0[0].image.data[(4 * i) + 2] = this.inputPos[(id0 * 3) + 2];
            this.elems0[1].image.data[(4 * i)    ] = this.inputPos[(id1 * 3) + 0];
            this.elems0[1].image.data[(4 * i) + 1] = this.inputPos[(id1 * 3) + 1];
            this.elems0[1].image.data[(4 * i) + 2] = this.inputPos[(id1 * 3) + 2];
            this.elems0[2].image.data[(4 * i)    ] = this.inputPos[(id2 * 3) + 0];
            this.elems0[2].image.data[(4 * i) + 1] = this.inputPos[(id2 * 3) + 1];
            this.elems0[2].image.data[(4 * i) + 2] = this.inputPos[(id2 * 3) + 2];
            this.elems0[3].image.data[(4 * i)    ] = this.inputPos[(id3 * 3) + 0];
            this.elems0[3].image.data[(4 * i) + 1] = this.inputPos[(id3 * 3) + 1];
            this.elems0[3].image.data[(4 * i) + 2] = this.inputPos[(id3 * 3) + 2];
            // Initialize quaternions
            this.quats0.image.data[(4 * i)    ] = 0.0; // Quaternion
            this.quats0.image.data[(4 * i) + 1] = 0.0;
            this.quats0.image.data[(4 * i) + 2] = 0.0;
            this.quats0.image.data[(4 * i) + 3] = 1.0;

            // The forward table
            this.elemToParticlesTable.image.data[(4 * i)    ] = id0;
            this.elemToParticlesTable.image.data[(4 * i) + 1] = id1;
            this.elemToParticlesTable.image.data[(4 * i) + 2] = id2;
            this.elemToParticlesTable.image.data[(4 * i) + 3] = id3;

            // Construct the particleToElemVertsTables
            // Encode this particle's vert index within the elemTexture[4]
            // int elemId = int(floor(value / 4.0));
            // int vertId = int(  mod(value , 4.0));
            let ids = [id0, id1, id2, id3];
            for (let id = 0; id < 4; id++) {
                let assigned = false;
                for (let t = 0; t < this.particleToElemVertsTable.length; t++) {
                    for (let c = 0; c < 4; c++) {
                        if (this.particleToElemVertsTable[t].image.data[(4 * ids[id]) + c] <= 0.0) {
                            this.particleToElemVertsTable[t].image.data[(4 * ids[id]) + c] = (4.0 * i) + id;
                            biggestT = Math.max(biggestT, t);
                            if (t == this.numTables-1) { console.log(ids[id], (4 * ids[id]) + c);}
                            assigned = true; break;
                        }
                    }
                    if (assigned) break;
                }
            }

            this.vecSetDiff(this.oldrestingPose, 3 * i    , this.inputPos, id1, this.inputPos, id0);
            this.vecSetDiff(this.oldrestingPose, 3 * i + 1, this.inputPos, id2, this.inputPos, id0);
            this.vecSetDiff(this.oldrestingPose, 3 * i + 2, this.inputPos, id3, this.inputPos, id0);
            let V = this.matGetDeterminant(this.oldrestingPose, i) / 6.0;

            let pm = V / 4.0 * density;
            this.invMass      .image.data[id0 * 4] += pm;
            this.invMass      .image.data[id1 * 4] += pm;
            this.invMass      .image.data[id2 * 4] += pm;
            this.invMass      .image.data[id3 * 4] += pm;
            this.invRestVolumeAndColor.image.data[(i * 4) + 0] = 1.0 / V; // Set InvMass
            this.invRestVolumeAndColor.image.data[(i * 4) + 1] = -1.0;    // Mark Color as Undefined
        }

        for (let i = 0; i < this.invMass.image.data.length; i++) {
            if (this.invMass.image.data[i] != 0.0) { this.invMass.image.data[i] = 1.0 / this.invMass.image.data[i]; }
        }

        console.log(biggestT);
    }

    simulate(dt, physicsParams) {
        physicsParams.dt = dt;

        // First, upload the new shader uniforms to the GPU
        if (this.xpbdIntegratePass) {
            this.xpbdIntegratePass.material.uniforms['dt'] = { value: physicsParams.dt };
            this.xpbdIntegratePass.material.uniformsNeedUpdate = true;
            this.xpbdIntegratePass.material.needsUpdate = true;
        }
        if (this.gatherElemPass) {
            this.gatherElemPass.material.uniforms['dt'] = { value: physicsParams.dt };
            this.gatherElemPass.material.uniformsNeedUpdate = true;
            this.gatherElemPass.material.needsUpdate = true;
        }
        if (this.collisionPass) {
            this.collisionPass.material.uniforms['dt'] = { value: physicsParams.dt };
            this.collisionPass.material.uniforms['friction'] = { value: physicsParams.friction };
            this.collisionPass.material.uniforms['grabId' ] = { value: this.grabId };
            this.collisionPass.material.uniforms['grabPos'] = { value: new THREE.Vector3(this.grabPos[0], this.grabPos[1], this.grabPos[2]) };
            this.collisionPass.material.uniformsNeedUpdate = true;
            this.collisionPass.material.needsUpdate = true;
        }
        if (this.xpbdVelocityPass) {
            this.xpbdVelocityPass.material.uniforms['dt'] = { value: physicsParams.dt };
            this.xpbdVelocityPass.material.uniforms['gravity'] = { value: physicsParams.gravity };
            this.xpbdVelocityPass.material.uniformsNeedUpdate = true;
            this.xpbdVelocityPass.material.needsUpdate = true;
        }

        // Run a substep!
        this.gpuCompute.compute();
    }

    endFrame() {
        //this.updateEdgeMesh();
        //this.updateVisMesh();
        this.edgeMesh.visible = this.physicsParams.ShowTetMesh;
    }

    readToCPU(gpuComputeVariable, buffer) {
        this.renderer.readRenderTargetPixels(
            this.gpuCompute.getCurrentRenderTarget(gpuComputeVariable),
            0, 0, this.texDim, this.texDim, buffer);
    }

    updateEdgeMesh() {
        // Read tetrahedron positions back from the GPU
        this.readToCPU(this.pos, this.tetPositionsArray);

        let positionIndex = 0;
        const positions = this.edgeMesh.geometry.attributes.position.array;
        for (let i = 0; i < this.tetPositionsArray.length; i+=4) {
            positions[positionIndex++] = this.tetPositionsArray[i  ];
            positions[positionIndex++] = this.tetPositionsArray[i+1];
            positions[positionIndex++] = this.tetPositionsArray[i+2];
        }
        this.edgeMesh.geometry.attributes.position.needsUpdate = true;
        this.edgeMesh.geometry.computeBoundingSphere();
    }

    updateVisMesh() {
        const positions = this.visMesh.geometry.attributes.position.array;
        const tetpositions = this.edgeMesh.geometry.attributes.position.array;
        let nr = 0;
        for (let i = 0; i < this.numVisVerts; i++) {
            let tetNr = this.visVerts[nr++] * 4;
            let b0 = this.visVerts[nr++];
            let b1 = this.visVerts[nr++];
            let b2 = this.visVerts[nr++];
            let b3 = 1.0 - b0 - b1 - b2;
            this.vecSetZero(positions, i);
            this.vecAdd(positions, i, tetpositions, this.tetIds[tetNr++], b0);
            this.vecAdd(positions, i, tetpositions, this.tetIds[tetNr++], b1);
            this.vecAdd(positions, i, tetpositions, this.tetIds[tetNr++], b2);
            this.vecAdd(positions, i, tetpositions, this.tetIds[tetNr++], b3);
        }
        // This is colossally slow; move to GPU soon...
        if (this.physicsParams.computeNormals) { this.visMesh.geometry.computeVertexNormals(); }
        this.visMesh.geometry.attributes.position.needsUpdate = true;
        this.visMesh.geometry.computeBoundingSphere();
    }

    startGrab(pos) {
        let p = [pos.x, pos.y, pos.z];
        let minD2 = Number.MAX_VALUE;
        this.grabId = -1;
        let particles = this.edgeMesh.geometry.attributes.position.array;
        for (let i = 0; i < this.numParticles; i++) {
            let d2 = this.vecDistSquared(p, 0, particles, i);
            if (d2 < minD2) {
                minD2 = d2;
                this.grabId = i;
            }
        }
        this.vecCopy(this.grabPos, 0, p, 0);
    }

    moveGrabbed(pos) {
        let p = [pos.x, pos.y, pos.z];
        this.vecCopy(this.grabPos, 0, p, 0);
    }

    endGrab() { this.grabId = -1; }

    // ----- vector math -------------------------------------------------------------

    vecSetZero(a, anr) {
        anr *= 3;
        a[anr++] = 0.0;
        a[anr++] = 0.0;
        a[anr] = 0.0;
    }

    vecCopy(a, anr, b, bnr) {
        anr *= 3; bnr *= 3;
        a[anr++] = b[bnr++];
        a[anr++] = b[bnr++];
        a[anr] = b[bnr];
    }
            
    vecAdd(a, anr, b, bnr, scale = 1.0) {
        anr *= 3; bnr *= 3;
        a[anr++] += b[bnr++] * scale;
        a[anr++] += b[bnr++] * scale;
        a[anr  ] += b[bnr  ] * scale;
    }

    vecSetDiff(dst, dnr, a, anr, b, bnr, scale = 1.0) {
        dnr *= 3; anr *= 3; bnr *= 3;
        dst[dnr++] = (a[anr++] - b[bnr++]) * scale;
        dst[dnr++] = (a[anr++] - b[bnr++]) * scale;
        dst[dnr  ] = (a[anr  ] - b[bnr  ]) * scale;
    }

    vecDistSquared(a, anr, b, bnr) {
        anr *= 3; bnr *= 3;
        let a0 = a[anr] - b[bnr], a1 = a[anr + 1] - b[bnr + 1], a2 = a[anr + 2] - b[bnr + 2];
        return a0 * a0 + a1 * a1 + a2 * a2;
    }

    // ----- matrix math ----------------------------------

    matGetDeterminant(A, anr) {
        anr *= 9;
        let a11 = A[anr + 0], a12 = A[anr + 3], a13 = A[anr + 6];
        let a21 = A[anr + 1], a22 = A[anr + 4], a23 = A[anr + 7];
        let a31 = A[anr + 2], a32 = A[anr + 5], a33 = A[anr + 8];
        return a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a13 * a22 * a31 - a12 * a21 * a33 - a11 * a23 * a32;
    }

}

export class GPUGrabber {
    constructor(scene, renderer, camera, container, controls) {
        this.scene = scene;
        this.renderer = renderer;
        this.camera = camera;
        this.mousePos = new THREE.Vector2();
        this.raycaster = new THREE.Raycaster();
        this.raycaster.layers.set(1);
        //					this.raycaster.params.Mesh.threshold = 3;
        this.raycaster.params.Line.threshold = 0.1;
        this.grabDistance = 0.0;
        this.active = false;
        this.physicsObject = null;
        this.controls = controls;

        container.addEventListener( 'pointerdown', this.onPointer.bind(this), true );
        container.addEventListener( 'pointermove', this.onPointer.bind(this), true );
        container.addEventListener( 'pointerup'  , this.onPointer.bind(this), true );
        container.addEventListener( 'pointerout' , this.onPointer.bind(this), true );
    }
    updateRaycaster(x, y) {
        var rect = this.renderer.domElement.getBoundingClientRect();
        this.mousePos.x = ((x - rect.left) / rect.width) * 2 - 1;
        this.mousePos.y = -((y - rect.top) / rect.height) * 2 + 1;
        this.raycaster.setFromCamera(this.mousePos, this.camera);
    }
    start(x, y) {
        // Reads the Mesh Geometry Back from the GPU
        for (let child in this.scene.children) {
            if (this.scene.children[child].userData instanceof SoftBodyGPU) {
                this.scene.children[child].userData.updateEdgeMesh();
                this.scene.children[child].userData.endFrame();
            }
        }
        this.physicsObject = null;
        this.updateRaycaster(x, y);
        var intersects = this.raycaster.intersectObjects(this.scene.children);
        if (intersects.length > 0) {
            var obj = intersects[0].object.userData;
            if (obj instanceof SoftBodyGPU) {
                this.physicsObject = obj;
                this.grabDistance = intersects[0].distance;
                let hit = this.raycaster.ray.origin.clone();
                hit.addScaledVector(this.raycaster.ray.direction, this.grabDistance);
                this.physicsObject.startGrab(hit);
                this.active = true;
                this.controls.enabled = false;
            }
        }
    }
    move(x, y) {
        if (this.active) {
            this.updateRaycaster(x, y);
            let hit = this.raycaster.ray.origin.clone();
            hit.addScaledVector(this.raycaster.ray.direction, this.grabDistance);
            if (this.physicsObject != null)
                this.physicsObject.moveGrabbed(hit);
        }
    }
    end(evt) {
        if (this.active) {
            if (this.physicsObject != null) {
                this.physicsObject.endGrab();
                this.physicsObject = null;
            }
            this.active = false;
            this.controls.enabled = true;
            //this.controls.onPointerUp(evt);
        }
    }

    onPointer(evt) {
        //evt.preventDefault();
        if (evt.type == "pointerdown") {
            this.start(evt.clientX, evt.clientY);
            this.mouseDown = true;
        } else if (evt.type == "pointermove" && this.mouseDown) {
            if (this.active)
                this.move(evt.clientX, evt.clientY);
        } else if (evt.type == "pointerup" /*|| evt.type == "pointerout"*/) {
            this.controls.enabled = true;
            this.end(evt);
            this.mouseDown = false;
        }
    }
}
