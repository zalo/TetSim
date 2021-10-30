import * as THREE from '../node_modules/three/build/three.module.js';
import { MultiTargetGPUComputationRenderer } from './MultiTargetGPUComputationRenderer.js';

export class SoftBodyGPU {
    constructor(vertices, tetIds, tetEdgeIds, physicsParams,
        visVerts, visTriIds, visMaterial, world) {
        this.physicsParams = physicsParams; // Set the Uniforms using these later
        /** @type {THREE.WebGLRenderer} */ 
        this.renderer = world.renderer;

        this.numParticles      = vertices.length / 3;
        this.numElems          = tetIds.length / 4;
        console.log(this.numParticles, this.numElems);
        this.texDim            = Math.ceil(Math.sqrt(this.numElems));
        this.tetPositionsArray = new Float32Array(this.texDim * this.texDim * 4); // Used for GPU Readback
        this.elemPositionsArray = new Float32Array(this.texDim * this.texDim * 4); // Used for GPU Readback
        this.inputPos          = vertices.slice(0);

        // Initialize the General Purpose GPU Computation Renderer
        this.gpuCompute = new MultiTargetGPUComputationRenderer(this.texDim, this.texDim, this.renderer)

        // Allocate static textures that are used to initialize the simulation
        this.pos0                  = this.gpuCompute.createTexture(); // Set to vertices
        this.vel0                  = this.gpuCompute.createTexture(); // Leave as 0s for zero velocity
        this.invMass               = this.gpuCompute.createTexture(); // Inverse Mass Per Particle
        this.invRestVolumeAndColor = this.gpuCompute.createTexture(); // Inverse Volume and Graph Color Per Element
        this.invRestPoseX          = this.gpuCompute.createTexture(); // Split the 3x3 restpose into 3 textures
        this.invRestPoseY          = this.gpuCompute.createTexture(); // Split the 3x3 restpose into 3 textures
        this.invRestPoseZ          = this.gpuCompute.createTexture(); // Split the 3x3 restpose into 3 textures
        this.elemToParticlesTable  = this.gpuCompute.createTexture(); // Maps from elems to the 4 tet vertex positions for the gather step
        this.particleToElemVertsTable = [this.gpuCompute.createTexture(), // Maps from vertices back to the elems gbuffer for the scatter step
                                     this.gpuCompute.createTexture(), // There is more than one because a particle may have a bunch of elems sharing it
                                     this.gpuCompute.createTexture(),
                                     this.gpuCompute.createTexture(),
                                     this.gpuCompute.createTexture(),
                                     this.gpuCompute.createTexture(),
                                     this.gpuCompute.createTexture(),
                                     this.gpuCompute.createTexture(),
                                     this.gpuCompute.createTexture()];

        // Fill in the above textures with the appropriate data
        this.tetIds = tetIds;
        this.initPhysics(this.physicsParams.density);

        // Allocate the variables that are computed at runtime
        this.pos     = this.gpuCompute.addVariable("texturePos"    , this.pos0);
        this.prevPos = this.gpuCompute.addVariable("texturePrevPos", this.pos0);
        this.vel     = this.gpuCompute.addVariable("textureVel"    , this.vel0);
        // Create a multi target element texture; this temporarily stores the 4 vertex results of solveElem
        // (before the gather step where they are accumulated back into pos via the particleToElemVertsTable)
        this.elems   = this.gpuCompute.addVariable("textureElem"   , this.vel0, 4);
        this.debugElem1 = this.gpuCompute.addVariable("textureDebug1", this.vel0);
        this.debugElem2 = this.gpuCompute.addVariable("textureDebug2", this.vel0);
        this.debugElem3 = this.gpuCompute.addVariable("textureDebug3", this.vel0);
        this.debugElem4 = this.gpuCompute.addVariable("textureDebug4", this.vel0);

        // Set up the 6 GPGPU Passes for each substep of the FEM Simulation
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

        // Steps 3 and 4 are going to be the toughest
        // Need to take special care when precomputing 
        // ElemToParticlesTable, particleToElemVertsTable, InvMassAndInvRestVolume, and InvRestPose[3]
        // Ensure the Uniforms are set (Grab Point, Collision Domain, Gravity, Compliance, etc.)

        // 3. Gather into the Elements and Enforce Element Shape Constraint
        this.solveElemPass = this.gpuCompute.addPass(this.elems, [this.pos], `
            uniform float dt;
            uniform sampler2D elemToParticlesTable, invRestVolume,
                    invRestPoseX, invRestPoseY, invRestPoseZ, invMassTex;
            vec3[4] g, id;
            float[4] invMass;

            layout(location = 0) out vec4 vert1;
            layout(location = 1) out vec4 vert2;
            layout(location = 2) out vec4 vert3;
            layout(location = 3) out vec4 vert4;

            vec2 uvFromIndex(int index) {
                return vec2(  index % int(resolution.x),
                             (index / int(resolution.x))) / (resolution - 1.0); }

            void applyToElem(float C, float compliance, float dt, float invRestVolume, inout vec3[4] id, in float[4] invMass) {
                if (C == 0.0)
                    return;

                g[0] = vec3(0);
                g[0] -= g[1];
                g[0] -= g[2];
                g[0] -= g[3];

                float w = 0.0;
                for (int i = 0; i < 4; i++) { w += dot(g[i], g[i]) * invMass[i]; }

                if (w == 0.0) { return; }
                float alpha = compliance / dt / dt * invRestVolume;
                float dlambda = -C / (w + alpha);

                for (int i = 0; i < 4; i++) {
                    id[i].xyz += g[i] * dlambda * invMass[i];
                }
            }

            void solveElement(in mat3 invRestPose, in float invRestVolume, inout vec3[4] id, in float[4] invMass) {
                float C = 0.0;
                float devCompliance = 1.0/100000.0;
                float volCompliance = 0.0;

                // tr(F) = 3
                //P     = new Float32Array(9);
                //F     = new Float32Array(9);
                //dF    = new Float32Array(9);
                //grads = new Float32Array(12); // vec3[4]
                mat3 ir = invRestPose;
        
                // Watch out for transpose issues here
                mat3 P = mat3(
                    id[1] - id[0],
                    id[2] - id[0],
                    id[3] - id[0]);
        
                mat3 F = P * ir;
        
                float r_s = sqrt(
                    dot(F[0], F[0]) +
                    dot(F[1], F[1]) +
                    dot(F[2], F[2]));
                float r_s_inv = 1.0 / r_s;
        
                ir = transpose(ir);
                g[1] = vec3(0); g[2] = vec3(0); g[3] = vec3(0);
                g[1] += F[0] * r_s_inv * ir[0][0];
                g[1] += F[1] * r_s_inv * ir[0][1];
                g[1] += F[2] * r_s_inv * ir[0][2];
                g[2] += F[0] * r_s_inv * ir[1][0];
                g[2] += F[1] * r_s_inv * ir[1][1];
                g[2] += F[2] * r_s_inv * ir[1][2];
                g[3] += F[0] * r_s_inv * ir[2][0];
                g[3] += F[1] * r_s_inv * ir[2][1];
                g[3] += F[2] * r_s_inv * ir[2][2];
                ir = transpose(ir);
        
                C = r_s;
        
                // Non gradient pass?
                applyToElem(C, devCompliance, dt, invRestVolume, id, invMass); //
        
                // det F = 1
        
                P = mat3(
                    id[1] - id[0],
                    id[2] - id[0],
                    id[3] - id[0]);
        
                F = P * ir;
        
                mat3 dF = mat3(
                    cross(F[1], F[2]),
                    cross(F[2], F[0]),
                    cross(F[0], F[1]));
        
                ir = transpose(ir);
                g[1] = vec3(0); g[2] = vec3(0); g[3] = vec3(0);
                g[1] += dF[0] * ir[0][0];
                g[1] += dF[1] * ir[0][1];
                g[1] += dF[2] * ir[0][2];
                g[2] += dF[0] * ir[1][0];
                g[2] += dF[1] * ir[1][1];
                g[2] += dF[2] * ir[1][2];
                g[3] += dF[0] * ir[2][0];
                g[3] += dF[1] * ir[2][1];
                g[3] += dF[2] * ir[2][2];
                ir = transpose(ir);
        
                float vol = determinant(F);
                C = vol - 1.0 - volCompliance / devCompliance;
                applyToElem(C, volCompliance, dt, invRestVolume, id, invMass);
            }

            void main()	{
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                // Grab the Relevant Element Variables
                float invVolume  = texture2D( invRestVolume, uv ).x;
                mat3 invRestPose = /*transpose*/(mat3(
                    texture2D( invRestPoseX, uv).xyz,
                    texture2D( invRestPoseY, uv).xyz,
                    texture2D( invRestPoseZ, uv).xyz));

                // Gather this tetrahedron's 4 vertices
                vec4 tetIndices = texture2D( elemToParticlesTable, uv );
                id[0] = texture2D( texturePos, uvFromIndex(int(tetIndices.x))).xyz;
                id[1] = texture2D( texturePos, uvFromIndex(int(tetIndices.y))).xyz;
                id[2] = texture2D( texturePos, uvFromIndex(int(tetIndices.z))).xyz;
                id[3] = texture2D( texturePos, uvFromIndex(int(tetIndices.w))).xyz;

                invMass[0] = texture2D( invMassTex, uvFromIndex(int(tetIndices.x))).x;
                invMass[1] = texture2D( invMassTex, uvFromIndex(int(tetIndices.y))).x;
                invMass[2] = texture2D( invMassTex, uvFromIndex(int(tetIndices.z))).x;
                invMass[3] = texture2D( invMassTex, uvFromIndex(int(tetIndices.w))).x;

                // TODO: Perform the NeoHookean Tet Constraint Resolution Step
                solveElement(invRestPose, invVolume, id, invMass);

                // Ultra Simplified experiment: Use the rest pose directly without any rotation
                // Looks funny because each tet has the same mass despite being different sizes
                //mat3 restPose = inverse(invRestPose);
                //vec3 curCentroid  = (id[0] + id[1] + id[2] + id[3]) * 0.25;
                //vec3 restCentroid = (restPose[0] + restPose[1] + restPose[2]) * 0.25;
                //id[0] = -restCentroid + curCentroid;
                //id[1] = (restPose[0] - restCentroid) + curCentroid;
                //id[2] = (restPose[1] - restCentroid) + curCentroid;
                //id[3] = (restPose[2] - restCentroid) + curCentroid;

                // Write out the new positions
                vert1 = vec4(id[0], 0);
                vert2 = vec4(id[1], 0);
                vert3 = vec4(id[2], 0);
                vert4 = vec4(id[3], 0);
            }`);
        this.solveElemPass.material.uniforms['dt'                  ] = { value: this.physicsParams.dt };
        this.solveElemPass.material.uniforms['elemToParticlesTable'] = { value: this.elemToParticlesTable };
        this.solveElemPass.material.uniforms['invRestVolume'       ] = { value: this.invRestVolumeAndColor };
        this.solveElemPass.material.uniforms['invMassTex'          ] = { value: this.invMass      };
        this.solveElemPass.material.uniforms['invRestPoseX'        ] = { value: this.invRestPoseX };
        this.solveElemPass.material.uniforms['invRestPoseY'        ] = { value: this.invRestPoseY };
        this.solveElemPass.material.uniforms['invRestPoseZ'        ] = { value: this.invRestPoseZ };
        this.solveElemPass.material.uniformsNeedUpdate = true;
        this.solveElemPass.material.needsUpdate = true;

        // Debug: Copy the elems textures into a debug texture for reading from the GPU
        this.copyElems1Pass = this.gpuCompute.addPass(this.debugElem1, [this.elems], `
            out highp vec4 pc_fragColor;
            void main()	{ pc_fragColor = texture2D( textureElem[0], gl_FragCoord.xy / resolution.xy ); }`);
        this.copyElems2Pass = this.gpuCompute.addPass(this.debugElem2, [this.elems], `
            out highp vec4 pc_fragColor;
            void main()	{ pc_fragColor = texture2D( textureElem[1], gl_FragCoord.xy / resolution.xy ); }`);
        this.copyElems3Pass = this.gpuCompute.addPass(this.debugElem3, [this.elems], `
            out highp vec4 pc_fragColor;
            void main()	{ pc_fragColor = texture2D( textureElem[2], gl_FragCoord.xy / resolution.xy ); }`);
        this.copyElems4Pass = this.gpuCompute.addPass(this.debugElem4, [this.elems], `
            out highp vec4 pc_fragColor;
            void main()	{ pc_fragColor = texture2D( textureElem[3], gl_FragCoord.xy / resolution.xy ); }`);

        // 4. Gather the particles back from the elements
        this.applyElemPass = this.gpuCompute.addPass(this.pos, [this.elems, this.pos],
        `
            out highp vec4 pc_fragColor;
            uniform float dt;
            uniform sampler2D particleToElemVertsTable[9];

            vec2 uvFromIndex(int index) {
                return vec2(  index % int(resolution.x),
                             (index / int(resolution.x))) / (resolution - 1.0); }
            vec4 textureElemSample(int index, vec2 uv) {
                 vec4 result = texture2D( textureElem[0], uv );
                 if (index == 1) {
                     result  = texture2D( textureElem[1], uv );
                 } else if (index == 2) {
                     result  = texture2D( textureElem[2], uv );
                 } else if (index == 3) {
                     result  = texture2D( textureElem[3], uv );
                 }
                 return result;
            }
            vec4 particleToElemVertsTableSample(int index, vec2 uv) {
                vec4 result = texture2D( particleToElemVertsTable[0], uv );
                if (index == 1) {
                    result  = texture2D( particleToElemVertsTable[1], uv );
                } else if (index == 2) {
                    result  = texture2D( particleToElemVertsTable[2], uv );
                } else if (index == 3) {
                    result  = texture2D( particleToElemVertsTable[3], uv );
                } else if (index == 4) {
                    result  = texture2D( particleToElemVertsTable[4], uv );
                } else if (index == 5) {
                    result  = texture2D( particleToElemVertsTable[5], uv );
                } else if (index == 6) {
                    result  = texture2D( particleToElemVertsTable[6], uv );
                } else if (index == 7) {
                    result  = texture2D( particleToElemVertsTable[7], uv );
                } else if (index == 7) {
                    result  = texture2D( particleToElemVertsTable[8], uv );
                }
                return result;
           }
            void main()	{
                vec2 uv = gl_FragCoord.xy / resolution.xy;

                vec4 sumVertex = vec4(0.0);
                for(int tableId = 0; tableId < 8; tableId++) {
                    vec4 vertIndices = particleToElemVertsTableSample(tableId, uv);
                    for(int component = 0; component < 4; component++) {
                        if(vertIndices[component] > 0.0) {
                            int   elemId = int(vertIndices[component]) / 4;
                            int   vertId = int(vertIndices[component]) % 4;
                            sumVertex   += vec4(textureElemSample(vertId, uvFromIndex(elemId) ).xyz, 1.0);
                        } else { break; }
                    }
                }
                pc_fragColor = vec4((sumVertex / sumVertex.w).xyz, 0); // Output the average vertex position
            }`);
        this.applyElemPass.material.uniforms['particleToElemVertsTable'] = { value: this.particleToElemVertsTable };
        this.applyElemPass.material.uniformsNeedUpdate = true;
        this.applyElemPass.material.needsUpdate = true;

        // 5. Enforce Collisions (TODO: Also Apply Grab Forces via Uniforms here)
        this.collisionPass = this.gpuCompute.addPass(this.pos, [this.pos, this.prevPos],  `
            out highp vec4 pc_fragColor;
            uniform float dt, friction;
            void main()	{
                vec2 uv  = gl_FragCoord.xy / resolution.xy;
                vec3 pos = texture2D( texturePos    , uv ).xyz;
                pos      = clamp(pos, vec3(-2.5, -1.0, -2.5), vec3(2.5, 10.0, 2.5));
                // simple friction
                if(pos.y < 0.0) {
                    pos.y = 0.0;
                    vec3 F = texture2D( texturePrevPos, uv ).xyz - pos;
                    pos.xz += F.xz * min(1.0, dt * friction);
                }
                pc_fragColor = vec4(pos, 0.0 );
            }`);
        this.collisionPass.material.uniforms['dt'      ] = { value: this.physicsParams.dt };
        this.collisionPass.material.uniforms['friction'] = { value: this.physicsParams.friction };
        this.collisionPass.material.uniformsNeedUpdate = true;
        this.collisionPass.material.needsUpdate = true;

        //// 6. XPBD Velocity + Gravity Update
        this.xpbdVelocityPass = this.gpuCompute.addPass(this.vel, [this.pos, this.prevPos], `
            out highp vec4 pc_fragColor;
            uniform float dt, gravity;
            void main()	{
                vec2 uv      = gl_FragCoord.xy / resolution.xy;
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
                { map: this.gpuCompute.getCurrentRenderTarget(this.debugElem4).texture, side: THREE.DoubleSide });
            this.labelPlane = new THREE.PlaneGeometry(1, 1);
            this.labelMesh = new THREE.Mesh(this.labelPlane, this.labelMaterial);
            this.labelMesh.position.set(0, 2.5, 0);
            world.scene.add(this.labelMesh);
        }

        // TODO: Finish the implementation! ---------------------------------------------------------------------------------------

        this.grabPos = new Float32Array(3);
        this.grabId  = -1;

        // solve data: define here to avoid memory allocation during solve

        // visual edge mesh

        this.geometry = new THREE.BufferGeometry();
        this.geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        this.geometry.setIndex(tetEdgeIds);
        this.edgeMesh = new THREE.LineSegments(this.geometry);
        this.edgeMesh.userData = this;    // for raycasting
        //                    this.edgeMesh.layers.enable(1);
        this.edgeMesh.visible = true;

        // visual embedded mesh

        this.visVerts = visVerts;
        this.numVisVerts = visVerts.length / 4;
        this.geometry = new THREE.BufferGeometry();
        this.geometry.setAttribute('position', new THREE.BufferAttribute(
            new Float32Array(3 * this.numVisVerts), 3));
        this.geometry.setIndex(visTriIds);
        this.visMesh = new THREE.Mesh(this.geometry, visMaterial);
        this.visMesh.castShadow = true;
        this.visMesh.userData = this;    // for raycasting
        this.visMesh.layers.enable(1);
        this.geometry.computeVertexNormals();
        this.updateVisMesh();
    }

    initPhysics(density) {
        // and fill in here the texture data from vertices

        // Initialize velocities and masses to 0
        for (let i = 0; i < this.vel0.image.data.length; i++){
            this.vel0                       .image.data[i] = 0.0;
            this.invMass                    .image.data[i] = 0.0;
            this.particleToElemVertsTable[0].image.data[i] = -1.0;
            this.particleToElemVertsTable[1].image.data[i] = -1.0;
            this.particleToElemVertsTable[2].image.data[i] = -1.0;
            this.particleToElemVertsTable[3].image.data[i] = -1.0;
            this.particleToElemVertsTable[4].image.data[i] = -1.0;
            this.particleToElemVertsTable[5].image.data[i] = -1.0;
            this.particleToElemVertsTable[6].image.data[i] = -1.0;
            this.particleToElemVertsTable[7].image.data[i] = -1.0;
            this.particleToElemVertsTable[8].image.data[i] = -1.0;
        }

        // Initialize the positions of the vertices
        //this.pos0                     // Set to vertices
        let posIndex = 0;
        for (let i = 0; i < this.pos0.image.data.length; i += 4){
            this.pos0.image.data[i  ] = this.inputPos[posIndex++];
            this.pos0.image.data[i+1] = this.inputPos[posIndex++];
            this.pos0.image.data[i+2] = this.inputPos[posIndex++];
        }

        this.oldInvRestPose = new Float32Array(9 * this.numElems);
        let biggestT = 0;
        for (let i = 0; i < this.numElems; i++) {
            let id0 = this.tetIds[(4 * i)    ];
            let id1 = this.tetIds[(4 * i) + 1];
            let id2 = this.tetIds[(4 * i) + 2];
            let id3 = this.tetIds[(4 * i) + 3];

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
                            if (t == 8) { console.log((4 * ids[id]) + c);}
                            assigned = true; break;
                        }
                    }
                    if (assigned) break;
                }
            }

            this.vecSetDiff(this.oldInvRestPose, 3 * i    , this.inputPos, id1, this.inputPos, id0);
            this.vecSetDiff(this.oldInvRestPose, 3 * i + 1, this.inputPos, id2, this.inputPos, id0);
            this.vecSetDiff(this.oldInvRestPose, 3 * i + 2, this.inputPos, id3, this.inputPos, id0);
            let V = this.matGetDeterminant(this.oldInvRestPose, i) / 6.0;
            this.matSetInverse(this.oldInvRestPose, i);

            // Copy the oldInvRestPose into invRestPoseX, invRestPoseY, and invRestPoseZ
            // TODO: Monitor whether this needs to be transposed
            this.invRestPoseX.image.data[(4 * i) + 0] = this.oldInvRestPose[(9 * i) + 0];
            this.invRestPoseX.image.data[(4 * i) + 1] = this.oldInvRestPose[(9 * i) + 1];
            this.invRestPoseX.image.data[(4 * i) + 2] = this.oldInvRestPose[(9 * i) + 2];
            this.invRestPoseY.image.data[(4 * i) + 0] = this.oldInvRestPose[(9 * i) + 3];
            this.invRestPoseY.image.data[(4 * i) + 1] = this.oldInvRestPose[(9 * i) + 4];
            this.invRestPoseY.image.data[(4 * i) + 2] = this.oldInvRestPose[(9 * i) + 5];
            this.invRestPoseZ.image.data[(4 * i) + 0] = this.oldInvRestPose[(9 * i) + 6];
            this.invRestPoseZ.image.data[(4 * i) + 1] = this.oldInvRestPose[(9 * i) + 7];
            this.invRestPoseZ.image.data[(4 * i) + 2] = this.oldInvRestPose[(9 * i) + 8];

            let pm = V / 4.0 * density;
            this.invMass      .image.data[id0 * 4] += pm;
            this.invMass      .image.data[id1 * 4] += pm;
            this.invMass      .image.data[id2 * 4] += pm;
            this.invMass      .image.data[id3 * 4] += pm;
            this.invRestVolumeAndColor.image.data[(i * 4) + 0] =  1.0 / V; // Set InvMass
            this.invRestVolumeAndColor.image.data[(i * 4) + 1] = -1.0;     // Mark Color as Undefined
        }

        for (let i = 0; i < this.invMass.image.data.length; i++) {
            if (this.invMass.image.data[i] != 0.0) { this.invMass.image.data[i] = 1.0 / this.invMass.image.data[i]; }
        }

        // Colors graph with mutually disconnected tetrahedra. 
        // This keeps the tetrahedra from stepping on eachother while the constraints are being satisfied.
        //this.colorGraph();

        console.log(biggestT);
        console.log(this.particleToElemVertsTable[8].image.data);
    }

    // Yikes! 31 Passes!  This is unacceptably high connectivity; I refuse to simulate it this way...
    //colorGraph() {
    //    for (let currentGraphColor = 0; currentGraphColor < 100; currentGraphColor++) {
    //        let vertexAccounting = {}; let numTetsAddedThisPass = 0;
    //        //for (let i = 0; i < this.numElems; i++) {
    //        for (let i = this.numElems-1; i >= 0; i--) {
    //            let id0 = this.tetIds[(4 * i)];
    //            let id1 = this.tetIds[(4 * i) + 1];
    //            let id2 = this.tetIds[(4 * i) + 2];
    //            let id3 = this.tetIds[(4 * i) + 3];
    //            if (!(id0 in vertexAccounting) &&
    //                !(id1 in vertexAccounting) &&
    //                !(id2 in vertexAccounting) &&
    //                !(id3 in vertexAccounting) &&
    //                this.invRestVolumeAndColor.image.data[(i * 4) + 1] < 0.0) {
    //                vertexAccounting[id0] = true;
    //                vertexAccounting[id1] = true;
    //                vertexAccounting[id2] = true;
    //                vertexAccounting[id3] = true;
    //                this.invRestVolumeAndColor.image.data[(i * 4) + 1] = currentGraphColor;
    //                numTetsAddedThisPass += 1;
    //            }
    //        }
    //        console.log(currentGraphColor, numTetsAddedThisPass);
    //        if (numTetsAddedThisPass == 0) { break; }
    //    }
    //}

    // ----------------- begin solver -----------------------------------------------------                

    simulate(dt, physicsParams) {
        // Run a substep!
        this.gpuCompute.compute();
    }

    // ----------------- end solver -----------------------------------------------------                

    endFrame() {
        this.updateEdgeMesh();
        this.updateVisMesh();
    }

    readIntoBuffer(gpuComputeVariable, buffer) {
        let target = this.gpuCompute.getCurrentRenderTarget(gpuComputeVariable);
        if (target.isWebGLMultipleRenderTargets) {
            // BROKEN BAH
            this.renderer.readRenderTargetPixels(
                this.gpuCompute.getCurrentRenderTarget(gpuComputeVariable).texture[0],
                0, 0, this.texDim, this.texDim, buffer);
        } else {
            this.renderer.readRenderTargetPixels(
                this.gpuCompute.getCurrentRenderTarget(gpuComputeVariable),
                0, 0, this.texDim, this.texDim, buffer);
        }

    }

    updateEdgeMesh() {
        // Read tetrahedron positions back from the GPU
        this.readIntoBuffer(this.pos, this.tetPositionsArray);

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
        this.visMesh.geometry.computeVertexNormals();
        this.visMesh.geometry.attributes.position.needsUpdate = true;
        this.visMesh.geometry.computeBoundingSphere();
    }

    //startGrab(pos) {
    //    let p = [pos.x, pos.y, pos.z];
    //    let minD2 = Number.MAX_VALUE;
    //    this.grabId = -1;
    //    for (let i = 0; i < this.numParticles; i++) {
    //        let d2 = this.vecDistSquared(p, 0, this.pos, i);
    //        if (d2 < minD2) {
    //            minD2 = d2;
    //            this.grabId = i;
    //        }
    //    }
    //    this.vecCopy(this.grabPos, 0, p, 0);
    //}

    //moveGrabbed(pos) {
    //    let p = [pos.x, pos.y, pos.z];
    //    this.vecCopy(this.grabPos, 0, p, 0);
    //}

    //endGrab() { this.grabId = -1; }

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

    vecLengthSquared(a, anr) {
        anr *= 3;
        let a0 = a[anr], a1 = a[anr + 1], a2 = a[anr + 2];
        return a0 * a0 + a1 * a1 + a2 * a2;
    }

    vecDistSquared(a, anr, b, bnr) {
        anr *= 3; bnr *= 3;
        let a0 = a[anr] - b[bnr], a1 = a[anr + 1] - b[bnr + 1], a2 = a[anr + 2] - b[bnr + 2];
        return a0 * a0 + a1 * a1 + a2 * a2;
    }

    /// a = b x c (nr = index)
    vecSetCross(a, anr, b, bnr, c, cnr) {
        anr *= 3; bnr *= 3; cnr *= 3;
        a[anr++] = b[bnr + 1] * c[cnr + 2] - b[bnr + 2] * c[cnr + 1];
        a[anr++] = b[bnr + 2] * c[cnr + 0] - b[bnr + 0] * c[cnr + 2];
        a[anr] = b[bnr + 0] * c[cnr + 1] - b[bnr + 1] * c[cnr + 0];
    }

    vecSetClamped(dst, dnr, a, anr, b, bnr) {
        dnr *= 3; anr *= 3; bnr *= 3;
        dst[dnr] = Math.max(a[anr++], Math.min(b[bnr++], dst[dnr++]));
        dst[dnr] = Math.max(a[anr++], Math.min(b[bnr++], dst[dnr++]));
        dst[dnr] = Math.max(a[anr++], Math.min(b[bnr++], dst[dnr++]));
    }

    // ----- matrix math ----------------------------------

    matIJ(A, anr, row, col) {
        return A[9 * anr + 3 * col + row];
    }

    matSetVecProduct(dst, dnr, A, anr, b, bnr) {
        bnr *= 3; anr *= 3;
        let b0 = b[bnr++];
        let b1 = b[bnr++];
        let b2 = b[bnr];
        this.vecSetZero(dst, dnr);
        this.vecAdd(dst, dnr, A, anr++, b0);
        this.vecAdd(dst, dnr, A, anr++, b1);
        this.vecAdd(dst, dnr, A, anr, b2);
    }

    matSetMatProduct(Dst, dnr, A, anr, B, bnr) {
        dnr *= 3; bnr *= 3;
        this.matSetVecProduct(Dst, dnr++, A, anr, B, bnr++);
        this.matSetVecProduct(Dst, dnr++, A, anr, B, bnr++);
        this.matSetVecProduct(Dst, dnr++, A, anr, B, bnr++);
    }

    matGetDeterminant(A, anr) {
        anr *= 9;
        let a11 = A[anr + 0], a12 = A[anr + 3], a13 = A[anr + 6];
        let a21 = A[anr + 1], a22 = A[anr + 4], a23 = A[anr + 7];
        let a31 = A[anr + 2], a32 = A[anr + 5], a33 = A[anr + 8];
        return a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a13 * a22 * a31 - a12 * a21 * a33 - a11 * a23 * a32;
    }

    matSetInverse(A, anr) {
        let det = this.matGetDeterminant(A, anr);
        if (det == 0.0) {
            for (let i = 0; i < 9; i++)
                A[anr + i] = 0.0;
            return;
        }
        let invDet = 1.0 / det;
        anr *= 9;
        let a11 = A[anr + 0], a12 = A[anr + 3], a13 = A[anr + 6];
        let a21 = A[anr + 1], a22 = A[anr + 4], a23 = A[anr + 7];
        let a31 = A[anr + 2], a32 = A[anr + 5], a33 = A[anr + 8]
        A[anr + 0] = (a22 * a33 - a23 * a32) * invDet;
        A[anr + 3] = -(a12 * a33 - a13 * a32) * invDet;
        A[anr + 6] = (a12 * a23 - a13 * a22) * invDet;
        A[anr + 1] = -(a21 * a33 - a23 * a31) * invDet;
        A[anr + 4] = (a11 * a33 - a13 * a31) * invDet;
        A[anr + 7] = -(a11 * a23 - a13 * a21) * invDet;
        A[anr + 2] = (a21 * a32 - a22 * a31) * invDet;
        A[anr + 5] = -(a11 * a32 - a12 * a31) * invDet;
        A[anr + 8] = (a11 * a22 - a12 * a21) * invDet;
    }

}

//export class Grabber {
//    constructor(scene, renderer, camera, container, controls) {
//        this.scene = scene;
//        this.renderer = renderer;
//        this.camera = camera;
//        this.mousePos = new THREE.Vector2();
//        this.raycaster = new THREE.Raycaster();
//        this.raycaster.layers.set(1);
//        //					this.raycaster.params.Mesh.threshold = 3;
//        this.raycaster.params.Line.threshold = 0.1;
//        this.grabDistance = 0.0;
//        this.active = false;
//        this.physicsObject = null;
//        this.controls = controls;
//
//        container.addEventListener( 'pointerdown', this.onPointer.bind(this), false );
//        container.addEventListener( 'pointermove', this.onPointer.bind(this), false );
//        container.addEventListener( 'pointerup'  , this.onPointer.bind(this), false );
//        container.addEventListener( 'pointerout' , this.onPointer.bind(this), false );
//    }
//    updateRaycaster(x, y) {
//        var rect = this.renderer.domElement.getBoundingClientRect();
//        this.mousePos.x = ((x - rect.left) / rect.width) * 2 - 1;
//        this.mousePos.y = -((y - rect.top) / rect.height) * 2 + 1;
//        this.raycaster.setFromCamera(this.mousePos, this.camera);
//    }
//    start(x, y) {
//        this.physicsObject = null;
//        this.updateRaycaster(x, y);
//        var intersects = this.raycaster.intersectObjects(this.scene.children);
//        if (intersects.length > 0) {
//            var obj = intersects[0].object.userData;
//            if (obj instanceof SoftBody) {
//                this.physicsObject = obj;
//                this.grabDistance = intersects[0].distance;
//                let hit = this.raycaster.ray.origin.clone();
//                hit.addScaledVector(this.raycaster.ray.direction, this.grabDistance);
//                this.physicsObject.startGrab(hit);
//                this.active = true;
//                this.controls.enabled = false;
//            }
//        }
//    }
//    move(x, y) {
//        if (this.active) {
//            this.updateRaycaster(x, y);
//            let hit = this.raycaster.ray.origin.clone();
//            hit.addScaledVector(this.raycaster.ray.direction, this.grabDistance);
//            if (this.physicsObject != null)
//                this.physicsObject.moveGrabbed(hit);
//        }
//    }
//    end() {
//        if (this.active) {
//            if (this.physicsObject != null) {
//                this.physicsObject.endGrab();
//                this.physicsObject = null;
//            }
//            this.active = false;
//            this.controls.enabled = true;
//        }
//    }
//
//    onPointer(evt) {
//        evt.preventDefault();
//        if (evt.type == "pointerdown") {
//            this.start(evt.clientX, evt.clientY);
//            this.mouseDown = true;
//        } else if (evt.type == "pointermove" && this.mouseDown) {
//            if (this.active)
//                this.move(evt.clientX, evt.clientY);
//        } else if (evt.type == "pointerup" || evt.type == "pointerout") {
//            this.end();
//            this.mouseDown = false;
//        }
//    }
//}
