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
        this.invRestVolume         = this.gpuCompute.createTexture(); // Inverse Volume Per Element
        this.invRestPoseX          = this.gpuCompute.createTexture(); // Split the 3x3 restpose into 3 textures
        this.invRestPoseY          = this.gpuCompute.createTexture(); // Split the 3x3 restpose into 3 textures
        this.invRestPoseZ          = this.gpuCompute.createTexture(); // Split the 3x3 restpose into 3 textures
        this.elemToParticlesTable  = this.gpuCompute.createTexture(); // Maps from elems to the 4 tet vertex positions for the gather step
        this.particleToElemVertsTable = [this.gpuCompute.createTexture(), // Maps from vertices back to the elems gbuffer for the scatter step
                                     this.gpuCompute.createTexture(), // There is more than one because a particle may have a bunch of elems sharing it
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
                    invRestPoseX, invRestPoseY, invRestPoseZ;

            layout(location = 0) out vec4 vert1;
            layout(location = 1) out vec4 vert2;
            layout(location = 2) out vec4 vert3;
            layout(location = 3) out vec4 vert4;

            vec2 uvFromIndex(int index) {
                return vec2(  index % int(resolution.x),
                             (index / int(resolution.x))) / (resolution - 1.0); }

            void main()	{
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                // Grab the Relevant Element Variables
                float invVolume   = texture2D( invRestVolume, uv ).x;
                mat3  invRestPose = mat3(
                    texture2D( invRestPoseX, uv).xyz,
                    texture2D( invRestPoseY, uv).xyz,
                    texture2D( invRestPoseZ, uv).xyz);

                // Gather this tetrahedron's 4 vertices
                vec4 tetIndices = texture2D( elemToParticlesTable, uv );
                vec4 vert1Pos   = texture2D( texturePos, uvFromIndex(int(tetIndices.x)));
                vec4 vert2Pos   = texture2D( texturePos, uvFromIndex(int(tetIndices.y)));
                vec4 vert3Pos   = texture2D( texturePos, uvFromIndex(int(tetIndices.z)));
                vec4 vert4Pos   = texture2D( texturePos, uvFromIndex(int(tetIndices.w)));

                // TODO: Perform the NeoHookean Tet Constraint Resolution Step

                vert1 = vert1Pos;
                vert2 = vert2Pos;
                vert3 = vert3Pos;
                vert4 = vert4Pos;
            }`);
        this.solveElemPass.material.uniforms['dt'                  ] = { value: this.physicsParams.dt };
        this.solveElemPass.material.uniforms['elemToParticlesTable'] = { value: this.elemToParticlesTable };
        this.solveElemPass.material.uniforms['invRestVolume'       ] = { value: this.invRestVolume };
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
            uniform sampler2D particleToElemVertsTable1, particleToElemVertsTable2,
                              particleToElemVertsTable3, particleToElemVertsTable4;
            vec2 uvFromIndex(int index) {
                return vec2(  index % int(resolution.x),
                             (index / int(resolution.x))) / (resolution - 1.0); }
            vec4 elemByVertIndex(int index, vec2 uv) {
                 vec4 result = texture2D( textureElem[0], uv );
                 if (index == 1) {
                     result = texture2D( textureElem[1], uv );
                 } else if (index == 2) {
                     result = texture2D( textureElem[2], uv );
                 } else if (index == 3) {
                     result = texture2D( textureElem[3], uv );
                 }
                 return result;
            }
            void main()	{
                vec2 uv = gl_FragCoord.xy / resolution.xy;

                vec4 vertIndices1 = texture2D( particleToElemVertsTable1, uv );
                //vec4 vertIndices2 = texture2D( particleToElemVertsTable2, uv );
                //vec4 vertIndices3 = texture2D( particleToElemVertsTable3, uv );
                //vec4 vertIndices4 = texture2D( particleToElemVertsTable4, uv );

                vec4 sumVertex = vec4(0.0);
                for(int i = 0; i < 4; i++) {
                    if(vertIndices1[i] > 0.0) {
                        int   elemId = int(vertIndices1[i]) / 4;
                        int   vertId = int(vertIndices1[i]) % 4;
                        sumVertex   += vec4(elemByVertIndex(vertId, uvFromIndex(elemId) ).xyz, 1.0);
                    } else { break; }
                }
                sumVertex /= sumVertex.w;

                //int   elemId = int(vertIndices1[0]) / 4;
                //int   vertId = int(vertIndices1[0]) % 4;
                //pc_fragColor = elemByVertIndex(vertId, uvFromIndex(elemId));

                pc_fragColor = vec4(sumVertex.xyz, 0);
                //pc_fragColor = texture2D(texturePos, uv); //
            }`);
        this.applyElemPass.material.uniforms['particleToElemVertsTable1'] = { value: this.particleToElemVertsTable[0] };
        this.applyElemPass.material.uniforms['particleToElemVertsTable2'] = { value: this.particleToElemVertsTable[1] };
        this.applyElemPass.material.uniforms['particleToElemVertsTable3'] = { value: this.particleToElemVertsTable[2] };
        this.applyElemPass.material.uniforms['particleToElemVertsTable4'] = { value: this.particleToElemVertsTable[3] };
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
            this.labelMesh.position.set(0, 1, 0);
            world.scene.add(this.labelMesh);
        }

        // BELOW THIS POINT THE SCRIPT IS LARGELY UNCHANGED AS OF NOW
        // TODO: Finish the implementation! ---------------------------------------------------------------------------------------

        this.grabPos = new Float32Array(3);
        this.grabId  = -1;

        // solve data: define here to avoid memory allocation during solve

        //this.P     = new Float32Array(9);
        //this.F     = new Float32Array(9);
        //this.dF    = new Float32Array(9);
        //this.grads = new Float32Array(12);

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

        //// Run a substep!
        //this.gpuCompute.compute();
        //// Verify that the compute framebuffer is correct
        //let numDifferences = [];
        //// Read tetrahedron and debug positions back from the GPU
        //this.readIntoBuffer(this.pos       , this. tetPositionsArray);
        //this.readIntoBuffer(this.debugElem1, this.elemPositionsArray);
        //for (let i = 0; i < this.numElems; i++){
        //    let id0 = this.elemToParticlesTable.image.data[4 * i + 0] * 4;
        //    let id1 = this.elemToParticlesTable.image.data[4 * i + 1] * 4;
        //    let id2 = this.elemToParticlesTable.image.data[4 * i + 2] * 4;
        //    let id3 = this.elemToParticlesTable.image.data[4 * i + 3] * 4; 
        //    numDifferences.push(
        //        this.elemPositionsArray[(i*4) + 0] - this.tetPositionsArray[id0 + 0],
        //        this.elemPositionsArray[(i*4) + 1] - this.tetPositionsArray[id0 + 1],
        //        this.elemPositionsArray[(i*4) + 2] - this.tetPositionsArray[id0 + 2]);
        //}
        //console.log(numDifferences);
    }

    initPhysics(density) {
        // and fill in here the texture data from vertices

        // Initialize velocities and masses to 0
        //this.vel0                     // Leave as 0s for zero velocity
        //this.invMass                  // Inverse Volume Per Element
        for (let i = 0; i < this.numElems; i++){
            this.vel0.image.data[i] = 0.0;
            this.invMass.image.data[i] = 0.0;
            this.particleToElemVertsTable[0].image.data[i] = -1.0;
            this.particleToElemVertsTable[1].image.data[i] = -1.0;
            this.particleToElemVertsTable[2].image.data[i] = -1.0;
            this.particleToElemVertsTable[3].image.data[i] = -1.0;
        }

        // Initialize the positions of the vertices
        //this.pos0                     // Set to vertices
        let posIndex = 0;
        for (let i = 0; i < this.pos0.image.data.length; i += 4){
            this.pos0.image.data[i  ] = this.inputPos[posIndex++];
            this.pos0.image.data[i+1] = this.inputPos[posIndex++];
            this.pos0.image.data[i+2] = this.inputPos[posIndex++];
        }

        //this.invRestVolume            // Have Mass and Rest Volume Share a Texture
        //this.invRestPoseX             // Split the 3x3 restpose into 3 textures
        //this.invRestPoseY             // Split the 3x3 restpose into 3 textures
        //this.invRestPoseZ             // Split the 3x3 restpose into 3 textures
        //this.elemToParticlesTable     // Maps from elems to the 4 tet vertex positions for the gather step
        //this.particleToElemVertsTable[]   // Maps from vertices back to the elems gbuffer for the scatter step
                                        // There is more than one because a particle may have a bunch of elems sharing it
        this.oldInvRestPose = new Float32Array(9 * this.numElems);
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
            //let ids = [id0, id1, id2, id3];
            //for (let id = 0; id < 4; id++) {
            //    let assigned = false;
            //    for (let t = 0; t < this.particleToElemVertsTable.length; t++) {
            //        for (let c = 0; c < 4; c++) {
            //            if (this.particleToElemVertsTable[t].image.data[(4 * ids[id]) + c] < 0.0) {
            //                this.particleToElemVertsTable[t].image.data[(4 * ids[id]) + c] = (4.0*i) + id;
            //                assigned = true; break;
            //            }
            //        }
            //        if (assigned) break;
            //    }
            //}
            //if (this.particleToElemVertsTable[0].image.data[4 * id0] < 0.0) {
                this.particleToElemVertsTable[0].image.data[4 * id0] = (4.0 * i) + 0;
            //}
            //if (this.particleToElemVertsTable[0].image.data[4 * id1] < 0.0) {
                this.particleToElemVertsTable[0].image.data[4 * id1] = (4.0 * i) + 1;
            //}
            //if (this.particleToElemVertsTable[0].image.data[4 * id2] < 0.0) {
                this.particleToElemVertsTable[0].image.data[4 * id2] = (4.0 * i) + 2;
            //}
            //if (this.particleToElemVertsTable[0].image.data[4 * id3] < 0.0) {
                this.particleToElemVertsTable[0].image.data[4 * id3] = (4.0 * i) + 3;
            //}

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
            this.invRestVolume.image.data[i   * 4] = 1.0 / V;
        }

        for (let i = 0; i < this.invMass.image.data.length; i++) {
            if (this.invMass[i] != 0.0) { this.invMass[i] = 1.0 / this.invMass[i]; }
        }

    }

    // ----------------- begin solver -----------------------------------------------------                

    solveElem(elemNr, dt) {
        let C = 0.0;
        let g = this.grads;
        let ir = this.invRestPose;

        // tr(F) = 3

        let id0 = this.tetIds[4 * elemNr];
        let id1 = this.tetIds[4 * elemNr + 1];
        let id2 = this.tetIds[4 * elemNr + 2];
        let id3 = this.tetIds[4 * elemNr + 3];

        this.vecSetDiff(this.P, 0, this.pos, id1, this.pos, id0);
        this.vecSetDiff(this.P, 1, this.pos, id2, this.pos, id0);
        this.vecSetDiff(this.P, 2, this.pos, id3, this.pos, id0);

        this.matSetMatProduct(this.F, 0, this.P, 0, this.invRestPose, elemNr);

        let r_s = Math.sqrt(this.vecLengthSquared(this.F, 0) + this.vecLengthSquared(this.F, 1) + this.vecLengthSquared(this.F, 2));
        let r_s_inv = 1.0 / r_s;

        this.vecSetZero(g, 1);
        this.vecAdd(g, 1, this.F, 0, r_s_inv * this.matIJ(ir, elemNr, 0, 0));
        this.vecAdd(g, 1, this.F, 1, r_s_inv * this.matIJ(ir, elemNr, 0, 1));
        this.vecAdd(g, 1, this.F, 2, r_s_inv * this.matIJ(ir, elemNr, 0, 2));

        this.vecSetZero(g, 2);
        this.vecAdd(g, 2, this.F, 0, r_s_inv * this.matIJ(ir, elemNr, 1, 0));
        this.vecAdd(g, 2, this.F, 1, r_s_inv * this.matIJ(ir, elemNr, 1, 1));
        this.vecAdd(g, 2, this.F, 2, r_s_inv * this.matIJ(ir, elemNr, 1, 2));

        this.vecSetZero(g, 3);
        this.vecAdd(g, 3, this.F, 0, r_s_inv * this.matIJ(ir, elemNr, 2, 0));
        this.vecAdd(g, 3, this.F, 1, r_s_inv * this.matIJ(ir, elemNr, 2, 1));
        this.vecAdd(g, 3, this.F, 2, r_s_inv * this.matIJ(ir, elemNr, 2, 2));

        C = r_s;


        this.applyToElem(elemNr, C, this.physicsParams.devCompliance, dt);
        
        // det F = 1

        this.vecSetDiff(this.P, 0, this.pos, id1, this.pos, id0);
        this.vecSetDiff(this.P, 1, this.pos, id2, this.pos, id0);
        this.vecSetDiff(this.P, 2, this.pos, id3, this.pos, id0);

        this.matSetMatProduct(this.F, 0, this.P, 0, this.invRestPose, elemNr);

        this.vecSetCross(this.dF, 0, this.F, 1, this.F, 2);
        this.vecSetCross(this.dF, 1, this.F, 2, this.F, 0);
        this.vecSetCross(this.dF, 2, this.F, 0, this.F, 1);

        this.vecSetZero(g, 1);
        this.vecAdd(g, 1, this.dF, 0, this.matIJ(ir, elemNr, 0, 0));
        this.vecAdd(g, 1, this.dF, 1, this.matIJ(ir, elemNr, 0, 1));
        this.vecAdd(g, 1, this.dF, 2, this.matIJ(ir, elemNr, 0, 2));

        this.vecSetZero(g, 2);
        this.vecAdd(g, 2, this.dF, 0, this.matIJ(ir, elemNr, 1, 0));
        this.vecAdd(g, 2, this.dF, 1, this.matIJ(ir, elemNr, 1, 1));
        this.vecAdd(g, 2, this.dF, 2, this.matIJ(ir, elemNr, 1, 2));

        this.vecSetZero(g, 3);
        this.vecAdd(g, 3, this.dF, 0, this.matIJ(ir, elemNr, 2, 0));
        this.vecAdd(g, 3, this.dF, 1, this.matIJ(ir, elemNr, 2, 1));
        this.vecAdd(g, 3, this.dF, 2, this.matIJ(ir, elemNr, 2, 2));

        let vol = this.matGetDeterminant(this.F, 0);
        //C = vol - 1.0 - ((1.0 / this.physicsParams.devCompliance) / (1.0 / this.physicsParams.volCompliance));
        C = vol - 1.0 - this.physicsParams.volCompliance / this.physicsParams.devCompliance;

        //this.volError += vol - 1.0;
        
        this.applyToElem(elemNr, C, this.physicsParams.volCompliance, dt);
    }

    applyToElem(elemNr, C, compliance, dt) {
        if (C == 0.0)
            return;
        let g = this.grads;

        this.vecSetZero(g, 0);
        this.vecAdd(g, 0, g, 1, -1.0);
        this.vecAdd(g, 0, g, 2, -1.0);
        this.vecAdd(g, 0, g, 3, -1.0);

        let w = 0.0;
        for (let i = 0; i < 4; i++) {
            let id = this.tetIds[4 * elemNr + i];
            w += this.vecLengthSquared(g, i) * this.invMass[id];
        }

        if (w == 0.0)
            return;
        let alpha = compliance / dt / dt * this.invRestVolume[elemNr];
        let dlambda = -C / (w + alpha);

        for (let i = 0; i < 4; i++) {
            let id = this.tetIds[4 * elemNr + i];
            this.vecAdd(this.pos, id, g, i, dlambda * this.invMass[id]);
        }
    }

    simulate(dt, physicsParams) {
        // TODO: Update the Simulation Uniforms

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
        //const positions = this.visMesh.geometry.attributes.position.array;
        //let nr = 0;
        //for (let i = 0; i < this.numVisVerts; i++) {
        //    let tetNr = this.visVerts[nr++] * 4;
        //    let b0 = this.visVerts[nr++];
        //    let b1 = this.visVerts[nr++];
        //    let b2 = this.visVerts[nr++];
        //    let b3 = 1.0 - b0 - b1 - b2;
        //    this.vecSetZero(positions, i);
        //    this.vecAdd(positions, i, this.pos, this.tetIds[tetNr++], b0);
        //    this.vecAdd(positions, i, this.pos, this.tetIds[tetNr++], b1);
        //    this.vecAdd(positions, i, this.pos, this.tetIds[tetNr++], b2);
        //    this.vecAdd(positions, i, this.pos, this.tetIds[tetNr++], b3);
        //}
        //this.visMesh.geometry.computeVertexNormals();
        //this.visMesh.geometry.attributes.position.needsUpdate = true;
        //this.visMesh.geometry.computeBoundingSphere();
    }

    startGrab(pos) {
        let p = [pos.x, pos.y, pos.z];
        let minD2 = Number.MAX_VALUE;
        this.grabId = -1;
        for (let i = 0; i < this.numParticles; i++) {
            let d2 = this.vecDistSquared(p, 0, this.pos, i);
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

export class Grabber {
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

        container.addEventListener( 'pointerdown', this.onPointer.bind(this), false );
        container.addEventListener( 'pointermove', this.onPointer.bind(this), false );
        container.addEventListener( 'pointerup'  , this.onPointer.bind(this), false );
        container.addEventListener( 'pointerout' , this.onPointer.bind(this), false );
    }
    updateRaycaster(x, y) {
        var rect = this.renderer.domElement.getBoundingClientRect();
        this.mousePos.x = ((x - rect.left) / rect.width) * 2 - 1;
        this.mousePos.y = -((y - rect.top) / rect.height) * 2 + 1;
        this.raycaster.setFromCamera(this.mousePos, this.camera);
    }
    start(x, y) {
        this.physicsObject = null;
        this.updateRaycaster(x, y);
        var intersects = this.raycaster.intersectObjects(this.scene.children);
        if (intersects.length > 0) {
            var obj = intersects[0].object.userData;
            if (obj instanceof SoftBody) {
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
    end() {
        if (this.active) {
            if (this.physicsObject != null) {
                this.physicsObject.endGrab();
                this.physicsObject = null;
            }
            this.active = false;
            this.controls.enabled = true;
        }
    }

    onPointer(evt) {
        evt.preventDefault();
        if (evt.type == "pointerdown") {
            this.start(evt.clientX, evt.clientY);
            this.mouseDown = true;
        } else if (evt.type == "pointermove" && this.mouseDown) {
            if (this.active)
                this.move(evt.clientX, evt.clientY);
        } else if (evt.type == "pointerup" || evt.type == "pointerout") {
            this.end();
            this.mouseDown = false;
        }
    }
}
