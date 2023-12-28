import * as THREE from '../node_modules/three/build/three.module.js';
import { GUI } from '../node_modules/three/examples/jsm/libs/lil-gui.module.min.js';
import { SoftBody, Grabber } from './Softbody.js';
import { SoftBodyGPU, GPUGrabber } from './SoftbodyGPU.js';
import { dragonTetVerts, dragonTetIds, dragonTetEdgeIds, dragonAttachedVerts, dragonAttachedTriIds } from './Dragon.js';
import World from './World.js';
import ManifoldModule from '../node_modules/manifold-3d/manifold.js';
//import { PhysX } from './physx-js-webidl/dist/physx-js-webidl.js';

// Tetrahedralize the test shape
const px = await PhysX();
let version    = px.PHYSICS_VERSION;
var allocator  = new px.PxDefaultAllocator();
var errorCb    = new px.PxDefaultErrorCallback();
var foundation = px.CreateFoundation(version, allocator, errorCb);
console.log('PhysX loaded! Version: ' + ((version >> 24) & 0xff) + '.' + ((version >> 16) & 0xff) + '.' + ((version >> 8) & 0xff));

/** The fundamental set up and animation structures for 3D Visualization */
export default class Main {

    constructor() {
        // Intercept Main Window Errors
        window.realConsoleError = console.error;
        window.addEventListener('error', (event) => {
            let path = event.filename.split("/");
            this.display((path[path.length - 1] + ":" + event.lineno + " - " + event.message));
        });
        console.error = this.fakeError.bind(this);
        this.physicsScene = { softBodies: [] };
        this.deferredConstructor();
    }
    async deferredConstructor() {
        // Configure Settings
        let cpuSim = new URLSearchParams(window.location.search).get('cpu') === 'true';
        this.physicsParams = {
            gravity       : -9.81,
            timeScale     : 1.0,
            timeStep      : 1.0 / 60.0,
            numSubsteps   : cpuSim?5:20,
            dt            : 1.0 / (60.0 * (cpuSim?5:20)),
            friction      : 1000.0,
            density       : 1000.0,
            devCompliance : 1.0/100000.0,
            volCompliance : 0.0,
            worldBounds   : [-2.5,-1.0, -2.5, 2.5, 10.0, 2.5],
            computeNormals: true,
            ShowTetMesh   : false,
            cpuSim        : cpuSim
        };
        this.gui = new GUI();
        this.gui.add(this.physicsParams, 'gravity', -20.0, 0.0, 1);
        this.gui.add(this.physicsParams, 'timeScale', 0.1, 2.0, 0.01);
        this.gui.add(this.physicsParams, 'numSubsteps', 1, cpuSim?10:100, 1);
        this.gui.add(this.physicsParams, 'friction', 0.0, 6000.0, 100.0);
        this.gui.add(this.physicsParams, 'ShowTetMesh');
        //this.gui.add(this.physicsParams, 'density', 0.0, 10000.0, 100.0);
        //this.gui.add(this.physicsParams, 'devCompliance', 1.0 / 2000000.0, 1.0 / 1000.0, 0.00001);
        //this.gui.add(this.physicsParams, 'volCompliance', 0.0, 0.001, 0.00001);

        // Construct the render world
        this.world = new World(this);

        // Construct Test Shape
        const manifold = await ManifoldModule();
        manifold.setup();
        /** @type {manifold.Manifold} */
        let sphere = new manifold.Manifold.sphere(0.6, 32);
        /** @type {manifold.Manifold} */
        let box    = new manifold.Manifold.cube([1.0, 1.0, 1.0], true);
        /** @type {manifold.Manifold} */
        let spherelessBox = new manifold.Manifold.difference(box, sphere).translate([0.0, 1.0, 0.0]);
        /** @type {manifold.Mesh} */
        let sphereMesh = spherelessBox.getMesh();
        if(sphereMesh.numProp == 3){
            let bufferGeo = new THREE.BufferGeometry();
            bufferGeo.setAttribute('position', new THREE.BufferAttribute(sphereMesh.vertProperties, 3));
            bufferGeo.setIndex(new THREE.BufferAttribute(sphereMesh.triVerts, 1));
            bufferGeo.computeVertexNormals();
            let threeMesh = new THREE.Mesh(bufferGeo, new THREE.MeshPhysicalMaterial({ color: 0x00ff00, wireframe: true }));
            threeMesh.position.set(0.75, 0.0, 0.0);
            this.world.scene.add(threeMesh);
            sphere.delete();
            box.delete();
            spherelessBox.delete();
        }
        
        let remeshedGeo   = this.remesh(sphereMesh.vertProperties, sphereMesh.triVerts, 8);
        let simplifiedGeo = this.simplifyMesh(remeshedGeo.getAttribute("position").array, remeshedGeo.getIndex().array, 500, 100.0);
        let remeshedThreeMesh = new THREE.Mesh(simplifiedGeo, new THREE.MeshPhysicalMaterial({ color: 0x00ff00, wireframe: true }));
        remeshedThreeMesh.position.set(2.0, 0.0, 0.0);
        this.world.scene.add(remeshedThreeMesh);


        let tetrahedronGeo = this.createConformingTetrahedronMesh(simplifiedGeo.getAttribute("position").array, simplifiedGeo.getIndex().array);
        let edgeMesh = new THREE.LineSegments(tetrahedronGeo, new THREE.LineBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide }));
        edgeMesh.userData = this;    // for raycasting
        edgeMesh.layers.enable(1);
        edgeMesh.visible = true;
        edgeMesh.position.set(3.1, 0.0, 0.0);
        this.world.scene.add(edgeMesh);

        // Construct the physics world
        if (this.physicsParams.cpuSim) {
            this.dragon = new SoftBody(dragonTetVerts, dragonTetIds, dragonTetEdgeIds, this.physicsParams,
                dragonAttachedVerts, dragonAttachedTriIds, new THREE.MeshPhysicalMaterial({ color: 0xf78a1d }));
            this.physicsScene.softBodies.push(this.dragon);
            this.grabber = new Grabber(
                this.world.scene, this.world.renderer, this.world.camera,
                this.world.container.parentElement, this.world.controls);
        } else {
            this.dragon = new SoftBodyGPU(dragonTetVerts, dragonTetIds, dragonTetEdgeIds, this.physicsParams,
                dragonAttachedVerts, dragonAttachedTriIds, new THREE.MeshPhysicalMaterial({ color: 0xf78a1d, roughness:0.4 }), this.world);
            this.physicsScene.softBodies.push(this.dragon);
            this.grabber = new GPUGrabber(
                this.world.scene, this.world.renderer, this.world.camera,
                this.world.container.parentElement, this.world.controls);
        }
        this.world.scene.add(this.dragon.edgeMesh);
        this.world.scene.add(this.dragon.visMesh);

        //this.previousTime = (performance.now()*0.001) - 1/60.0;
    }

    /** Update the simulation */
    update() {
        //this.physicsParams.timeStep += (((performance.now()*0.001) - this.previousTime) - this.physicsParams.timeStep) * 0.01;
        //this.previousTime = performance.now()*0.001;

        if(this.physicsScene.softBodies.length == 0) { return; }

        // Simulate all of the soft bodies in the scene
        let dt = (this.physicsParams.timeScale * this.physicsParams.timeStep) / this.physicsParams.numSubsteps;
        for (let step = 0; step < this.physicsParams.numSubsteps; step++) {
            for (let i = 0; i < this.physicsScene.softBodies.length; i++) {
                this.physicsScene.softBodies[i].simulate(dt, this.physicsParams);
            }
        }

        // Update their visual representations
        for (let i = 0; i < this.physicsScene.softBodies.length; i++) {
            this.physicsScene.softBodies[i].endFrame();
        }

        // Render the scene and update the framerate counter
        this.world.controls.update();
        this.world.renderer.render(this.world.scene, this.world.camera);
        this.world.stats.update();

    }

    // Log Errors as <div>s over the main viewport
    fakeError(...args) {
        if (args.length > 0 && args[0]) { this.display(JSON.stringify(args[0])); }
        window.realConsoleError.apply(console, arguments);
    }

    display(text) {
        let errorNode = window.document.createElement("div");
        errorNode.innerHTML = text.fontcolor("red");
        window.document.getElementById("info").appendChild(errorNode);
    }

    /** @returns {THREE.BufferGeometry} */
    remesh(vertices, indices, remesherGridResolution = 20){
        let inputVertices  = new px.PxArray_PxVec3(vertices.length/3);
        let inputIndices   = new px.PxArray_PxU32 (indices.length);
        for(let i = 0; i < vertices.length; i+=3){
            inputVertices.set(i/3, new px.PxVec3(vertices[i], vertices[i+1], vertices[i+2]));
        }
        for(let i = 0; i < indices.length; i++){
            inputIndices.set(i, indices[i]);
        }

        let outputVertices = new px.PxArray_PxVec3();
        let outputIndices  = new px.PxArray_PxU32 ();
        let vertexMap      = new px.PxArray_PxU32 ();
        px.PxTetMaker.prototype.remeshTriangleMesh(inputVertices, inputIndices, remesherGridResolution, outputVertices, outputIndices, vertexMap);

        // Transform From PxVec3 to THREE.Vector3
        let triIndices = new Uint32Array(outputIndices.size());
        for(let i = 0; i < triIndices.length; i++){
            triIndices[i] = outputIndices.get(i);
        }
        let vertPositions = new Float32Array(outputVertices.size() * 3);
        for(let i = 0; i < outputVertices.size(); i++){
            let vec3 = outputVertices.get(i);
            vertPositions[i*3+0] = vec3.get_x();
            vertPositions[i*3+1] = vec3.get_y();
            vertPositions[i*3+2] = vec3.get_z();
        }
        let remeshedBufferGeo = new THREE.BufferGeometry();
        remeshedBufferGeo.setAttribute('position', new THREE.BufferAttribute(vertPositions, 3));
        remeshedBufferGeo.setIndex(new THREE.BufferAttribute(triIndices, 1));
        remeshedBufferGeo.computeVertexNormals();
        inputVertices .__destroy__();
        inputIndices  .__destroy__();
        outputVertices.__destroy__();
        outputIndices .__destroy__();
        vertexMap     .__destroy__();
        return remeshedBufferGeo;
    }

    /** @returns {THREE.BufferGeometry} */
    simplifyMesh(vertices, indices, targetTriangleCount = 5000, maximalTriangleEdgeLength = 110.0){
        let inputVertices  = new px.PxArray_PxVec3(vertices.length/3);
        let inputIndices   = new px.PxArray_PxU32 (indices.length);
        for(let i = 0; i < vertices.length; i+=3){
            inputVertices.set(i/3, new px.PxVec3(vertices[i], vertices[i+1], vertices[i+2]));
        }
        for(let i = 0; i < indices.length; i++){
            inputIndices.set(i, indices[i]);
        }

        let outputVertices = new px.PxArray_PxVec3();
        let outputIndices  = new px.PxArray_PxU32 ();
        px.PxTetMaker.prototype.simplifyTriangleMesh(inputVertices, inputIndices, targetTriangleCount, maximalTriangleEdgeLength, outputVertices, outputIndices);

        console.log(inputVertices.size(), inputIndices.size(), outputVertices.size(), outputIndices.size());

        // Transform From PxVec3 to THREE.Vector3
        let triIndices = new Uint32Array(outputIndices.size());
        for(let i = 0; i < triIndices.length; i++){
            triIndices[i] = outputIndices.get(i);
        }
        let vertPositions = new Float32Array(outputVertices.size() * 3);
        for(let i = 0; i < outputVertices.size(); i++){
            let vec3 = outputVertices.get(i);
            vertPositions[i*3+0] = vec3.get_x();
            vertPositions[i*3+1] = vec3.get_y();
            vertPositions[i*3+2] = vec3.get_z();
        }
        let remeshedBufferGeo = new THREE.BufferGeometry();
        remeshedBufferGeo.setAttribute('position', new THREE.BufferAttribute(vertPositions, 3));
        remeshedBufferGeo.setIndex(new THREE.BufferAttribute(triIndices, 1));
        remeshedBufferGeo.computeVertexNormals();
        inputVertices .__destroy__();
        inputIndices  .__destroy__();
        outputVertices.__destroy__();
        outputIndices .__destroy__();
        return remeshedBufferGeo;
    }

    /** @returns {THREE.BufferGeometry} */
    createConformingTetrahedronMesh(vertices, indices){
        // First need to get the data into PhysX
        let inputVertices = new px.PxArray_PxVec3(vertices.length/3);
        let inputIndices  = new px.PxArray_PxU32 (indices.length);
        for(let i = 0; i < vertices.length; i+=3){
            inputVertices.set(i/3, new px.PxVec3(vertices[i], vertices[i+1], vertices[i+2]));
        }
        for(let i = 0; i < indices.length; i++){
            inputIndices.set(i, indices[i]);
            if(indices[i] < 0 || indices[i] >= inputVertices.size()){
                console.log("Index out of range!", i, indices[i], inputVertices.size());
            }
        }

        // Next need to make the PxBoundedData for both the vertices and indices to make the 'Simple'TriangleMesh
        let vertexData = new px.PxBoundedData();
        let indexData  = new px.PxBoundedData();
        vertexData.set_count(inputVertices.size ());
        vertexData.set_data (inputVertices.begin());
        indexData .set_count(inputIndices .size ());
        indexData .set_data (inputIndices .begin());
        let simpleMesh = new px.PxSimpleTriangleMesh();
        simpleMesh.set_points   (vertexData);
        simpleMesh.set_triangles( indexData);

        let analysis = px.PxTetMaker.prototype.validateTriangleMesh(simpleMesh);
        if (!analysis.isSet(px.PxTriangleMeshAnalysisResultEnum.eVALID) || analysis.isSet(px.PxTriangleMeshAnalysisResultEnum.eMESH_IS_INVALID)){
            console.log(  "eVALID",                                  analysis.isSet(px.PxTriangleMeshAnalysisResultEnum.eVALID),
                        "\neZERO_VOLUME",                            analysis.isSet(px.PxTriangleMeshAnalysisResultEnum.eZERO_VOLUME),
                        "\neOPEN_BOUNDARIES",                        analysis.isSet(px.PxTriangleMeshAnalysisResultEnum.eOPEN_BOUNDARIES),
                        "\neSELF_INTERSECTIONS",                     analysis.isSet(px.PxTriangleMeshAnalysisResultEnum.eSELF_INTERSECTIONS),
                        "\neINCONSISTENT_TRIANGLE_ORIENTATION",      analysis.isSet(px.PxTriangleMeshAnalysisResultEnum.eINCONSISTENT_TRIANGLE_ORIENTATION),
                        "\neCONTAINS_ACUTE_ANGLED_TRIANGLES",        analysis.isSet(px.PxTriangleMeshAnalysisResultEnum.eCONTAINS_ACUTE_ANGLED_TRIANGLES),
                        "\neEDGE_SHARED_BY_MORE_THAN_TWO_TRIANGLES", analysis.isSet(px.PxTriangleMeshAnalysisResultEnum.eEDGE_SHARED_BY_MORE_THAN_TWO_TRIANGLES),
                        "\neCONTAINS_DUPLICATE_POINTS",              analysis.isSet(px.PxTriangleMeshAnalysisResultEnum.eCONTAINS_DUPLICATE_POINTS),
                        "\neCONTAINS_INVALID_POINTS",                analysis.isSet(px.PxTriangleMeshAnalysisResultEnum.eCONTAINS_INVALID_POINTS),
                        "\neREQUIRES_32BIT_INDEX_BUFFER",            analysis.isSet(px.PxTriangleMeshAnalysisResultEnum.eREQUIRES_32BIT_INDEX_BUFFER),
                        "\neTRIANGLE_INDEX_OUT_OF_RANGE",            analysis.isSet(px.PxTriangleMeshAnalysisResultEnum.eTRIANGLE_INDEX_OUT_OF_RANGE),
                        "\neMESH_IS_PROBLEMATIC",                    analysis.isSet(px.PxTriangleMeshAnalysisResultEnum.eMESH_IS_PROBLEMATIC),
                        "\neMESH_IS_INVALID",                        analysis.isSet(px.PxTriangleMeshAnalysisResultEnum.eMESH_IS_INVALID));
        }
        
        // Now we should be able to make the Conforming Tetrahedron Mesh
        let outputVertices = new px.PxArray_PxVec3();
        let outputIndices  = new px.PxArray_PxU32 ();
        px.PxTetMaker.prototype.createConformingTetrahedronMesh(simpleMesh, outputVertices, outputIndices, true, 0.0001);

        // Transform From PxVec3 to THREE.Vector3
        let tetIndices = new Uint32Array(outputIndices.size());
        for(let i = 0; i < tetIndices.length; i++){
            tetIndices[i] = outputIndices.get(i);
        }
        let vertPositions = new Float32Array(outputVertices.size() * 3);
        for(let i = 0; i < outputVertices.size(); i++){
            let vec3 = outputVertices.get(i);
            vertPositions[i*3+0] = vec3.get_x();
            vertPositions[i*3+1] = vec3.get_y();
            vertPositions[i*3+2] = vec3.get_z();
        }
        let remeshedBufferGeo = new THREE.BufferGeometry();
        remeshedBufferGeo.setAttribute('position', new THREE.BufferAttribute(vertPositions, 3));
        remeshedBufferGeo.setIndex(new THREE.BufferAttribute(tetIndices, 1));
        inputVertices .__destroy__();
        inputIndices  .__destroy__();
        vertexData    .__destroy__();
        indexData     .__destroy__();
        simpleMesh    .__destroy__();
        outputVertices.__destroy__();
        outputIndices .__destroy__();
        return remeshedBufferGeo;
    }
}

new Main();
