import * as THREE from '../node_modules/three/build/three.module.js';
import { GUI } from '../node_modules/three/examples/jsm/libs/dat.gui.module.js';
import { SoftBody, Grabber } from './Softbody.js';
import { SoftBodyGPU, GPUGrabber } from './SoftbodyGPU.js';
import { dragonTetVerts, dragonTetIds, dragonTetEdgeIds, dragonAttachedVerts, dragonAttachedTriIds } from './Dragon.js';
import World from './World.js';
import { MSHLoader } from './MSHLoader.js';

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

        // Load in a sample fTetWild Mesh
        this.mshLoader = new MSHLoader();
        this.mshLoader.load('./src/Test.obj_.msh', (geometry) => {
            geometry.computeVertexNormals();
            this.tetMat = new THREE.MeshBasicMaterial(
                { color: 0xffffff, wireframe: false, transparent: true, opacity: 0.3, side: THREE.DoubleSide });
            this.tetMesh = new THREE.Mesh(geometry, this.tetMat);
            this.tetMesh.scale.set(0.02, 0.02, 0.02);
            this.world.scene.add( this.tetMesh );
        });

        // Construct the physics world
        this.physicsScene = { softBodies: [] };
        if (this.physicsParams.cpuSim) {
            this.dragon = new SoftBody(dragonTetVerts, dragonTetIds, dragonTetEdgeIds, this.physicsParams,
                dragonAttachedVerts, dragonAttachedTriIds, new THREE.MeshPhongMaterial({ color: 0xf78a1d }));
            this.physicsScene.softBodies.push(this.dragon);
            this.grabber = new Grabber(
                this.world.scene, this.world.renderer, this.world.camera,
                this.world.container.parentElement, this.world.controls);
        } else {
            this.dragon = new SoftBodyGPU(dragonTetVerts, dragonTetIds, dragonTetEdgeIds, this.physicsParams,
                dragonAttachedVerts, dragonAttachedTriIds, new THREE.MeshPhongMaterial({ color: 0xf78a1d }), this.world);
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

}

new Main();