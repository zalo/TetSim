import * as THREE from '../node_modules/three/build/three.module.js';
import { GUI } from '../node_modules/three/examples/jsm/libs/dat.gui.module.js';
import { SoftBody, Grabber } from './Softbody.js';
import { dragonTetVerts, dragonTetIds, dragonTetEdgeIds, dragonAttachedVerts, dragonAttachedTriIds } from './Dragon.js';
import World from './World.js';

/** The fundamental set up and animation structures for 3D Visualization */
export default class Main {

    constructor() {
        // Configure Settings
        this.physicsParams = {
            gravity       : -9.81,
            timeStep      : 1.0 / 60.0,
            numSubsteps   : 5,
            dt            : 1.0 / (60.0 * 5.0),
            friction      : 1000.0,
            density       : 1000.0,
            devCompliance : 1.0/100000.0,
            volCompliance : 0.0,
            worldBounds   : [-2.5,-1.0, -2.5, 2.5, 10.0, 2.5],
        };
        this.gui = new GUI();
        this.gui.add(this.physicsParams, 'gravity', -100.0, 0.0, 1);
        this.gui.add(this.physicsParams, 'timeStep', 0.001, 0.016, 0.001);
        this.gui.add(this.physicsParams, 'numSubsteps', 1, 10, 1);
        this.gui.add(this.physicsParams, 'friction', 0.0, 10000.0, 100.0);
        //this.gui.add(this.physicsParams, 'density', 0.0, 10000.0, 100.0);
        this.gui.add(this.physicsParams, 'devCompliance', 1.0 / 200000.0, 1.0 / 1000.0, 0.00001);
        //this.gui.add(this.physicsParams, 'volCompliance', 0.0, 0.001, 0.00001);

        // Construct the render world
        this.world = new World(this);

        // Construct the physics world
        this.physicsScene = { softBodies : [] };
        this.dragon = new SoftBody(dragonTetVerts, dragonTetIds, dragonTetEdgeIds, this.physicsParams, 
            dragonAttachedVerts, dragonAttachedTriIds, new THREE.MeshPhongMaterial({color: 0xf78a1d}));
        this.physicsScene.softBodies.push(this.dragon);

        this.grabber = new Grabber(
            this.world.scene, this.world.renderer, this.world.camera,
            this.world.container.parentElement, this.world.controls);
        this.world.scene.add(this.dragon.edgeMesh);
        this.world.scene.add(this.dragon.visMesh);
    }

    /** Update the simulation */
    update() {
        // Simulate all of the soft bodies in the scene
        let dt = this.physicsParams.timeStep / this.physicsParams.numSubsteps;
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

}

new Main();