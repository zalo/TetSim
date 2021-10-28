# [TetSim](https://zalo.github.io/TetSim/)

<p align="left">
  <a href="https://github.com/zalo/TetSim/deployments/activity_log?environment=github-pages">
      <img src="https://img.shields.io/github/deployments/zalo/TetSim/github-pages?label=Github%20Pages%20Deployment" title="Github Pages Deployment"></a>
  <a href="https://github.com/zalo/TetSim/commits/master">
      <img src="https://img.shields.io/github/last-commit/zalo/TetSim" title="Last Commit Date"></a>
  <!--<a href="https://github.com/zalo/TetSim/blob/master/LICENSE">
      <img src="https://img.shields.io/github/license/zalo/TetSim" title="License: Apache V2"></a>-->  <!-- No idea what license this should be! -->
</p>

This is a rehosting of Miles Macklin and Matthias Müller's [A Constraint-based Formulation of Stable Neo-Hookean Materials](http://blog.mmacklin.com/publications/#:~:text=A%20Constraint-based%20Formulation%20of%20Stable%20Neo-Hookean%20Materials) demo.  I've broken out the script into several ES6 Module pieces with the intention of improving reusability and porting to gpu.js.

 ## Dependencies
 - [three.js](https://github.com/mrdoob/three.js/) (3D Rendering Engine)
 - [esbuild](https://github.com/evanw/esbuild/) (Bundler)
