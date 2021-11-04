import {
	BufferAttribute,
	BufferGeometry,
	FileLoader,
	Float32BufferAttribute,
	Loader,
	LoaderUtils,
	Vector3
} from '../node_modules/three/build/three.module.js';

/**
 * Description: A THREE loader for MSH ASCII files, as created by fTetWild and other Tetrahedralization programs.
 *
 * Only supports ASCII encoded files.
 *
 * The loader returns an indexed buffer geometry.
 *
 * Limitations:
 *  ASCII decoding assumes file is UTF-8.
 *
 * Usage:
 *  const loader = new MSHLoader();
 *  loader.load( './models/MSH/slotted_disk.msh', function ( geometry ) {
 *    scene.add( new THREE.Mesh( geometry ) );
 *  });
 *
 */

class MSHLoader extends Loader {

	constructor( manager ) {

		super( manager );

	}

	load( url, onLoad, onProgress, onError ) {

		const scope = this;

		const loader = new FileLoader( this.manager );
		loader.setPath( this.path );
		loader.setResponseType( 'arraybuffer' );
		loader.setRequestHeader( this.requestHeader );
		loader.setWithCredentials( this.withCredentials );

		loader.load( url, function ( text ) {

			try {

				onLoad( scope.parse( text ) );

			} catch ( e ) {

				if ( onError ) {

					onError( e );

				} else {

					console.error( e );

				}

				scope.manager.itemError( url );

			}

		}, onProgress, onError );

	}

	parse( data ) {

		function parseASCII( data ) {

			const geometry = new BufferGeometry();

			const indices = [];
			const tetindices = [];
			const vertices = [];

			let numVertices = 0;
			let numElements = 0;

			let inNodes = false;
			let inElements = false;

			let lines = data.split('\n');
			for (let l = 0; l < lines.length; l++) {
				if(lines[l].startsWith('$Nodes')) {
					inNodes = true; numVertices = parseInt(lines[++l].trim()); 
					console.log(numVertices); continue;
				} else if(lines[l].startsWith('$EndNodes')) {
					inNodes = false; continue;
				}
				if(lines[l].startsWith('$Elements')) {
					inElements = true;
					numElements = parseInt(lines[++l].trim());
					console.log(numElements);
					continue;
				} else if(lines[l].startsWith('$EndElements')) {
					inElements = false;  continue;
				}

				if (inNodes) {
					let currentLine = lines[l].trim().split(' ');
					if (currentLine.length === 4) {
						//indices.push(parseInt(currentLine[0]));
						vertices.push(parseFloat(currentLine[1]), parseFloat(currentLine[2]), parseFloat(currentLine[3]));
					} else {
						console.error('Invalid node line: ' + lines[l]);
					}
				}

				if (inElements) {
					let currentLine = lines[l].trim().split(' ');
					if (currentLine.length === 7) {
						//indices.push(parseInt(currentLine[0]));
						//1 4 0 1185 1192 1440 1135
						let verts = [parseInt(currentLine[3])-1, parseInt(currentLine[4])-1, parseInt(currentLine[5])-1, parseInt(currentLine[6])-1];
						indices.push(verts[0], verts[1]); // Edge Ids
						indices.push(verts[0], verts[2]);
						indices.push(verts[0], verts[3]);
						indices.push(verts[1], verts[2]);
						indices.push(verts[2], verts[3]);
						indices.push(verts[3], verts[1]);
						tetindices.push(verts[0], verts[1], verts[2], verts[3]);
					} else {
						console.error('Invalid Element line: ' + lines[l]);
					}
				}

			}

			geometry.setIndex( indices );
			geometry.setAttribute( 'position', new Float32BufferAttribute( vertices, 3 ) );
			geometry.setAttribute( 'tetIndices', new Float32BufferAttribute( tetindices, 4 ) );
			geometry.userData.vertices = vertices;
			geometry.userData.tetIndices = tetindices;
			geometry.userData.index = indices;

			return geometry;

		}

		function ensureString( buffer ) {

			if ( typeof buffer !== 'string' ) {

				return LoaderUtils.decodeText( new Uint8Array( buffer ) );

			}

			return buffer;

		}

		// start

		return parseASCII( ensureString( data ) );

	}

}

export { MSHLoader };
