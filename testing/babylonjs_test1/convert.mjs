import { promises as fs } from 'fs';
import pkg from 'xhr2';
const { XMLHttpRequest } = pkg;
global.XMLHttpRequest = XMLHttpRequest;

async function convert() {
    const BABYLON = await import('@babylonjs/core');
    await import('@babylonjs/loaders/glTF/index.js');

    const engine = new BABYLON.NullEngine();
    const scene = new BABYLON.Scene(engine);

    const data = await fs.readFile('models/building1/scene.gltf');
    const base64Data = Buffer.from(data).toString('base64');
    const dataUrl = `data:model/gltf+json;base64,${base64Data}`;


    await BABYLON.SceneLoader.ImportMeshAsync(null, '', dataUrl, scene);

    const gltf = await BABYLON.GLTF2Export.GLBAsync(scene, 'model');
    const glb = gltf.glTFFiles['model.glb'];
    await fs.writeFile('model.babylon', Buffer.from(glb));

    engine.dispose();
}

convert().catch(console.error);