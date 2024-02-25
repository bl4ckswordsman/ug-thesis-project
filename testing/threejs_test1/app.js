import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { PMREMGenerator } from 'three';

let scene, camera, renderer, controls, loader;

function init() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer();

    controls = new OrbitControls(camera, renderer.domElement);
    controls.minPolarAngle = 0; // radians
    controls.maxPolarAngle = Math.PI / 2; // radians
    controls.maxDistance = 50; // Limit the maximum distance for zooming out

    loader = new GLTFLoader();

    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // Create a PMREMGenerator
    const pmremGenerator = new PMREMGenerator(renderer);

    // Load HDR environment map
    const rgbeLoader = new RGBELoader();
    rgbeLoader.load('models/env/venice_dawn.hdr', function (texture) {
        const envMap = pmremGenerator.fromEquirectangular(texture).texture;
        scene.background = envMap;
        scene.environment = envMap;
        texture.dispose();
        pmremGenerator.dispose();
    });

    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);

    // Add point light
    const pointLight = new THREE.PointLight(0xffffff, 1, 100);
    pointLight.position.set(50, 50, 50);
    scene.add(pointLight);

    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(1, 1, 1).normalize();
    scene.add(light);

    // Create ground
    const groundGeometry = new THREE.PlaneGeometry(100, 100);
    const groundMaterial = new THREE.MeshStandardMaterial({ color: 0x888888 });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2; // Rotate the ground to be horizontal
    ground.position.y = -13; // Lower the ground level
    scene.add(ground);

    loader.load('models/model1.gltf', function(gltf) {
        console.log('Model loaded'); // Log when the model is loaded
        scene.add(gltf.scene);
        camera.position.z = 30; // Move the camera back
        gltf.scene.position.y = -12; // Move the model down
    }, undefined, function(error) {
        console.error(error);
    });
}

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

init();
animate();