import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { PMREMGenerator } from 'three';

let scene, camera, renderer, controls, loader, raycaster, mouse, INTERSECTED;

// Default  z camera position
let z_def = 10;
// Default y offset for the model
let y_def = -2;
// Default ground position
let ground_y = -3.3;

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

    // Create a Raycaster and a Vector2 for the mouse position
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    // Add a listener for the mouse move event
    window.addEventListener('mousemove', onMouseMove, false);

    // Create a PMREMGenerator
    const pmremGenerator = new PMREMGenerator(renderer);

    // Load HDR environment map
    const rgbeLoader = new RGBELoader();
    rgbeLoader.load('models/env/kloppenheim_puresky.hdr', function (texture) {
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
    ground.position.y = ground_y; // Lower the ground level
    scene.add(ground);

    loader.load('models/model2-3.glb', function(gltf) {
        console.log('Model loaded'); // Log when the model is loaded
        scene.add(gltf.scene);
        camera.position.z = z_def; // Move the camera back
        gltf.scene.position.y = y_def; // Move the model down

        // Store the loaded model in a variable accessible in the animate function
        window.model = gltf.scene;
    }, undefined, function(error) {
        console.error(error);
    });
}

function onMouseMove(event) {
    // Calculate mouse position in normalized device coordinates (-1 to +1) for both components
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
}

function animate() {
    requestAnimationFrame(animate);

    // Update the picking ray with the camera and mouse position
    raycaster.setFromCamera(mouse, camera);

    // Calculate objects intersecting the picking ray
    const intersects = raycaster.intersectObjects(window.model.children, true);

    if (intersects.length > 0) {
        if (INTERSECTED != intersects[0].object) {
            if (INTERSECTED && INTERSECTED.material && INTERSECTED.material.emissive)
                INTERSECTED.material.emissive.setHex(INTERSECTED.currentHex);

            INTERSECTED = intersects[0].object;
            if (INTERSECTED.material && INTERSECTED.material.emissive) {
                INTERSECTED.currentHex = INTERSECTED.material.emissive.getHex();
                INTERSECTED.material.emissive.setHex(0xff0000);
            }
        }
    } else {
        if (INTERSECTED && INTERSECTED.material && INTERSECTED.material.emissive)
            INTERSECTED.material.emissive.setHex(INTERSECTED.currentHex);
        INTERSECTED = null;
    }

    renderer.render(scene, camera);
}

init();
animate();