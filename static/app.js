// Global Three.js Setup
const container = document.getElementById('viewer3d');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf1f5f9);

const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 1, 1000);
camera.position.set(0, -100, 100);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// Lighting for dental models
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(50, 50, 50);
scene.add(dirLight);
const backLight = new THREE.DirectionalLight(0xffffff, 0.5);
backLight.position.set(-50, -50, -50);
scene.add(backLight);

// STL Loader
const loader = new THREE.STLLoader();

const jawMaterial = new THREE.MeshStandardMaterial({ 
    color: 0xe5e7eb, 
    roughness: 0.6, 
    metalness: 0.1 
});

const toothMaterial = new THREE.MeshStandardMaterial({ 
    color: 0x3b82f6, 
    roughness: 0.4, 
    metalness: 0.2,
    emissive: 0x1d4ed8,
    emissiveIntensity: 0.2
});

let currentJawMesh = null;
let currentPredictedMesh = null;

// Handle File Select
const scanUpload = document.getElementById('scanUpload');
const predictBtn = document.getElementById('predictBtn');
const logs = document.getElementById('logs');
let selectedFile = null;

function log(msg) {
    logs.innerHTML += `\n> ${msg}`;
    logs.scrollTop = logs.scrollHeight;
}

// Fit Camera to Object
function fitCameraToObject(camera, object, offset = 1.5) {
    const boundingBox = new THREE.Box3();
    boundingBox.setFromObject(object);
    const center = new THREE.Vector3();
    boundingBox.getCenter(center);
    const size = new THREE.Vector3();
    boundingBox.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
    cameraZ *= offset;
    
    // Set camera position roughly keeping current angles but fitting
    camera.position.set(center.x, center.y - (cameraZ * 0.5), center.z + cameraZ);
    camera.far = cameraZ * 10;
    camera.updateProjectionMatrix();
    
    controls.target.copy(center);
    controls.update();
}

scanUpload.addEventListener('change', (e) => {
    selectedFile = e.target.files[0];
    if (selectedFile) {
        predictBtn.disabled = false;
        predictBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        
        log(`Loaded ${selectedFile.name} (${(selectedFile.size/1024).toFixed(1)} KB)`);
        
        // Preview locally
        const reader = new FileReader();
        reader.onload = function(event) {
            const contents = event.target.result;
            const geometry = loader.parse(contents);
            
            // We do NOT center the geometry here! Exocad relies on global coordinates.
            // Rendering it strictly where it is in 3D space.
            
            if (currentJawMesh) scene.remove(currentJawMesh);
            if (currentPredictedMesh) scene.remove(currentPredictedMesh);
            currentPredictedMesh = null;
            
            currentJawMesh = new THREE.Mesh(geometry, jawMaterial);
            
            // Rotate the mesh to standard up-axis for viewing if necessary, 
            // but usually raw coordinate view is better for clinical accuracy.
            currentJawMesh.rotation.x = -Math.PI / 2; // Common adjustment for dental scans
            
            scene.add(currentJawMesh);
            fitCameraToObject(camera, currentJawMesh);
            log("Scan visualized in 3D space.");
        };
        reader.readAsArrayBuffer(selectedFile);
    }
});

predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    // UI Update
    predictBtn.disabled = true;
    predictBtn.classList.add('opacity-50', 'cursor-not-allowed');
    document.getElementById('btnText').innerText = "Processing Engine...";
    document.getElementById('loadingSpinner').classList.remove('hidden');
    log("Initializing Core inference pipeline (PyTorch -> Open3D)...");
    
    const formData = new FormData();
    formData.append("jaw_scan", selectedFile);
    
    try {
        log("Uploading Scan. Constructing Point Cloud...");
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === "success") {
            log(`Inference Complete. Processing output...`);
            log(`Exporting Transform Matrix for Exocad.`);
            
            // Load predicted tooth
            loader.load(data.predicted_tooth_url, (geometry) => {
                if (currentPredictedMesh) scene.remove(currentPredictedMesh);
                
                currentPredictedMesh = new THREE.Mesh(geometry, toothMaterial);
                currentPredictedMesh.rotation.x = -Math.PI / 2; // same adjustment as jaw
                
                scene.add(currentPredictedMesh);
                
                const group = new THREE.Group();
                group.add(currentJawMesh);
                group.add(currentPredictedMesh);
                
                fitCameraToObject(camera, group);
                log("Prediction successfully aligned onto input scan.");
            });
        } else {
            log(`Error: ${data.error}`);
        }
    } catch (err) {
        log(`API Error: ${err.message}`);
    } finally {
        predictBtn.disabled = false;
        predictBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        document.getElementById('btnText').innerText = "Predict Missing Tooth";
        document.getElementById('loadingSpinner').classList.add('hidden');
    }
});

// Window resize handler
window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});

// Render Loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();
