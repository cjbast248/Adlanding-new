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

const jawMaterial = new THREE.MeshStandardMaterial({ color: 0xe5e7eb, roughness: 0.6, metalness: 0.1 });
const toothMaterial = new THREE.MeshStandardMaterial({ color: 0x3b82f6, roughness: 0.4, metalness: 0.2 });

let currentJawMesh = null;
let currentPredictedMesh = null;

// Handle File Select
const apiUrlInput = document.getElementById('apiUrl');
const scanUpload = document.getElementById('scanUpload');
const predictBtn = document.getElementById('predictBtn');
const logs = document.getElementById('logs');
let selectedFile = null;

function log(msg) {
    logs.innerHTML += `\n> ${msg}`;
    logs.scrollTop = logs.scrollHeight;
}

function fitCameraToObject(camera, object, offset = 1.5) {
    const boundingBox = new THREE.Box3();
    boundingBox.setFromObject(object);
    const center = new THREE.Vector3();
    boundingBox.getCenter(center);
    const size = new THREE.Vector3();
    boundingBox.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    let cameraZ = Math.abs(maxDim * offset);
    
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
        log(`Loaded ${selectedFile.name}`);
        const reader = new FileReader();
        reader.onload = function(event) {
            const contents = event.target.result;
            const geometry = loader.parse(contents);
            if (currentJawMesh) scene.remove(currentJawMesh);
            if (currentPredictedMesh) scene.remove(currentPredictedMesh);
            currentPredictedMesh = null;
            
            currentJawMesh = new THREE.Mesh(geometry, jawMaterial);
            currentJawMesh.rotation.x = -Math.PI / 2;
            scene.add(currentJawMesh);
            fitCameraToObject(camera, currentJawMesh);
        };
        reader.readAsArrayBuffer(selectedFile);
    }
});

predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    predictBtn.disabled = true;
    predictBtn.classList.add('opacity-50', 'cursor-not-allowed');
    document.getElementById('btnText').innerText = "Running PyTorch Model...";
    document.getElementById('loadingSpinner').classList.remove('hidden');
    
    let apiUrl = apiUrlInput.value.replace(/\/$/, ""); // trim trailing slash
    log(`Connecting to ML Backend: ${apiUrl}`);
    
    const formData = new FormData();
    formData.append("jaw_scan", selectedFile);
    
    try {
        const response = await fetch(`${apiUrl}/api/predict`, {
            method: 'POST',
            body: formData
        });
        
        let data;
        try {
            data = await response.json();
        } catch (je) {
            throw new Error(`Server returned non-JSON! URL: ${apiUrl}`);
        }
        
        if (data.status === "success" || data.predicted_tooth_url) {
            log(`Inference Complete. Processing output...`);
            let toothUrl = `${apiUrl}${data.predicted_tooth_url}`;
            log(`Downloading STL from ${toothUrl}`);
            
            loader.load(toothUrl, (geometry) => {
                if (currentPredictedMesh) scene.remove(currentPredictedMesh);
                currentPredictedMesh = new THREE.Mesh(geometry, toothMaterial);
                currentPredictedMesh.rotation.x = -Math.PI / 2;
                scene.add(currentPredictedMesh);
                const group = new THREE.Group();
                group.add(currentJawMesh);
                group.add(currentPredictedMesh);
                fitCameraToObject(camera, group);
                log("Prediction successfully aligned.");
            });
        } else {
            log(`Error: ${data.error || 'Unknown API failure'}`);
        }
    } catch (err) {
        log(`API Error: ${err.message}. Make sure the FastAPI python server is running and CORS is enabled.`);
    } finally {
        predictBtn.disabled = false;
        predictBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        document.getElementById('btnText').innerText = "Predict Missing Tooth";
        document.getElementById('loadingSpinner').classList.add('hidden');
    }
});

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();
