// Global Three.js Setup
const container = document.getElementById('viewer3d');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a); // Deep professional black

const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 1, 1000);
camera.position.set(0, -100, 100);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// Professional Medical Lighting (Studio Setup)
const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambientLight);

const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
mainLight.position.set(100, 100, 100);
scene.add(mainLight);

const fillLight = new THREE.DirectionalLight(0x93c5fd, 0.5); // Subtle blue fill
fillLight.position.set(-100, 50, 50);
scene.add(fillLight);

const rimLight = new THREE.PointLight(0xffffff, 0.6);
rimLight.position.set(0, -100, -50);
scene.add(rimLight);

// Loaders
const stlLoader = new THREE.STLLoader();
const plyLoader = new THREE.PLYLoader();

// Materials - PLY files can have vertex colors
const jawMaterial = new THREE.MeshPhongMaterial({ 
    color: 0xcccccc,       // Pearl white bone color
    vertexColors: false,    // Will switch to true for PLY
    specular: 0x444444, 
    shininess: 50,
    flatShading: false
}); 

const jawMaterialPLY = new THREE.MeshPhongMaterial({ 
    vertexColors: true,     // Use embedded PLY vertex colors
    specular: 0x222222, 
    shininess: 50,
    flatShading: false
}); 

const toothMaterial = new THREE.MeshPhongMaterial({ 
    color: 0x2563eb,
    emissive: 0x1e40af,
    emissiveIntensity: 0.3,
    transparent: true, 
    opacity: 0.80,
    shininess: 120
});

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

function getLoaderForFile(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    if (ext === 'ply') return plyLoader;
    return stlLoader;
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
            const ext = selectedFile.name.split('.').pop().toLowerCase();
            const loader = getLoaderForFile(selectedFile.name);
            const geometry = loader.parse(contents);
            if (!geometry.attributes.normal) geometry.computeVertexNormals();
            
            if (currentJawMesh) scene.remove(currentJawMesh);
            if (currentPredictedMesh) scene.remove(currentPredictedMesh);
            currentPredictedMesh = null;
            
            // Auto-select material: PLY with colors uses vertex colors, STL uses pearl white
            const hasColors = geometry.attributes && geometry.attributes.color;
            const mat = (ext === 'ply' && hasColors) ? jawMaterialPLY : jawMaterial;
            
            currentJawMesh = new THREE.Mesh(geometry, mat);
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
            log(`Downloading Predicted Tooth from ${toothUrl}`);
            
            // Predicted tooth is always STL currently
            stlLoader.load(toothUrl, (geometry) => {
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
