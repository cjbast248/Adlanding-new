from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
from dotenv import load_dotenv

from predictor import TeethPositionPredictor
from exocad_exporter import export_to_exocad

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client, Client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Connected to Supabase (Cloud Deployment Mode).")
    except Exception as e:
        print(f"Failed to initialize Supabase client: {e}")
else:
    print("Supabase credentials not found. Running in Local Mode.")

app = FastAPI(title="Dental AI Predictor (PoC)")

# Enable CORS for frontend hosted on GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, change to Github Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# For UI backwards compatibility, we can still serve the frontend locally
app.mount("/static", StaticFiles(directory="static"), name="static")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize AI Predictor (PointNet / Heuristic stub)
library_path = os.path.join(DATA_DIR, "library_tooth.stl")
predictor = TeethPositionPredictor(library_path)

@app.get("/")
def serve_frontend():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return JSONResponse(status_code=404, content={"message": "Frontend not found locally. Please use GitHub Pages URL."})

@app.post("/api/predict")
async def predict_missing_tooth(jaw_scan: UploadFile = File(...)):
    # Unique ID for tracking
    file_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(jaw_scan.filename)[1].lower()
    
    # Save uploaded file locally
    input_filename = f"scan_{file_id}{ext}"
    input_path = os.path.join(DATA_DIR, input_filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(jaw_scan.file, buffer)
        
    # Optional: Upload to Supabase Storage before processing
    if supabase_client:
        try:
            with open(input_path, 'rb') as f:
                supabase_client.storage.from_("scans").upload(file=f, path=input_filename)
        except Exception as e:
            print(f"Supabase Upload Warning: {e}. Note: Did you create the 'scans' storage bucket?")
        
    # Run ML Inference
    output_filename = f"predicted_{file_id}.stl"
    output_path = os.path.join(DATA_DIR, output_filename)
    
    try:
        result = predictor.run_inference(input_path, output_path)
        
        # Matrix Export
        export_to_exocad(tooth_id=46, transformation_matrix=result["matrix"], export_dir=DATA_DIR)
        
        # Upload prediction back to Supabase if Cloud Deployment Mode
        if supabase_client:
            try:
                with open(output_path, 'rb') as f:
                    supabase_client.storage.from_("predictions").upload(file=f, path=output_filename)
                
                # Fetch public URLs if needed
                # pub_url = supabase_client.storage.from_("predictions").get_public_url(output_filename)
            except Exception as e:
                print(f"Supabase Prediction Upload Warning: {e}")
        
        return JSONResponse(content={
            "status": "success",
            "message": "AI Inference completed.",
            "predicted_tooth_url": f"/api/download/{output_filename}",
            "exocad_data": f"/api/download/exocad_matrix_46.json",
            "matrix": result["matrix"]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/octet-stream", filename=filename)
    return JSONResponse(status_code=404, content={"error": "File not found locally."})

if __name__ == "__main__":
    import uvicorn
    # If running on Render, the port is provided via PORT env var.
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
