import urllib.request
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

os.makedirs('data', exist_ok=True)
os.makedirs('test_data', exist_ok=True)

try:
    print("Downloading mandible...")
    urllib.request.urlretrieve(
        "https://3dprint.nih.gov/system/files/3d_model/3DPX-001002/Human_Mandible.stl", 
        "test_data/Human_Mandible_NIH.stl"
    )
    print("Downloading molar...")
    urllib.request.urlretrieve(
        "https://3dprint.nih.gov/system/files/3d_model/3DPX-000571/Human_Mandibular_Molar.stl", 
        "data/library_tooth.stl"
    )
    # Also put a copy of molar in test_data for the frontend download buttons
    import shutil
    shutil.copy("data/library_tooth.stl", "test_data/Human_Mandibular_Molar_NIH.stl")
    print("Downloads complete.")
except Exception as e:
    print(f"Error: {e}")
