import urllib.request
import os

# Create directories
os.makedirs("data/real_scans", exist_ok=True)

# List of open source STL files from GitHub (sample mandibles/teeth/skulls)
urls = {
    "sample_jaw_1.stl": "https://raw.githubusercontent.com/ArsoVukicevic/OpenMandible/main/stl/Mandible.stl",
    "sample_jaw_2.stl": "https://raw.githubusercontent.com/ArsoVukicevic/OpenMandible/main/stl/Cortical_bone.stl"
}

print("Downloading real open-source Mandible (Jaw) STL scans for training...")

for filename, url in urls.items():
    filepath = os.path.join("data/real_scans", filename)
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f" -> Saved to {filepath}")
    except Exception as e:
        print(f" -> Failed to download {filename}: {e}")

print("\nDownload complete. You can now train the model using these scans!")
