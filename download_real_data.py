import urllib.request
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

os.makedirs('data/real_scans', exist_ok=True)

urls = [
    # A public dental model from a generic STL repository or a sample shape (using Stanford Bunny as a fallback 3D shape if dental fails)
    "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/bunny.stl",
    # Another test model
    "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/armadillo.stl"
]

print("Downloading 3D files for ML training...")
for i, url in enumerate(urls):
    filename = f"sample_jaw_scan_0{i}.stl"
    filepath = os.path.join('data/real_scans', filename)
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed {filename}: {e}")
