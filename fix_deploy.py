import os
import shutil
import subprocess

# 1. Move frontend to the root so GitHub Pages finds index.html
if os.path.exists("frontend/index.html"):
    shutil.move("frontend/index.html", ".")
if os.path.exists("frontend/app.js"):
    shutil.move("frontend/app.js", ".")
if os.path.exists("frontend/index.css"):
    shutil.move("frontend/index.css", ".")

# 2. Update setup_vps.sh to install missing libgl1 for Open3D
with open("setup_vps.sh", "r", encoding="utf-8") as f:
    vps_script = f.read()

if "libgl1" not in vps_script:
    vps_script = vps_script.replace("apt-get install -y python3", "apt-get install -y libgl1 libglib2.0-0 python3")
    with open("setup_vps.sh", "w", encoding="utf-8") as f:
        f.write(vps_script)

# 3. Add to git and push
os.system("git add .")
os.system('git commit -m "Fix frontend root and add missing libGL for Open3D"')
os.system("git push origin main")
