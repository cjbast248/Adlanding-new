#!/bin/bash

# Dental AI Predictor - 1-Click VPS Deployment Script
# Supports Ubuntu 20.04/22.04/24.04 and Debian 11/12.
# Run this as ROOT!

set -e

echo "=========================================================="
echo "🚀 Starting Dental AI Backend Deployment on VPS 🚀"
echo "=========================================================="

if [ "$EUID" -ne 0 ]; then
  echo "Please run this script as root (e.g., sudo bash setup_vps.sh)"
  exit 1
fi

PROJECT_DIR="/opt/dental_ai_poc"
REPO_URL="https://github.com/cjbast248/Adlanding-new.git"
SUPABASE_TOKEN="sbp_7d73fbd608cb164949347b346335683ff7e0f5fe"

echo "[1/6] Updating system and installing dependencies..."
apt-get update -y
apt-get install -y libgl1 libglib2.0-0 python3 python3-pip python3-venv git curl ufw

echo "[2/6] Cloning the repository (Adlanding-new)..."
if [ -d "$PROJECT_DIR" ]; then
    echo "Directory $PROJECT_DIR already exists. Cleaning up..."
    rm -rf "$PROJECT_DIR"
fi
git clone "$REPO_URL" "$PROJECT_DIR"

cd "$PROJECT_DIR"

# Wait, the repo structure has frontend/ and ml/. Where is server.py?
# If the user pushed server.py to the root of Adlanding-new, it will run.
# Otherwise, we warn them.
if [ ! -f "server.py" ]; then
    echo "WARNING: server.py not found in the root of the repository!"
    echo "Please ensure you pushed the entire backend code to GitHub."
fi

echo "[3/6] Setting up Python Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

echo "[4/6] Installing Heavy AI Dependencies (PyTorch CPU, FastAPI, Open3D)..."
# We force CPU PyTorch to save massive amounts of RAM and disk space on a typical VPS.
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn python-multipart supabase python-dotenv
# Re-install other potential requirements without relying on full requirements.txt which might fetch GPU torch
pip install trimesh open3d numpy pydantic

echo "[ML-FIX] Setting up 4GB Swapfile to prevent Out of Memory (OOM) crashes..."
if [ ! -f /swapfile ]; then
    # Some VPS kernels don't support fallocate for swap, default to dd
    fallocate -l 4G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=4096
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile || true
    grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "Swapfile created successfully!"
else
    echo "Swapfile already exists."
fi

echo "[5/6] Creating Configuration Files..."
cat <<EOF > .env
SUPABASE_URL=https://<ВАШ_ПРОЕКТ>.supabase.co
SUPABASE_KEY=${SUPABASE_TOKEN}
PORT=80
EOF

echo "[6/6] Configuring Systemd Daemon for 24/7 uptime..."
cat <<EOF > /etc/systemd/system/dental_ai.service
[Unit]
Description=Dental AI FastAPI Server
After=network.target

[Service]
User=root
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PROJECT_DIR}/venv/bin/uvicorn server:app --host 0.0.0.0 --port 80
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable dental_ai
systemctl restart dental_ai

echo "Configuring firewall (allowing port 80)..."
ufw allow 80/tcp || true
ufw allow 22/tcp || true
ufw --force enable || true

echo "=========================================================="
echo "✅ DEPLOYMENT SUCCESSFUL! ✅"
echo "=========================================================="
echo "Your AI Predictor API is now running on Port 80."
echo "You can test it by going to: http://$(curl -s ifconfig.me)/"
echo "To view live logs, run: sudo journalctl -u dental_ai -f"
echo "=========================================================="
