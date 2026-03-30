import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DentalJawDataset
from model import PointNetRegressor
import os

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{device.type.upper()}] Hardware acceleration utilized.")
    
    # Hyperparameters
    epochs = 50
    batch_size = 8
    lr = 0.001
    
    # Dataset Preparation
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "real_scans")
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        dataset = DentalJawDataset(data_dir=data_dir, train=True, num_points=2048)
        if len(dataset) == 0:
            print(f"CRITICAL: No STL/PLY scans found in \n{os.path.abspath(data_dir)}\n"
                  f"Please download real scans (e.g. from 3DTeethSeg22 dataset) and put them there.")
            return
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}")
        return
        
    print(f"Loaded {len(dataset)} dynamically augmented samples for training.")

    # Model Setup
    model = PointNetRegressor(output_dim=7).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # We use MSELoss to compare [tx,ty,tz,qx,qy,qz,qw] with truth
    criterion = nn.MSELoss()
    
    print("Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (points, targets) in enumerate(dataloader):
            points, targets = points.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(points)
            
            # Loss computation
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], MSE Loss: {avg_loss:.5f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/pointnet_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"--> Saved checkpoint: {ckpt_path}")

    # Final Save
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/pointnet_final.pth")
    print("Optimization Complete! Network weights saved to checkpoints/pointnet_final.pth")

if __name__ == "__main__":
    train()
