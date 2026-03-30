import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DentalJawDataset
from model import PointNetRegressor
import os

import argparse

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/real_scans", help="Path to STL/PLY dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{device.type.upper()}] Hardware acceleration utilized.")
    
    # Dataset Preparation
    data_dir = args.data_dir
    os.makedirs("checkpoints", exist_ok=True)
    
    try:
        dataset = DentalJawDataset(data_dir=data_dir, train=True, num_points=2048)
        if len(dataset) == 0:
            print(f"CRITICAL: No scans found in {os.path.abspath(data_dir)}")
            return
            
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return
        
    print(f"Loaded {len(dataset)} dynamically augmented samples for training.")

    # Model Setup
    model = PointNetRegressor(output_dim=7).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    criterion = nn.MSELoss()
    
    print("Starting Training Loop...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for i, (points, targets) in enumerate(dataloader):
            points, targets = points.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(points)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}], MSE Loss: {avg_loss:.5f}")
        
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
