import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetRegressor(nn.Module):
    def __init__(self, output_dim=7):
        super(PointNetRegressor, self).__init__()
        # output_dim = 3 (translation) + 4 (quaternion rotation) = 7
        
        # PointNet Base
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # MLP Regressor Head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # x shape: (Batch, NumPoints, 3) 
        # Conv1d expects (Batch, Channels, Length) -> transpose x
        x = x.transpose(1, 2)
        
        # PointNet Feature Extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Max Pooling (Global Feature Vector)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # Regression Head
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # x is now (Batch, 7) [tx, ty, tz, qx, qy, qz, qw]
        # Normalize quaternion to ensure valid rotation (last 4 dims)
        translation = x[:, :3]
        quaternion = F.normalize(x[:, 3:], p=2, dim=1)
        
        return torch.cat([translation, quaternion], dim=1)

if __name__ == "__main__":
    # Test model shape output
    model = PointNetRegressor(output_dim=7)
    dummy_input = torch.rand(4, 2048, 3) # Batch 4, 2048 points, 3 coords
    out = model(dummy_input)
    print("Output shape:", out.shape) # Should be [4, 7]
