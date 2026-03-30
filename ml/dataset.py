import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
import os
import glob

class DentalJawDataset(Dataset):
    """
    Self-Supervised Dataset.
    Loads real jaw STLs/PLYs, randomly "masks" (removes points) in a specific region simulating a missing tooth.
    The target is the centroid of the removed points and a dummy rotation.
    This creates an infinite variety of training data from a small set of healthy jaw scans.
    """
    def __init__(self, data_dir, num_points=2048, train=True):
        self.data_dir = data_dir
        self.num_points = num_points
        self.file_paths = glob.glob(os.path.join(data_dir, "*.stl")) + glob.glob(os.path.join(data_dir, "*.ply"))
        self.train = train

    def __len__(self):
        # We multiply length for training to reuse same scans with different "missing" teeth
        return max(len(self.file_paths) * 10 if self.train else len(self.file_paths), 0)

    def __getitem__(self, idx):
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No STL/PLY files found in {self.data_dir}. Please upload real medical scans.")

        file_idx = idx % len(self.file_paths)
        file_path = self.file_paths[file_idx]
        
        # Load mesh
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertices():
            # fallback if file corrupt (generates random noise just so DataLoader doesn't crash)
            points = np.random.rand(self.num_points, 3).astype(np.float32)
            return torch.tensor(points), torch.zeros(7, dtype=torch.float32)

        # Sample points to work with Point Cloud (simulating a scanner input)
        pcd = mesh.sample_points_uniformly(number_of_points=10000)
        points = np.asarray(pcd.points)

        # --- Self-Supervised Masking Logic ---
        # 1. Pick a random point on the jaw to be the "center" of the missing tooth
        center_idx = np.random.randint(0, len(points))
        tooth_center = points[center_idx]
        
        # 2. Mask points within a certain radius (simulating 1 tooth gap: radius ~ 6-8mm)
        # Using 8.0 units (assuming scan is in mm)
        radius = 8.0 
        distances = np.linalg.norm(points - tooth_center, axis=1)
        valid_indices = distances > radius
        
        # The 'jaw with missing tooth'
        masked_points = points[valid_indices]
        
        # Ground truth target: [tx, ty, tz, qx, qy, qz, qw]
        # For this network, we want it to predict the tooth_center!
        # Identity quaternion [0,0,0,1] since we are only estimating position for now in this dataset simulation
        quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        target = np.concatenate([tooth_center, quaternion]).astype(np.float32)

        # 3. Downsample or pad masked points to fixed size (num_points = 2048) for PointNet
        if len(masked_points) >= self.num_points:
            choice = np.random.choice(len(masked_points), self.num_points, replace=False)
            final_points = masked_points[choice]
        else:
            # If the scan was too small, sample with replacement
            choice = np.random.choice(len(masked_points), self.num_points, replace=True)
            final_points = masked_points[choice]

        # 4. Normalize jaw to center (PointNet needs centered input)
        centroid = np.mean(final_points, axis=0)
        final_points -= centroid
        
        # The target translation must also be shifted proportionally!
        target[:3] -= centroid
        
        return torch.tensor(final_points, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="../data", help="Directory with STLs to test dataset")
    args = parser.parse_args()

    # Test dataset instantiation
    print(f"Testing loader on directory: {args.test_dir}")
    try:
        ds = DentalJawDataset(data_dir=args.test_dir)
        if len(ds) > 0:
            pts, tgt = ds[0]
            print(f"Dataset generated properly!")
            print(f"Output Pts shape: {pts.shape}, Target shape: {tgt.shape}")
        else:
            print("Dataset empty.")
    except Exception as e:
        print(f"Dataset test failed (ensure files exist): {e}")
