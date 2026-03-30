"""
Tooth-Aware Self-Supervised Dataset
─────────────────────────────────────
Strategy:
  1. For each jaw scan, detect the REAL tooth positions (Z-protrusions on the alveolar arch).
  2. For each detected tooth: create a masked point cloud (that tooth removed).
  3. Target = the removed tooth's 3-D centroid (normalised to jaw centroid).

This makes PointNet learn the mapping:
  "jaw with a hole" → 3-D position of the missing tooth

Because we detect multiple teeth per scan, each scan produces many training pairs.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
import os
import glob


# ─── Tooth detection on an alveolar surface ─────────────────────────────────
def detect_teeth_on_arch(points: np.ndarray, n_candidates: int = 16) -> list:
    """
    Find approximate tooth centroid positions on a jaw point cloud.

    Method:
      • Filter to the alveolar process (upper-mid Z band).
      • Project to XY, identify the U-shaped arch.
      • Scan the arch in angular sectors and find Z-local-maxima (= crown tips).
      • Return up to `n_candidates` highest Z-protrusions.
    """
    if len(points) < 100:
        return []

    # ── 1. Alveolar region filter ────────────────────────────────────────────
    z = points[:, 2]
    z_min, z_max = z.min(), z.max()
    z_h = z_max - z_min

    x = points[:, 0]
    x_min, x_max = x.min(), x.max()
    x_w = x_max - x_min

    # Height band: 25-70%, X band: inner 80% (exclude ramus wings)
    mask = (
        (z >= z_min + z_h * 0.25) & (z <= z_min + z_h * 0.70) &
        (x >= x_min + x_w * 0.10) & (x <= x_max - x_w * 0.10)
    )
    alv = points[mask]
    if len(alv) < 100:
        return []

    # ── 2. Polar scan from arch centroid ─────────────────────────────────────
    xy = alv[:, :2]
    cx, cy = np.median(xy, axis=0)
    dx, dy = xy[:, 0] - cx, xy[:, 1] - cy
    angles = np.arctan2(dy, dx)

    # Find the back of the U (lowest density sector)
    n_bins = 36
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    cnts = np.array([np.sum((angles >= bins[i]) & (angles < bins[i+1])) for i in range(n_bins)])
    back_mid = (bins[np.argmin(cnts)] + bins[np.argmin(cnts) + 1]) / 2
    front_ang = back_mid + np.pi

    # ── 3. Scan 16 tooth positions, find Z-maxima (crown tips) ──────────────
    offsets = [4, 13, 26, 40, 52, 63, 73, 83]   # degrees from front
    tooth_positions = []

    for off_deg in offsets:
        for side in (+1, -1):
            target_ang = front_ang + np.radians(off_deg * side)
            ang_diff = np.abs(((angles - target_ang + np.pi) % (2 * np.pi)) - np.pi)
            near = alv[ang_diff < np.radians(9.0)]
            if len(near) < 5:
                continue

            # Take the TOP 10% highest Z points in this sector → crown tips
            z_near = near[:, 2]
            top_mask = z_near >= np.percentile(z_near, 90)
            crown_tip = np.mean(near[top_mask], axis=0)

            # Exclude if Z is not elevated enough (flat bone, no tooth)
            z_mean_sector = np.mean(z_near)
            if crown_tip[2] < z_mean_sector + 0.5:  # < 0.5 mm above mean → not a tooth
                continue

            tooth_positions.append(crown_tip)

    return tooth_positions


# ─── Dataset ─────────────────────────────────────────────────────────────────
class DentalJawDataset(Dataset):
    """
    Self-supervised dataset.  For each jaw scan we detect real teeth, then
    create one training example per tooth (remove that tooth → predict position).
    """

    TOOTH_MASK_RADIUS = 7.5  # mm — typical molar crown radius

    def __init__(self, data_dir: str, num_points: int = 2048, train: bool = True):
        self.num_points = num_points
        self.train      = train

        file_paths = (
            glob.glob(os.path.join(data_dir, "*.ply")) +
            glob.glob(os.path.join(data_dir, "*.stl"))
        )

        if not file_paths:
            raise FileNotFoundError(f"No PLY/STL files found in: {data_dir}")

        # Pre-cache: for each file, detect teeth → build (file, tooth_pos) pairs
        print(f"Scanning {len(file_paths)} jaw files for tooth positions...")
        self.samples = []  # list of (full_points_ndarray, tooth_center_ndarray)

        for fp in file_paths:
            try:
                mesh = o3d.io.read_triangle_mesh(fp)
                if not mesh.has_vertices():
                    continue
                pcd = mesh.sample_points_uniformly(number_of_points=12000)
                pts = np.asarray(pcd.points, dtype=np.float32)

                teeth = detect_teeth_on_arch(pts)
                if teeth:
                    for tc in teeth:
                        self.samples.append((pts, tc.astype(np.float32)))
                    print(f"  {os.path.basename(fp)}: {len(teeth)} teeth detected")
                else:
                    # Fallback: use random sampling if no teeth found (older approach)
                    for _ in range(3):
                        idx = np.random.randint(0, len(pts))
                        self.samples.append((pts, pts[idx]))
                    print(f"  {os.path.basename(fp)}: no teeth detected, using fallback")
            except Exception as e:
                print(f"  SKIP {os.path.basename(fp)}: {e}")

        if not self.samples:
            raise RuntimeError("No training samples could be extracted.")

        print(f"\n✅ Total training pairs: {len(self.samples)}")

        # In training mode, multiply the dataset by random augmentations
        self._len = len(self.samples) * (5 if train else 1)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        pts, tooth_center = self.samples[idx % len(self.samples)]
        pts = pts.copy()

        # ── Augmentation: random rotation around Z (dental arch orientation varies) ──
        if self.train:
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            pts          = pts @ R.T
            tooth_center = (R @ tooth_center).astype(np.float32)

        # ── Mask the tooth (simulate missing tooth) ──────────────────────────
        dists = np.linalg.norm(pts - tooth_center, axis=1)
        keep  = dists > self.TOOTH_MASK_RADIUS
        masked = pts[keep]

        # Ensure fixed size
        if len(masked) >= self.num_points:
            choice = np.random.choice(len(masked), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(masked), self.num_points, replace=True)
        final = masked[choice]

        # ── Normalize ────────────────────────────────────────────────────────
        centroid = np.mean(final, axis=0)
        final    -= centroid
        target_pos = tooth_center - centroid   # <- what PointNet must predict

        # Output: 7-dim [tx, ty, tz, qx, qy, qz, qw]  (identity quaternion)
        target = np.concatenate([target_pos, [0., 0., 0., 1.]]).astype(np.float32)
        return torch.tensor(final, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../test_data/M_mandible"
    ds = DentalJawDataset(data_dir=data_dir)
    pts, tgt = ds[0]
    print(f"Sample → points: {pts.shape}, target: {tgt}")
