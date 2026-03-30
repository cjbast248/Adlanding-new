import open3d as o3d
import numpy as np
import copy
import os
import torch
import urllib.request

try:
    from ml.model import PointNetRegressor
except ImportError:
    PointNetRegressor = None

class TeethPositionPredictor:
    def __init__(self, library_model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.data_dir = os.path.dirname(library_model_path)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Clinical NIH models urls
        
        

        # Auto-download professional molar for library if missing
        if not os.path.exists(library_model_path):
            print("Procedurally generating an ANATOMICAL molar crown on-the-fly...")
            try:
                import trimesh
                # MEDICAL Z-UP CONVENTION: Z = vertical axis (up), matching PLY jaw files
                # Crown body: wide in XY plane, height (small) in Z
                crown_body = trimesh.creation.box(extents=(10.0, 8.0, 5.0))  # X=mesio-distal, Y=buccal-lingual, Z=crown height
                crown_body.apply_translation([0.0, 0.0, 2.5])  # Lift crown above zero in Z

                # 4 anatomical cusps on the occlusal surface (at Z+)
                cusps = []
                for cx in [-2.5, 2.5]:
                    for cy in [-2.0, 2.0]:
                        cusp = trimesh.creation.icosphere(radius=2.8, subdivisions=2)
                        cusp.apply_scale([1.0, 1.0, 0.6])  # Flatten on Z (low cusp profile)
                        cusp.apply_translation([cx, cy, 6.5])  # Place on top of crown in Z+
                        cusps.append(cusp)

                # 2 root stubs pointing into bone (Z-)
                for rx in [-2.5, 2.5]:
                    root = trimesh.creation.cone(radius=2.0, height=7.0)
                    root.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1,0,0]))  # Flip tip to Z-
                    root.apply_translation([rx, 0.0, -2.5])  # Roots below crown in Z-
                    cusps.append(root)

                molar = trimesh.util.concatenate([crown_body] + cusps)
                molar.export(library_model_path)
                print(f"Z-up anatomical molar saved: {library_model_path}")
            except Exception as e:
                print(f"Molar generation failed: {e}")

        # Load the newly trained ML weights from Colab if they exist
        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "pointnet_final.pth")
        
        if PointNetRegressor and os.path.exists(checkpoint_path):
            print("🚀 LOADING REAL NEURAL NETWORK WEIGHTS...")
            self.model = PointNetRegressor(output_dim=7).to(self.device)
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.model.eval()
        else:
            print("AI Model weights not found. Using Heuristic simulation Mode.")

        # Load library tooth for visualization
        if os.path.exists(library_model_path):
            self.library_tooth = o3d.io.read_triangle_mesh(library_model_path)
            self.library_tooth.compute_vertex_normals()
        else:
            self.library_tooth = o3d.geometry.TriangleMesh.create_box()
            self.library_tooth.compute_vertex_normals()

    def _detect_arch_gaps(self, jaw_mesh) -> list:
        """
        Geometric gap detection along the dental arch.
        Finds ALL positions where teeth are missing by analyzing
        the density of points along the U-shaped alveolar ridge.
        Returns list of 3D gap positions.
        """
        pcd = jaw_mesh.sample_points_uniformly(number_of_points=25000)
        points = np.asarray(pcd.points)

        bbox = jaw_mesh.get_axis_aligned_bounding_box()
        min_b = bbox.min_bound
        max_b = bbox.max_bound
        z_h = max_b[2] - min_b[2]

        # === 1. Filter to alveolar process: upper 35-90% of total height ===
        # This excludes the inferior border (too low) and condyle tips (too high)
        z_min = min_b[2] + z_h * 0.35
        z_max = min_b[2] + z_h * 0.90
        mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        alv = points[mask]
        if len(alv) < 300:
            return []

        # === 2. Project onto XY plane, find the dental arch center ===
        xy = alv[:, :2]
        arch_center = np.median(xy, axis=0)

        dx = xy[:, 0] - arch_center[0]
        dy = xy[:, 1] - arch_center[1]
        angles = np.arctan2(dy, dx)
        radii  = np.sqrt(dx**2 + dy**2)

        # === 3. Scan the arch in 20 angular bins (~9° each, covers ±90°) ===
        n_bins = 20
        # Focus only on the "body" of the jaw, not ramus wings (roughly ±90° from front)
        angle_min, angle_max = np.percentile(angles, 5), np.percentile(angles, 95)
        bins = np.linspace(angle_min, angle_max, n_bins + 1)

        bin_counts   = np.zeros(n_bins)
        bin_centers3d = [None] * n_bins
        mean_z_alv   = np.mean(alv[:, 2])

        for i in range(n_bins):
            m = (angles >= bins[i]) & (angles < bins[i+1])
            cnt = np.sum(m)
            bin_counts[i] = cnt
            if cnt > 0:
                sector_pts = alv[m]
                r_sec = radii[m]
                # Use the outermost 40% of points (the arch ridge surface, not interior)
                outer_thr = np.percentile(r_sec, 60)
                outer = sector_pts[r_sec >= outer_thr]
                pos3d = np.mean(outer if len(outer) > 0 else sector_pts, axis=0)
                bin_centers3d[i] = pos3d

        if np.max(bin_counts) == 0:
            return []

        # === 4. Identify gaps: bins with <40% of median count, flanked by dense bins ===
        median_count = np.median(bin_counts[bin_counts > 0])
        gap_threshold = median_count * 0.40
        gaps = []

        for i in range(1, n_bins - 1):
            left  = bin_counts[max(0, i - 1)]
            here  = bin_counts[i]
            right = bin_counts[min(n_bins - 1, i + 1)]
            if here < gap_threshold and left > median_count * 0.5 and right > median_count * 0.5:
                # Interpolate between neighbors
                lp = bin_centers3d[i-1]
                rp = bin_centers3d[i+1]
                if lp is not None and rp is not None:
                    gaps.append((lp + rp) / 2.0)
                elif bin_centers3d[i] is not None:
                    gaps.append(bin_centers3d[i])

        return gaps

    def run_inference(self, jaw_scan_path: str, output_path: str) -> dict:
        """
        Predicts optimal positions for ALL missing teeth using geometric arch analysis.
        """
        jaw_mesh = o3d.io.read_triangle_mesh(jaw_scan_path)
        if not jaw_mesh.has_vertices():
            raise ValueError("Input jaw mesh is empty or invalid.")
        jaw_mesh.compute_vertex_normals()

        # Adaptive tooth size
        jaw_extent = jaw_mesh.get_axis_aligned_bounding_box().get_extent()
        avg_jaw_dim = np.mean(jaw_extent[:2])
        tooth_size_mm = max(avg_jaw_dim / 7.0, 8.0)

        # === Gap detection ===
        gap_positions = self._detect_arch_gaps(jaw_mesh)
        model_used_name = "Geometric Arch Gap Detector"

        # Fallback: PointNet or simple heuristic if no geometric gaps found
        if not gap_positions:
            model_used_name = "PointNet3D (Fallback)"
            if self.model:
                pcd = jaw_mesh.sample_points_uniformly(number_of_points=2048)
                pts = np.asarray(pcd.points)
                centroid = np.mean(pts, axis=0)
                pts -= centroid
                t = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.model(t).cpu().numpy()[0]
                gap_positions = [out[:3] + centroid]
            else:
                c = jaw_mesh.get_center()
                bb = jaw_mesh.get_axis_aligned_bounding_box()
                mn, mx = bb.min_bound, bb.max_bound
                gap_positions = [np.array([c[0], c[1], mn[2] + (mx[2]-mn[2])*0.70])]

        # === Build one (or multiple) teeth and combine into one output STL ===
        all_teeth = []
        for gap_pos in gap_positions:
            tooth = copy.deepcopy(self.library_tooth)
            tc = tooth.get_center()
            tooth.translate(-tc)
            te = tooth.get_axis_aligned_bounding_box().get_extent()
            s = tooth_size_mm / max(te)
            tooth.scale(s, center=(0, 0, 0))
            tooth.translate(gap_pos)
            all_teeth.append(tooth)

        if len(all_teeth) == 1:
            result_mesh = all_teeth[0]
        else:
            result_mesh = all_teeth[0]
            for t in all_teeth[1:]:
                result_mesh += t

        o3d.io.write_triangle_mesh(output_path, result_mesh)

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, 3] = gap_positions[0]

        return {
            "status": "success",
            "model_used": model_used_name,
            "gaps_found": len(gap_positions),
            "output_stl": output_path,
            "matrix": transformation_matrix.tolist()
        }
