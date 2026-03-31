import open3d as o3d
import numpy as np
import copy
import os
import torch

try:
    from ml.model import PointNetRegressor
except ImportError:
    PointNetRegressor = None

# ─────────────────────────────────────────────
#   TOOTH TYPE DEFINITIONS (Z-up medical coords)
# ─────────────────────────────────────────────
TOOTH_TYPES = {
    #             (min°, max°)  MD    BL   Ht  cusps  r
    "incisor":  {"arch_angle_deg": (0,  18),  "mesio_distal": 7.0,  "buccal_lingual": 7.5, "crown_h": 5.5, "cusps": 1, "cusp_r": 2.5},
    "canine":   {"arch_angle_deg": (18, 34),  "mesio_distal": 8.0,  "buccal_lingual": 8.0, "crown_h": 7.0, "cusps": 1, "cusp_r": 3.0},
    "premolar": {"arch_angle_deg": (34, 54),  "mesio_distal": 9.0,  "buccal_lingual": 9.0, "crown_h": 6.0, "cusps": 2, "cusp_r": 2.8},
    "molar":    {"arch_angle_deg": (54, 110), "mesio_distal": 12.0, "buccal_lingual": 11.0,"crown_h": 5.5, "cusps": 4, "cusp_r": 2.8},
}

def _build_tooth_mesh(kind: str):
    """Procedurally generate an anatomical tooth in Z-up convention."""
    import trimesh
    info = TOOTH_TYPES[kind]
    md = info["mesio_distal"]    # X
    bl = info["buccal_lingual"]  # Y
    ch = info["crown_h"]         # Z crown height
    n  = info["cusps"]
    cr = info["cusp_r"]

    # Crown body (raised above zero so roots go into Z-)
    crown = trimesh.creation.box(extents=(md, bl, ch))
    crown.apply_translation([0, 0, ch / 2])
    parts = [crown]

    # Cusps on top (Z+)
    if n == 1:
        cusp = trimesh.creation.icosphere(radius=cr, subdivisions=2)
        cusp.apply_scale([1.0, 0.8, 0.7])
        cusp.apply_translation([0, 0, ch + cr * 0.5])
        parts.append(cusp)
    elif n == 2:
        for cx in [-md * 0.22, md * 0.22]:
            cusp = trimesh.creation.icosphere(radius=cr, subdivisions=2)
            cusp.apply_scale([1.0, 0.9, 0.65])
            cusp.apply_translation([cx, 0, ch + cr * 0.45])
            parts.append(cusp)
    elif n == 4:
        for cx in [-md * 0.20, md * 0.20]:
            for cy in [-bl * 0.18, bl * 0.18]:
                cusp = trimesh.creation.icosphere(radius=cr, subdivisions=2)
                cusp.apply_scale([1.0, 1.0, 0.62])
                cusp.apply_translation([cx, cy, ch + cr * 0.4])
                parts.append(cusp)

    # Root stubs (Z-)
    n_roots = 1 if kind in ("incisor", "canine") else (2 if kind == "premolar" else 2)
    root_h  = 8.0 if kind in ("molar", "canine") else 7.0
    root_r  = md * 0.18
    if n_roots == 1:
        root = trimesh.creation.cone(radius=root_r * 1.4, height=root_h)
        root.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        root.apply_translation([0, 0, -root_h * 0.15])
        parts.append(root)
    else:
        for rx in [-md * 0.22, md * 0.22]:
            root = trimesh.creation.cone(radius=root_r, height=root_h)
            root.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
            root.apply_translation([rx, 0, -root_h * 0.15])
            parts.append(root)

    return trimesh.util.concatenate(parts)


class TeethPositionPredictor:
    def __init__(self, library_model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = None
        self.data_dir = os.path.dirname(library_model_path)
        os.makedirs(self.data_dir, exist_ok=True)

        # ── Build tooth library for all 4 types ──────────────────────────────
        self.tooth_library = {}
        for kind in TOOTH_TYPES:
            stl_path = os.path.join(self.data_dir, f"tooth_{kind}.stl")
            if not os.path.exists(stl_path):
                print(f"Generating {kind} tooth model...")
                try:
                    mesh = _build_tooth_mesh(kind)
                    mesh.export(stl_path)
                except Exception as e:
                    print(f"  Failed: {e}")
            if os.path.exists(stl_path):
                m = o3d.io.read_triangle_mesh(stl_path)
                m.compute_vertex_normals()
                self.tooth_library[kind] = m
            else:
                self.tooth_library[kind] = o3d.geometry.TriangleMesh.create_box()

        # Backwards compat: keep self.library_tooth pointing to molar
        self.library_tooth = self.tooth_library["molar"]

        # ── Load PointNet weights if available ────────────────────────────────
        ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "pointnet_final.pth")
        if PointNetRegressor and os.path.exists(ckpt):
            print("🚀 LOADING REAL NEURAL NETWORK WEIGHTS...")
            self.model = PointNetRegressor(output_dim=7).to(self.device)
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
            self.model.eval()
        else:
            print("PointNet weights not found – using geometric arch detector.")

    # ─── Gap Detection ──────────────────────────────────────────────────────
    def _detect_arch_gaps(self, jaw_mesh):
        """
        Template-based gap detection using the dental arch U-shape geometry.
        Finds ALL positions where teeth are missing.
        Returns list of (position_3d, tooth_type_str).
        """
        pcd    = jaw_mesh.sample_points_uniformly(number_of_points=30000)
        points = np.asarray(pcd.points)

        bbox  = jaw_mesh.get_axis_aligned_bounding_box()
        min_b = bbox.min_bound
        max_b = bbox.max_bound
        z_h   = max_b[2] - min_b[2]
        x_w   = max_b[0] - min_b[0]

        # 1. Filter to alveolar process (25-70% of height, and EXCLUDE ramus wings)
        # The ramus is at far-left/right X AND high Z. Filter both.
        z_min = min_b[2] + z_h * 0.25
        z_max = min_b[2] + z_h * 0.70   # 70% cuts off most of ramus
        # Also exclude the outer 20% of X (ramus attachment zones)
        x_min = min_b[0] + x_w * 0.10
        x_max = max_b[0] - x_w * 0.10

        mask = ((points[:, 2] >= z_min) & (points[:, 2] <= z_max) &
                (points[:, 0] >= x_min) & (points[:, 0] <= x_max))
        alv  = points[mask]
        if len(alv) < 200:
            return []

        # 2. Arch alignment using PCA (Symmetry Axis)
        xy = alv[:, :2]
        center = np.mean(xy, axis=0)
        u, s, vh = np.linalg.svd(xy - center)
        # Primary axis (pc1) is along the length of the mandible (back to front)
        pc1 = vh[0]
        pc2 = vh[1]
        
        # Ensure pc1 points to the FRONT (higher point density)
        proj = (xy - center) @ pc1
        if np.sum(proj > 0) < np.sum(proj < 0):
            pc1 = -pc1
            
        front_ang = np.arctan2(pc1[1], pc1[0])
        arch_center = center - pc1 * (np.max(proj) * 0.1) # Shift center back slightly
        
        # Recalculate polar coordinates for template matching
        dx_new = xy[:, 0] - arch_center[0]
        dy_new = xy[:, 1] - arch_center[1]
        angles = np.arctan2(dy_new, dx_new)
        radii  = np.sqrt(dx_new**2 + dy_new**2)

        # 4. Standard dental arch template (symmetric, 8 teeth per side)
        ARCH_TEMPLATE = [
            # Central & lateral incisors (0–18°)
            ( 4,  "incisor",  +1), ( 4,  "incisor",  -1),
            (13,  "incisor",  +1), (13,  "incisor",  -1),
            # Canines (18–34°)
            (26,  "canine",   +1), (26,  "canine",   -1),
            # 1st & 2nd Premolars (34–54°)
            (40,  "premolar", +1), (40,  "premolar", -1),
            (52,  "premolar", +1), (52,  "premolar", -1),
            # 1st, 2nd, 3rd Molars (54–90°, max 88° to avoid ramus)
            (63,  "molar",    +1), (63,  "molar",    -1),
            (73,  "molar",    +1), (73,  "molar",    -1),
            (83,  "molar",    +1), (83,  "molar",    -1),
        ]

        # 5. Check each template position for tooth presence
        window_rad = np.radians(7.5)
        wide_rad   = np.radians(13.0)
        dense_count        = np.percentile(bin_cnt[bin_cnt > 0], 75) if np.any(bin_cnt > 0) else 1
        presence_threshold = dense_count * 0.40

        results = []
        for (offset_deg, tooth_kind, side) in ARCH_TEMPLATE:
            offset_rad = np.radians(offset_deg) * side
            target_ang = front_ang + offset_rad

            # Find points in this angular sector
            ang_diff = np.abs(((angles - target_ang + np.pi) % (2 * np.pi)) - np.pi)
            mask_near = ang_diff < window_rad
            
            if np.sum(mask_near) < 10:
                continue # No bone data here at all

            sector_points = alv[mask_near]
            z_near = sector_points[:, 2]
            
            z_mean_sector = np.mean(z_near)
            top_10_percent = np.percentile(z_near, 90)
            crown_tip_z = np.mean(z_near[z_near >= top_10_percent])
            
            # If the peak is significantly higher than the mean, a tooth is PRESENT.
            # Otherwise, the bone is flat = tooth is MISSING.
            if crown_tip_z < z_mean_sector + 1.5:
                # Tooth is MISSING — find coordinates on the bone surface
                r_sec = radii[mask_near]
                # Anchor crown to the buccal/labial side (outer 40% of the ridge)
                outer_mask = r_sec >= np.percentile(r_sec, 60)
                if np.sum(outer_mask) > 0:
                    pos3d = np.mean(sector_points[outer_mask], axis=0)
                else:
                    pos3d = np.mean(sector_points, axis=0)
                
                pos3d[2] = np.clip(pos3d[2], z_min, z_max)
                results.append((pos3d.copy(), tooth_kind))

        return results




    def _classify_tooth(self, delta_deg: float) -> str:
        """Map angular offset from arch front to dental tooth type."""
        for kind, info in TOOTH_TYPES.items():
            lo, hi = info["arch_angle_deg"]
            if lo <= delta_deg < hi:
                return kind
        return "molar"  # Default to molar for extreme posterior positions

    def _scale_tooth(self, tooth_mesh, target_md_mm: float):
        """Scale tooth so its mesio-distal (X) width equals target_md_mm."""
        tc = tooth_mesh.get_center()
        tooth_mesh.translate(-tc)
        ext = tooth_mesh.get_axis_aligned_bounding_box().get_extent()
        s   = target_md_mm / max(ext)
        tooth_mesh.scale(s, center=(0, 0, 0))
        return tooth_mesh

    # ─── Main Inference ─────────────────────────────────────────────────────
    def run_inference(self, jaw_scan_path: str, output_path: str) -> dict:
        """Detects ALL missing teeth, places anatomically correct crown type at each gap."""
        jaw_mesh = o3d.io.read_triangle_mesh(jaw_scan_path)
        if not jaw_mesh.has_vertices():
            raise ValueError("Input jaw mesh is empty or invalid.")
        jaw_mesh.compute_vertex_normals()

        jaw_extent   = jaw_mesh.get_axis_aligned_bounding_box().get_extent()
        jaw_width_mm = np.mean(jaw_extent[:2])  # Approximate jaw width

        # ── 1. Geometric gap detection ────────────────────────────────────
        gap_results  = self._detect_arch_gaps(jaw_mesh)
        model_used   = "Geometric Arch Detector"

        # ── 2. PointNet iterative multi-tooth prediction ──────────────────
        if not gap_results:
            model_used = "PointNet3D Iterative"
            if self.model:
                pcd      = jaw_mesh.sample_points_uniformly(number_of_points=8000)
                all_pts  = np.asarray(pcd.points).copy()
                MASK_R   = 9.0    # mm — radius to mask after each prediction
                MAX_ITER = 8      # max number of teeth to find
                MIN_DIST = 6.0    # mm — ignore if too close to previous prediction

                for iteration in range(MAX_ITER):
                    if len(all_pts) < 500:
                        break

                    # Sample fixed size for PointNet
                    if len(all_pts) >= 2048:
                        idx = np.random.choice(len(all_pts), 2048, replace=False)
                    else:
                        idx = np.random.choice(len(all_pts), 2048, replace=True)
                    pts = all_pts[idx].copy()
                    cen = np.mean(pts, axis=0);  pts -= cen

                    t = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        out = self.model(t).cpu().numpy()[0]
                    pos = out[:3] + cen

                    # Skip if too close to an already-found tooth
                    too_close = any(
                        np.linalg.norm(pos - prev[0]) < MIN_DIST
                        for prev in gap_results
                    )
                    if too_close:
                        break

                    # Clamp to jaw bounding box
                    bb  = jaw_mesh.get_axis_aligned_bounding_box()
                    pos = np.clip(pos, bb.min_bound, bb.max_bound)

                    # Classify tooth type by angular position from jaw center
                    jaw_c  = np.array(jaw_mesh.get_center())
                    delta  = pos[:2] - jaw_c[:2]
                    ang    = abs(np.degrees(np.arctan2(delta[1], delta[0])))
                    tooth_kind = self._classify_tooth(ang % 90)
                    gap_results.append((pos, tooth_kind))

                    # Mask the predicted region so next iteration finds a different tooth
                    dists   = np.linalg.norm(all_pts - pos, axis=1)
                    all_pts = all_pts[dists > MASK_R]

                if not gap_results:
                    gap_results = [(jaw_mesh.get_center(), "molar")]
            else:
                c   = jaw_mesh.get_center()
                bb  = jaw_mesh.get_axis_aligned_bounding_box()
                mn, mx = bb.min_bound, bb.max_bound
                pos = np.array([c[0], c[1], mn[2] + (mx[2] - mn[2]) * 0.70])
                gap_results = [(pos, "molar")]

        # ── 3. Build a crown for each gap ────────────────────────────────
        all_teeth = []
        gap_summary = []
        for (gap_pos, tooth_kind) in gap_results:
            info     = TOOTH_TYPES[tooth_kind]
            base_lib = self.tooth_library.get(tooth_kind, self.library_tooth)
            tooth    = copy.deepcopy(base_lib)
            # Scale to anatomically correct mesio-distal size (proportional to jaw)
            target_mm = max(jaw_width_mm * (info["mesio_distal"] / 80.0), info["mesio_distal"] * 0.8)
            tooth     = self._scale_tooth(tooth, target_mm)
            tooth.translate(gap_pos)
            all_teeth.append(tooth)
            gap_summary.append(f"{tooth_kind}@{gap_pos.round(1).tolist()}")
            print(f"  → Placing {tooth_kind} at {gap_pos.round(1)}")

        # ── 4. Combine all into one output STL ───────────────────────────
        if len(all_teeth) == 1:
            result = all_teeth[0]
        else:
            result = all_teeth[0]
            for t in all_teeth[1:]:
                result += t

        o3d.io.write_triangle_mesh(output_path, result)

        mat = np.eye(4)
        mat[:3, 3] = gap_results[0][0]

        return {
            "status":      "success",
            "model_used":  model_used,
            "gaps_found":  len(gap_results),
            "teeth_placed": gap_summary,
            "output_stl":  output_path,
            "matrix":      mat.tolist()
        }
