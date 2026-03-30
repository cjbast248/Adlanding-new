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
    "incisor":  {"arch_angle_deg": (0,  20),  "mesio_distal": 7.0,  "buccal_lingual": 7.5, "crown_h": 5.5, "cusps": 1, "cusp_r": 2.5},
    "canine":   {"arch_angle_deg": (20, 38),  "mesio_distal": 8.0,  "buccal_lingual": 8.0, "crown_h": 7.0, "cusps": 1, "cusp_r": 3.0},
    "premolar": {"arch_angle_deg": (38, 58),  "mesio_distal": 9.0,  "buccal_lingual": 9.0, "crown_h": 6.0, "cusps": 2, "cusp_r": 2.8},
    "molar":    {"arch_angle_deg": (58, 100), "mesio_distal": 12.0, "buccal_lingual": 11.0,"crown_h": 5.5, "cusps": 4, "cusp_r": 2.8},
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
        Template-based gap detection: places 16 expected tooth positions on
        the dental arch and checks point density at each one.
        Returns list of (position_3d, tooth_type_str) for every missing tooth.
        """
        pcd    = jaw_mesh.sample_points_uniformly(number_of_points=30000)
        points = np.asarray(pcd.points)

        bbox  = jaw_mesh.get_axis_aligned_bounding_box()
        min_b = bbox.min_bound
        max_b = bbox.max_bound
        z_h   = max_b[2] - min_b[2]

        # 1. Filter to alveolar process (35–90% of jaw height)
        z_min = min_b[2] + z_h * 0.35
        z_max = min_b[2] + z_h * 0.90
        alv   = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]
        if len(alv) < 200:
            return []

        # 2. Project to XY, find arch centroid and radius
        xy          = alv[:, :2]
        arch_center = np.median(xy, axis=0)
        dx = xy[:, 0] - arch_center[0]
        dy = xy[:, 1] - arch_center[1]
        angles = np.arctan2(dy, dx)
        radii  = np.sqrt(dx**2 + dy**2)

        # 3. Find the arch "front" direction (smallest radius sector = front of U)
        n_scan = 36
        scan_bins = np.linspace(np.percentile(angles, 2), np.percentile(angles, 98), n_scan + 1)
        bin_r_mean = np.full(n_scan, np.inf)
        bin_cnt    = np.zeros(n_scan)
        for i in range(n_scan):
            m = (angles >= scan_bins[i]) & (angles < scan_bins[i+1])
            if np.sum(m) > 0:
                bin_r_mean[i] = np.mean(radii[m])
                bin_cnt[i]    = np.sum(m)

        front_bin  = int(np.argmin(bin_r_mean))
        front_ang  = (scan_bins[front_bin] + scan_bins[front_bin + 1]) / 2
        arch_radius = np.median(radii)
        alv_z_mean  = np.mean(alv[:, 2])

        # 4. Standard dental arch template (half-arch angles from front, mirrored)
        # Each tuple: (offset_deg_from_front, tooth_type, side)  — covers full arch
        ARCH_TEMPLATE = [
            # Incisors (front)
            (5,  "incisor", +1), (5,  "incisor", -1),
            (15, "incisor", +1), (15, "incisor", -1),
            # Canines
            (28, "canine",  +1), (28, "canine",  -1),
            # Premolars
            (42, "premolar", +1), (42, "premolar", -1),
            (55, "premolar", +1), (55, "premolar", -1),
            # Molars
            (68, "molar", +1), (68, "molar", -1),
            (82, "molar", +1), (82, "molar", -1),
            (94, "molar", +1), (94, "molar", -1),
        ]

        # 5. For each template position, check if a tooth is PRESENT
        window_deg = 7.0  # ±7° window around each expected tooth
        window_rad = np.radians(window_deg)

        # Compute "expected" point count if a tooth is present
        # Use the DENSEST sector as reference for "tooth present"
        dense_count = np.percentile(bin_cnt[bin_cnt > 0], 75)
        # A tooth is "missing" if density < 45% of what a present tooth would give
        presence_threshold = dense_count * 0.45

        results = []
        for (offset_deg, tooth_kind, side) in ARCH_TEMPLATE:
            # Target angle for this tooth position
            offset_rad = np.radians(offset_deg) * side
            target_ang = front_ang + offset_rad

            # Count points in this window
            ang_diff = np.abs(((angles - target_ang + np.pi) % (2 * np.pi)) - np.pi)
            mask = ang_diff < window_rad
            cnt  = np.sum(mask)

            if cnt < presence_threshold:
                # Tooth is missing! Compute 3D position
                # Use outer alveolar ridge at this angle
                sec = alv[mask] if cnt > 5 else None

                # Compute expected 3D coordinates from arch geometry
                ex = arch_center[0] + arch_radius * np.cos(target_ang)
                ey = arch_center[1] + arch_radius * np.sin(target_ang)
                ez = alv_z_mean

                if sec is not None and len(sec) > 0:
                    # Refine with actual points
                    r_sec = np.sqrt((sec[:, 0] - arch_center[0])**2 +
                                    (sec[:, 1] - arch_center[1])**2)
                    outer = sec[r_sec >= np.percentile(r_sec, 50)]
                    if len(outer) > 0:
                        ex, ey = np.mean(outer[:, :2], axis=0)
                        ez = np.mean(outer[:, 2])

                results.append((np.array([ex, ey, ez]), tooth_kind))

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

        # ── 2. PointNet fallback if no geometric gaps found ───────────────
        if not gap_results:
            model_used = "PointNet3D / Heuristic (fallback)"
            if self.model:
                pcd = jaw_mesh.sample_points_uniformly(number_of_points=2048)
                pts = np.asarray(pcd.points)
                cen = np.mean(pts, axis=0);  pts -= cen
                t   = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.model(t).cpu().numpy()[0]
                pos = out[:3] + cen
            else:
                c   = jaw_mesh.get_center()
                bb  = jaw_mesh.get_axis_aligned_bounding_box()
                mn, mx = bb.min_bound, bb.max_bound
                pos = np.array([c[0], c[1], mn[2] + (mx[2]-mn[2]) * 0.70])
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
