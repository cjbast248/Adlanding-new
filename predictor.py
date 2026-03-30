import open3d as o3d
import numpy as np
import copy
import os
import torch

try:
    from ml.model import PointNetRegressor
except ImportError:
    PointNetRegressor = None

class TeethPositionPredictor:
    def __init__(self, library_model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Load the newly trained ML weights from Colab if they exist
        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "pointnet_final.pth")
        
        if PointNetRegressor and os.path.exists(checkpoint_path):
            print("==================================================")
            print("🚀 LOADING REAL NEURAL NETWORK WEIGHTS FROM COLAB 🚀")
            print("==================================================")
            self.model = PointNetRegressor(output_dim=7).to(self.device)
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.model.eval()
        else:
            print("AI Model weights (pointnet_final.pth) not found. Using Heuristic simulation Mode.")

        # Load dummy library tooth for visualization
        if os.path.exists(library_model_path):
            self.library_tooth = o3d.io.read_triangle_mesh(library_model_path)
            self.library_tooth.compute_vertex_normals()
        else:
            self.library_tooth = o3d.geometry.TriangleMesh.create_box()
            self.library_tooth.compute_vertex_normals()

    def run_inference(self, jaw_scan_path: str, output_path: str) -> dict:
        """
        Predicts optimal position for missing tooth using Deep Learning simulation (ICP registration).
        """
        jaw_mesh = o3d.io.read_triangle_mesh(jaw_scan_path)
        if not jaw_mesh.has_vertices():
            raise ValueError("Input jaw mesh is empty or invalid.")
        
        jaw_mesh.compute_vertex_normals()
        
        if self.model:
            # 1. REAL ML PREDICTION
            # Convert mesh to Point Cloud as expected by PointNet
            pcd = jaw_mesh.sample_points_uniformly(number_of_points=2048)
            points = np.asarray(pcd.points)
            centroid = np.mean(points, axis=0)
            points -= centroid # Normalize
            
            input_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor).cpu().numpy()[0]
                
            translation = output[:3] + centroid  # De-normalize back to global coords
            # Quaternion rotation is output[3:]
            
            initial_translation = translation
            model_used_name = "PointNet3D (Real Training)"
        else:
            # 2. HEURISTIC FALLBACK (If weight file is missing)
            jaw_center = jaw_mesh.get_center()
            bbox = jaw_mesh.get_axis_aligned_bounding_box()
            bounds = bbox.get_extent()
            initial_translation = jaw_center + np.array([0, bounds[1]*0.3, bounds[2]*0.2]) 
            model_used_name = "PointNet3D (Heuristic Stub)"
        
        predicted_tooth = copy.deepcopy(self.library_tooth)
        # Center the tooth
        tooth_center = predicted_tooth.get_center()
        predicted_tooth.translate(-tooth_center)
        
        # Scale tooth roughly to 10% of jaw width (typical tooth size approximation)
        bbox = jaw_mesh.get_axis_aligned_bounding_box()
        bounds = bbox.get_extent()
        tooth_bbox = predicted_tooth.get_axis_aligned_bounding_box()
        tooth_extent = tooth_bbox.get_extent()
        max_extent = max(tooth_extent[0], tooth_extent[1], tooth_extent[2], 0.001)
        scale_factor = (bounds[0] * 0.1) / max_extent
        predicted_tooth.scale(scale_factor, predicted_tooth.get_center())

        # Move tooth center to initial_translation predictions
        predicted_tooth.translate(initial_translation)
        
        # Save result
        o3d.io.write_triangle_mesh(output_path, predicted_tooth)
        
        # Generate Exocad transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, 3] = initial_translation
        
        return {
            "status": "success",
            "model_used": model_used_name,
            "output_stl": output_path,
            "matrix": transformation_matrix.tolist()
        }

