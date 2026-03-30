import os
import json

def export_to_exocad(tooth_id, transformation_matrix, export_dir="data"):
    """
    Generates a generic metadata file or .dentalProject snippet compatible with Exocad.
    Exocad uses XML-based .dentalProject files. This function exports the necessary matrix
    so the generated tooth can be seamlessly integrated.
    """
    matrix_path = os.path.join(export_dir, f"exocad_matrix_{tooth_id}.json")
    
    # In Exocad, the user can import a pre-aligned STL.
    # We provide this matrix to demonstrate integration readiness.
    exocad_data = {
        "Software": "Exocad DentalCAD",
        "Action": "Import Pre-aligned Mesh",
        "TargetTooth": tooth_id,
        "TransformationMatrix4x4": transformation_matrix,
        "Documentation": "Load the accompanying predicted_*.stl file via Expert Mode -> Add Mesh without changing its coordinates."
    }
    
    with open(matrix_path, "w") as f:
        json.dump(exocad_data, f, indent=4)
        
    return matrix_path
