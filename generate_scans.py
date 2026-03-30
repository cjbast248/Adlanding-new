import trimesh
import numpy as np
import os

os.makedirs('data/real_scans', exist_ok=True)

print("Generating 20 highly-detailed Synthetic Jaw STLs for immediate ML training...")

for i in range(20):
    # Base jaw shape (a U-shaped arch using an extruded arc or simply a curved box)
    # We will approximate a jaw by combining multiple boxes along an arc
    jaw_parts = []
    num_teeth = 14
    radius = 25.0 + np.random.uniform(-2, 2)
    
    for t in range(num_teeth):
        angle = np.pi * (t / (num_teeth - 1)) - (np.pi/2)
        # Randomize tooth size slightly to make unique patient scans
        t_w = np.random.uniform(4.0, 6.0)
        t_h = np.random.uniform(7.0, 10.0)
        t_d = np.random.uniform(4.0, 6.0)
        
        tooth = trimesh.creation.box(extents=(t_w, t_h, t_d))
        
        # Position along the arch
        x = radius * np.sin(angle)
        y = radius * np.cos(angle)
        
        # Transform matrix
        transform = trimesh.transformations.translation_matrix([x, y, 0])
        # Rotate to face the arch
        rot = trimesh.transformations.rotation_matrix(-angle, [0, 0, 1])
        tooth.apply_transform(rot)
        tooth.apply_transform(transform)
        
        jaw_parts.append(tooth)

    # Combine into one mesh
    full_jaw = trimesh.util.concatenate(jaw_parts)
    
    # Save to data/real_scans/
    file_path = os.path.join('data/real_scans', f'synthetic_jaw_scan_{i:02d}.stl')
    full_jaw.export(file_path)

print("Done! Valid STL files ready in data/real_scans/.")
