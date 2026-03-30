import trimesh
import os

os.makedirs('data', exist_ok=True)

# Create a primitive tooth (a slightly transformed box to look like a placeholder)
tooth = trimesh.creation.box(extents=(8, 10, 12))
# Make it look slightly more tooth-like
tooth.export('data/library_tooth.stl')

# Create a dummy jaw (a U-shaped structure)
left = trimesh.creation.box(extents=(10, 30, 15))
left.apply_translation([-20, 0, 0])

right = trimesh.creation.box(extents=(10, 30, 15))
right.apply_translation([20, 0, 0])

front = trimesh.creation.box(extents=(50, 10, 15))
front.apply_translation([0, 20, 0])

jaw = trimesh.util.concatenate([left, right, front])
jaw.export('data/sample_jaw.stl')

print("Dummy STL files generated successfully in data/ folder:")
print(" - data/library_tooth.stl")
print(" - data/sample_jaw.stl")
