#!/usr/bin/env python3
"""
PENGWIN 2026 Challenge - Task 3 Fragment Mesh Loader, Visualizer, and Assembler
(Direct Run Version)
"""

import os
import json
import numpy as np
import trimesh

# Define official visualization color palette (Dark 10)
_DARK10_COLORS = [
    [0.121568, 0.466667, 0.705882],  # Blue
    [1.000000, 0.498039, 0.054902],  # Orange
    [0.172549, 0.627451, 0.172549],  # Green
    [0.839216, 0.152941, 0.156863],  # Red
    [0.580392, 0.403922, 0.741176],  # Purple
    [0.549020, 0.337255, 0.294118],  # Brown
    [0.890196, 0.466667, 0.760784],  # Pink
    [0.498039, 0.498039, 0.498039],  # Gray
    [0.737255, 0.741176, 0.133333],  # Olive
    [0.090196, 0.745098, 0.811765]  # Cyan
]


def load_obj_fragments(obj_path, verbose=True):
    """
    Load and split independent fragments from a single OBJ file using trimesh.
    """
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"Cannot find file: {obj_path}")

    # split_object=True is the key to parsing our custom group/object IDs
    scene = trimesh.load(obj_path, split_object=True)

    if isinstance(scene, trimesh.Scene) and verbose:
        print(f"Successfully loaded Scene with {len(scene.geometry)} geometry objects.")

    meshdict = {"SA": {}, "LI": {}, "RI": {}}

    for key, geom in scene.geometry.items():
        try:
            key_int = int(key)
        except (ValueError, TypeError):
            continue

        if 1 <= key_int <= 100:
            bone_type = "SA"
        elif 101 <= key_int <= 200:
            bone_type = "LI"
        elif 201 <= key_int <= 300:
            bone_type = "RI"
        else:
            continue

        if isinstance(geom, trimesh.Trimesh):
            mesh = geom
        elif hasattr(geom, 'merged_vertices'):
            mesh = geom.merged_vertices
        else:
            continue

        volume = mesh.volume if hasattr(mesh, 'volume') and mesh.volume > 0 else 0.001

        meshdict[bone_type][str(key)] = {
            'mesh': mesh,
            'volume': volume
        }

        if verbose:
            print(f"  Identified fragment ID: {key} -> Bone: {bone_type}, Vertices: {len(mesh.vertices)}")

    return meshdict


def load_transformations(json_path, verbose=True):
    """
    Load reduction transformation matrices from the JSON file.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find JSON file: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    transformations = {}
    for item in data:
        frag_id = str(item["fragment_id"])
        matrix = np.array(item["transformation"])
        transformations[frag_id] = matrix

    if verbose:
        print(f"Loaded {len(transformations)} transformation matrices from JSON.")

    return transformations


def apply_transformations(meshdict, transformations, verbose=True):
    """
    Apply the loaded 4x4 transformation matrices to the respective fragments.
    """
    if verbose:
        print("\n--- Applying Reduction Matrices ---")

    applied_count = 0
    for bone_type, fragments in meshdict.items():
        for frag_name, frag_data in fragments.items():
            if frag_name in transformations:
                matrix = transformations[frag_name]
                # Apply the 4x4 matrix to the mesh vertices
                frag_data['mesh'].apply_transform(matrix)
                applied_count += 1
                if verbose:
                    print(f"  Applied matrix to fragment {frag_name} ({bone_type})")
            else:
                if verbose:
                    print(f"  Warning: No matrix found for fragment {frag_name} ({bone_type})")

    if verbose:
        print(f"Successfully reduced {applied_count} fragments.")


def visualize_fragments(meshdict, save_path=None, title="3D Viewer"):
    """
    Assign a unique color to each fragment and visualize in 3D.
    """
    scene = trimesh.Scene()
    color_idx = 0

    for bone_type in ["SA", "LI", "RI"]:
        fragments = meshdict.get(bone_type, {})
        for frag_name, frag_data in fragments.items():
            mesh = frag_data['mesh']

            # Assign colors cyclically
            fragment_color = _DARK10_COLORS[color_idx % len(_DARK10_COLORS)]

            # Apply vertex colors
            mesh.visual.vertex_colors[:, :3] = (np.array(fragment_color) * 255).astype(np.uint8)
            scene.add_geometry(mesh, geom_name=f"{bone_type}_{frag_name}")
            color_idx += 1

    if save_path:
        scene.save_image(save_path)
        print(f"Visualization screenshot saved to: {save_path}")

    # Show interactive 3D window
    print(f"Launching {title} (use mouse to rotate/zoom, close window to continue)...")
    scene.show()
    return scene


# =====================================================================
# Participant Execution Area: Modify paths here and run the script
# =====================================================================
if __name__ == "__main__":

    # 1. Path to the input fractured OBJ file  peripelvic-fracture-fragments.obj
    OBJ_FILE_PATH = 'data/sample/peripelvic-fracture-fragments.obj'

    # 2. Path to the output reduction poses JSON file
    JSON_FILE_PATH = 'data/sample/plan_pl_gt.json'

    # 3. Optional path to save screenshot (e.g., 'output.png')
    SAVE_SCREENSHOT_PATH = None

    print(f"--- Processing Input OBJ: {OBJ_FILE_PATH} ---")

    if not os.path.exists(OBJ_FILE_PATH):
        print(f"Error: Cannot find OBJ file! Please check OBJ_FILE_PATH.")
    else:
        # Step 1: Load and parse the initial fractured fragments
        parsed_meshdict = load_obj_fragments(OBJ_FILE_PATH, verbose=True)

        if any(parsed_meshdict[b] for b in parsed_meshdict):
            print("\n✅ Read successful! Showing initial FRACTURED state...")
            # Display pre-reduction state
            visualize_fragments(parsed_meshdict, save_path=None, title="Initial Fractured State")

            # Step 2: Load the JSON transformations
            if os.path.exists(JSON_FILE_PATH):
                transformations = load_transformations(JSON_FILE_PATH, verbose=True)

                # Step 3: Apply the transformations (Assemble the bones)
                apply_transformations(parsed_meshdict, transformations, verbose=True)

                print("\n✅ Assembly complete! Showing final REDUCED state...")
                # Display post-reduction state
                visualize_fragments(parsed_meshdict, save_path=SAVE_SCREENSHOT_PATH, title="Final Reduced State")
            else:
                print(f"\n❌ JSON file not found at {JSON_FILE_PATH}. Skipping assembly step.")
        else:
            print("\n❌ Parsing failed: No valid independent fragment structures found in the OBJ.")