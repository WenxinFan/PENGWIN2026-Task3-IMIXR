"""
Apply predicted reduction transforms to a fracture OBJ, producing an assembled
(reduced) OBJ file with per-fragment group tags preserved.

Usage:
    python tools/apply_reduction.py input.obj prediction.json -o reduced.obj
    python tools/apply_reduction.py input.obj prediction.json  # writes next to input

The prediction JSON must follow the competition format:
    [{"fragment_id": "1", "transformation": [[4x4], ...]}, ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh


def apply_reduction(obj_path, prediction_path, output_path=None):
    """Load OBJ, apply predicted transforms, save reduced OBJ with groups."""
    scene = trimesh.load(obj_path, split_object=True)
    pred = load_predictions(prediction_path)

    # Collect fragments in input order
    fragments = []
    if isinstance(scene, trimesh.Scene):
        for key, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                fragments.append((str(key), geom))
    elif isinstance(scene, trimesh.Trimesh):
        fragments.append(("1", scene))

    if not fragments:
        raise RuntimeError("No meshes found in input OBJ.")

    # Apply transforms
    transformed = []
    for fid, geom in fragments:
        m = geom.copy()
        if fid in pred:
            m.apply_transform(pred[fid])
        else:
            print(f"Warning: no prediction for fragment {fid}, keeping original position")
        transformed.append((fid, m))

    # Write OBJ manually to preserve g-tags for 3D Slicer
    if output_path is None:
        base = Path(obj_path).stem
        output_path = Path(obj_path).with_name(f"{base}_reduced.obj")

    _write_obj_with_groups(transformed, output_path)
    print(f"Saved reduced mesh: {output_path}  ({len(transformed)} fragments)")
    return output_path


def _write_obj_with_groups(fragments, path):
    """Write an OBJ file with g-tags between fragments."""
    with open(path, "w") as f:
        f.write("# Reduced fracture assembly\n")
        vertex_offset = 0
        normal_offset = 0

        for fid, mesh in fragments:
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            has_normals = hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None

            f.write(f"\ng {fid}\n")

            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            if has_normals:
                norms = np.asarray(mesh.vertex_normals, dtype=np.float64)
                for vn in norms:
                    f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")

            for face in faces:
                if has_normals:
                    f.write(f"f {face[0]+1+vertex_offset}//{face[0]+1+normal_offset} "
                            f"{face[1]+1+vertex_offset}//{face[1]+1+normal_offset} "
                            f"{face[2]+1+vertex_offset}//{face[2]+1+normal_offset}\n")
                else:
                    f.write(f"f {face[0]+1+vertex_offset} "
                            f"{face[1]+1+vertex_offset} "
                            f"{face[2]+1+vertex_offset}\n")

            vertex_offset += len(verts)
            if has_normals:
                normal_offset += len(norms)


def load_predictions(path):
    """Load prediction JSON in competition format."""
    with open(path) as f:
        data = json.load(f)

    result = {}
    if isinstance(data, list):
        for item in data:
            result[str(item["fragment_id"])] = np.array(item["transformation"], dtype=np.float64)
    elif isinstance(data, dict) and "fragments" in data:
        for item in data["fragments"]:
            result[str(item["fragment_id"])] = np.array(item["transformation"], dtype=np.float64)
    return result


def main():
    parser = argparse.ArgumentParser(description="Apply reduction transforms to fracture OBJ")
    parser.add_argument("input", help="Fracture OBJ file")
    parser.add_argument("prediction", help="Prediction JSON (reduction-poses-matrices.json)")
    parser.add_argument("-o", "--output", default=None, help="Output OBJ path (default: <input>_reduced.obj)")
    args = parser.parse_args()

    apply_reduction(args.input, args.prediction, args.output)

if __name__ == "__main__":
    main()
