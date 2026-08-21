"""Preprocess clinical OBJ meshes → per-bone NPZ files (one-time conversion).

Usage:
    python tools/clinical_mesh2points.py

Source: .../clinical_fractures_train/mesh/
Target: .../clinical_fractures_train/points/

Output per sample:
    points/
      001/
        SA_2.npz   ← SA bone, 2 fragments
        LI_1.npz   ← LI bone, 1 fragment
        RI_3.npz   ← RI bone, 3 fragments
      002/
        ...

Each NPZ contains:
    points:        (N, 3)   assembled GT coords
    normals:       (N, 3)   assembled GT normals
    offsets:       (M+1,)   cumulative per-fragment offsets
    per_frag_T_inv: (M,4,4) np.linalg.inv(T_gt) — assembled→scattered
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import trimesh

from datasets.fractures.clinical import (
    _iter_fragments_with_gt_ids,
    _load_gt_poses,
)

# Reuse constants from clinical dataset
BONE_RANGES = {"SA": (1, 100), "LI": (101, 200), "RI": (201, 300)}


def load_gt_poses(json_path: str) -> dict:
    return _load_gt_poses(json_path)


def load_obj_fragment_ids(obj_path: str) -> list[str]:
    fragment_ids = []
    with open(obj_path, errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line.startswith(("o ", "g ")):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            fid = parts[1].strip()
            if fid and (not fragment_ids or fragment_ids[-1] != fid):
                fragment_ids.append(fid)
    return fragment_ids


def iter_fragments_with_gt_ids(obj_path: str, scene, gt_poses: dict):
    return _iter_fragments_with_gt_ids(obj_path, scene, gt_poses)


def apply_T(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((len(pts), 1), dtype=pts.dtype)
    return (T @ np.concatenate([pts, ones], axis=1).T).T[:, :3]


def apply_T_normals(normals: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ normals.T).T


def sample_mesh_proportional(mesh: trimesh.Trimesh, n_points: int, min_points: int = 20) -> tuple:
    n_points = max(n_points, min_points)
    pts, face_idx = trimesh.sample.sample_surface_even(mesh, n_points)
    normals = mesh.face_normals[face_idx]

    current = len(pts)
    if current < n_points:
        extra = np.random.choice(current, n_points - current, replace=True)
        pts = np.vstack([pts, pts[extra]])
        normals = np.vstack([normals, normals[extra]])
    elif current > n_points:
        pts = pts[:n_points]
        normals = normals[:n_points]

    return pts.astype(np.float32), normals.astype(np.float32)


def process_sample(sample_dir: str, output_dir: str, npoints_per_bone: int = 5000):
    sample_name = os.path.basename(sample_dir)

    # Find OBJ and JSON
    obj_files = list(Path(sample_dir).glob("*.obj"))
    json_path = os.path.join(sample_dir, "plan_pl_gt.json")
    if not obj_files or not os.path.exists(json_path):
        print(f"  [SKIP] {sample_name}: missing OBJ or plan_pl_gt.json")
        return

    obj_path = str(obj_files[0])
    scene = trimesh.load(obj_path, split_object=True, process=False)
    gt_poses = load_gt_poses(json_path)

    # Group fragments by bone type
    bone_fragments: dict[str, list] = {bt: [] for bt in ["SA", "LI", "RI"]}
    for frag_key, geom in iter_fragments_with_gt_ids(obj_path, scene, gt_poses):
        key_int = int(frag_key)
        for bt, (lo, hi) in BONE_RANGES.items():
            if lo <= key_int <= hi:
                bone_fragments[bt].append((frag_key, geom))
                break

    # Create sample output directory
    sample_out = os.path.join(output_dir, sample_name)
    os.makedirs(sample_out, exist_ok=True)

    for bt in ["SA", "LI", "RI"]:
        frags = sorted(bone_fragments[bt], key=lambda x: int(x[0]))
        if not frags:
            continue

        n_frags = len(frags)
        pts_per_frag = max(npoints_per_bone // n_frags, 20)

        all_pts_gt, all_normals_gt = [], []
        offsets = [0]
        all_T_inv = []

        for frag_key, mesh in frags:
            T_gt = gt_poses[frag_key]
            pts, normals = sample_mesh_proportional(mesh, pts_per_frag)

            # Apply GT pose: scattered → assembled
            pts_gt = apply_T(pts, T_gt).astype(np.float32)
            normals_gt = apply_T_normals(normals, T_gt).astype(np.float32)

            all_pts_gt.append(pts_gt)
            all_normals_gt.append(normals_gt)
            offsets.append(offsets[-1] + len(pts_gt))

            # Inverse transform: assembled → scattered
            T_inv = np.linalg.inv(T_gt).astype(np.float32)
            all_T_inv.append(T_inv)

        npz_path = os.path.join(sample_out, f"{bt}_{n_frags}.npz")
        np.savez_compressed(
            npz_path,
            points=np.concatenate(all_pts_gt, axis=0),
            normals=np.concatenate(all_normals_gt, axis=0),
            offsets=np.array(offsets, dtype=np.int64),
            per_frag_T_inv=np.stack(all_T_inv, axis=0),
        )

    print(f"  {sample_name}: " + "  ".join(
        f"{bt}:{len(bone_fragments[bt])}" for bt in ["SA", "LI", "RI"] if bone_fragments[bt]
    ))


def main():
    parser = argparse.ArgumentParser(description="Convert clinical OBJ meshes to NPZ point cache.")
    parser.add_argument(
        "--input", type=str,
        default="./data/train/mesh",
    )
    parser.add_argument(
        "--output", type=str,
        default="./data/train/points",
    )
    parser.add_argument("--npoints", type=int, default=5000)
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"[ERROR] Input directory not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    sample_dirs = sorted(
        d for d in Path(args.input).iterdir() if d.is_dir()
    )
    print(f"Found {len(sample_dirs)} samples in {args.input}")
    print(f"Output: {args.output}")

    for sample_dir in sample_dirs:
        process_sample(str(sample_dir), args.output, npoints_per_bone=args.npoints)

    print("Done.")


if __name__ == "__main__":
    main()
