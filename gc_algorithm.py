"""Grand Challenge container entrypoint for PENGWIN Task 3.

The container reads one OBJ fracture case from /input and writes:
    /output/reduction-poses-matrices.json
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import trimesh

from datasets.fractures.clinical import (
    _iter_fragments_with_gt_ids,
    _load_obj_fragment_ids,
)
from models.AssemblyNet.transformer import AssemblyTransformer
from models.AssemblyNet.utils import quaternion_to_matrix


OUTPUT_TYPE = "coords"
MAX_PARTS = 50
EMBED_DIM = 384
NUM_LAYERS = 12
NUM_HEADS = 8
DROPOUT_RATE = 0.0
DEFAULT_NPOINTS = 1800
DEFAULT_MAX_ITERS = 1
DEFAULT_ALPHA = 1.0
DEFAULT_TTA = 1
DEFAULT_SEED = 42
DEFAULT_CONVERGENCE_THRESHOLD = 2.0


def parse_args():
    parser = argparse.ArgumentParser(description="PENGWIN Task 3 Grand Challenge inference")
    parser.add_argument("--input", type=str, default="/input", help="Input OBJ file or directory")
    parser.add_argument("--output", type=str, default="/output/reduction-poses-matrices.json",
                        help="Output JSON path")
    parser.add_argument("--config", type=str, default="/opt/app/configs/test_gc.yaml",
                        help="Accepted for compatibility; this fixed container uses baked-in model settings")
    parser.add_argument("--checkpoint", type=str, default="/opt/app/model_state_dict.pt",
                        help="Model checkpoint")
    parser.add_argument("--npoints", type=int, default=None, help="Override sampled point count")
    parser.add_argument("--max_iters", type=int, default=None, help="Override iterative inference steps")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Blend predicted relative poses toward identity")
    parser.add_argument("--rot_alpha", type=float, default=None,
                        help="Scale rotations toward/from identity while preserving transformed mesh centroids")
    parser.add_argument("--rot_alpha_li", type=float, default=None,
                        help="Override --rot_alpha for left-ilium (LI) fragments")
    parser.add_argument("--rot_alpha_ri", type=float, default=None,
                        help="Override --rot_alpha for right-ilium (RI) fragments")
    parser.add_argument("--kabsch_robust", choices=["none", "huber"], default="none",
                        help="Use robust Huber IRLS Kabsch for coordinate outputs")
    parser.add_argument("--kabsch_huber_delta", type=float, default=0.02,
                        help="Huber residual cutoff in normalized coordinates")
    parser.add_argument("--kabsch_irls_iters", type=int, default=5,
                        help="Number of Huber IRLS reweighting iterations")
    parser.add_argument("--diagnostics_output", type=str, default=None,
                        help="Optional JSON sidecar with per-fragment SVD/TTA diagnostics")
    parser.add_argument("--tta", type=int, default=None,
                        help="Number of stochastic surface-sampling test-time augmentations")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Base random seed for deterministic TTA sampling")
    parser.add_argument("--allocation", choices=["equal", "volume"], default="volume",
                        help="Point allocation within each bone")
    parser.add_argument("--min_points_per_fragment", type=int, default=20,
                        help="Minimum sampled points per fragment within each bone")
    parser.add_argument("--min_points_per_fragment_sa", type=int, default=None,
                        help="Optional SA-specific minimum; defaults to --min_points_per_fragment")
    parser.add_argument("--min_points_per_fragment_li", type=int, default=None,
                        help="Optional LI-specific minimum; defaults to --min_points_per_fragment")
    parser.add_argument("--min_points_per_fragment_ri", type=int, default=None,
                        help="Optional RI-specific minimum; defaults to --min_points_per_fragment")
    parser.add_argument("--surface_sample", choices=["even", "random"], default="random",
                        help="Surface sampling strategy")
    parser.add_argument("--output_type", choices=["coords", "pose"], default=OUTPUT_TYPE,
                        help="Model output type used by the checkpoint")
    parser.add_argument("--inference_precision", choices=["fp16", "bf16", "fp32"], default="fp16",
                        help="CUDA inference precision; use fp32 for numerically sensitive pose heads")
    parser.add_argument("--rotation_rerank_angles", type=str, default="",
                        help="Comma-separated degrees for inference-only rotation candidate reranking; empty disables")
    parser.add_argument("--rotation_rerank_sample_points", type=int, default=512,
                        help="Surface points per fragment used by rotation candidate reranking")
    parser.add_argument("--rotation_rerank_contact_fraction", type=float, default=0.08,
                        help="Fraction of closest surface points used in candidate contact score")
    parser.add_argument("--rotation_rerank_angle_weight", type=float, default=0.002,
                        help="Penalty per squared candidate angle in degrees")
    return parser.parse_args()


def load_model(device, checkpoint_path, output_type=OUTPUT_TYPE):
    model = AssemblyTransformer(
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        max_parts=MAX_PARTS,
        output_type=output_type,
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    transformer_state = {}
    for key, value in state_dict.items():
        if key.startswith("transformer_model."):
            transformer_state[key[len("transformer_model."):]] = value
    if not transformer_state:
        transformer_state = state_dict
    model.load_state_dict(transformer_state)
    return model.to(device).eval()


def find_obj(input_path):
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() == ".obj":
        return path
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    obj_files = sorted(path.rglob("*.obj"))
    if not obj_files:
        raise FileNotFoundError(f"No OBJ files found under: {path}")
    preferred = [p for p in obj_files if p.name == "peripelvic-fracture-fragments.obj"]
    return preferred[0] if preferred else obj_files[0]


def _parse_obj_groups(obj_path):
    names = []
    with open(obj_path, errors="ignore") as f:
        for line in f:
            if line.startswith(("o ", "g ")):
                name = line[2:].strip().split()[0]
                if name and (not names or names[-1] != name):
                    names.append(name)
    return names


def _iter_meshes_with_obj_ids(obj_path, scene):
    valid_ids = {fragment_id: None for fragment_id in _load_obj_fragment_ids(obj_path)}
    return _iter_fragments_with_gt_ids(obj_path, scene, valid_ids)


def load_obj_fragments(obj_path):
    scene = trimesh.load(obj_path, split_object=True, process=False)
    meshdict = {"SA": {}, "LI": {}, "RI": {}}

    for key, geom in _iter_meshes_with_obj_ids(obj_path, scene):
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
        elif hasattr(geom, "merged_vertices"):
            mesh = geom.merged_vertices
        else:
            continue

        volume = mesh.volume if hasattr(mesh, "volume") and mesh.volume > 0 else 0.001
        meshdict[bone_type][str(key)] = {"mesh": mesh, "volume": volume}
    return meshdict


def _allocate_equal(total_points, n_fragments, min_points=20):
    if total_points < n_fragments * min_points:
        raise ValueError(
            f"{total_points} points cannot provide min_points={min_points} "
            f"to {n_fragments} fragments"
        )
    base, remainder = divmod(total_points, n_fragments)
    return [base + (idx < remainder) for idx in range(n_fragments)]


def _allocate_by_volume(fragments, frag_keys, total_points, min_points=20):
    if total_points < len(frag_keys) * min_points:
        raise ValueError(
            f"{total_points} points cannot provide min_points={min_points} "
            f"to {len(frag_keys)} fragments"
        )
    volumes = np.asarray([fragments[k]["volume"] for k in frag_keys], dtype=np.float64)
    volume_sum = float(volumes.sum())
    if not np.isfinite(volume_sum) or volume_sum <= 0:
        volumes = np.ones(len(frag_keys), dtype=np.float64)
        volume_sum = float(volumes.sum())

    allocations = [max(int(vol / volume_sum * total_points), min_points) for vol in volumes]
    diff = total_points - sum(allocations)
    if diff > 0:
        max_vol_idx = int(np.argmax(volumes))
        allocations[max_vol_idx] += diff
    elif diff < 0:
        deficit = -diff
        for idx in np.argsort(-volumes):
            reducible = allocations[int(idx)] - min_points
            reduction = min(reducible, deficit)
            allocations[int(idx)] -= reduction
            deficit -= reduction
            if deficit == 0:
                break
    if sum(allocations) != total_points:
        raise ValueError(
            f"Unable to preserve total_points={total_points} with "
            f"min_points={min_points}; got allocations={allocations}"
        )
    return allocations


def _remove_close_without_scipy(points, radius):
    """Match trimesh.points.remove_close without requiring scipy.cKDTree."""
    if len(points) == 0:
        return points, np.zeros(0, dtype=bool)

    cell_size = float(radius)
    if cell_size <= 0:
        return points, np.ones(len(points), dtype=bool)

    cells = {}
    pairs = []
    radius_sq = cell_size * cell_size
    cell_coords = np.floor(points / cell_size).astype(np.int64)

    for idx, cell in enumerate(cell_coords):
        cx, cy, cz = cell.tolist()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor = (cx + dx, cy + dy, cz + dz)
                    for prev_idx in cells.get(neighbor, []):
                        delta = points[idx] - points[prev_idx]
                        if float(delta @ delta) < radius_sq:
                            pairs.append((prev_idx, idx))
        cells.setdefault((cx, cy, cz), []).append(idx)

    if not pairs:
        return points, np.ones(len(points), dtype=bool)

    pairs = np.asarray(sorted(pairs), dtype=np.int64)
    count = np.bincount(pairs.ravel(), minlength=len(points))
    column = count[pairs].argmax(axis=1)
    highest = pairs.ravel()[column + 2 * np.arange(len(column))]

    mask = np.ones(len(points), dtype=bool)
    mask[highest] = False
    return points[mask], mask


def sample_surface_even_no_scipy(mesh, count, rng):
    radius = np.sqrt(mesh.area / (3 * count))
    points, face_indices = trimesh.sample.sample_surface(mesh, count * 3, seed=rng)
    points, mask = _remove_close_without_scipy(points, radius)
    face_indices = face_indices[mask]
    if len(points) >= count:
        return points[:count], face_indices[:count]
    return points, face_indices


def sub_sample(meshdict, npoints=3000, min_points=20, rng=None,
               allocation="equal", surface_sample="even",
               min_points_by_bone=None):
    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED)
    sampled_data = {}
    for bone_type, fragments in meshdict.items():
        sampled_data[bone_type] = {}
        frag_keys = sorted(fragments.keys(), key=lambda name: int(name))
        if not frag_keys:
            continue

        bone_min_points = min_points
        if min_points_by_bone is not None:
            bone_min_points = min_points_by_bone.get(bone_type, min_points)

        if allocation == "volume":
            allocations = _allocate_by_volume(
                fragments, frag_keys, npoints, bone_min_points
            )
        else:
            allocations = _allocate_equal(
                npoints, len(frag_keys), bone_min_points
            )

        for frag_name, target_count in zip(frag_keys, allocations):
            mesh = fragments[frag_name]["mesh"]
            if surface_sample == "random":
                points, face_indices = trimesh.sample.sample_surface(
                    mesh, target_count, seed=rng
                )
            else:
                points, face_indices = sample_surface_even_no_scipy(mesh, target_count, rng)
            normals = mesh.face_normals[face_indices]

            current_count = len(points)
            if current_count < target_count:
                needed = target_count - current_count
                if current_count > 0:
                    indices = rng.integers(0, current_count, size=needed)
                    points = np.vstack([points, points[indices]])
                    normals = np.vstack([normals, normals[indices]])
                else:
                    points, face_indices = trimesh.sample.sample_surface(
                        mesh, target_count, seed=rng
                    )
                    normals = mesh.face_normals[face_indices]
            elif current_count > target_count:
                points = points[:target_count]
                normals = normals[:target_count]

            sampled_data[bone_type][frag_name] = {
                "coords": np.asarray(points, dtype=np.float32),
                "normals": np.asarray(normals, dtype=np.float32),
            }
    return sampled_data


def build_fragment_data_from_meshdict(meshdict, npoints, rng,
                                      allocation="equal", surface_sample="even",
                                      min_points_per_fragment=20,
                                      min_points_by_bone=None):
    sampled = sub_sample(
        meshdict,
        npoints=npoints,
        min_points=min_points_per_fragment,
        rng=rng,
        allocation=allocation,
        surface_sample=surface_sample,
        min_points_by_bone=min_points_by_bone,
    )
    fragment_data = []
    for bone_type in ["SA", "LI", "RI"]:
        for frag_name in sorted(
            sampled.get(bone_type, {}).keys(), key=lambda name: int(name)
        ):
            frag = sampled[bone_type][frag_name]
            fragment_data.append((frag["coords"], frag["normals"]))
    return fragment_data


def fragment_names_from_meshdict(meshdict):
    names = []
    for bone_type in ["SA", "LI", "RI"]:
        for frag_name in sorted(
            meshdict.get(bone_type, {}).keys(), key=lambda name: int(name)
        ):
            names.append(frag_name)
    return names


def fragment_bone_types_from_meshdict(meshdict):
    bone_types = []
    for bone_type in ["SA", "LI", "RI"]:
        for _ in sorted(
            meshdict.get(bone_type, {}).keys(), key=lambda name: int(name)
        ):
            bone_types.append(bone_type)
    return bone_types


def fragment_centroids_from_meshdict(meshdict):
    centroids = []
    for bone_type in ["SA", "LI", "RI"]:
        for frag_name in sorted(
            meshdict.get(bone_type, {}).keys(), key=lambda name: int(name)
        ):
            mesh = meshdict[bone_type][frag_name]["mesh"]
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            if vertices.size == 0:
                raise ValueError(f"Fragment {frag_name} has no mesh vertices")
            centroids.append(vertices.mean(axis=0))
    return centroids


def recenter_pc(pc):
    centroid = (pc.max(axis=0) + pc.min(axis=0)) / 2
    return pc - centroid[None], centroid[None]


def rescale_pc(pc):
    scale = np.linalg.norm(pc, axis=1).max()
    return pc / scale, scale


def transform_pose_to_original_space(T_pred, centroid, scale):
    T_init = np.eye(4, dtype=np.float32)
    T_init[:3, :3] = np.diag([1.0 / scale] * 3)
    T_init[:3, 3] = -centroid / scale
    return np.linalg.inv(T_init) @ T_pred @ T_init


def apply_transform_to_points(points, T):
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    return (T @ pts_h.T).T[:, :3]


def axis_angle_matrix(axis, angle_deg):
    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0:
        raise ValueError("Rotation rerank axis must be non-zero")
    axis = axis / axis_norm
    angle = np.deg2rad(float(angle_deg))
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rotate_transform_about_point(transform, rotation, point):
    out = np.asarray(transform, dtype=np.float64).copy()
    point = np.asarray(point, dtype=np.float64)
    out[:3, :3] = rotation @ out[:3, :3]
    out[:3, 3] = rotation @ (out[:3, 3] - point) + point
    return out.astype(np.float32)


def parse_rotation_rerank_angles(text):
    if text is None or text.strip() == "":
        return []
    angles = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            raise ValueError("--rotation_rerank_angles contains an empty value")
        angle = float(raw)
        if not np.isfinite(angle) or angle <= 0.0:
            raise ValueError("--rotation_rerank_angles must contain positive finite degrees")
        angles.append(angle)
    return angles


def nearest_distances_chunked(query, target, chunk_size=256):
    query = np.asarray(query, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if query.ndim != 2 or target.ndim != 2 or query.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("nearest_distances_chunked expects Nx3 and Mx3 point arrays")
    if len(query) == 0 or len(target) == 0:
        raise ValueError("nearest_distances_chunked received an empty point array")

    out = np.empty(len(query), dtype=np.float32)
    for start in range(0, len(query), chunk_size):
        chunk = query[start:start + chunk_size]
        diff = chunk[:, None, :] - target[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        out[start:start + len(chunk)] = np.sqrt(np.min(dist2, axis=1))
    return out


def contact_score(candidate_points, context_points, contact_fraction):
    if not 0.0 < contact_fraction <= 1.0:
        raise ValueError("--rotation_rerank_contact_fraction must be in (0, 1]")
    forward = nearest_distances_chunked(candidate_points, context_points)
    backward = nearest_distances_chunked(context_points, candidate_points)
    k_forward = max(1, int(np.ceil(len(forward) * contact_fraction)))
    k_backward = max(1, int(np.ceil(len(backward) * contact_fraction)))
    return 0.5 * (
        float(np.partition(forward, k_forward - 1)[:k_forward].mean())
        + float(np.partition(backward, k_backward - 1)[:k_backward].mean())
    )


def _bone_type_from_fragment_id(fragment_id):
    value = int(fragment_id)
    if 1 <= value <= 100:
        return "SA"
    if 101 <= value <= 200:
        return "LI"
    if 201 <= value <= 300:
        return "RI"
    raise ValueError(f"Unsupported fragment id for reranking: {fragment_id}")


def rotation_candidate_rerank(transforms, fragment_data, fragment_names, angles,
                              contact_fraction, angle_weight):
    if not angles:
        return transforms
    if len(transforms) != len(fragment_data) or len(transforms) != len(fragment_names):
        raise ValueError("Rerank inputs have inconsistent fragment counts")
    if angle_weight < 0.0 or not np.isfinite(angle_weight):
        raise ValueError("--rotation_rerank_angle_weight must be a non-negative finite value")

    axes = [
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ]
    transformed_points = [
        apply_transform_to_points(points, transform)
        for (points, _), transform in zip(fragment_data, transforms)
    ]
    bone_types = [_bone_type_from_fragment_id(name) for name in fragment_names]
    reranked = [np.asarray(T, dtype=np.float32).copy() for T in transforms]

    for part_idx, (points, transform, bone_type) in enumerate(
        zip(transformed_points, transforms, bone_types)
    ):
        if part_idx == 0:
            continue

        context_indices = [
            idx for idx, other_bone in enumerate(bone_types)
            if idx != part_idx and (other_bone == bone_type or other_bone == "SA" or bone_type == "SA")
        ]
        if not context_indices:
            raise ValueError(f"No geometry context available for fragment {fragment_names[part_idx]}")
        context = np.concatenate([transformed_points[idx] for idx in context_indices], axis=0)
        center = points.mean(axis=0)

        best_transform = np.asarray(transform, dtype=np.float32)
        best_score = contact_score(points, context, contact_fraction)
        best_label = "base"

        for angle in angles:
            for sign in (-1.0, 1.0):
                signed_angle = sign * angle
                for axis_idx, axis in enumerate(axes):
                    rotation = axis_angle_matrix(axis, signed_angle)
                    candidate_transform = rotate_transform_about_point(transform, rotation, center)
                    candidate_points = apply_transform_to_points(fragment_data[part_idx][0], candidate_transform)
                    score = (
                        contact_score(candidate_points, context, contact_fraction)
                        + angle_weight * angle * angle
                    )
                    if score < best_score:
                        best_score = score
                        best_transform = candidate_transform
                        best_label = f"axis{axis_idx}:{signed_angle:+.1f}"

        reranked[part_idx] = best_transform
        print(
            f"Rotation rerank fragment={fragment_names[part_idx]} "
            f"choice={best_label} score={best_score:.6f}",
            flush=True,
        )
    return reranked


def extract_poses_from_coords(input_coords, pred_coords, part_batch_ids,
                              kabsch_robust="none", huber_delta=0.02,
                              irls_iters=5):
    inp = input_coords.detach().float()
    pred = pred_coords.detach().float()
    ids = part_batch_ids.detach()
    num_valid_parts = int(ids.max().item()) + 1

    quat = torch.zeros(num_valid_parts, 4, dtype=torch.float32, device=input_coords.device)
    trans = torch.zeros(num_valid_parts, 3, dtype=torch.float32, device=input_coords.device)
    fit_diagnostics = []

    with torch.autocast(device_type=input_coords.device.type, enabled=False):
        for part_id in range(num_valid_parts):
            mask = ids == part_id
            src = inp[mask]
            dst = pred[mask]
            if src.shape[0] < 3:
                quat[part_id] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=input_coords.device)
                fit_diagnostics.append({"npoints": int(src.shape[0])})
                continue

            src_mean = src.mean(dim=0, keepdim=True)
            dst_mean = dst.mean(dim=0, keepdim=True)
            weights = torch.ones(src.shape[0], dtype=src.dtype, device=src.device)
            n_irls = int(irls_iters) if kabsch_robust == "huber" else 0
            for irls_idx in range(max(0, n_irls) + 1):
                wsum = weights.sum().clamp(min=1e-8)
                src_mean = (src * weights[:, None]).sum(dim=0, keepdim=True) / wsum
                dst_mean = (dst * weights[:, None]).sum(dim=0, keepdim=True) / wsum
                H = ((src - src_mean) * weights[:, None]).T @ (dst - dst_mean)
                U, singular_values, Vt = torch.linalg.svd(H)
                R = Vt.T @ U.T
                if torch.det(R) < 0:
                    Vt = Vt.clone()
                    Vt[2, :] *= -1
                    R = Vt.T @ U.T
                trans_est = dst_mean.squeeze() - src_mean.squeeze() @ R.T
                if kabsch_robust != "huber" or irls_idx == n_irls:
                    break
                residuals_iter = torch.linalg.norm(src @ R.T + trans_est - dst, dim=1)
                cutoff = max(float(huber_delta), 1e-6)
                weights = torch.where(
                    residuals_iter <= cutoff,
                    torch.ones_like(residuals_iter),
                    cutoff / residuals_iter.clamp(min=1e-8),
                )
            trans[part_id] = trans_est
            residuals = torch.linalg.norm(src @ R.T + trans[part_id] - dst, dim=1)
            fit_diagnostics.append({
                "npoints": int(src.shape[0]),
                "svd_residual_mean_norm": float(residuals.mean().item()),
                "svd_residual_median_norm": float(residuals.median().item()),
                "svd_residual_p90_norm": float(torch.quantile(residuals, 0.9).item()),
                "svd_residual_max_norm": float(residuals.max().item()),
            })

            trace = R.trace()
            if trace > 0:
                s = 0.5 / torch.sqrt(trace + 1.0)
                w = 0.25 / s
                x = (R[2, 1] - R[1, 2]) * s
                y = (R[0, 2] - R[2, 0]) * s
                z = (R[1, 0] - R[0, 1]) * s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
            q = torch.stack([w, x, y, z])
            quat[part_id] = q / q.norm().clamp(min=1e-8)
    return quat, trans, fit_diagnostics


def run_iterative_inference(model, fragment_data, device, output_type, max_parts,
                            max_iters=1, convergence_threshold=2.0,
                            inference_precision="fp16", kabsch_robust="none",
                            huber_delta=0.02, irls_iters=5):
    num_parts = len(fragment_data)
    cumulative_transforms = [np.eye(4, dtype=np.float32) for _ in range(num_parts)]
    current_coords = [fd[0].copy() for fd in fragment_data]
    current_normals = [fd[1].copy() for fd in fragment_data]
    inference_diagnostics = []

    print(f"Starting inference: parts={num_parts}, max_iters={max_iters}, mode={output_type}")
    for iter_idx in range(max_iters):
        all_points = np.concatenate(current_coords, axis=0).astype(np.float32)
        all_points, centroid = recenter_pc(all_points)
        all_points, scale = rescale_pc(all_points)

        offsets = [0]
        for coords in current_coords:
            offsets.append(offsets[-1] + len(coords))
        norm_coords = [all_points[offsets[i]:offsets[i + 1]] for i in range(num_parts)]

        all_normals = np.concatenate(current_normals, axis=0).astype(np.float32)
        points_per_part = np.array([len(coords) for coords in norm_coords], dtype=np.int32)
        points_per_part_padded = np.zeros(max_parts, dtype=np.int32)
        points_per_part_padded[:num_parts] = points_per_part

        tensor_dict = {
            "pointclouds": torch.from_numpy(all_points).to(device),
            "pointclouds_normals": torch.from_numpy(all_normals).to(device),
            "points_per_part": torch.from_numpy(points_per_part_padded).unsqueeze(0).to(device),
        }
        use_amp = device == "cuda" and inference_precision != "fp32"
        amp_dtype = torch.bfloat16 if inference_precision == "bf16" else torch.float16
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            output = model(
                input_coords=tensor_dict["pointclouds"],
                input_normals=tensor_dict["pointclouds_normals"],
                points_per_part=tensor_dict["points_per_part"],
            )

        if output_type == "coords":
            quat, trans, fit_diagnostics = extract_poses_from_coords(
                tensor_dict["pointclouds"],
                output["pred_coords"],
                output["part_batch_ids"],
                kabsch_robust=kabsch_robust,
                huber_delta=huber_delta,
                irls_iters=irls_iters,
            )
        else:
            quat, trans = output["quat"], output["trans"]
            fit_diagnostics = [{} for _ in range(num_parts)]
        for part_id, stats in enumerate(fit_diagnostics):
            stats.update({"iteration": iter_idx + 1, "part_index": part_id, "scale_mm": float(scale)})
            for key in ("mean", "median", "p90", "max"):
                norm_key = f"svd_residual_{key}_norm"
                if norm_key in stats:
                    stats[f"svd_residual_{key}_mm"] = stats[norm_key] * float(scale)
        inference_diagnostics.append(fit_diagnostics)
        if not torch.isfinite(quat).all() or not torch.isfinite(trans).all():
            raise RuntimeError(
                f"Non-finite pose prediction in {inference_precision} inference; "
                "rerun with --inference_precision fp32"
            )
        quat = quat.float().detach().cpu()
        trans = trans.float().detach().cpu()
        rot = quaternion_to_matrix(quat)
        del tensor_dict, output
        if device == "cuda":
            torch.cuda.empty_cache()

        max_delta = 0.0
        for i in range(num_parts):
            T_pred = np.eye(4, dtype=np.float32)
            T_pred[:3, :3] = rot[i].numpy()
            T_pred[:3, 3] = trans[i].numpy()
            T_i = transform_pose_to_original_space(T_pred, centroid, scale)

            cumulative_transforms[i] = T_i @ cumulative_transforms[i]
            current_coords[i] = apply_transform_to_points(current_coords[i], T_i)
            current_normals[i] = (T_i[:3, :3] @ current_normals[i].T).T
            max_delta = max(max_delta, float(np.linalg.norm(T_i[:3, 3])))

        print(f"Iter {iter_idx + 1}: max_trans_delta={max_delta:.6f}")
        if max_delta < convergence_threshold:
            break
    return cumulative_transforms, inference_diagnostics


def normalize_transforms_by_first_sa(cumulative_transforms, meshdict):
    if not meshdict.get("SA"):
        return cumulative_transforms
    T_sa0_inv = np.linalg.inv(cumulative_transforms[0])
    return [T_sa0_inv @ T for T in cumulative_transforms]


def project_to_rotation(matrix):
    U, _, Vt = np.linalg.svd(matrix)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def fuse_transform_set(transform_sets):
    """Fuse multiple TTA predictions using robust translation and chordal rotation mean."""
    if len(transform_sets) == 1:
        return transform_sets[0]

    num_parts = len(transform_sets[0])
    fused = []
    for part_idx in range(num_parts):
        matrices = [np.asarray(ts[part_idx], dtype=np.float64) for ts in transform_sets]
        out = np.eye(4, dtype=np.float64)
        out[:3, 3] = np.median(np.stack([m[:3, 3] for m in matrices], axis=0), axis=0)
        out[:3, :3] = project_to_rotation(np.mean(np.stack([m[:3, :3] for m in matrices], axis=0), axis=0))
        fused.append(out.astype(np.float32))
    return fused


def build_final_results(transforms, fragment_names):
    return [
        {"fragment_id": name, "transformation": transforms[i].tolist()}
        for i, name in enumerate(fragment_names)
    ]


def write_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved prediction: {output_path}")


def write_inference_diagnostics(
    output_path, fragment_names, bone_types, centroids, final_transforms,
    transform_sets, pass_diagnostics,
):
    records = []
    fused_uncalibrated = fuse_transform_set(transform_sets)
    for part_idx, (fragment_id, bone_type, centroid, final_transform) in enumerate(
        zip(fragment_names, bone_types, centroids, final_transforms)
    ):
        final_rot = np.asarray(final_transform, dtype=np.float64)[:3, :3]
        reference_rot = np.asarray(fused_uncalibrated[part_idx], dtype=np.float64)[:3, :3]
        centroid = np.asarray(centroid, dtype=np.float64)
        seed_rot_deviations = []
        seed_centroid_positions = []
        for transforms in transform_sets:
            transform = np.asarray(transforms[part_idx], dtype=np.float64)
            seed_rot_deviations.append(
                float(np.degrees(np.linalg.norm(so3_log(transform[:3, :3] @ reference_rot.T))))
            )
            seed_centroid_positions.append(transform[:3, :3] @ centroid + transform[:3, 3])

        seed_centroid_positions = np.stack(seed_centroid_positions, axis=0)
        centroid_mean = seed_centroid_positions.mean(axis=0)
        centroid_deviations = np.linalg.norm(seed_centroid_positions - centroid_mean, axis=1)
        last_fit_stats = [
            diagnostics[-1][part_idx]
            for diagnostics in pass_diagnostics
            if diagnostics and part_idx < len(diagnostics[-1])
        ]

        def mean_stat(name):
            values = [float(stats[name]) for stats in last_fit_stats if name in stats]
            return float(np.mean(values)) if values else None

        records.append({
            "fragment_id": str(fragment_id),
            "bone": bone_type,
            "predicted_angle_deg": float(np.degrees(np.linalg.norm(so3_log(final_rot)))),
            "svd_residual_mean_norm": mean_stat("svd_residual_mean_norm"),
            "svd_residual_p90_norm": mean_stat("svd_residual_p90_norm"),
            "svd_residual_mean_mm": mean_stat("svd_residual_mean_mm"),
            "svd_residual_p90_mm": mean_stat("svd_residual_p90_mm"),
            "tta_count": len(transform_sets),
            "tta_rotation_deviation_mean_deg": float(np.mean(seed_rot_deviations)),
            "tta_rotation_deviation_max_deg": float(np.max(seed_rot_deviations)),
            "tta_centroid_deviation_mean_mm": float(np.mean(centroid_deviations)),
            "tta_centroid_deviation_max_mm": float(np.max(centroid_deviations)),
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"fragments": records}, f, indent=2)
    print(f"Saved inference diagnostics: {output_path}")


def so3_log(R):
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return theta / (2.0 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64)


def so3_exp(rotvec):
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def blend_transform_to_identity(transform, alpha):
    matrix = np.asarray(transform, dtype=np.float64)
    blended = np.eye(4, dtype=np.float64)
    blended[:3, :3] = so3_exp(alpha * so3_log(matrix[:3, :3]))
    blended[:3, 3] = alpha * matrix[:3, 3]
    return blended.astype(np.float32)


def apply_alpha(transforms, alpha):
    if alpha is None or abs(alpha - 1.0) < 1e-8:
        return transforms
    return [blend_transform_to_identity(T, alpha) for T in transforms]


def blend_rotation_preserve_centroid(transform, centroid, rot_alpha):
    matrix = np.asarray(transform, dtype=np.float64)
    centroid = np.asarray(centroid, dtype=np.float64)
    blended = matrix.copy()
    old_rot = matrix[:3, :3]
    old_trans = matrix[:3, 3]
    new_rot = so3_exp(rot_alpha * so3_log(old_rot))
    transformed_centroid = old_rot @ centroid + old_trans
    blended[:3, :3] = new_rot
    blended[:3, 3] = transformed_centroid - new_rot @ centroid
    return blended.astype(np.float32)


def apply_rot_alpha_preserve_centroids(
    transforms,
    centroids,
    rot_alpha,
    bone_types=None,
    rot_alpha_li=None,
    rot_alpha_ri=None,
):
    if len(transforms) != len(centroids):
        raise ValueError(
            f"rot_alpha needs one centroid per transform: "
            f"{len(centroids)} centroids for {len(transforms)} transforms"
        )
    if bone_types is None:
        bone_types = [None] * len(transforms)
    if len(transforms) != len(bone_types):
        raise ValueError(
            f"rot_alpha needs one bone type per transform: "
            f"{len(bone_types)} bone types for {len(transforms)} transforms"
        )

    bone_overrides = {"LI": rot_alpha_li, "RI": rot_alpha_ri}
    calibrated = []
    for transform, centroid, bone_type in zip(transforms, centroids, bone_types):
        effective_alpha = bone_overrides.get(bone_type)
        if effective_alpha is None:
            effective_alpha = rot_alpha
        if effective_alpha is None or abs(effective_alpha - 1.0) < 1e-8:
            calibrated.append(transform)
        else:
            calibrated.append(
                blend_rotation_preserve_centroid(transform, centroid, effective_alpha)
            )
    return calibrated


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(device, args.checkpoint, output_type=args.output_type)
    npoints = args.npoints if args.npoints is not None else DEFAULT_NPOINTS
    max_iters = args.max_iters if args.max_iters is not None else DEFAULT_MAX_ITERS
    alpha = args.alpha if args.alpha is not None else DEFAULT_ALPHA
    rot_alpha = args.rot_alpha
    rot_alpha_li = args.rot_alpha_li
    rot_alpha_ri = args.rot_alpha_ri
    min_points_by_bone = {
        "SA": (
            args.min_points_per_fragment_sa
            if args.min_points_per_fragment_sa is not None
            else args.min_points_per_fragment
        ),
        "LI": (
            args.min_points_per_fragment_li
            if args.min_points_per_fragment_li is not None
            else args.min_points_per_fragment
        ),
        "RI": (
            args.min_points_per_fragment_ri
            if args.min_points_per_fragment_ri is not None
            else args.min_points_per_fragment
        ),
    }
    tta = args.tta if args.tta is not None else DEFAULT_TTA
    tta = max(1, int(tta))
    rerank_angles = parse_rotation_rerank_angles(args.rotation_rerank_angles)

    obj_path = find_obj(args.input)
    print(f"Input OBJ: {obj_path}")
    print(
        f"Device: {device}, npoints={npoints}, max_iters={max_iters}, "
        f"alpha={alpha}, rot_alpha={rot_alpha}, rot_alpha_li={rot_alpha_li}, "
        f"rot_alpha_ri={rot_alpha_ri}, tta={tta}, allocation={args.allocation}, "
        f"min_points_per_fragment={args.min_points_per_fragment}, "
        f"min_points_by_bone={min_points_by_bone}, "
        f"surface_sample={args.surface_sample}, precision={args.inference_precision}"
    )

    meshdict = load_obj_fragments(str(obj_path))
    transform_sets = []
    pass_diagnostics = []
    for tta_idx in range(tta):
        rng = np.random.default_rng(args.seed + tta_idx)
        fragment_data = build_fragment_data_from_meshdict(
            meshdict,
            npoints=npoints,
            rng=rng,
            allocation=args.allocation,
            surface_sample=args.surface_sample,
            min_points_per_fragment=args.min_points_per_fragment,
            min_points_by_bone=min_points_by_bone,
        )
        if not fragment_data:
            raise RuntimeError(f"No valid fragments were loaded from {obj_path}")

        print(f"TTA pass {tta_idx + 1}/{tta}")
        transforms, diagnostics = run_iterative_inference(
            model,
            fragment_data,
            device,
            output_type=args.output_type,
            max_parts=MAX_PARTS,
            max_iters=max_iters,
            convergence_threshold=DEFAULT_CONVERGENCE_THRESHOLD,
            inference_precision=args.inference_precision,
            kabsch_robust=args.kabsch_robust,
            huber_delta=args.kabsch_huber_delta,
            irls_iters=args.kabsch_irls_iters,
        )
        transform_sets.append(normalize_transforms_by_first_sa(transforms, meshdict))
        pass_diagnostics.append(diagnostics)

    transforms = fuse_transform_set(transform_sets)
    transforms = apply_alpha(transforms, alpha)
    fragment_names = fragment_names_from_meshdict(meshdict)
    fragment_bone_types = fragment_bone_types_from_meshdict(meshdict)
    fragment_centroids = fragment_centroids_from_meshdict(meshdict)
    transforms = apply_rot_alpha_preserve_centroids(
        transforms,
        fragment_centroids,
        rot_alpha,
        bone_types=fragment_bone_types,
        rot_alpha_li=rot_alpha_li,
        rot_alpha_ri=rot_alpha_ri,
    )
    if rerank_angles:
        rerank_rng = np.random.default_rng(args.seed + 10000)
        rerank_fragment_data = build_fragment_data_from_meshdict(
            meshdict,
            npoints=args.rotation_rerank_sample_points,
            rng=rerank_rng,
            allocation="equal",
            surface_sample=args.surface_sample,
            min_points_per_fragment=args.min_points_per_fragment,
            min_points_by_bone=min_points_by_bone,
        )
        transforms = rotation_candidate_rerank(
            transforms,
            rerank_fragment_data,
            fragment_names,
            rerank_angles,
            args.rotation_rerank_contact_fraction,
            args.rotation_rerank_angle_weight,
        )
    if args.diagnostics_output:
        write_inference_diagnostics(
            args.diagnostics_output,
            fragment_names,
            fragment_bone_types,
            fragment_centroids,
            transforms,
            transform_sets,
            pass_diagnostics,
        )
    results = build_final_results(transforms, fragment_names)

    write_results(results, args.output)


if __name__ == "__main__":
    main()
