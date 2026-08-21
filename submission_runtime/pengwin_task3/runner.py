"""Grand Challenge container entrypoint for PENGWIN Task 3.

The container reads one OBJ fracture case from /input and writes:
    /output/reduction-poses-matrices.json
"""

import argparse
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import trimesh

from models.AssemblyNet.transformer import AssemblyTransformer
from models.AssemblyNet.utils import quaternion_to_matrix


OUTPUT_TYPE = "coords"
MAX_PARTS = 50
EMBED_DIM = 384
NUM_LAYERS = 12
NUM_HEADS = 8
DROPOUT_RATE = 0.0
DEFAULT_NPOINTS = 3000
DEFAULT_MIN_POINTS_SA = 75
DEFAULT_MIN_POINTS_LI = 75
DEFAULT_MIN_POINTS_RI = 50
DEFAULT_MAX_ITERS = 1
DEFAULT_ALPHA = 1.0
DEFAULT_ROT_ALPHA = 0.975
DEFAULT_TTA = 1
DEFAULT_SEED = 42
DEFAULT_CONVERGENCE_THRESHOLD = 2.0


def select_device():
    """Use CUDA only when this PyTorch build contains the visible GPU arch."""
    if not torch.cuda.is_available():
        return "cpu"

    major, minor = torch.cuda.get_device_capability(0)
    visible_arch = f"sm_{major}{minor}"
    compiled_arches = set(torch.cuda.get_arch_list())
    if visible_arch not in compiled_arches:
        print(
            f"CUDA architecture {visible_arch} is not included in this PyTorch "
            f"build ({sorted(compiled_arches)}); using CPU instead."
        )
        return "cpu"
    return "cuda"


def _load_obj_fragment_sections(obj_path):
    """Recover original OBJ object/group IDs and geometry signatures."""
    sections = []
    vertices = []
    current = None
    with open(obj_path, errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith(("o ", "g ")):
                parts = line.split(maxsplit=1)
                if len(parts) != 2 or not parts[1].strip():
                    continue
                fid = parts[1].strip()
                if current is None or current["id"] != fid:
                    current = {"id": fid, "face_count": 0, "vertex_indices": set()}
                    sections.append(current)
                continue
            if current is None:
                continue
            if line.startswith("v "):
                values = line.split()[1:4]
                if len(values) == 3:
                    vertices.append(np.asarray(values, dtype=np.float64))
            elif line.startswith("f "):
                face = line.split()[1:]
                current["face_count"] += max(len(face) - 2, 0)
                for item in face:
                    raw_index = item.split("/", 1)[0]
                    if not raw_index:
                        continue
                    index = int(raw_index)
                    index = index - 1 if index > 0 else len(vertices) + index
                    current["vertex_indices"].add(index)
    for section in sections:
        indices = sorted(section.pop("vertex_indices"))
        section["vertex_count"] = len(indices)
        section["vertex_mean"] = np.asarray([vertices[index] for index in indices]).mean(axis=0)
    return sections


def _load_obj_fragment_ids(obj_path):
    return [section["id"] for section in _load_obj_fragment_sections(obj_path)]


def _iter_fragments_with_gt_ids(obj_path, scene, gt_poses):
    if isinstance(scene, trimesh.Scene):
        geometries = [geom for geom in scene.geometry.values() if isinstance(geom, trimesh.Trimesh)]
    elif isinstance(scene, trimesh.Trimesh):
        geometries = [scene]
    else:
        geometries = []

    fragments = []
    if isinstance(scene, trimesh.Scene):
        for key, geom in scene.geometry.items():
            key = str(key)
            if isinstance(geom, trimesh.Trimesh) and key in gt_poses:
                fragments.append((key, geom))
        if len(fragments) == len(geometries):
            return fragments
    elif isinstance(scene, trimesh.Trimesh) and "1" in gt_poses:
        return [("1", scene)]

    sections = [section for section in _load_obj_fragment_sections(obj_path) if section["id"] in gt_poses]
    if len(sections) != len(geometries):
        return fragments
    unmatched = set(range(len(sections)))
    recovered = []
    for geom in geometries:
        candidates = [
            index for index in unmatched
            if sections[index]["vertex_count"] == len(geom.vertices)
            and sections[index]["face_count"] == len(geom.faces)
        ]
        if not candidates:
            return fragments
        geom_mean = np.asarray(geom.vertices, dtype=np.float64).mean(axis=0)
        index = min(
            candidates,
            key=lambda candidate: np.linalg.norm(sections[candidate]["vertex_mean"] - geom_mean),
        )
        unmatched.remove(index)
        recovered.append((sections[index]["id"], geom))
    return recovered


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
    parser.add_argument("--min_points_per_fragment_sa", type=int,
                        default=DEFAULT_MIN_POINTS_SA)
    parser.add_argument("--min_points_per_fragment_li", type=int,
                        default=DEFAULT_MIN_POINTS_LI)
    parser.add_argument("--min_points_per_fragment_ri", type=int,
                        default=DEFAULT_MIN_POINTS_RI)
    parser.add_argument("--max_iters", type=int, default=None, help="Override iterative inference steps")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Blend predicted relative poses toward identity")
    parser.add_argument("--rot_alpha", type=float, default=None,
                        help="Scale rotations toward/from identity while preserving transformed mesh centroids")
    parser.add_argument("--tta", type=int, default=None,
                        help="Number of stochastic surface-sampling test-time augmentations")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Base random seed for deterministic TTA sampling")
    parser.add_argument("--allocation", choices=["equal", "volume"], default="volume",
                        help="Point allocation within each bone")
    parser.add_argument("--surface_sample", choices=["even", "random"], default="random",
                        help="Surface sampling strategy")
    return parser.parse_args()


def load_model(device, checkpoint_path):
    model = AssemblyTransformer(
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        max_parts=MAX_PARTS,
        output_type=OUTPUT_TYPE,
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
    volumes = np.asarray([fragments[k]["volume"] for k in frag_keys], dtype=np.float64)
    volume_sum = float(volumes.sum())
    if not np.isfinite(volume_sum) or volume_sum <= 0:
        volumes = np.ones(len(frag_keys), dtype=np.float64)
        volume_sum = float(volumes.sum())

    allocations = [max(int(vol / volume_sum * total_points), min_points) for vol in volumes]
    diff = total_points - sum(allocations)
    if diff != 0:
        max_vol_idx = int(np.argmax(volumes))
        allocations[max_vol_idx] = max(allocations[max_vol_idx] + diff, min_points)
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


def sub_sample(meshdict, npoints=3000, min_points_by_bone=None, rng=None,
               allocation="equal", surface_sample="even"):
    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED)
    sampled_data = {}
    for bone_type, fragments in meshdict.items():
        sampled_data[bone_type] = {}
        frag_keys = sorted(fragments.keys(), key=lambda name: int(name))
        if not frag_keys:
            continue
        min_points = 20
        if min_points_by_bone is not None:
            min_points = int(min_points_by_bone.get(bone_type, min_points))

        if allocation == "volume":
            allocations = _allocate_by_volume(fragments, frag_keys, npoints, min_points)
        else:
            allocations = _allocate_equal(npoints, len(frag_keys), min_points)

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
                                      min_points_by_bone=None,
                                      allocation="equal", surface_sample="even"):
    sampled = sub_sample(
        meshdict,
        npoints=npoints,
        min_points_by_bone=min_points_by_bone,
        rng=rng,
        allocation=allocation,
        surface_sample=surface_sample,
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


def extract_poses_from_coords(input_coords, pred_coords, part_batch_ids):
    inp = input_coords.detach().float()
    pred = pred_coords.detach().float()
    ids = part_batch_ids.detach()
    num_valid_parts = int(ids.max().item()) + 1

    quat = torch.zeros(num_valid_parts, 4, dtype=torch.float32, device=input_coords.device)
    trans = torch.zeros(num_valid_parts, 3, dtype=torch.float32, device=input_coords.device)

    with torch.autocast(device_type=input_coords.device.type, enabled=False):
        for part_id in range(num_valid_parts):
            mask = ids == part_id
            src = inp[mask]
            dst = pred[mask]
            if src.shape[0] < 3:
                quat[part_id] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=input_coords.device)
                continue

            src_mean = src.mean(dim=0, keepdim=True)
            dst_mean = dst.mean(dim=0, keepdim=True)
            H = (src - src_mean).T @ (dst - dst_mean)
            U, _, Vt = torch.linalg.svd(H)
            R = Vt.T @ U.T
            if torch.det(R) < 0:
                Vt = Vt.clone()
                Vt[2, :] *= -1
                R = Vt.T @ U.T
            trans[part_id] = dst_mean.squeeze() - src_mean.squeeze() @ R.T

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
    return quat, trans


def run_iterative_inference(model, fragment_data, device, output_type, max_parts,
                            max_iters=1, convergence_threshold=2.0):
    num_parts = len(fragment_data)
    cumulative_transforms = [np.eye(4, dtype=np.float32) for _ in range(num_parts)]
    current_coords = [fd[0].copy() for fd in fragment_data]
    current_normals = [fd[1].copy() for fd in fragment_data]

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
        use_amp = device == "cuda"
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = model(
                input_coords=tensor_dict["pointclouds"],
                input_normals=tensor_dict["pointclouds_normals"],
                points_per_part=tensor_dict["points_per_part"],
            )

        if output_type == "coords":
            quat, trans = extract_poses_from_coords(
                tensor_dict["pointclouds"],
                output["pred_coords"],
                output["part_batch_ids"],
            )
        else:
            quat, trans = output["quat"], output["trans"]
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
    return cumulative_transforms


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


def write_identity_fallback(args, exc):
    print("Inference failed; writing identity fallback so the container exits cleanly.")
    print("Failure traceback:")
    traceback.print_exception(type(exc), exc, exc.__traceback__)

    try:
        obj_path = find_obj(args.input)
        meshdict = load_obj_fragments(str(obj_path))
        names = fragment_names_from_meshdict(meshdict)
    except Exception as parse_exc:
        print("Could not parse input fragments for fallback output.")
        traceback.print_exception(type(parse_exc), parse_exc, parse_exc.__traceback__)
        names = []

    identity = np.eye(4, dtype=np.float32)
    results = [
        {"fragment_id": name, "transformation": identity.tolist()}
        for name in names
    ]
    write_results(results, args.output)


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


def apply_rot_alpha_preserve_centroids(transforms, centroids, rot_alpha):
    if rot_alpha is None or abs(rot_alpha - 1.0) < 1e-8:
        return transforms
    if len(transforms) != len(centroids):
        raise ValueError(
            f"rot_alpha needs one centroid per transform: "
            f"{len(centroids)} centroids for {len(transforms)} transforms"
        )
    return [
        blend_rotation_preserve_centroid(transform, centroid, rot_alpha)
        for transform, centroid in zip(transforms, centroids)
    ]


def main():
    args = parse_args()
    device = select_device()

    model = load_model(device, args.checkpoint)
    npoints = args.npoints if args.npoints is not None else DEFAULT_NPOINTS
    max_iters = args.max_iters if args.max_iters is not None else DEFAULT_MAX_ITERS
    alpha = args.alpha if args.alpha is not None else DEFAULT_ALPHA
    rot_alpha = args.rot_alpha if args.rot_alpha is not None else DEFAULT_ROT_ALPHA
    tta = args.tta if args.tta is not None else DEFAULT_TTA
    tta = max(1, int(tta))
    min_points_by_bone = {
        "SA": args.min_points_per_fragment_sa,
        "LI": args.min_points_per_fragment_li,
        "RI": args.min_points_per_fragment_ri,
    }

    obj_path = find_obj(args.input)
    print(f"Input OBJ: {obj_path}")
    print(
        f"Device: {device}, npoints={npoints}, max_iters={max_iters}, "
        f"alpha={alpha}, rot_alpha={rot_alpha}, tta={tta}, allocation={args.allocation}, "
        f"surface_sample={args.surface_sample}, min_points={min_points_by_bone}"
    )

    meshdict = load_obj_fragments(str(obj_path))
    transform_sets = []
    for tta_idx in range(tta):
        rng = np.random.default_rng(args.seed + tta_idx)
        fragment_data = build_fragment_data_from_meshdict(
            meshdict,
            npoints=npoints,
            rng=rng,
            min_points_by_bone=min_points_by_bone,
            allocation=args.allocation,
            surface_sample=args.surface_sample,
        )
        if not fragment_data:
            raise RuntimeError(f"No valid fragments were loaded from {obj_path}")

        print(f"TTA pass {tta_idx + 1}/{tta}")
        transforms = run_iterative_inference(
            model,
            fragment_data,
            device,
            output_type=OUTPUT_TYPE,
            max_parts=MAX_PARTS,
            max_iters=max_iters,
            convergence_threshold=DEFAULT_CONVERGENCE_THRESHOLD,
        )
        transform_sets.append(normalize_transforms_by_first_sa(transforms, meshdict))

    transforms = fuse_transform_set(transform_sets)
    transforms = apply_alpha(transforms, alpha)
    transforms = apply_rot_alpha_preserve_centroids(
        transforms,
        fragment_centroids_from_meshdict(meshdict),
        rot_alpha,
    )
    results = build_final_results(transforms, fragment_names_from_meshdict(meshdict))

    write_results(results, args.output)


def cli():
    """Parse container arguments and execute one case with a safe fallback."""
    global args
    args = parse_args()
    try:
        main()
    except Exception as exc:
        write_identity_fallback(args, exc)


if __name__ == "__main__":
    cli()
