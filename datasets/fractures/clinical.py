"""Clinical (real) fracture dataset for point cloud assembly training.

Loads OBJ fragment meshes and GT reduction poses from `plan_pl_gt.json`.
Applies GT poses to obtain assembled ground-truth positions, then feeds
through the same augmentation pipeline as the simulation dataset.

Expected directory structure:
    data_root_real/
        sample_001/
            fragments.obj          (all fragments; keys 1-100=SA, 101-200=LI, 201-300=RI)
            plan_pl_gt.json        (GT 4x4 reduction matrices)
        sample_002/
            ...
"""

import json
import os
import hashlib
import random
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from .sample import FragmentSamples
from .mesh_sampling import sample_fragments_equal
from ..transform import recenter_pc, rescale_pc


GT_JSON_NAME = "plan_pl_gt.json"
BONE_RANGES = {
    "SA": (1, 100),
    "LI": (101, 200),
    "RI": (201, 300),
}
BONE_TYPE_MAP = {"SA": 0, "LI": 1, "RI": 2}


def _load_gt_poses(json_path: str) -> dict:
    """Load GT 4x4 matrices from plan_pl_gt.json.

    Returns:
        dict mapping fragment_id (str) -> np.ndarray (4, 4)
    """
    with open(json_path) as f:
        data = json.load(f)

    poses = {}
    if isinstance(data, list):
        for entry in data:
            fid = str(entry["fragment_id"])
            poses[fid] = np.array(entry["transformation"], dtype=np.float64)
    elif isinstance(data, dict):
        for fid, mat in data.items():
            poses[str(fid)] = np.array(mat, dtype=np.float64)
    return poses


def _load_obj_fragment_sections(obj_path: str) -> list[dict]:
    """Read original OBJ object/group metadata in file order.

    Newer trimesh versions may rename repeated scene keys as 1_1, 1_2, ...
    even when the OBJ groups are 1, 101, 201.  The GT json uses the original
    ids, so recover them directly from the file.  Geometry statistics let us
    match ids without assuming that ``scene.geometry`` preserves file order.
    """
    sections = []
    vertices = []
    current = None
    with open(obj_path, errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith(("o ", "g ")):
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                fid = parts[1].strip()
                if not fid:
                    continue
                if current is None or current["id"] != fid:
                    current = {
                        "id": fid,
                        "face_count": 0,
                        "vertex_indices": set(),
                    }
                    sections.append(current)
                continue

            if current is None:
                continue
            if line.startswith("v "):
                values = line.split()[1:4]
                if len(values) == 3:
                    vertices.append(np.asarray(values, dtype=np.float64))
            elif line.startswith("f "):
                # Trimesh triangulates an OBJ polygon with n vertices into n-2 faces.
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


def _load_obj_fragment_ids(obj_path: str) -> list[str]:
    return [section["id"] for section in _load_obj_fragment_sections(obj_path)]


def _iter_fragments_with_gt_ids(obj_path: str, scene, gt_poses: dict) -> list[tuple[str, trimesh.Trimesh]]:
    if isinstance(scene, trimesh.Scene):
        geometries = [geom for geom in scene.geometry.values() if isinstance(geom, trimesh.Trimesh)]
    elif isinstance(scene, trimesh.Trimesh):
        geometries = [scene]
    else:
        geometries = []

    # Trimesh 4.8.x preserves the original OBJ ids as geometry keys.  Trust
    # those keys before attempting compatibility recovery for newer versions.
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

    # Trimesh >=4.9 can replace every OBJ id with names such as 1_1 while also
    # returning geometries in reverse file order.  Match each mesh to its OBJ
    # section by geometry statistics instead of zipping two unrelated orders.
    sections = [section for section in _load_obj_fragment_sections(obj_path) if section["id"] in gt_poses]
    if len(sections) != len(geometries):
        return fragments

    unmatched = set(range(len(sections)))
    recovered = []
    for geom in geometries:
        candidates = [
            index
            for index in unmatched
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


def _apply_T(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 homogeneous transform to (N, 3) points."""
    ones = np.ones((len(pts), 1), dtype=pts.dtype)
    return (T @ np.concatenate([pts, ones], axis=1).T).T[:, :3]


def _apply_T_normals(normals: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply rotation part of T to normals."""
    return (T[:3, :3] @ normals.T).T


class ClinicalFragmentSamples(FragmentSamples):
    """Clinical fracture dataset.

    Loads OBJ + plan_pl_gt.json pairs, applies GT poses to obtain assembled
    ground-truth positions, then runs the same augmentation pipeline as
    FragmentSamples (clinical pose variation, random rotation, etc.).
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        data_root_real: str = "",
        npoints_per_bone: int = 5000,
        min_parts: int = 2,
        max_parts: int = 50,
        max_bone: int = 3,
        random_anchor: bool = True,
        merge_prob: float = 0.15,
        frac_prob: float = 0.0,   # not used for clinical (no template)
        drop_prob: float = 0.1,
        region_dropout_prob: float = 0.0,
        point_jitter_std: float = 0.0,
        use_real_poses: bool = True,
        use_all_train: bool = False,
        split_json: str = "",
        split_fold: str = "fold0",
        mesh_allocation: str = "equal",
        surface_sample: str = "even",
        small_fragment_boost_factor: float = 1.0,
        small_fragment_area_percentile: float = 30.0,
        hard_pose_aug_prob: float = 0.0,
        hard_pose_aug_max_angle_deg: float = 35.0,
        hard_pose_aug_max_translation_mm: float = 30.0,
        hard_pose_aug_bones: list[str] | tuple[str, ...] = ("LI", "RI"),
        hard_pose_aug_preserve_anchor: bool = True,
    ):
        # Bypass FragmentDataBase.__init__ — we build our own index
        Dataset.__init__(self)

        self.data_root = data_root_real
        self.split = split
        self.npoints_per_bone = npoints_per_bone
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.max_bone = max_bone
        self.random_anchor = random_anchor
        self.merge_prob = merge_prob
        self.frac_prob = frac_prob
        self.drop_prob = drop_prob
        self.region_dropout_prob = region_dropout_prob
        self.point_jitter_std = point_jitter_std
        self.use_real_poses = use_real_poses
        self.use_all_train = use_all_train
        self.split_json = split_json
        self.split_fold = split_fold
        self.mesh_allocation = mesh_allocation
        self.surface_sample = surface_sample
        self.small_fragment_boost_factor = small_fragment_boost_factor
        self.small_fragment_area_percentile = small_fragment_area_percentile
        self.hard_pose_aug_prob = hard_pose_aug_prob
        self.hard_pose_aug_max_angle_deg = hard_pose_aug_max_angle_deg
        self.hard_pose_aug_max_translation_mm = hard_pose_aug_max_translation_mm
        self.hard_pose_aug_bones = set(hard_pose_aug_bones)
        self.hard_pose_aug_preserve_anchor = hard_pose_aug_preserve_anchor
        self.BONE_TYPE_MAP = BONE_TYPE_MAP

        if not os.path.exists(data_root_real):
            if os.environ.get("LOCAL_RANK", "0") == "0":
                print(f"[Warning] Clinical data root does not exist: {data_root_real}")

        self.samples = self._discover_samples()
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"[Clinical] Split: {split}, Samples: {len(self.samples)}")

        # Build index_map (list of sample paths) — reshuffle randomises order
        self.index_map = list(self.samples)
        self.reshuffle()

    def _discover_samples(self) -> list:
        """Return list of (sample_name, obj_path, json_path) for valid samples."""
        root = Path(self.data_root)
        if not root.is_dir():
            return []

        all_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        n = len(all_dirs)
        split_idx = int(n * 0.9)

        if self.split_json:
            with open(self.split_json) as f:
                manifest = json.load(f)
            split_payload = manifest["splits"][self.split_fold]
            key = self.split if self.split in split_payload else "test"
            selected = set(str(x) for x in split_payload[key])
            dirs = [d for d in all_dirs if d.name in selected]
        elif self.split == "train" and self.use_all_train:
            dirs = all_dirs
        elif self.split == "train":
            dirs = all_dirs[:split_idx]
        elif self.split == "val":
            dirs = all_dirs[split_idx:]
        else:
            dirs = all_dirs

        samples = []
        for d in dirs:
            obj_files = sorted(d.glob("*.obj"))
            json_path = d / GT_JSON_NAME
            if not obj_files or not json_path.exists():
                continue
            samples.append((d.name, str(obj_files[0]), str(json_path)))
        return samples

    def reshuffle(self):
        if self.split not in ["val", "test"]:
            random.shuffle(self.index_map)
        else:
            rng = random.Random(42)
            rng.shuffle(self.index_map)

    def get_relative_rotation_hardness_deg(self) -> np.ndarray:
        """Return per-case maximum GT rotation relative to the first SA.

        Values follow the current ``index_map`` order so they can be used as
        sampler weights without loading or sampling any mesh geometry.  This
        method is intentionally restricted to discovered samples; callers must
        only invoke it on the training split to avoid validation/test leakage.
        """
        hardness = []
        for sample_name, _, json_path in self.index_map:
            gt_poses = _load_gt_poses(json_path)
            sa_ids = sorted(
                (fid for fid in gt_poses if BONE_RANGES["SA"][0] <= int(fid) <= BONE_RANGES["SA"][1]),
                key=int,
            )
            if not sa_ids:
                raise RuntimeError(f"Case {sample_name} has no SA anchor in {json_path}")

            anchor_rot = np.asarray(gt_poses[sa_ids[0]][:3, :3], dtype=np.float64)
            self._validate_rotation_matrix(
                anchor_rot, sample_name=sample_name, fragment_id=sa_ids[0]
            )
            case_angles = []
            for fid, pose in gt_poses.items():
                if fid == sa_ids[0]:
                    continue
                fragment_rot = np.asarray(pose[:3, :3], dtype=np.float64)
                self._validate_rotation_matrix(
                    fragment_rot, sample_name=sample_name, fragment_id=fid
                )
                rel_rot = anchor_rot.T @ fragment_rot
                cos_angle = 0.5 * (float(np.trace(rel_rot)) - 1.0)
                # The matrices have already been validated. This only removes
                # floating-point round-off at the acos boundary; it never
                # repairs an invalid transform.
                if cos_angle < -1.0 - 1e-10 or cos_angle > 1.0 + 1e-10:
                    raise ValueError(
                        f"Invalid relative rotation in case {sample_name}, "
                        f"fragment {fid}: trace-derived cosine={cos_angle}"
                    )
                angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
                case_angles.append(angle_deg)
            hardness.append(max(case_angles, default=0.0))
        return np.asarray(hardness, dtype=np.float64)

    @staticmethod
    def _validate_rotation_matrix(
        rotation: np.ndarray, *, sample_name: str, fragment_id: str,
        atol: float = 1e-5,
    ) -> None:
        """Fail fast on malformed GT rotations; no projection/fallback is allowed."""
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError(
                f"Invalid rotation matrix in case {sample_name}, fragment {fragment_id}"
            )
        # Keep this pre-worker check free of NumPy/SciPy linear algebra.  The
        # DataLoader uses fork on Linux, and initializing a threaded BLAS
        # backend in the parent can deadlock child workers.
        orthogonality_error = max(
            abs(sum(float(rotation[k, i] * rotation[k, j]) for k in range(3)) - (1.0 if i == j else 0.0))
            for i in range(3)
            for j in range(3)
        )
        determinant = float(
            rotation[0, 0] * (rotation[1, 1] * rotation[2, 2] - rotation[1, 2] * rotation[2, 1])
            - rotation[0, 1] * (rotation[1, 0] * rotation[2, 2] - rotation[1, 2] * rotation[2, 0])
            + rotation[0, 2] * (rotation[1, 0] * rotation[2, 1] - rotation[1, 1] * rotation[2, 0])
        )
        if orthogonality_error > atol or abs(determinant - 1.0) > atol:
            raise ValueError(
                f"GT rotation is not in SO(3) for case {sample_name}, fragment {fragment_id}: "
                f"orthogonality_error={orthogonality_error:.3e}, det={determinant:.12f}"
            )

    def _get_rng(self, key: str):
        if self.split in ["val", "test"]:
            seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)
            return random.Random(seed), np.random.RandomState(seed)
        return random, np.random

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        sample_name, obj_path, json_path = self.index_map[index]
        rng, np_rng = self._get_rng(f"{sample_name}_{index}")

        # Load meshes and GT poses
        scene = trimesh.load(obj_path, split_object=True, process=False)
        gt_poses = _load_gt_poses(json_path)

        # Collect fragments per bone type
        bone_fragments: dict[str, list] = {bt: [] for bt in ["SA", "LI", "RI"]}
        for frag_key, geom in _iter_fragments_with_gt_ids(obj_path, scene, gt_poses):
            key_int = int(frag_key)
            for bt, (lo, hi) in BONE_RANGES.items():
                if lo <= key_int <= hi:
                    bone_fragments[bt].append((frag_key, geom))
                    break

        # Build PENGWIN task tensors from the real fractured OBJ and GT poses.
        all_points_gt, all_normals_gt = [], []
        all_points_input, all_normals_input = [], []
        all_bone_types, all_points_per_part, all_points_per_bone = [], [], []
        all_point_bone_ids = []
        all_gt_poses = []
        num_parts = 0

        for bone_idx, bt in enumerate(["SA", "LI", "RI"]):
            frags = sorted(bone_fragments[bt], key=lambda x: int(x[0]))
            if not frags:
                continue

            if self.surface_sample == "original":
                # Exact e50a959 clinical path: floor-divide the per-bone budget
                # and call trimesh directly without a replacement sampler.
                n_points = max(self.npoints_per_bone // len(frags), 20)
                sampled_fragments = []
                for _, mesh in frags:
                    pts, face_indices = trimesh.sample.sample_surface_even(
                        mesh, n_points
                    )
                    normals = mesh.face_normals[face_indices]
                    sampled_fragments.append(
                        (pts.astype(np.float32), normals.astype(np.float32))
                    )
            else:
                sampled_fragments = sample_fragments_equal(
                    [mesh for _, mesh in frags],
                    self.npoints_per_bone,
                    np_rng,
                    allocation=self.mesh_allocation,
                    surface_sample=self.surface_sample,
                    small_fragment_boost_factor=(
                        self.small_fragment_boost_factor
                        if self.split == "train" else 1.0
                    ),
                    small_fragment_area_percentile=self.small_fragment_area_percentile,
                )
            bone_total = 0

            for (frag_key, mesh), (pts, normals) in zip(frags, sampled_fragments):
                T_gt = gt_poses[frag_key]

                # Apply GT pose: scattered → assembled
                pts_gt = _apply_T(pts, T_gt).astype(np.float32)
                normals_gt = _apply_T_normals(normals, T_gt).astype(np.float32)

                all_points_input.append(pts.astype(np.float32))
                all_normals_input.append(normals.astype(np.float32))
                all_points_gt.append(pts_gt)
                all_normals_gt.append(normals_gt)
                all_bone_types.append(BONE_TYPE_MAP[bt])
                all_points_per_part.append(len(pts_gt))
                all_point_bone_ids.append(np.full(len(pts_gt), bone_idx, dtype=np.int32))
                all_gt_poses.append(T_gt.astype(np.float64))
                bone_total += len(pts_gt)
                num_parts += 1

            if bone_total > 0:
                all_points_per_bone.append(bone_total)

        if num_parts == 0:
            raise RuntimeError(f"No valid fragments found in {obj_path}")

        frac_dict = {
            "name": sample_name,
            "num_parts": num_parts,
            "pointclouds": np.concatenate(all_points_input, axis=0),
            "pointclouds_normals": np.concatenate(all_normals_input, axis=0),
            "pointclouds_gt": np.concatenate(all_points_gt, axis=0),
            "pointclouds_normals_gt": np.concatenate(all_normals_gt, axis=0),
            "boneType": all_bone_types,
            "points_per_part": all_points_per_part,
            "points_per_bone": all_points_per_bone,
            "point_bone_ids": np.concatenate(all_point_bone_ids, axis=0),
            "gt_poses": all_gt_poses,
            "anon_bone_ids_per_part": [
                i for i, bt in enumerate(["SA", "LI", "RI"])
                for _ in range(len(bone_fragments[bt]))
                if bone_fragments[bt]
            ],
        }

        # Reuse simulation augmentation pipeline
        from .base import remove_extra_parts
        frac_dict = remove_extra_parts(frac_dict, max_parts=self.max_parts)
        if self.use_real_poses:
            return self.transform_real_pengwin(frac_dict, rng=rng, np_rng=np_rng)
        return self.transform(frac_dict, rng=rng, np_rng=np_rng)

    @staticmethod
    def _random_hard_pose(points: np.ndarray, max_angle_deg: float, max_translation_mm: float, np_rng):
        """Return an extra scattered-space perturbation around a fragment center."""
        center = points.mean(axis=0).astype(np.float64)
        axis = np_rng.normal(size=3)
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm < 1e-8:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = axis / norm

        angle = np.deg2rad(np_rng.uniform(-max_angle_deg, max_angle_deg))
        rot_mat = R.from_rotvec(axis * angle).as_matrix()

        direction = np_rng.normal(size=3)
        direction_norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction_norm) or direction_norm < 1e-8:
            direction = np.zeros(3, dtype=np.float64)
        else:
            direction = direction / direction_norm
        radius = np_rng.uniform(0.0, max_translation_mm)
        translation = direction * radius

        # x_aug = R_aug @ (x - center) + center + translation
        t_aug = center - rot_mat @ center + translation
        return rot_mat.astype(np.float64), t_aug.astype(np.float64)

    def _apply_hard_pose_augmentation(
        self,
        pointclouds: np.ndarray,
        pointclouds_normals: np.ndarray,
        offsets: np.ndarray,
        bone_type: list[int],
        gt_poses: list[np.ndarray],
        num_parts: int,
        np_rng,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        if self.hard_pose_aug_prob <= 0:
            return pointclouds, pointclouds_normals, gt_poses

        target_bone_ids = {
            BONE_TYPE_MAP[bone]
            for bone in self.hard_pose_aug_bones
            if bone in BONE_TYPE_MAP
        }
        if not target_bone_ids:
            return pointclouds, pointclouds_normals, gt_poses

        updated_poses = list(gt_poses)
        for part_idx in range(num_parts):
            if self.hard_pose_aug_preserve_anchor and part_idx == 0:
                continue
            if bone_type[part_idx] not in target_bone_ids:
                continue
            if np_rng.rand() >= self.hard_pose_aug_prob:
                continue

            start = int(offsets[part_idx])
            end = int(offsets[part_idx + 1])
            if end <= start:
                continue

            rot_aug, trans_aug = self._random_hard_pose(
                pointclouds[start:end],
                self.hard_pose_aug_max_angle_deg,
                self.hard_pose_aug_max_translation_mm,
                np_rng,
            )
            pointclouds[start:end] = (
                rot_aug @ pointclouds[start:end].astype(np.float64).T
            ).T.astype(np.float32) + trans_aug.astype(np.float32)
            pointclouds_normals[start:end] = (
                rot_aug @ pointclouds_normals[start:end].astype(np.float64).T
            ).T.astype(np.float32)

            T_gt = updated_poses[part_idx].astype(np.float64)
            T_new = T_gt.copy()
            T_new[:3, :3] = T_gt[:3, :3] @ rot_aug.T
            T_new[:3, 3] = T_gt[:3, 3] - T_new[:3, :3] @ trans_aug
            updated_poses[part_idx] = T_new

        return pointclouds, pointclouds_normals, updated_poses

    def transform_real_pengwin(self, data, rng=None, np_rng=None):
        """Prepare true PENGWIN clinical samples.

        The input is the observed fractured OBJ. The target is obtained by
        applying plan_pl_gt.json to the same sampled points. Normalization
        matches inference.py: recenter/rescale by the fractured input.
        """
        if rng is None:
            rng = random
        if np_rng is None:
            np_rng = np.random

        pointclouds = data["pointclouds"].astype(np.float32)
        pointclouds_normals = data["pointclouds_normals"].astype(np.float32)
        pointclouds_gt = data["pointclouds_gt"].astype(np.float32)
        pointclouds_normals_gt = data["pointclouds_normals_gt"].astype(np.float32)
        point_bone_ids = data["point_bone_ids"].astype(np.int32)
        points_per_part = np.asarray(data["points_per_part"], dtype=np.int32)
        points_per_bone = np.asarray(data["points_per_bone"], dtype=np.int32)
        bone_type = list(data["boneType"])
        gt_poses = list(data["gt_poses"])
        num_parts = int(data["num_parts"])

        offsets = np.concatenate([[0], np.cumsum(points_per_part[:num_parts])])

        if self.random_anchor:
            order = np_rng.permutation(num_parts)
            parts_in, normals_in, parts_gt, normals_gt = [], [], [], []
            bids_parts, poses_new, bone_new, ppp_new = [], [], [], []
            for idx in order:
                s, e = int(offsets[idx]), int(offsets[idx + 1])
                parts_in.append(pointclouds[s:e])
                normals_in.append(pointclouds_normals[s:e])
                parts_gt.append(pointclouds_gt[s:e])
                normals_gt.append(pointclouds_normals_gt[s:e])
                bids_parts.append(point_bone_ids[s:e])
                poses_new.append(gt_poses[idx])
                bone_new.append(bone_type[idx])
                ppp_new.append(points_per_part[idx])
            pointclouds = np.concatenate(parts_in, axis=0)
            pointclouds_normals = np.concatenate(normals_in, axis=0)
            pointclouds_gt = np.concatenate(parts_gt, axis=0)
            pointclouds_normals_gt = np.concatenate(normals_gt, axis=0)
            point_bone_ids = np.concatenate(bids_parts, axis=0)
            gt_poses = poses_new
            bone_type = bone_new
            points_per_part = np.asarray(ppp_new, dtype=np.int32)
            offsets = np.concatenate([[0], np.cumsum(points_per_part[:num_parts])])

        if self.split == "train":
            if self.region_dropout_prob > 0:
                (pointclouds, pointclouds_normals, pointclouds_gt,
                 pointclouds_normals_gt, point_bone_ids, offsets) = self._apply_region_dropout(
                    pointclouds, pointclouds_normals,
                    pointclouds_gt, pointclouds_normals_gt, point_bone_ids,
                    offsets, num_parts, np_rng
                )
                points_per_part = np.asarray(offsets[1:] - offsets[:-1], dtype=np.int32)

            if self.hard_pose_aug_prob > 0:
                pointclouds, pointclouds_normals, gt_poses = self._apply_hard_pose_augmentation(
                    pointclouds,
                    pointclouds_normals,
                    offsets,
                    bone_type,
                    gt_poses,
                    num_parts,
                    np_rng,
                )

            if self.point_jitter_std > 0:
                pointclouds = pointclouds + np_rng.normal(
                    0, self.point_jitter_std, pointclouds.shape
                ).astype(np.float32)

        pointclouds, translation_final = recenter_pc(pointclouds)
        pointclouds, scale = rescale_pc(pointclouds)
        pointclouds_gt = (pointclouds_gt - translation_final) / scale

        center = translation_final.reshape(3).astype(np.float64)
        quaternions, translations = [], []
        for T_gt in gt_poses:
            R_gt = T_gt[:3, :3]
            t_gt = T_gt[:3, 3]
            q_xyzw = R.from_matrix(R_gt).as_quat()
            quaternions.append(np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32))
            t_norm = (R_gt @ center + t_gt - center) / float(scale)
            translations.append(t_norm.astype(np.float32))

        return {
            "name": data["name"],
            "num_parts": num_parts,
            "pointclouds": pointclouds.astype(np.float32),
            "pointclouds_normals": pointclouds_normals.astype(np.float32),
            "points_per_part": self._pad_data(points_per_part).astype(np.int32),
            "points_per_bone": self._pad_bone(points_per_bone).astype(np.int32),
            "bonetype": self._pad_data(bone_type).astype(np.int32),
            "point_bone_ids": point_bone_ids,
            "pointclouds_gt": pointclouds_gt.astype(np.float32),
            "pointclouds_normals_gt": pointclouds_normals_gt.astype(np.float32),
            "quat_gt": self._pad_data(np.stack(quaternions)).astype(np.float32),
            "trans_gt": self._pad_data(np.stack(translations)).astype(np.float32),
        }
