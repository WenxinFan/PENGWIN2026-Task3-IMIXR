"""Lightning DataModule for fragment assembly training.

Supports five data modes via `data_type`:
    'simu'         — simulation NPZ data only (data_root_sim)
    'real'         — clinical data only (data_root_real), auto-detects OBJ vs NPZ
    'mix'          — both sources fully concatenated
    'mix_balanced' — each epoch: randomly sample len(real) from sim, concat with
                      real so every epoch has equal sim/real counts
    'mix_train_real_val' — balanced sim+real for training, clinical-only val/test

Typical workflow:
    1. Pre-train on simulation:  data_type: simu
    2. Fine-tune on clinical:    data_type: real  (with training_mode: lora)
    3. Joint training:           data_type: mix_balanced
"""

import random
from typing import Literal, List, Optional, Union

import lightning as L
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate

from pathlib import Path

from .sample import FragmentSamples
from .clinical import ClinicalFragmentSamples
from .clinical_npz import ClinicalFragmentSamplesCached


class BalancedMixDataset(Dataset):
    """Each epoch randomly samples sim_multiplier * len(real) from sim.

    Pass seed=None for training (random per epoch), seed=42 for val/test (deterministic).
    """

    def __init__(self, sim_dataset: Dataset, real_dataset: Dataset, seed: Optional[int] = None,
                 sim_multiplier: int = 1):
        super().__init__()
        self.sim_dataset = sim_dataset
        self.real_dataset = real_dataset
        self.split = "train" if seed is None else "val"
        self._rng = random.Random(seed)
        self.sim_multiplier = sim_multiplier
        self._resample()

    def _resample(self):
        n_real = len(self.real_dataset)
        n_sim_target = min(self.sim_multiplier * n_real, len(self.sim_dataset))
        self._sim_indices = self._rng.sample(range(len(self.sim_dataset)), n_sim_target)

    def reshuffle(self):
        if self.split != "train":
            return
        if hasattr(self.sim_dataset, "reshuffle"):
            self.sim_dataset.reshuffle()
        if hasattr(self.real_dataset, "reshuffle"):
            self.real_dataset.reshuffle()
        self._resample()

    def __len__(self):
        return len(self.real_dataset) + len(self._sim_indices)

    def __getitem__(self, idx):
        n_real = len(self.real_dataset)
        if idx < n_real:
            return self.real_dataset[idx]
        else:
            return self.sim_dataset[self._sim_indices[idx - n_real]]


class RotationStratifiedDistributedSampler(DistributedSampler):
    """Fixed-length DDP sampler with exact easy/medium/hard proportions."""

    def __init__(
        self,
        dataset: Dataset,
        bin_ids: np.ndarray,
        target_probs: List[float],
        seed: int = 42,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "RotationStratifiedDistributedSampler requires an initialized "
                "torch.distributed process group; single-process fallback is disabled."
            )
        world_size = torch.distributed.get_world_size()
        distributed_rank = torch.distributed.get_rank()
        if num_replicas is None:
            num_replicas = world_size
        elif num_replicas != world_size:
            raise ValueError(
                f"num_replicas={num_replicas} disagrees with distributed world_size={world_size}"
            )
        if rank is None:
            rank = distributed_rank
        elif rank != distributed_rank:
            raise ValueError(
                f"rank={rank} disagrees with distributed rank={distributed_rank}"
            )
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=False,
        )
        self.bin_ids = torch.as_tensor(bin_ids, dtype=torch.long)
        probs = torch.as_tensor(target_probs, dtype=torch.float64)
        if self.bin_ids.numel() != len(dataset):
            raise ValueError("rotation bin count must equal dataset length")
        if probs.numel() < 2 or torch.any(probs <= 0):
            raise ValueError("rotation target probabilities must all be positive")
        self.target_probs = probs / probs.sum()
        self.bin_members = [
            torch.nonzero(self.bin_ids == bin_id, as_tuple=False).flatten()
            for bin_id in range(probs.numel())
        ]
        if any(members.numel() == 0 for members in self.bin_members):
            raise ValueError("rotation stratification produced an empty bin")

    def _target_counts(self) -> torch.Tensor:
        raw = self.target_probs * self.total_size
        counts = torch.floor(raw).long()
        remainder = self.total_size - int(counts.sum().item())
        if remainder > 0:
            order = torch.argsort(raw - counts, descending=True)
            counts[order[:remainder]] += 1
        return counts

    @staticmethod
    def _sample_bin(members: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
        pieces = []
        remaining = count
        while remaining > 0:
            take = min(remaining, members.numel())
            order = torch.randperm(members.numel(), generator=generator)[:take]
            pieces.append(members[order])
            remaining -= take
        return torch.cat(pieces)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        counts = self._target_counts()
        global_indices = torch.cat([
            self._sample_bin(members, int(count.item()), generator)
            for members, count in zip(self.bin_members, counts)
        ])
        global_indices = global_indices[torch.randperm(global_indices.numel(), generator=generator)]
        rank_indices = global_indices[self.rank:self.total_size:self.num_replicas]
        if rank_indices.numel() != self.num_samples:
            raise RuntimeError("rotation sampler generated an invalid per-rank length")
        return iter(rank_indices.tolist())


class FragmentDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root_sim: str = "",
        data_root_real: str = "",
        data_type: Literal["simu", "real", "mix", "mix_balanced", "mix_train_real_val"] = "simu",
        min_parts: int = 2,
        max_parts: int = 50,
        batch_size: int = 5,
        num_workers: int = 5,
        random_anchor: bool = True,
        merge_prob: float = 0.15,
        frac_prob: float = 0.70,
        drop_prob: float = 0.1,
        region_dropout_prob: float = 0.0,
        point_jitter_std: float = 0.0,
        npoints_per_bone: int = 5000,
        sim_combos_per_case: int = 0,
        real_pose_ratio: float = 0.2,
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
        hard_pose_aug_bones: Optional[List[str]] = None,
        hard_pose_aug_preserve_anchor: bool = True,
        rotation_stratified_sampling: bool = False,
        rotation_strata_quantiles: Optional[List[float]] = None,
        rotation_strata_target_probs: Optional[List[float]] = None,
        rotation_sampling_seed: int = 42,
    ):
        super().__init__()
        self.data_root_sim = data_root_sim
        self.data_root_real = data_root_real
        self.data_type = data_type
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_anchor = random_anchor
        self.merge_prob = merge_prob
        self.frac_prob = frac_prob
        self.drop_prob = drop_prob
        self.region_dropout_prob = region_dropout_prob
        self.point_jitter_std = point_jitter_std
        self.npoints_per_bone = npoints_per_bone
        self.sim_combos_per_case = sim_combos_per_case
        self.real_pose_ratio = real_pose_ratio
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
        self.hard_pose_aug_bones = hard_pose_aug_bones or ["LI", "RI"]
        self.hard_pose_aug_preserve_anchor = hard_pose_aug_preserve_anchor
        self.rotation_stratified_sampling = rotation_stratified_sampling
        self.rotation_strata_quantiles = (
            [0.5, 0.8]
            if rotation_strata_quantiles is None
            else list(rotation_strata_quantiles)
        )
        self.rotation_strata_target_probs = (
            [0.4, 0.3, 0.3]
            if rotation_strata_target_probs is None
            else list(rotation_strata_target_probs)
        )
        self.rotation_sampling_seed = rotation_sampling_seed
        self._rotation_bin_ids: Optional[np.ndarray] = None
        self._rotation_thresholds: Optional[np.ndarray] = None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def _make_sim_dataset(self, split: str) -> FragmentSamples:
        return FragmentSamples(
            split=split,
            data_root=self.data_root_sim,
            min_parts=self.min_parts,
            max_parts=self.max_parts,
            random_anchor=self.random_anchor,
            merge_prob=self.merge_prob,
            frac_prob=self.frac_prob,
            drop_prob=self.drop_prob,
            region_dropout_prob=self.region_dropout_prob,
            point_jitter_std=self.point_jitter_std,
            npoints_per_bone=self.npoints_per_bone,
            sim_combos_per_case=self.sim_combos_per_case,
        )

    def _make_real_dataset(self, split: str) -> Dataset:
        # Auto-detect: .npz files → cached NPZ loader, otherwise → OBJ loader
        root = Path(self.data_root_real)
        has_npz = root.is_dir() and any(root.rglob("*.npz"))
        if has_npz:
            return ClinicalFragmentSamplesCached(
                split=split,
                data_root_real=self.data_root_real,
                npoints_per_bone=self.npoints_per_bone,
                min_parts=self.min_parts,
                max_parts=self.max_parts,
                random_anchor=self.random_anchor,
                merge_prob=self.merge_prob,
                drop_prob=self.drop_prob,
                region_dropout_prob=self.region_dropout_prob,
                point_jitter_std=self.point_jitter_std,
                real_pose_ratio=self.real_pose_ratio,
                use_all_train=self.use_all_train,
                split_json=self.split_json,
                split_fold=self.split_fold,
            )
        else:
            return ClinicalFragmentSamples(
                split=split,
                data_root_real=self.data_root_real,
                npoints_per_bone=self.npoints_per_bone,
                min_parts=self.min_parts,
                max_parts=self.max_parts,
                random_anchor=self.random_anchor,
                merge_prob=self.merge_prob,
                drop_prob=self.drop_prob,
                region_dropout_prob=self.region_dropout_prob,
                point_jitter_std=self.point_jitter_std,
                use_real_poses=self.use_real_poses,
                use_all_train=self.use_all_train,
                split_json=self.split_json,
                split_fold=self.split_fold,
                mesh_allocation=self.mesh_allocation,
                surface_sample=self.surface_sample,
                small_fragment_boost_factor=self.small_fragment_boost_factor,
                small_fragment_area_percentile=self.small_fragment_area_percentile,
                hard_pose_aug_prob=self.hard_pose_aug_prob,
                hard_pose_aug_max_angle_deg=self.hard_pose_aug_max_angle_deg,
                hard_pose_aug_max_translation_mm=self.hard_pose_aug_max_translation_mm,
                hard_pose_aug_bones=self.hard_pose_aug_bones,
                hard_pose_aug_preserve_anchor=self.hard_pose_aug_preserve_anchor,
            )

    def _make_dataset(self, split: str) -> Dataset:
        if self.data_type == "simu":
            return self._make_sim_dataset(split)
        elif self.data_type == "real":
            return self._make_real_dataset(split)
        elif self.data_type == "mix":
            datasets = []
            if self.data_root_sim:
                datasets.append(self._make_sim_dataset(split))
            if self.data_root_real:
                datasets.append(self._make_real_dataset(split))
            if not datasets:
                raise ValueError("data_type='mix' requires at least one of data_root_sim or data_root_real")
            return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        elif self.data_type == "mix_balanced":
            sim = self._make_sim_dataset(split)
            real = self._make_real_dataset(split)
            if split == "train":
                return BalancedMixDataset(sim, real)
            else:
                return BalancedMixDataset(sim, real, seed=42)
        elif self.data_type == "mix_train_real_val":
            if split == "train":
                sim = self._make_sim_dataset(split)
                real = self._make_real_dataset(split)
                return BalancedMixDataset(sim, real)
            return self._make_real_dataset(split)
        else:
            raise ValueError(f"Unknown data_type: {self.data_type!r}. "
                             f"Choose 'simu', 'real', 'mix', 'mix_balanced', "
                             f"or 'mix_train_real_val'.")

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self._make_dataset("train")
            self.val_dataset = self._make_dataset("val")
            self._validate_non_empty("train", self.train_dataset)
            self._validate_non_empty("val", self.val_dataset)
            if self.rotation_stratified_sampling:
                self._prepare_rotation_stratification()
            self._log_dataset_info()
        elif stage in ("test", "predict"):
            self.test_dataset = self._make_dataset("test")
            self._validate_non_empty("test", self.test_dataset)

    def _validate_non_empty(self, split: str, dataset: Dataset):
        if len(dataset) > 0:
            return

        roots = []
        if self.data_type in {"simu", "mix", "mix_balanced", "mix_train_real_val"}:
            roots.append(f"data_root_sim={self.data_root_sim!r}")
        if self.data_type in {"real", "mix", "mix_balanced", "mix_train_real_val"}:
            roots.append(f"data_root_real={self.data_root_real!r}")
        split_info = f", split_json={self.split_json!r}, split_fold={self.split_fold!r}" if self.split_json else ""
        raise ValueError(
            f"{split!r} dataset is empty for data_type={self.data_type!r} "
            f"({', '.join(roots)}{split_info}). Check the configured data roots and split settings."
        )

    def _log_dataset_info(self):
        from lightning.pytorch.utilities.rank_zero import rank_zero_only

        @rank_zero_only
        def _print():
            n_train = len(self.train_dataset)
            n_val = len(self.val_dataset)
            aug = []
            if self.region_dropout_prob > 0:
                aug.append(f"region_dropout={self.region_dropout_prob}")
            if self.point_jitter_std > 0:
                aug.append(f"jitter={self.point_jitter_std}")
            if self.small_fragment_boost_factor > 1.0:
                aug.append(
                    f"small_frag_boost={self.small_fragment_boost_factor}"
                    f"@p{self.small_fragment_area_percentile:g}"
                )
            if self.hard_pose_aug_prob > 0:
                aug.append(
                    f"hard_pose={self.hard_pose_aug_prob}"
                    f"/{self.hard_pose_aug_max_angle_deg:g}deg"
                    f"/{self.hard_pose_aug_max_translation_mm:g}mm"
                    f"/bones={','.join(self.hard_pose_aug_bones)}"
                )
            if self.use_all_train:
                aug.append("use_all_train=True")
            if self.split_json:
                aug.append(f"split={self.split_fold}")
            if self.sim_combos_per_case > 0 and self.data_type in {"simu", "mix", "mix_balanced", "mix_train_real_val"}:
                aug.append(f"sim_combos_per_case={self.sim_combos_per_case}")
            if self.rotation_stratified_sampling:
                counts = np.bincount(
                    self._rotation_bin_ids,
                    minlength=len(self.rotation_strata_target_probs),
                ).tolist()
                aug.append(
                    "rotation_stratified="
                    f"counts{counts}/thresholds{np.round(self._rotation_thresholds, 2).tolist()}"
                    f"/target{self.rotation_strata_target_probs}"
                )
            aug_str = f"  augmentations: {', '.join(aug)}" if aug else ""
            print(f"[DataModule] data_type={self.data_type}  train={n_train}  val={n_val}{aug_str}")

        _print()

    def _prepare_rotation_stratification(self):
        if self.data_type != "real" or not isinstance(self.train_dataset, ClinicalFragmentSamples):
            raise ValueError("rotation_stratified_sampling currently requires OBJ clinical data_type='real'")
        if len(self.rotation_strata_target_probs) != len(self.rotation_strata_quantiles) + 1:
            raise ValueError("rotation strata need one more target probability than quantile thresholds")
        quantiles = np.asarray(self.rotation_strata_quantiles, dtype=np.float64)
        if np.any(quantiles <= 0) or np.any(quantiles >= 1) or np.any(np.diff(quantiles) <= 0):
            raise ValueError("rotation_strata_quantiles must be strictly increasing inside (0, 1)")
        hardness = self.train_dataset.get_relative_rotation_hardness_deg()
        self._rotation_thresholds = np.quantile(hardness, quantiles)
        self._rotation_bin_ids = np.digitize(hardness, self._rotation_thresholds, right=True)
        counts = np.bincount(self._rotation_bin_ids, minlength=len(self.rotation_strata_target_probs))
        if np.any(counts == 0):
            raise ValueError(f"rotation stratification has empty bins: counts={counts.tolist()}")
        self.rotation_hardness_deg = hardness

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if self.rotation_stratified_sampling:
            sampler = RotationStratifiedDistributedSampler(
                self.train_dataset,
                self._rotation_bin_ids,
                self.rotation_strata_target_probs,
                seed=self.rotation_sampling_seed,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=sampler is None,
            sampler=sampler,
            persistent_workers=False,
            collate_fn=variable_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=False,
            collate_fn=variable_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            persistent_workers=False,
            collate_fn=variable_collate_fn,
        )


def variable_collate_fn(batch: list) -> dict:
    """Collate variable-length point cloud samples into a single batch dict."""
    if not batch:
        return {}

    batch_size = len(batch)
    has_keys = set()
    for sample in batch:
        has_keys.update(sample.keys())

    has_points = "pointclouds" in has_keys
    has_normals = "pointclouds_normals" in has_keys
    has_gt = "pointclouds_gt" in has_keys
    has_normals_gt = "pointclouds_normals_gt" in has_keys
    has_bone_ids = "point_bone_ids" in has_keys
    has_ppp = "points_per_part" in has_keys

    total_points = sum(len(s["pointclouds"]) for s in batch if "pointclouds" in s)
    total_points_gt = sum(len(s["pointclouds_gt"]) for s in batch if "pointclouds_gt" in s)

    if has_points:    batch_points = torch.empty((total_points, 3), dtype=torch.float32)
    if has_normals:   batch_normals = torch.empty((total_points, 3), dtype=torch.float32)
    if has_gt:        batch_gt = torch.empty((total_points_gt, 3), dtype=torch.float32)
    if has_normals_gt: batch_normals_gt = torch.empty((total_points_gt, 3), dtype=torch.float32)
    if has_bone_ids:
        batch_bone_ids = torch.empty((total_points_gt,), dtype=torch.int64)
        batch_global_bone_ids = torch.empty((total_points_gt,), dtype=torch.int64)
    if has_ppp:
        batch_part_ids = torch.empty((total_points,), dtype=torch.int64)

    points_per_sample = torch.empty(batch_size, dtype=torch.int32)
    other_keys: dict = {}
    start_idx = 0
    start_idx_gt = 0
    current_bone_offset = 0

    for i, sample in enumerate(batch):
        num_points = len(sample["pointclouds"])
        end_idx = start_idx + num_points

        if has_points:
            batch_points[start_idx:end_idx] = torch.from_numpy(sample["pointclouds"].astype(np.float32))
        if has_normals and "pointclouds_normals" in sample:
            batch_normals[start_idx:end_idx] = torch.from_numpy(sample["pointclouds_normals"].astype(np.float32))

        if has_gt and "pointclouds_gt" in sample:
            num_gt = len(sample["pointclouds_gt"])
            end_idx_gt = start_idx_gt + num_gt
            batch_gt[start_idx_gt:end_idx_gt] = torch.from_numpy(sample["pointclouds_gt"].astype(np.float32))
            if has_normals_gt and "pointclouds_normals_gt" in sample:
                batch_normals_gt[start_idx_gt:end_idx_gt] = torch.from_numpy(sample["pointclouds_normals_gt"].astype(np.float32))
            if has_bone_ids and "point_bone_ids" in sample:
                p_ids = sample["point_bone_ids"]
                if isinstance(p_ids, np.ndarray):
                    p_ids = torch.from_numpy(p_ids.astype(np.int64)).view(-1)
                batch_bone_ids[start_idx_gt:end_idx_gt] = p_ids
                batch_global_bone_ids[start_idx_gt:end_idx_gt] = p_ids + current_bone_offset
                current_bone_offset += p_ids.max().item() + 1
            start_idx_gt = end_idx_gt

        if has_ppp and "points_per_part" in sample:
            ppp = sample["points_per_part"]
            if isinstance(ppp, (list, np.ndarray)):
                ppp = torch.tensor(ppp)
            ppp_valid = ppp[ppp > 0]
            p_part_ids = torch.repeat_interleave(
                torch.arange(len(ppp_valid), dtype=torch.int64), ppp_valid.to(torch.int64)
            )
            batch_part_ids[start_idx:end_idx] = p_part_ids

        points_per_sample[i] = num_points

        for k, v in sample.items():
            if k not in {"pointclouds", "pointclouds_normals", "pointclouds_gt",
                         "pointclouds_normals_gt", "point_bone_ids"}:
                other_keys.setdefault(k, []).append(v)

        start_idx = end_idx

    batch_out = {"points_per_sample": points_per_sample}
    if has_points:     batch_out["pointclouds"] = batch_points
    if has_normals:    batch_out["pointclouds_normals"] = batch_normals
    if has_gt:         batch_out["pointclouds_gt"] = batch_gt
    if has_normals_gt: batch_out["pointclouds_normals_gt"] = batch_normals_gt
    if has_bone_ids:
        batch_out["point_bone_ids"] = batch_bone_ids
        batch_out["global_bone_ids"] = batch_global_bone_ids
    if has_ppp:
        batch_out["point_part_ids"] = batch_part_ids
        batch_out["points_per_part"] = default_collate(
            other_keys.pop("points_per_part")) if "points_per_part" in other_keys else None

    for k, v_list in other_keys.items():
        try:
            batch_out[k] = default_collate(v_list)
        except Exception:
            batch_out[k] = v_list

    return batch_out
