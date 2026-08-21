"""Pure Transformer Lightning Module for Point Cloud Assembly."""

from __future__ import annotations

import os
from functools import partial
from typing import Optional
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *

class AssemblyLightningModule(L.LightningModule):
    def __init__(
            self,
            transformer_model: nn.Module,
            optimizer: "partial[torch.optim.Optimizer]",
            lr_scheduler: str = "onecycle",
            output_type: str = "coords",
            training_mode: str = "full",
            lora_rank: int = 16,
            lora_alpha: float = 32.0,
            save_vis_path: str = "./output",
            debug_vis: bool = False,
            checkpoint: Optional[str] = None,
            fragment_balanced_loss: bool = True,
            pose_loss_weight: float = 1.0,
            trans_loss_weight: float = 1.0,
            rot_loss_weight: float = 1.0,
            pose_loss_max_points_per_part: int = 256,
            rot_geom_loss_weight: float = 0.0,
            so3_geodesic_loss_weight: float = 0.0,
            relative_rot_loss_weight: float = 0.0,
            rigid_point_loss_weight: float = 0.0,
            rigid_hard_fragment_weight_gamma: float = 0.0,
            rigid_hard_fragment_weight_min: float = 0.75,
            rigid_hard_fragment_weight_max: float = 1.5,
            fragment_size_weight_gamma: float = 0.0,
            fragment_size_weight_min: float = 0.5,
            fragment_size_weight_max: float = 2.0,
            hard_fragment_weight_gamma: float = 0.0,
            hard_fragment_weight_min: float = 0.5,
            hard_fragment_weight_max: float = 2.0,
    ):
        super().__init__()
        self.transformer_model = transformer_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.output_type = output_type
        self.training_mode = training_mode
        self.debug_vis = debug_vis
        self.fragment_balanced_loss = fragment_balanced_loss
        self.pose_loss_weight = pose_loss_weight
        self.trans_loss_weight = trans_loss_weight
        self.rot_loss_weight = rot_loss_weight
        self.pose_loss_max_points_per_part = pose_loss_max_points_per_part
        self.rot_geom_loss_weight = rot_geom_loss_weight
        self.so3_geodesic_loss_weight = so3_geodesic_loss_weight
        self.relative_rot_loss_weight = relative_rot_loss_weight
        self.rigid_point_loss_weight = rigid_point_loss_weight
        self.rigid_hard_fragment_weight_gamma = rigid_hard_fragment_weight_gamma
        self.rigid_hard_fragment_weight_min = rigid_hard_fragment_weight_min
        self.rigid_hard_fragment_weight_max = rigid_hard_fragment_weight_max
        self.fragment_size_weight_gamma = fragment_size_weight_gamma
        self.fragment_size_weight_min = fragment_size_weight_min
        self.fragment_size_weight_max = fragment_size_weight_max
        self.hard_fragment_weight_gamma = hard_fragment_weight_gamma
        self.hard_fragment_weight_min = hard_fragment_weight_min
        self.hard_fragment_weight_max = hard_fragment_weight_max

        self.base_vis_path = save_vis_path
        self.save_vis_path_val = None

        if checkpoint is not None:
            self._load_pretrained_weights(checkpoint)

        if training_mode == "lora":
            from .lora import apply_lora_to_transformer
            apply_lora_to_transformer(self.transformer_model, rank=lora_rank, alpha=lora_alpha)
        elif training_mode == "head_only":
            for parameter in self.transformer_model.parameters():
                parameter.requires_grad = False
            for parameter in self.transformer_model.head.parameters():
                parameter.requires_grad = True
        elif training_mode != "full":
            raise ValueError(
                f"Unsupported training_mode={training_mode!r}; expected full, lora, or head_only"
            )

    def on_fit_start(self):
        if self.trainer.is_global_zero:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            mode_str = f"mode={self.output_type}  training={self.training_mode}"
            param_str = f"params={trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total"
            print(f"[Model] {mode_str}  {param_str}")
            print(
                "[Loss] "
                f"fragment_balanced={self.fragment_balanced_loss}  "
                f"pose={self.pose_loss_weight:g}  "
                f"trans={self.trans_loss_weight:g}  "
                f"rot={self.rot_loss_weight:g}  "
                f"rot_geom={self.rot_geom_loss_weight:g}  "
                f"so3_geo={self.so3_geodesic_loss_weight:g}  "
                f"rel_rot={self.relative_rot_loss_weight:g}  "
                f"rigid_point={self.rigid_point_loss_weight:g}  "
                f"rigid_hard_gamma={self.rigid_hard_fragment_weight_gamma:g}  "
                f"pose_points={self.pose_loss_max_points_per_part}  "
                f"size_gamma={self.fragment_size_weight_gamma:g}  "
                f"hard_gamma={self.hard_fragment_weight_gamma:g}"
            )

    def _reduce_point_loss_by_fragment(self, point_loss: torch.Tensor, part_batch_ids: torch.Tensor):
        """Reduce per-point losses with optional fragment-size/hardness weights."""
        if not self.fragment_balanced_loss:
            return point_loss.mean()

        part_losses, part_counts = [], []
        num_parts = int(part_batch_ids.max().item()) + 1
        for part_id in range(num_parts):
            mask = part_batch_ids == part_id
            if torch.any(mask):
                part_losses.append(point_loss[mask].mean())
                part_counts.append(mask.sum())

        if not part_losses:
            return point_loss.mean()

        part_losses = torch.stack(part_losses)
        part_counts = torch.stack(part_counts).to(part_losses.device).float()

        weights = torch.ones_like(part_losses)
        if self.fragment_size_weight_gamma > 0:
            size_weights = (
                part_counts.mean().clamp_min(1.0) / part_counts.clamp_min(1.0)
            ).pow(self.fragment_size_weight_gamma)
            size_weights = size_weights.clamp(
                min=self.fragment_size_weight_min,
                max=self.fragment_size_weight_max,
            )
            weights = weights * size_weights

        if self.hard_fragment_weight_gamma > 0:
            detached_losses = part_losses.detach()
            hard_weights = (
                detached_losses / detached_losses.mean().clamp_min(1e-8)
            ).pow(self.hard_fragment_weight_gamma)
            hard_weights = hard_weights.clamp(
                min=self.hard_fragment_weight_min,
                max=self.hard_fragment_weight_max,
            )
            weights = weights * hard_weights

        weights = weights.clamp(min=1e-8)
        return (weights * part_losses).sum() / weights.sum()

    def _rotation_geometry_loss(
            self,
            input_coords: torch.Tensor,
            gt_coords: torch.Tensor,
            part_batch_ids: torch.Tensor,
            pred_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Dense translation-free rotation loss, averaged equally by fragment."""
        rotations = quaternion_to_matrix(
            F.normalize(pred_quat.float(), p=2, dim=-1, eps=1e-6)
        )
        ids = part_batch_ids.detach()
        losses = []
        for part_id in range(rotations.shape[0]):
            mask = ids == part_id
            src = input_coords[mask].detach().float()
            dst = gt_coords[mask].detach().float()
            if src.shape[0] < 3:
                continue
            if self.pose_loss_max_points_per_part is not None and src.shape[0] > self.pose_loss_max_points_per_part:
                sample_idx = torch.linspace(
                    0,
                    src.shape[0] - 1,
                    steps=self.pose_loss_max_points_per_part,
                    device=src.device,
                ).long()
                src = src[sample_idx]
                dst = dst[sample_idx]

            src_centered = src - src.mean(dim=0, keepdim=True)
            dst_centered = dst - dst.mean(dim=0, keepdim=True)
            radius = torch.sqrt(
                src_centered.square().sum(dim=-1).mean().clamp_min(1e-8)
            )
            rotated_centered = src_centered @ rotations[part_id].transpose(0, 1)
            losses.append(F.smooth_l1_loss(
                rotated_centered / radius,
                dst_centered / radius,
                reduction="mean",
            ))

        if not losses:
            return pred_quat.sum() * 0.0
        return torch.stack(losses).mean()

    @staticmethod
    def _so3_geodesic_loss(
            pred_quat: torch.Tensor,
            gt_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Mean SO(3) geodesic angle, normalised to the range [0, 1].

        ``atan2(sin(theta), cos(theta))`` is used instead of acos so the loss
        remains well behaved close to both zero and pi.  Computing the loss
        from rotation matrices also makes it invariant to quaternion sign.
        """
        pred_rot = quaternion_to_matrix(
            F.normalize(pred_quat.float(), p=2, dim=-1, eps=1e-6)
        )
        gt_rot = quaternion_to_matrix(
            F.normalize(gt_quat.float(), p=2, dim=-1, eps=1e-6)
        )
        delta = pred_rot.transpose(1, 2) @ gt_rot
        cos_theta = (
            delta[:, 0, 0] + delta[:, 1, 1] + delta[:, 2, 2] - 1.0
        ) * 0.5
        cos_theta = cos_theta.clamp(min=-1.0, max=1.0)
        skew_vee = torch.stack(
            [
                delta[:, 2, 1] - delta[:, 1, 2],
                delta[:, 0, 2] - delta[:, 2, 0],
                delta[:, 1, 0] - delta[:, 0, 1],
            ],
            dim=-1,
        )
        sin_theta = 0.5 * torch.linalg.vector_norm(skew_vee, dim=-1)
        theta = torch.atan2(sin_theta, cos_theta)
        return (theta / torch.pi).mean()

    def _rigid_point_loss(
            self,
            input_coords: torch.Tensor,
            gt_coords: torch.Tensor,
            part_batch_ids: torch.Tensor,
            pred_quat: torch.Tensor,
            pred_trans: torch.Tensor,
    ) -> torch.Tensor:
        """Metric-aligned rigid point distance after differentiable Kabsch.

        In coords mode, deployment discards non-rigid point residuals and uses
        the rigid pose extracted from the predicted coordinates.  This loss
        supervises that exact rigid output.  It averages points within each
        fragment and then fragments equally, matching the PA-sensitive
        fragment-balanced training recipe.
        """
        pred_rigid_coords = apply_pose_to_points(
            input_coords.float(),
            pred_trans.float(),
            pred_quat.float(),
            part_batch_ids,
        )
        point_dist = torch.sqrt(
            (pred_rigid_coords - gt_coords.float()).square().sum(dim=-1) + 1e-8
        )

        part_losses = []
        num_parts = int(part_batch_ids.max().item()) + 1
        for part_id in range(num_parts):
            mask = part_batch_ids == part_id
            if torch.any(mask):
                part_losses.append(point_dist[mask].mean())

        if not part_losses:
            return point_dist.mean()

        part_losses = torch.stack(part_losses)
        weights = torch.ones_like(part_losses)
        if self.rigid_hard_fragment_weight_gamma > 0:
            relative_loss = (
                part_losses.detach() / part_losses.detach().mean().clamp_min(1e-8)
            )
            weights = relative_loss.pow(self.rigid_hard_fragment_weight_gamma).clamp(
                min=self.rigid_hard_fragment_weight_min,
                max=self.rigid_hard_fragment_weight_max,
            )

        return (weights * part_losses).sum() / weights.sum().clamp_min(1e-8)

    @staticmethod
    def _relative_rotation_loss(
            pred_quat: torch.Tensor,
            gt_quat: torch.Tensor,
            points_per_part: torch.Tensor,
            bonetype: torch.Tensor,
    ) -> torch.Tensor:
        """Rotation loss relative to each case's first SA, matching inference."""
        pred_rot = quaternion_to_matrix(
            F.normalize(pred_quat.float(), p=2, dim=-1, eps=1e-6)
        )
        gt_rot = quaternion_to_matrix(
            F.normalize(gt_quat.float(), p=2, dim=-1, eps=1e-6)
        )

        losses = []
        offset = 0
        for case_idx in range(points_per_part.shape[0]):
            num_parts = int((points_per_part[case_idx] > 0).sum().item())
            if num_parts == 0:
                continue
            case_bones = bonetype[case_idx, :num_parts]
            sa_indices = torch.nonzero(case_bones == 0, as_tuple=False).flatten()
            if sa_indices.numel() == 0:
                offset += num_parts
                continue

            anchor = int(sa_indices[0].item())
            pred_case = pred_rot[offset:offset + num_parts]
            gt_case = gt_rot[offset:offset + num_parts]
            pred_rel = pred_case[anchor].transpose(0, 1).unsqueeze(0) @ pred_case
            gt_rel = gt_case[anchor].transpose(0, 1).unsqueeze(0) @ gt_case

            keep = torch.ones(num_parts, dtype=torch.bool, device=pred_quat.device)
            keep[anchor] = False
            if torch.any(keep):
                delta = pred_rel[keep].transpose(1, 2) @ gt_rel[keep]
                trace = delta[:, 0, 0] + delta[:, 1, 1] + delta[:, 2, 2]
                losses.append(((3.0 - trace) * 0.5).clamp(min=0.0, max=2.0))
            offset += num_parts

        if not losses:
            return pred_quat.sum() * 0.0
        return torch.cat(losses).mean()

    def _load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained weights into the transformer before LoRA wrapping."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        prefix = "transformer_model."
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                stripped[k[len(prefix):]] = v
            else:
                stripped[k] = v

        current_state = self.transformer_model.state_dict()
        loadable = {}
        skipped_shape = []
        for key, value in stripped.items():
            if key in current_state and current_state[key].shape != value.shape:
                skipped_shape.append((key, tuple(value.shape), tuple(current_state[key].shape)))
                continue
            loadable[key] = value

        missing, unexpected = self.transformer_model.load_state_dict(loadable, strict=False)

        if os.environ.get("LOCAL_RANK", "0") == "0":
            loaded = len(loadable) - len(unexpected)
            print(f"[Pretrained] Loaded {loaded} keys from {checkpoint_path}")
            if skipped_shape:
                print("[Pretrained]   Skipped shape-mismatched keys:")
                for key, src_shape, dst_shape in skipped_shape:
                    print(f"[Pretrained]     {key}: checkpoint{src_shape} -> model{dst_shape}")
            if missing:
                print(f"[Pretrained]   Missing keys: {missing}")
            if unexpected:
                print(f"[Pretrained]   Unexpected keys: {unexpected}")

    def setup(self, stage: str):
        """Set up output directories for validation/testing stages."""
        if stage in ['fit', 'validate']:
            self.save_vis_path_val = os.path.join(self.base_vis_path, "valsets_results")
            os.makedirs(self.save_vis_path_val, exist_ok=True)

    def on_train_epoch_start(self):
        """Dynamic reshuffling at the start of each epoch to increase data variety."""
        super().on_train_epoch_start()

        train_loader = self.trainer.train_dataloader
        if train_loader is not None:
            dataset = train_loader.dataset
            if hasattr(dataset, 'datasets'):
                for ds in dataset.datasets:
                    if getattr(ds, 'split', None) == 'train' and hasattr(ds, 'reshuffle'):
                        ds.reshuffle()
            else:
                if getattr(dataset, 'split', None) == 'train' and hasattr(dataset, 'reshuffle'):
                    dataset.reshuffle()

    def forward(self, data_dict: dict):
        """Single-step forward pass through the transformer."""
        points_per_part = data_dict["points_per_part"]
        input_coords = data_dict["pointclouds"]
        input_normals = data_dict["pointclouds_normals"]

        output_dict = self.transformer_model(
            input_coords=input_coords,
            input_normals=input_normals,
            points_per_part=points_per_part,
        )

        if "part_batch_ids" not in output_dict:
            part_valids = points_per_part != 0
            seq_len = points_per_part[part_valids]
            output_dict["part_batch_ids"] = torch.repeat_interleave(
                torch.arange(len(seq_len), device=self.device), seq_len
            )

        if self.output_type == "pose":
            if self.debug_vis:
                pred_assemble = apply_pose_to_points(
                    input_coords,
                    output_dict["trans"],
                    output_dict["quat"],
                    output_dict["part_batch_ids"]
                )
                debug_visualize_pred_assemble(input_coords, output_dict["part_batch_ids"])
                debug_visualize_coords_vs_pred(input_coords, pred_assemble, output_dict["part_batch_ids"])
        elif self.output_type == "coords":
            if not self.training or self.debug_vis:
                quat, trans = extract_poses_from_coords(
                    input_coords,
                    output_dict["pred_coords"],
                    output_dict["part_batch_ids"]
                )
                output_dict["quat"] = quat
                output_dict["trans"] = trans
            if self.debug_vis:
                debug_visualize_coords_vs_pred(
                    input_coords,
                    output_dict["pred_coords"],
                    output_dict["part_batch_ids"]
                )
        return output_dict

    def loss(self, output_dict: dict, data_dict: dict):
        """Compute the regression loss based on the output type."""
        gt_coords = data_dict["pointclouds_gt"]

        # point-level supervision 预测每个点的复位坐标，按点计算loss
        if self.output_type == "coords":
            pred_coords = output_dict["pred_coords"]
            input_coords = data_dict["pointclouds"]
            part_batch_ids = output_dict["part_batch_ids"]

            point_loss = F.mse_loss(pred_coords, gt_coords, reduction="none").mean(dim=-1)
            loss_coords = self._reduce_point_loss_by_fragment(point_loss, part_batch_ids)

            loss = loss_coords
            loss_dict = {"loss_coords": loss_coords}

            needs_rigid_pose = (
                self.pose_loss_weight > 0
                or self.rot_geom_loss_weight > 0
                or self.so3_geodesic_loss_weight > 0
                or self.relative_rot_loss_weight > 0
                or self.rigid_point_loss_weight > 0
            )
            if needs_rigid_pose and "quat_gt" in data_dict and "trans_gt" in data_dict:
                pred_quat, pred_trans = extract_poses_from_coords_grad(
                    input_coords,
                    pred_coords,
                    part_batch_ids,
                    max_points_per_part=self.pose_loss_max_points_per_part,
                )
                valid_mask = data_dict["points_per_part"] > 0
                gt_quat = data_dict["quat_gt"][valid_mask].float()
                gt_trans = data_dict["trans_gt"][valid_mask].float()

                pred_quat = F.normalize(pred_quat.float(), p=2, dim=-1, eps=1e-6)
                gt_quat = F.normalize(gt_quat.float(), p=2, dim=-1, eps=1e-6)

                loss_trans = F.smooth_l1_loss(pred_trans.float(), gt_trans, reduction="mean")
                quat_dot = torch.sum(pred_quat * gt_quat, dim=-1).abs().clamp(max=1.0)
                loss_rot = (1.0 - quat_dot).mean()
                loss_pose = self.trans_loss_weight * loss_trans + self.rot_loss_weight * loss_rot
                loss = loss + self.pose_loss_weight * loss_pose

                loss_dict.update({
                    "loss_pose": loss_pose,
                    "loss_trans": loss_trans,
                    "loss_rot": loss_rot,
                })

                if self.rot_geom_loss_weight > 0:
                    loss_rot_geom = self._rotation_geometry_loss(
                        input_coords,
                        gt_coords,
                        part_batch_ids,
                        pred_quat,
                    )
                    loss = loss + self.rot_geom_loss_weight * loss_rot_geom
                    loss_dict["loss_rot_geom"] = loss_rot_geom

                if self.so3_geodesic_loss_weight > 0:
                    loss_so3_geodesic = self._so3_geodesic_loss(
                        pred_quat,
                        gt_quat,
                    )
                    loss = loss + self.so3_geodesic_loss_weight * loss_so3_geodesic
                    loss_dict["loss_so3_geodesic"] = loss_so3_geodesic

                if self.relative_rot_loss_weight > 0:
                    loss_relative_rot = self._relative_rotation_loss(
                        pred_quat,
                        gt_quat,
                        data_dict["points_per_part"],
                        data_dict["bonetype"],
                    )
                    loss = loss + self.relative_rot_loss_weight * loss_relative_rot
                    loss_dict["loss_relative_rot"] = loss_relative_rot

                if self.rigid_point_loss_weight > 0:
                    loss_rigid_point = self._rigid_point_loss(
                        input_coords,
                        gt_coords,
                        part_batch_ids,
                        pred_quat,
                        pred_trans,
                    )
                    loss = loss + self.rigid_point_loss_weight * loss_rigid_point
                    loss_dict["loss_rigid_point"] = loss_rigid_point

            loss_dict["loss"] = loss
            return loss_dict

        # fragment-level supervision 每个fragment预测一个刚体变换，对整个fragment计算loss
        elif self.output_type == "pose":
            pred_quat, pred_trans = output_dict["quat"], output_dict["trans"]

            pred_coords = apply_pose_to_points(
                data_dict["pointclouds"],
                pred_trans,
                pred_quat,
                output_dict["part_batch_ids"]
            )
            point_loss = F.mse_loss(pred_coords, gt_coords, reduction="none").mean(dim=-1)
            part_batch_ids = output_dict["part_batch_ids"]
            loss_coords = self._reduce_point_loss_by_fragment(point_loss, part_batch_ids)

            loss = loss_coords
            loss_dict = {"loss_coords": loss_coords}
            if (
                self.pose_loss_weight > 0 or self.so3_geodesic_loss_weight > 0
            ) and "quat_gt" in data_dict and "trans_gt" in data_dict:
                valid_mask = data_dict["points_per_part"] > 0
                gt_quat = F.normalize(data_dict["quat_gt"][valid_mask].float(), p=2, dim=-1, eps=1e-6)
                gt_trans = data_dict["trans_gt"][valid_mask].float()
                pred_quat_n = F.normalize(pred_quat.float(), p=2, dim=-1, eps=1e-6)

                loss_trans = F.smooth_l1_loss(pred_trans.float(), gt_trans, reduction="mean")
                quat_dot = torch.sum(pred_quat_n * gt_quat, dim=-1).abs().clamp(max=1.0)
                loss_rot = (1.0 - quat_dot).mean()
                loss_pose = self.trans_loss_weight * loss_trans + self.rot_loss_weight * loss_rot
                loss = loss + self.pose_loss_weight * loss_pose
                loss_dict.update({
                    "loss_pose": loss_pose,
                    "loss_trans": loss_trans,
                    "loss_rot": loss_rot,
                })

                if self.so3_geodesic_loss_weight > 0:
                    loss_so3_geodesic = self._so3_geodesic_loss(
                        pred_quat_n,
                        gt_quat,
                    )
                    loss = loss + self.so3_geodesic_loss_weight * loss_so3_geodesic
                    loss_dict["loss_so3_geodesic"] = loss_so3_geodesic

            loss_dict["loss"] = loss
            return loss_dict

    def training_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        output_dict = self.forward(data_dict)
        loss_dict = self.loss(output_dict, data_dict)

        loss_val = loss_dict["loss"]
        batch_size = data_dict["points_per_part"].shape[0]

        # Sync NaN state across ALL ranks — if ANY rank has non-finite loss, ALL skip
        is_finite_int = torch.isfinite(loss_val).int()
        if self.trainer.world_size > 1:
            torch.distributed.all_reduce(is_finite_int, op=torch.distributed.ReduceOp.MIN)
        is_finite = is_finite_int.bool()

        if not is_finite:
            print(f"[Rank {self.global_rank}] Non-finite loss at step {self.global_step}, all ranks skipping.")
            self.log("train/loss", float('nan'), on_step=True, on_epoch=True, prog_bar=True,
                     sync_dist=True, batch_size=batch_size)
            return torch.nan_to_num(loss_val, nan=0.0, posinf=0.0, neginf=0.0)

        self.log("train/loss", loss_val, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=batch_size)

        if "loss_coords" in loss_dict:
            self.log("train/loss_coords", loss_dict["loss_coords"], on_step=True, on_epoch=False, sync_dist=True,
                     batch_size=batch_size)
        if "loss_pose" in loss_dict:
            self.log("train/loss_pose", loss_dict["loss_pose"], on_step=True, on_epoch=False, sync_dist=True,
                     batch_size=batch_size)
        if "loss_rot" in loss_dict:
            self.log("train/loss_rot", loss_dict["loss_rot"], on_step=True, on_epoch=False, sync_dist=True,
                     batch_size=batch_size)
        if "loss_trans" in loss_dict:
            self.log("train/loss_trans", loss_dict["loss_trans"], on_step=True, on_epoch=False, sync_dist=True,
                     batch_size=batch_size)
        if "loss_rot_geom" in loss_dict:
            self.log("train/loss_rot_geom", loss_dict["loss_rot_geom"], on_step=True, on_epoch=False,
                     sync_dist=True, batch_size=batch_size)
        if "loss_relative_rot" in loss_dict:
            self.log("train/loss_relative_rot", loss_dict["loss_relative_rot"], on_step=True, on_epoch=False,
                     sync_dist=True, batch_size=batch_size)
        if "loss_rigid_point" in loss_dict:
            self.log("train/loss_rigid_point", loss_dict["loss_rigid_point"], on_step=True, on_epoch=False,
                     sync_dist=True, batch_size=batch_size)

        return loss_val

    def validation_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        output_dict = self.forward(data_dict)
        loss_dict = self.loss(output_dict, data_dict)
        batch_size = data_dict["points_per_part"].shape[0]

        self.log("val/loss", loss_dict["loss"], on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        if "loss_coords" in loss_dict:
            self.log("val/loss_coords", loss_dict["loss_coords"], on_epoch=True, sync_dist=True, batch_size=batch_size)
        if "loss_pose" in loss_dict:
            self.log("val/loss_pose", loss_dict["loss_pose"], on_epoch=True, sync_dist=True, batch_size=batch_size)
        if "loss_rot" in loss_dict:
            self.log("val/loss_rot", loss_dict["loss_rot"], on_epoch=True, sync_dist=True, batch_size=batch_size)
        if "loss_trans" in loss_dict:
            self.log("val/loss_trans", loss_dict["loss_trans"], on_epoch=True, sync_dist=True, batch_size=batch_size)
        if "loss_rot_geom" in loss_dict:
            self.log("val/loss_rot_geom", loss_dict["loss_rot_geom"], on_epoch=True, sync_dist=True,
                     batch_size=batch_size)
        if "loss_relative_rot" in loss_dict:
            self.log("val/loss_relative_rot", loss_dict["loss_relative_rot"], on_epoch=True, sync_dist=True,
                     batch_size=batch_size)
        if "loss_rigid_point" in loss_dict:
            self.log("val/loss_rigid_point", loss_dict["loss_rigid_point"], on_epoch=True, sync_dist=True,
                     batch_size=batch_size)

        if "quat" in output_dict and "trans" in output_dict:
            valid_mask = data_dict["points_per_part"] > 0
            gt_quat = data_dict["quat_gt"][valid_mask]
            gt_trans = data_dict["trans_gt"][valid_mask]
            pred_quat = output_dict["quat"]
            pred_trans = output_dict["trans"]

            with torch.autocast(device_type=self.device.type, enabled=False):
                rot_err = rot_geodesic_deg(pred_quat.float(), gt_quat.float())
                trans_err = torch.norm(pred_trans.float() - gt_trans.float(), dim=-1)

                if "pred_coords" in output_dict:
                    pred_assembled = output_dict["pred_coords"].float()
                else:
                    pred_assembled = apply_pose_to_points(
                        data_dict["pointclouds"].float(),
                        pred_trans.float(), pred_quat.float(),
                        output_dict["part_batch_ids"],
                    )
                tre = torch.sqrt(F.mse_loss(
                    pred_assembled, data_dict["pointclouds_gt"].float(), reduction="none"
                ).mean(dim=-1)).mean()

            n_parts = rot_err.shape[0]
            self.log("val/rot_err_deg", rot_err.mean(), on_epoch=True, sync_dist=True, batch_size=n_parts)
            self.log("val/trans_err", trans_err.mean(), on_epoch=True, sync_dist=True, batch_size=n_parts)
            self.log("val/tre", tre, on_epoch=True, sync_dist=True, batch_size=batch_size)

        if self.debug_vis and batch_idx < 10:
            if self.output_type == "coords":
                pred_assemble = enforce_rigid_transform_svd(
                    src_points=data_dict["pointclouds"],
                    pred_points=output_dict["pred_coords"],
                    part_batch_ids=output_dict["part_batch_ids"]
                )
            else:
                pred_assemble = apply_pose_to_points(
                    data_dict["pointclouds"],
                    output_dict["trans"],
                    output_dict["quat"],
                    output_dict["part_batch_ids"]
                )

            output_results(
                pred_assemble=pred_assemble,
                data_dict=data_dict,
                output_dict=output_dict,
                save_path=self.save_vis_path_val,
                save_all=False,
                save_gt=True
            )

        return loss_dict["loss"]

    def configure_optimizers(self):
        """Configure AdamW and the requested learning-rate schedule."""
        # Exclude biases and 1D params (LayerNorm, etc.) from weight decay
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        base_lr = self.optimizer.keywords.get("lr", 1e-4)
        wd = self.optimizer.keywords.get("weight_decay", 0.01)
        param_groups = [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = self.optimizer(param_groups)

        if self.lr_scheduler == "constant":
            return optimizer
        if self.lr_scheduler != "onecycle":
            raise ValueError(
                f"Unsupported lr_scheduler={self.lr_scheduler!r}; expected onecycle or constant"
            )

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=base_lr,
            total_steps=total_steps,
            pct_start=0.05,
            anneal_strategy='cos',
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
        }
