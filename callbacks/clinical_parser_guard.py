"""Fail fast when a clinical run does not preserve original OBJ fragment IDs."""

from __future__ import annotations

import json
from pathlib import Path

import lightning as L
import trimesh
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from datasets.fractures.clinical import (
    _iter_fragments_with_gt_ids,
    _load_obj_fragment_ids,
)


class ClinicalParserGuard(L.Callback):
    def __init__(
        self,
        case_id: str,
        expected_fragment_ids: list[str],
        expected_npoints_per_bone: int,
    ):
        self.case_id = str(case_id)
        self.expected_fragment_ids = {str(value) for value in expected_fragment_ids}
        self.expected_npoints_per_bone = int(expected_npoints_per_bone)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        datamodule = trainer.datamodule
        datasets = [datamodule.train_dataset, datamodule.val_dataset]
        record = None
        for dataset in datasets:
            for sample_name, obj_path, json_path in getattr(dataset, "index_map", []):
                if str(sample_name) == self.case_id:
                    record = (Path(obj_path), Path(json_path))
                    break
            if record is not None:
                break
        if record is None:
            raise RuntimeError(f"ClinicalParserGuard case {self.case_id} is absent from train/val")

        obj_path, json_path = record
        original_ids = set(_load_obj_fragment_ids(str(obj_path)))
        with json_path.open() as f:
            gt_data = json.load(f)
        if isinstance(gt_data, list):
            gt_ids = {str(item["fragment_id"]) for item in gt_data}
        else:
            gt_ids = {str(key) for key in gt_data}

        scene = trimesh.load(str(obj_path), split_object=True, process=False)
        recovered_ids = {
            str(fragment_id)
            for fragment_id, _ in _iter_fragments_with_gt_ids(
                str(obj_path), scene, {fragment_id: None for fragment_id in gt_ids}
            )
        }

        errors = []
        for label, observed in (
            ("original_obj", original_ids),
            ("ground_truth", gt_ids),
            ("recovered_geometry", recovered_ids),
        ):
            if observed != self.expected_fragment_ids:
                errors.append(
                    f"{label}_ids={sorted(observed)} "
                    f"expected={sorted(self.expected_fragment_ids)}"
                )
        if datamodule.npoints_per_bone != self.expected_npoints_per_bone:
            errors.append(
                f"npoints_per_bone={datamodule.npoints_per_bone} "
                f"expected={self.expected_npoints_per_bone}"
            )
        if errors:
            raise RuntimeError("ClinicalParserGuard failed: " + "; ".join(errors))

        rank_zero_info(
            "[ClinicalParserGuard] PASS "
            f"case={self.case_id} fragments={len(recovered_ids)} "
            f"ids={sorted(recovered_ids)} "
            f"npoints_per_bone={datamodule.npoints_per_bone} "
            f"trimesh={trimesh.__version__}"
        )
