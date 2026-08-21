"""Dump one deterministic clinical sample for cross-server comparison."""

import argparse
import hashlib
import inspect
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets.fractures.clinical as clinical


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _preview(array: np.ndarray):
    if array.ndim == 0:
        return array.item()
    if array.ndim == 1:
        return array[:20].tolist()
    return array.reshape(-1, array.shape[-1])[:5].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--case", default="191")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--npoints", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    kwargs = {
        "split": args.split,
        "data_root_real": args.data_root,
        "npoints_per_bone": args.npoints,
        "random_anchor": False,
        "merge_prob": 0.0,
        "region_dropout_prob": 0.0,
        "point_jitter_std": 0.0,
        "use_real_poses": True,
    }
    signature = inspect.signature(clinical.ClinicalFragmentSamples.__init__)
    if "mesh_allocation" in signature.parameters:
        kwargs["mesh_allocation"] = "original"
    if "surface_sample" in signature.parameters:
        kwargs["surface_sample"] = "original"

    captured = {}
    original_recenter = clinical.recenter_pc
    original_rescale = clinical.rescale_pc

    def capture_recenter(points):
        result, center = original_recenter(points)
        captured["center"] = np.asarray(center)
        return result, center

    def capture_rescale(points):
        result, scale = original_rescale(points)
        captured["scale"] = np.asarray(scale)
        return result, scale

    clinical.recenter_pc = capture_recenter
    clinical.rescale_pc = capture_rescale
    try:
        dataset = clinical.ClinicalFragmentSamples(**kwargs)
        index = next(
            i for i, sample in enumerate(dataset.index_map) if str(sample[0]) == args.case
        )
        # Reset immediately before sampling so both servers consume the same
        # legacy NumPy random stream.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        sample = dataset[index]
    finally:
        clinical.recenter_pc = original_recenter
        clinical.rescale_pc = original_rescale

    arrays = {}
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif np.isscalar(value) and not isinstance(value, str):
            arrays[key] = np.asarray(value)

    valid_ppp = np.asarray(sample["points_per_part"])
    valid_ppp = valid_ppp[valid_ppp > 0].astype(np.int64)
    arrays["point_part_ids"] = np.repeat(
        np.arange(len(valid_ppp), dtype=np.int64), valid_ppp
    )
    arrays["normalization_center"] = captured["center"]
    arrays["normalization_scale"] = captured["scale"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)

    summary = {
        "case": args.case,
        "dataset_index": index,
        "seed": args.seed,
        "trimesh_version": trimesh.__version__,
        "trimesh_file": trimesh.__file__,
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _hash(value),
                "first5": _preview(value),
            }
            for key, value in arrays.items()
        },
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"NPZ: {output}")
    print(f"JSON: {summary_path}")


if __name__ == "__main__":
    main()
