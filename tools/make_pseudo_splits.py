"""Create deterministic pseudo-test splits for clinical Task 3 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def discover_cases(data_dir: Path) -> list[str]:
    cases = []
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        has_obj = any(d.glob("*.obj"))
        has_gt = (d / "plan_pl_gt.json").exists()
        if has_obj and has_gt:
            cases.append(d.name)
    return cases


def main():
    parser = argparse.ArgumentParser(description="Make clinical pseudo-test split manifest")
    parser.add_argument(
        "--data_dir",
        default="/mnt/disk3/fan/ReductionData/PENGWIN26_task3_clinical_fractures_train/mesh",
        help="Clinical mesh root containing case folders",
    )
    parser.add_argument("--out", default="configs/splits/clinical_pseudo_3fold_seed42.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test_size", type=int, default=30)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cases = discover_cases(data_dir)
    if len(cases) <= args.test_size:
        raise ValueError(f"Need more than test_size cases: {len(cases)} <= {args.test_size}")

    rng = np.random.default_rng(args.seed)
    folds = {}
    for fold_idx in range(args.folds):
        perm = rng.permutation(cases)
        test = sorted(perm[: args.test_size].tolist(), key=lambda x: int(x) if x.isdigit() else x)
        train = sorted(perm[args.test_size :].tolist(), key=lambda x: int(x) if x.isdigit() else x)
        folds[f"fold{fold_idx}"] = {
            "train": train,
            "val": test,
            "test": test,
        }

    payload = {
        "data_dir": str(data_dir),
        "seed": args.seed,
        "folds": args.folds,
        "test_size": args.test_size,
        "num_cases": len(cases),
        "all_cases": cases,
        "splits": folds,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {out}")
    print(f"cases={len(cases)} folds={args.folds} test_size={args.test_size}")
    for name, split in folds.items():
        print(f"{name}: train={len(split['train'])} val/test={len(split['test'])}")


if __name__ == "__main__":
    main()
