"""Make a clinical split with separate validation and held-out test cases.

The test split is the GT-hard20 list by default.  Validation cases are sampled
from the remaining cases by hardness-rank strata, so it contains easy, medium,
and hard-but-not-test examples without overlapping the independent test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def _case_sort_key(name: str):
    return int(name) if str(name).isdigit() else str(name)


def _read_ranked_cases(path: Path) -> list[str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [str(row["sample"]).zfill(3) for row in rows]


def _discover_cases(data_dir: Path) -> list[str]:
    cases = []
    for d in sorted((p for p in data_dir.iterdir() if p.is_dir()), key=lambda p: _case_sort_key(p.name)):
        if any(d.glob("*.obj")) and (d / "plan_pl_gt.json").exists():
            cases.append(d.name)
    return cases


def _stratified_sample(ranked_pool: list[str], n_select: int, seed: int) -> list[str]:
    if n_select > len(ranked_pool):
        raise ValueError(f"Cannot select {n_select} validation cases from pool of {len(ranked_pool)}")

    rng = np.random.default_rng(seed)
    bins = np.array_split(np.array(ranked_pool, dtype=object), n_select)
    selected = []
    for bucket in bins:
        selected.append(str(rng.choice(bucket)))
    return sorted(selected, key=_case_sort_key)


def main():
    parser = argparse.ArgumentParser(description="Create separate clinical val/test split manifest")
    parser.add_argument(
        "--data_dir",
        default=os.environ.get(
            "PENGWIN_CLINICAL_ROOT",
            "./data/PENGWIN26_task3_clinical_fractures_train/mesh",
        ),
        help="Clinical mesh root containing case folders",
    )
    parser.add_argument(
        "--ranking_csv",
        default="hard_clinical_split/clinical_hardness_ranking.csv",
        help="CSV with columns hard_rank and sample",
    )
    parser.add_argument(
        "--hard_cases",
        default="hard_clinical_split/hard_cases.txt",
        help="Independent test cases, one case id per line",
    )
    parser.add_argument("--out", default="configs/splits/clinical_val30_hard20_seed42.json")
    parser.add_argument("--fold_name", default="val30_hard20_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=int, default=30)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ranked_cases = _read_ranked_cases(Path(args.ranking_csv))
    discovered = set(_discover_cases(data_dir))

    with Path(args.hard_cases).open() as f:
        test = [line.strip().zfill(3) for line in f if line.strip()]

    missing = sorted((set(ranked_cases) | set(test)) - discovered, key=_case_sort_key)
    if missing:
        raise ValueError(f"Cases missing from data_dir: {missing}")

    overlap = sorted(set(test) & set(ranked_cases), key=_case_sort_key)
    if len(overlap) != len(test):
        raise ValueError("Some test cases are not present in the ranking CSV")

    test_set = set(test)
    ranked_pool = [case for case in ranked_cases if case not in test_set]
    val = _stratified_sample(ranked_pool, args.val_size, args.seed)
    val_set = set(val)
    train = sorted([case for case in ranked_cases if case not in test_set and case not in val_set], key=_case_sort_key)

    if set(train) & val_set or set(train) & test_set or val_set & test_set:
        raise AssertionError("Split overlap detected")

    payload = {
        "description": (
            "Clinical split with separate val_select and hard_test. "
            "Validation is stratified by GT-only hardness among non-test cases; "
            "test is the fixed GT-hard20 and must not be used for epoch or hyperparameter selection."
        ),
        "data_dir": str(data_dir),
        "ranking_csv": args.ranking_csv,
        "hard_cases": args.hard_cases,
        "seed": args.seed,
        "val_size": args.val_size,
        "test_size": len(test),
        "num_cases": len(ranked_cases),
        "all_cases": sorted(ranked_cases, key=_case_sort_key),
        "splits": {
            args.fold_name: {
                "train": train,
                "val": val,
                "test": test,
            }
        },
        "usage": {
            "train": "used for fitting only",
            "val": "val_select: used for early stopping, epoch selection, alpha/points selection",
            "test": "hard_test: held out until final local comparison",
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Saved {out}")
    print(f"fold={args.fold_name} train={len(train)} val_select={len(val)} hard_test={len(test)}")
    print("val_select:", " ".join(val))
    print("hard_test :", " ".join(test))


if __name__ == "__main__":
    main()
