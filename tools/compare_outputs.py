#!/usr/bin/env python3
"""Compare two PENGWIN reduction-pose JSON outputs."""

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path):
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list")
    return data


def key(item):
    for candidate in ("fragment_id", "id", "name"):
        if candidate in item:
            return str(item[candidate])
    raise KeyError(f"No fragment identifier in {item}")


def matrix(item):
    for candidate in ("transformation", "matrix", "pose"):
        if candidate in item:
            value = np.asarray(item[candidate], dtype=np.float64)
            if value.shape == (4, 4):
                return value
    raise KeyError(f"No 4x4 transformation matrix in {item}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-7)
    args = parser.parse_args()

    reference = {key(item): matrix(item) for item in load(args.reference)}
    candidate = {key(item): matrix(item) for item in load(args.candidate)}
    if reference.keys() != candidate.keys():
        raise SystemExit(
            f"Fragment IDs differ: reference={sorted(reference)} "
            f"candidate={sorted(candidate)}"
        )

    max_abs = max(
        float(np.max(np.abs(reference[item] - candidate[item])))
        for item in reference
    )
    print(f"fragments={len(reference)} max_abs_matrix_diff={max_abs:.12g}")
    if max_abs > args.atol:
        raise SystemExit(f"Difference exceeds atol={args.atol}")


if __name__ == "__main__":
    main()
