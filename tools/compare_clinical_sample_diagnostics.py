"""Compare clinical sample dumps produced on two servers."""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()

    reference = np.load(args.reference)
    candidate = np.load(args.candidate)
    keys = sorted(set(reference.files) | set(candidate.files))
    failed = False

    print(f"reference: {Path(args.reference).resolve()}")
    print(f"candidate: {Path(args.candidate).resolve()}")
    for key in keys:
        if key not in reference.files or key not in candidate.files:
            print(f"MISSING {key}: reference={key in reference.files} candidate={key in candidate.files}")
            failed = True
            continue

        left = reference[key]
        right = candidate[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            print(
                f"STRUCT {key}: shape {left.shape} vs {right.shape}; "
                f"dtype {left.dtype} vs {right.dtype}"
            )
            failed = True
            continue

        equal = np.array_equal(left, right)
        if equal:
            print(f"OK     {key}: shape={left.shape} dtype={left.dtype}")
            continue

        failed = True
        mismatch = np.argwhere(left != right)
        first = tuple(mismatch[0]) if len(mismatch) else ()
        if np.issubdtype(left.dtype, np.number):
            delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
            print(
                f"DIFF   {key}: first={first} left={left[first]} right={right[first]} "
                f"max_abs={np.nanmax(delta):.9g} mean_abs={np.nanmean(delta):.9g}"
            )
        else:
            print(f"DIFF   {key}: first={first} left={left[first]} right={right[first]}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
