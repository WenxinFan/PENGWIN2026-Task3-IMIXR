"""Create a directory of symlinks/copies for a split from a manifest."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Materialize a split as case directories")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fold", default="fold0")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--copy", action="store_true", help="Copy directories instead of symlinking")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    data_dir = Path(manifest["data_dir"])
    case_names = manifest["splits"][args.fold][args.split]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in case_names:
        src = data_dir / str(name)
        dst = out_dir / str(name)
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        if args.copy:
            shutil.copytree(src, dst)
        else:
            dst.symlink_to(src, target_is_directory=True)

    print(f"Materialized {len(case_names)} cases: {args.fold}/{args.split} -> {out_dir}")


if __name__ == "__main__":
    main()
