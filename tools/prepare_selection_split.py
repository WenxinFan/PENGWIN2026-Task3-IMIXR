"""Materialize the fixed train150 selection split for official-style evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="configs/splits/clinical_train150_select20hard10_seed42.json",
    )
    parser.add_argument("--fold", default="train150_select20hard10_seed42")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--clinical_root", required=True)
    parser.add_argument(
        "--out_dir",
        default="clinical_valtest_split/mesh_select_val10_hard10_seed42",
    )
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    case_ids = manifest["splits"][args.fold][args.split]
    clinical_root = Path(args.clinical_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case_id in case_ids:
        source = clinical_root / str(case_id)
        if not (source / "peripelvic-fracture-fragments.obj").is_file():
            raise FileNotFoundError(f"Missing clinical OBJ for case {case_id}: {source}")
        if not (source / "plan_pl_gt.json").is_file():
            raise FileNotFoundError(f"Missing GT JSON for case {case_id}: {source}")
        destination = out_dir / str(case_id)
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        elif destination.exists():
            shutil.rmtree(destination)
        if args.copy:
            shutil.copytree(source, destination)
        else:
            destination.symlink_to(source, target_is_directory=True)

    print(f"Prepared {len(case_ids)} cases at {out_dir}")


if __name__ == "__main__":
    main()
