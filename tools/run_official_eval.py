"""Run production inference and official-style evaluation outside Lightning DDP."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


SUMMARY_RE = {
    "trans_mm": re.compile(r"Trans error\s+\(mm\)\s+:\s+([0-9.]+)"),
    "rot_deg": re.compile(r"Rot error\s+\(deg\)\s+:\s+([0-9.]+)"),
    "tre_mm": re.compile(r"TRE\s+\(mm\)\s+:\s+([0-9.]+)"),
    "cd_mm": re.compile(r"CD_raw\s+\(mm\)\s+:\s+([0-9.]+)"),
    "pa": re.compile(r"PA\s+\(%\)\s+:\s+([0-9.]+)%"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--npoints", type=int, default=3000)
    parser.add_argument("--allocation", choices=["equal", "volume"], default="equal")
    parser.add_argument("--min_points_per_fragment", type=int, default=20)
    parser.add_argument("--min_points_per_fragment_sa", type=int, default=None)
    parser.add_argument("--min_points_per_fragment_li", type=int, default=None)
    parser.add_argument("--min_points_per_fragment_ri", type=int, default=None)
    parser.add_argument("--surface_sample", choices=["even", "random"], default="even")
    parser.add_argument("--output_type", choices=["coords", "pose"], default="coords")
    parser.add_argument("--inference_precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max_iters", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--rot_alpha", type=float, default=None)
    parser.add_argument("--rot_alpha_li", type=float, default=None)
    parser.add_argument("--rot_alpha_ri", type=float, default=None)
    parser.add_argument("--kabsch_robust", choices=["none", "huber"], default="none")
    parser.add_argument("--kabsch_huber_delta", type=float, default=0.02)
    parser.add_argument("--kabsch_irls_iters", type=int, default=5)
    parser.add_argument("--save_inference_diagnostics", action="store_true")
    parser.add_argument("--tta", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout_per_case", type=int, default=600)
    parser.add_argument("--rotation_rerank_angles", default="")
    parser.add_argument("--rotation_rerank_sample_points", type=int, default=512)
    parser.add_argument("--rotation_rerank_contact_fraction", type=float, default=0.08)
    parser.add_argument("--rotation_rerank_angle_weight", type=float, default=0.002)
    return parser.parse_args()


def parse_summary(path: Path) -> dict[str, float]:
    text = path.read_text(errors="ignore")
    metrics = {}
    for name, pattern in SUMMARY_RE.items():
        match = pattern.search(text)
        if match:
            value = float(match.group(1))
            metrics[name] = value / 100.0 if name == "pa" else value
    return metrics


def upsert_summary(path: Path, epoch: int, metrics: dict[str, float], run_dir: Path):
    fieldnames = ["epoch", "trans_mm", "rot_deg", "tre_mm", "cd_mm", "pa", "run_dir"]
    rows = []
    if path.exists():
        with path.open(newline="") as f:
            rows = [row for row in csv.DictReader(f) if int(row["epoch"]) != epoch]

    rows.append({
        "epoch": epoch,
        "trans_mm": metrics.get("trans_mm", float("nan")),
        "rot_deg": metrics.get("rot_deg", float("nan")),
        "tre_mm": metrics.get("tre_mm", float("nan")),
        "cd_mm": metrics.get("cd_mm", float("nan")),
        "pa": metrics.get("pa", float("nan")),
        "run_dir": str(run_dir),
    })
    rows.sort(key=lambda row: int(row["epoch"]))

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint).resolve()
    split_root = Path(args.split_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    summary_csv = Path(args.summary_csv).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    cases = sorted(path for path in split_root.iterdir() if path.is_dir())
    if args.max_cases > 0:
        cases = cases[:args.max_cases]

    print(
        f"[OfficialEvalWorker] epoch={args.epoch} cases={len(cases)} "
        f"checkpoint={checkpoint}",
        flush=True,
    )
    for case_idx, case_dir in enumerate(cases, start=1):
        dst = run_dir / case_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("peripelvic-fracture-fragments.obj", "plan_pl_gt.json"):
            target = dst / name
            source = (case_dir / name).resolve()
            if target.is_symlink() and target.resolve() != source:
                target.unlink()
            if not target.exists():
                target.symlink_to(source)

        cmd = [
            sys.executable,
            "gc_algorithm.py",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(dst),
            "--output",
            str(dst / "reduction-poses-matrices.json"),
            "--npoints",
            str(args.npoints),
            "--allocation",
            args.allocation,
            "--min_points_per_fragment",
            str(args.min_points_per_fragment),
            "--surface_sample",
            args.surface_sample,
            "--output_type",
            args.output_type,
            "--inference_precision",
            args.inference_precision,
            "--max_iters",
            str(args.max_iters),
            "--alpha",
            str(args.alpha),
            "--rot_alpha",
            str(args.rot_alpha) if args.rot_alpha is not None else "1.0",
            "--tta",
            str(args.tta),
            "--seed",
            str(args.seed),
            "--kabsch_robust",
            args.kabsch_robust,
            "--kabsch_huber_delta",
            str(args.kabsch_huber_delta),
            "--kabsch_irls_iters",
            str(args.kabsch_irls_iters),
        ]
        if args.rot_alpha_li is not None:
            cmd.extend(["--rot_alpha_li", str(args.rot_alpha_li)])
        if args.rot_alpha_ri is not None:
            cmd.extend(["--rot_alpha_ri", str(args.rot_alpha_ri)])
        if args.min_points_per_fragment_sa is not None:
            cmd.extend([
                "--min_points_per_fragment_sa",
                str(args.min_points_per_fragment_sa),
            ])
        if args.min_points_per_fragment_li is not None:
            cmd.extend([
                "--min_points_per_fragment_li",
                str(args.min_points_per_fragment_li),
            ])
        if args.min_points_per_fragment_ri is not None:
            cmd.extend([
                "--min_points_per_fragment_ri",
                str(args.min_points_per_fragment_ri),
            ])
        if args.save_inference_diagnostics:
            cmd.extend(["--diagnostics_output", str(dst / "inference-diagnostics.json")])
        if args.rotation_rerank_angles:
            cmd.extend([
                "--rotation_rerank_angles",
                args.rotation_rerank_angles,
                "--rotation_rerank_sample_points",
                str(args.rotation_rerank_sample_points),
                "--rotation_rerank_contact_fraction",
                str(args.rotation_rerank_contact_fraction),
                "--rotation_rerank_angle_weight",
                str(args.rotation_rerank_angle_weight),
            ])
        print(
            f"[OfficialEvalWorker] [{case_idx}/{len(cases)}] case={case_dir.name}",
            flush=True,
        )
        infer_log = dst / "infer.log"
        with infer_log.open("w") as log:
            subprocess.run(
                cmd,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=args.timeout_per_case,
            )
        infer_text = infer_log.read_text(errors="ignore")
        if "Inference failed; writing identity fallback" in infer_text:
            raise RuntimeError(
                f"Inference failed for case {case_dir.name}; see {infer_log}"
            )

    evaluate_log = run_dir / "evaluate.log"
    evaluate_cmd = [
        sys.executable,
        "evaluate.py",
        "--data_dir",
        str(run_dir),
        "--methods",
        "baseline",
        "--save_csv",
        str(run_dir / "metrics.csv"),
    ]
    with evaluate_log.open("w") as log:
        subprocess.run(
            evaluate_cmd,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    metrics = parse_summary(evaluate_log)
    if len(metrics) != len(SUMMARY_RE):
        raise RuntimeError(f"Incomplete official metrics in {evaluate_log}: {metrics}")
    upsert_summary(summary_csv, args.epoch, metrics, run_dir)
    print(f"[OfficialEvalWorker] completed: {metrics}", flush=True)


if __name__ == "__main__":
    main()
