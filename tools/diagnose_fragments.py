"""Export per-fragment pose/geometry diagnostics for Task 3 predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import (
    PA_THRESHOLD,
    _bone_key_for_fragment,
    apply_T,
    chamfer_distance_euc,
    get_anchor_id,
    load_obj_meshes,
    load_poses_json,
    meshes_to_bone_dict,
    normalise_poses,
    rotation_geodesic_deg,
    sample_surface_area_proportional,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Case folders containing OBJ, GT and predictions")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prediction_file", default="reduction-poses-matrices.json")
    parser.add_argument("--diagnostics_dir", default=None,
                        help="Optional parallel case tree containing inference-diagnostics.json")
    parser.add_argument("--diagnostics_file", default="inference-diagnostics.json")
    parser.add_argument("--npoints_frag", type=int, default=2000)
    parser.add_argument("--pa_threshold", type=float, default=PA_THRESHOLD)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_sidecar(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open() as f:
        payload = json.load(f)
    return {str(row["fragment_id"]): row for row in payload.get("fragments", [])}


def safe_correlation(rows, x_key, y_key="rot_error_deg"):
    pairs = []
    for row in rows:
        try:
            x, y = float(row[x_key]), float(row[y_key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            pairs.append((x, y))
    if len(pairs) < 3:
        return len(pairs), float("nan"), float("nan")
    x, y = np.asarray(pairs, dtype=np.float64).T
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return len(pairs), float("nan"), float("nan")
    return len(pairs), float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)


def write_csv(path: Path, rows: list[dict]):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows: list[dict], output_path: Path):
    panels = [
        ("predicted_angle_deg", "Predicted angle (deg)"),
        ("fragment_diameter_mm", "Fragment diameter (mm)"),
        ("svd_residual_mean_mm", "SVD residual mean (mm)"),
        ("tta_rotation_deviation_mean_deg", "TTA rotation deviation (deg)"),
    ]
    colors = {"SA": "#4c78a8", "LI": "#f58518", "RI": "#54a24b"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (x_key, label) in zip(axes.ravel(), panels):
        for bone in ("SA", "LI", "RI"):
            points = []
            for row in rows:
                if row.get("bone") != bone:
                    continue
                try:
                    x, y = float(row[x_key]), float(row["rot_error_deg"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(x) and np.isfinite(y):
                    points.append((x, y))
            if points:
                x, y = np.asarray(points).T
                ax.scatter(x, y, s=20, alpha=0.7, color=colors[bone], label=bone)
        ax.set_xlabel(label)
        ax.set_ylabel("Rotation error (deg)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    diagnostics_dir = Path(args.diagnostics_dir).resolve() if args.diagnostics_dir else data_dir
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    case_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    for case_dir in case_dirs:
        pred_path = case_dir / args.prediction_file
        gt_path = case_dir / "plan_pl_gt.json"
        obj_files = sorted(case_dir.glob("*.obj"))
        if not pred_path.exists() or not gt_path.exists() or not obj_files:
            continue

        meshes = load_obj_meshes(str(obj_files[0]))
        gt_raw = load_poses_json(str(gt_path))
        pred_raw = load_poses_json(str(pred_path))
        bone_dict = meshes_to_bone_dict(meshes)
        common_ids = sorted(set(meshes) & set(gt_raw) & set(pred_raw), key=int)
        if not common_ids:
            continue
        anchor_id = get_anchor_id(gt_raw, bone_dict)
        if anchor_id not in pred_raw:
            anchor_id = common_ids[0]
        gt = normalise_poses(gt_raw, anchor_id)
        pred = normalise_poses(pred_raw, anchor_id)
        sidecar = load_sidecar(diagnostics_dir / case_dir.name / args.diagnostics_file)

        gt_vertices = np.concatenate([apply_T(meshes[fid].vertices, gt[fid]) for fid in common_ids])
        assembly_center = gt_vertices.mean(axis=0)
        assembly_radius = float(np.max(np.linalg.norm(gt_vertices - assembly_center, axis=1)))
        assembly_radius = max(assembly_radius, 1e-8)

        for fid in common_ids:
            mesh = meshes[fid]
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            centroid = verts.mean(axis=0)
            diameter = float(2.0 * np.max(np.linalg.norm(verts - centroid, axis=1)))
            rng = np.random.default_rng(args.seed + int(case_dir.name) * 1000 + int(fid))
            points = sample_surface_area_proportional(mesh, args.npoints_frag, rng)
            points_gt = apply_T(points, gt[fid])
            points_pred = apply_T(points, pred[fid])
            centroid_gt = apply_T(verts, gt[fid]).mean(axis=0)
            centroid_pred = apply_T(verts, pred[fid]).mean(axis=0)
            cd_raw = chamfer_distance_euc(points_pred, points_gt)
            cd_norm = chamfer_distance_euc(
                (points_pred - assembly_center) / assembly_radius,
                (points_gt - assembly_center) / assembly_radius,
            )
            row = {
                "case": case_dir.name,
                "fragment_id": fid,
                "bone": _bone_key_for_fragment(int(fid)),
                "is_anchor": int(fid == anchor_id),
                "n_vertices": len(verts),
                "n_faces": len(mesh.faces),
                "is_watertight": int(bool(mesh.is_watertight)),
                "surface_area_mm2": float(mesh.area),
                "abs_volume_mm3": float(abs(mesh.volume)),
                "fragment_diameter_mm": diameter,
                "predicted_angle_deg": rotation_geodesic_deg(pred[fid][:3, :3], np.eye(3)),
                "gt_angle_deg": rotation_geodesic_deg(gt[fid][:3, :3], np.eye(3)),
                "rot_error_deg": rotation_geodesic_deg(pred[fid][:3, :3], gt[fid][:3, :3]),
                "trans_error_mm": float(np.linalg.norm(centroid_pred - centroid_gt)),
                "fragment_tre_mm": float(np.mean(np.linalg.norm(points_pred - points_gt, axis=1))),
                "fragment_cd_raw_mm": cd_raw,
                "fragment_cd_norm": cd_norm,
                "pa_pass": int(cd_norm < args.pa_threshold),
            }
            for key, value in sidecar.get(fid, {}).items():
                if key not in ("fragment_id", "bone"):
                    row[key] = value
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No evaluable predictions found under {data_dir}")

    write_csv(output_dir / "per_fragment_metrics.csv", rows)
    correlation_rows = []
    variables = [
        "predicted_angle_deg", "gt_angle_deg", "fragment_diameter_mm",
        "surface_area_mm2", "abs_volume_mm3", "svd_residual_mean_mm",
        "svd_residual_p90_mm", "tta_rotation_deviation_mean_deg",
        "tta_centroid_deviation_mean_mm",
    ]
    for group in ("ALL", "SA", "LI", "RI"):
        selected = rows if group == "ALL" else [row for row in rows if row["bone"] == group]
        for variable in variables:
            n, pearson, spearman = safe_correlation(selected, variable)
            correlation_rows.append({
                "group": group, "x": variable, "y": "rot_error_deg",
                "n": n, "pearson_r": pearson, "spearman_rho": spearman,
            })
    write_csv(output_dir / "rot_error_correlations.csv", correlation_rows)
    make_plots(rows, output_dir / "rot_error_scatter.png")

    worst = sorted(rows, key=lambda row: float(row["rot_error_deg"]), reverse=True)[:15]
    print(f"Saved {len(rows)} fragment rows to {output_dir / 'per_fragment_metrics.csv'}")
    print(f"Saved correlations to {output_dir / 'rot_error_correlations.csv'}")
    print(f"Saved plots to {output_dir / 'rot_error_scatter.png'}")
    print("Worst rotation fragments:")
    for row in worst:
        print(
            f"  case={row['case']} fragment={row['fragment_id']} bone={row['bone']} "
            f"rot={row['rot_error_deg']:.2f}deg trans={row['trans_error_mm']:.2f}mm "
            f"tre={row['fragment_tre_mm']:.2f}mm cd={row['fragment_cd_raw_mm']:.2f}mm"
        )


if __name__ == "__main__":
    main()
