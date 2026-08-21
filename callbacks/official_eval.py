"""Official-style validation callback for PENGWIN Task 3.

This callback runs the production inference entrypoint plus evaluate.py on a
materialised split, so checkpoint selection can use challenge metrics instead
of only normalised Lightning losses.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import lightning as L
import torch


_SUMMARY_RE = {
    "trans_mm": re.compile(r"Trans error\s+\(mm\)\s+:\s+([0-9.]+)"),
    "rot_deg": re.compile(r"Rot error\s+\(deg\)\s+:\s+([0-9.]+)"),
    "tre_mm": re.compile(r"TRE\s+\(mm\)\s+:\s+([0-9.]+)"),
    "cd_mm": re.compile(r"CD_raw\s+\(mm\)\s+:\s+([0-9.]+)"),
    "pa": re.compile(r"PA\s+\(%\)\s+:\s+([0-9.]+)%"),
}


class OfficialEvalCallback(L.Callback):
    """Run gc_algorithm.py + evaluate.py on a fixed split after validation.

    The callback is rank-zero only and then synchronises all DDP ranks with a
    barrier.  By default it uses CPU inference (`cuda_visible_devices=""`) to
    avoid competing with training GPUs; override this if you have a free GPU.
    """

    def __init__(
        self,
        split_root: str = "clinical_valtest_split/mesh_val_select",
        output_dir: str = "output/${experiment_name}/official_val",
        every_n_epochs: int = 5,
        max_cases: int = 0,
        npoints: int = 3000,
        allocation: str = "volume",
        min_points_per_fragment: int = 20,
        surface_sample: str = "random",
        alpha: float = 1.0,
        rot_alpha: float | None = None,
        rot_alpha_li: float | None = None,
        rot_alpha_ri: float | None = None,
        tta: int = 1,
        max_iters: int = 1,
        seed: int = 42,
        cuda_visible_devices: str = "",
        timeout_per_case: int = 600,
        async_eval: bool = True,
    ):
        self.split_root = split_root
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs
        self.max_cases = max_cases
        self.npoints = npoints
        self.allocation = allocation
        self.min_points_per_fragment = min_points_per_fragment
        self.surface_sample = surface_sample
        self.alpha = alpha
        self.rot_alpha = rot_alpha
        self.rot_alpha_li = rot_alpha_li
        self.rot_alpha_ri = rot_alpha_ri
        self.tta = tta
        self.max_iters = max_iters
        self.seed = seed
        self.cuda_visible_devices = cuda_visible_devices
        self.timeout_per_case = timeout_per_case
        self.async_eval = async_eval
        self._eval_process = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        epoch = int(trainer.current_epoch)
        if self.every_n_epochs <= 0 or (epoch + 1) % self.every_n_epochs != 0:
            return

        if trainer.is_global_zero:
            if self.async_eval:
                self._launch_eval(trainer, pl_module, epoch)
            else:
                metrics = self._run_eval(trainer, pl_module, epoch)
                if trainer.logger is not None and metrics:
                    trainer.logger.log_metrics(
                        {f"official/{k}": v for k, v in metrics.items()},
                        step=trainer.global_step,
                    )
                    trainer.callback_metrics.update(
                        {f"official/{k}": torch.tensor(v, device=pl_module.device) for k, v in metrics.items()}
                    )

        if trainer.world_size > 1:
            trainer.strategy.barrier()

    def _launch_eval(self, trainer, pl_module, epoch: int):
        if self._eval_process is not None and self._eval_process.poll() is None:
            print(
                f"[OfficialEval] Previous async evaluation is still running "
                f"(pid={self._eval_process.pid}); skipping epoch {epoch}."
            )
            return

        root = Path.cwd()
        split_root = root / self.split_root
        out_root = Path(self.output_dir.replace("${experiment_name}", Path(trainer.default_root_dir).name))
        out_root = root / out_root if not out_root.is_absolute() else out_root
        run_dir = out_root / f"epoch_{epoch:04d}"
        ckpt_dir = out_root / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.ckpt"
        torch.save({"state_dict": pl_module.state_dict(), "epoch": epoch}, ckpt_path)

        cmd = [
            sys.executable,
            "tools/run_official_eval.py",
            "--checkpoint",
            str(ckpt_path),
            "--split_root",
            str(split_root),
            "--run_dir",
            str(run_dir),
            "--summary_csv",
            str(out_root / "official_metrics.csv"),
            "--epoch",
            str(epoch),
            "--max_cases",
            str(self.max_cases),
            "--npoints",
            str(self.npoints),
            "--allocation",
            self.allocation,
            "--min_points_per_fragment",
            str(self.min_points_per_fragment),
            "--surface_sample",
            self.surface_sample,
            "--output_type",
            getattr(pl_module, "output_type", "coords"),
            "--max_iters",
            str(self.max_iters),
            "--alpha",
            str(self.alpha),
            "--tta",
            str(self.tta),
            "--seed",
            str(self.seed),
            "--timeout_per_case",
            str(self.timeout_per_case),
        ]
        if self.rot_alpha is not None:
            cmd.extend(["--rot_alpha", str(self.rot_alpha)])
        if self.rot_alpha_li is not None:
            cmd.extend(["--rot_alpha_li", str(self.rot_alpha_li)])
        if self.rot_alpha_ri is not None:
            cmd.extend(["--rot_alpha_ri", str(self.rot_alpha_ri)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        worker_log = out_root / f"epoch_{epoch:04d}_worker.log"
        with worker_log.open("w") as log:
            self._eval_process = subprocess.Popen(
                cmd,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(
            f"[OfficialEval] Launched epoch {epoch} asynchronously "
            f"(pid={self._eval_process.pid}, log={worker_log})."
        )

    def _run_eval(self, trainer, pl_module, epoch: int) -> dict[str, float]:
        root = Path.cwd()
        split_root = root / self.split_root
        out_root = Path(self.output_dir.replace("${experiment_name}", Path(trainer.default_root_dir).name))
        out_root = root / out_root if not out_root.is_absolute() else out_root
        run_dir = out_root / f"epoch_{epoch:04d}"
        ckpt_dir = out_root / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.ckpt"
        torch.save({"state_dict": pl_module.state_dict(), "epoch": epoch}, ckpt_path)

        cases = sorted(p for p in split_root.iterdir() if p.is_dir())
        if self.max_cases > 0:
            cases = cases[: self.max_cases]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

        for case_dir in cases:
            dst = run_dir / case_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("peripelvic-fracture-fragments.obj", "plan_pl_gt.json"):
                target = dst / name
                if not target.exists():
                    target.symlink_to((case_dir / name).resolve())

            cmd = [
                sys.executable,
                "gc_algorithm.py",
                "--checkpoint",
                str(ckpt_path),
                "--input",
                str(dst),
                "--output",
                str(dst / "reduction-poses-matrices.json"),
                "--npoints",
                str(self.npoints),
                "--allocation",
                self.allocation,
                "--min_points_per_fragment",
                str(self.min_points_per_fragment),
                "--surface_sample",
                self.surface_sample,
                "--max_iters",
                str(self.max_iters),
                "--alpha",
                str(self.alpha),
                "--tta",
                str(self.tta),
                "--seed",
                str(self.seed),
            ]
            if self.rot_alpha is not None:
                cmd.extend(["--rot_alpha", str(self.rot_alpha)])
            if self.rot_alpha_li is not None:
                cmd.extend(["--rot_alpha_li", str(self.rot_alpha_li)])
            if self.rot_alpha_ri is not None:
                cmd.extend(["--rot_alpha_ri", str(self.rot_alpha_ri)])
            with (dst / "infer.log").open("w") as log:
                subprocess.run(
                    cmd,
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=self.timeout_per_case,
                )

        eval_log = run_dir / "evaluate.log"
        eval_cmd = [
            sys.executable,
            "evaluate.py",
            "--data_dir",
            str(run_dir),
            "--methods",
            "baseline",
            "--save_csv",
            str(run_dir / "metrics.csv"),
        ]
        with eval_log.open("w") as log:
            subprocess.run(eval_cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)

        metrics = self._parse_summary(eval_log)
        self._append_summary(out_root / "official_metrics.csv", epoch, metrics, run_dir)
        return metrics

    @staticmethod
    def _parse_summary(log_path: Path) -> dict[str, float]:
        text = log_path.read_text(errors="ignore")
        metrics = {}
        for name, pattern in _SUMMARY_RE.items():
            match = pattern.search(text)
            if match:
                value = float(match.group(1))
                metrics[name] = value / 100.0 if name == "pa" else value
        return metrics

    @staticmethod
    def _append_summary(path: Path, epoch: int, metrics: dict[str, float], run_dir: Path):
        exists = path.exists()
        with path.open("a") as f:
            if not exists:
                f.write("epoch,trans_mm,rot_deg,tre_mm,cd_mm,pa,run_dir\n")
            f.write(
                f"{epoch},"
                f"{metrics.get('trans_mm', float('nan'))},"
                f"{metrics.get('rot_deg', float('nan'))},"
                f"{metrics.get('tre_mm', float('nan'))},"
                f"{metrics.get('cd_mm', float('nan'))},"
                f"{metrics.get('pa', float('nan'))},"
                f"{run_dir}\n"
            )
