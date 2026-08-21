"""Fail fast when a strict reproduction run drifts from historical settings."""

import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class ReproductionGuard(L.Callback):
    def __init__(
        self,
        expected_world_size: int,
        expected_train_samples: int,
        expected_val_samples: int,
        expected_total_steps: int,
        require_flash_disabled: bool = True,
        expected_trimesh_version: str = "",
    ):
        self.expected_world_size = expected_world_size
        self.expected_train_samples = expected_train_samples
        self.expected_val_samples = expected_val_samples
        self.expected_total_steps = expected_total_steps
        self.require_flash_disabled = require_flash_disabled
        self.expected_trimesh_version = expected_trimesh_version

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        datamodule = trainer.datamodule
        observed = {
            "world_size": trainer.world_size,
            "train_samples": len(datamodule.train_dataset),
            "val_samples": len(datamodule.val_dataset),
            "total_steps": int(trainer.estimated_stepping_batches),
        }
        expected = {
            "world_size": self.expected_world_size,
            "train_samples": self.expected_train_samples,
            "val_samples": self.expected_val_samples,
            "total_steps": self.expected_total_steps,
        }

        errors = [
            f"{key}={observed[key]} (expected {value})"
            for key, value in expected.items()
            if observed[key] != value
        ]

        if self.require_flash_disabled:
            from models.AssemblyNet import layer

            if layer.flash_attn is not None:
                errors.append(
                    "FlashAttention is enabled; set ASSEMBLY_DISABLE_FLASH_ATTN=1"
                )

        if self.expected_trimesh_version:
            import trimesh

            if trimesh.__version__ != self.expected_trimesh_version:
                errors.append(
                    f"trimesh={trimesh.__version__} "
                    f"(expected {self.expected_trimesh_version})"
                )

        if errors:
            raise RuntimeError("Strict reproduction guard failed: " + "; ".join(errors))

        rank_zero_info(
            "[ReproductionGuard] PASS "
            + " ".join(f"{key}={value}" for key, value in observed.items())
            + " flash=disabled"
        )
