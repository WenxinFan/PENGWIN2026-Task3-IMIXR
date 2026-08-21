"""Persist the actual final training state independently of top-k selection."""

from pathlib import Path

import lightning as L


class FinalCheckpointCallback(L.Callback):
    def __init__(self, path: str):
        self.path = path

    def on_fit_end(self, trainer, pl_module):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(self.path)
