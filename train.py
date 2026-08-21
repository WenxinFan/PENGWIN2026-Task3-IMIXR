"""Training entry point for AssembleNet.

Usage:
    python train.py                                    # training
    python train.py --config-name train_pose           # pose training
    python train.py --config-name finetune      # LoRA fine-tune
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
# `expandable_segments` is only available in newer PyTorch builds.  The
# fracformer environment rejects it, so use the broadly-supported allocator
# split-size knob instead.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import lightning as L
from typing import List

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    L.seed_everything(cfg.get("seed", 42), workers=True)

    datamodule: L.LightningDataModule = instantiate(cfg.data)
    model: L.LightningModule = instantiate(cfg.model)

    callbacks: List[L.Callback] = [
        callback
        for callback in (instantiate(cb) for cb in cfg.get("callbacks", {}).values())
        if callback is not None
    ]

    trainer: L.Trainer = instantiate(cfg.trainer, callbacks=callbacks)
    ckpt_path = cfg.get("ckpt_path", None)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
    # # watch -n 0.1 -d nvidia-smi
    # CUDA_VISIBLE_DEVICES=2,3 // 0,1
