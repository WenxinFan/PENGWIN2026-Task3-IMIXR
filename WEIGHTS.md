# Included weights

All large files under `weights/` must be stored with Git LFS.

| File | Purpose | Bytes | SHA256 |
|---|---|---:|---|
| `sim_pretrain_e294.ckpt` | Stage 1 simulation checkpoint | 430253287 | `49d7180ee114c7a68e4f521d34641bdb3c41a69545303ab5314a8ac508e951a3` |
| `train120_rotstrat_e169.ckpt` | Stage 2 rotation-stratified clinical checkpoint | 143416563 | `33450edfd1c78a4a595364517b74dc9e02ed13a7e392f1709a15c015af1bf82c` |
| `train120_rotgeom_e19.ckpt` | Stage 3 rotation-geometry checkpoint | 430254117 | `6b538b067f90e37bfc7d8eb3bed41c2828d000e5a7263497ac69af9c045f9677` |
| `train150_e4.ckpt` | Selected Stage 4 Lightning checkpoint | 430252523 | `62860acb95dde21087485745cd0d1cf43c98c25a283aacd2eee6bb5ba14dfbc1` |
| `model_state_dict.pt` | Exported deployment checkpoint | 143395235 | `01b118590702d41dfa045fc563ef0afd7dcc90cd4541662cad7843d945ff1993` |

The final submitted inference uses `model_state_dict.pt`. The other checkpoints
allow the documented continuation stages to be reproduced or resumed without
rerunning all preceding stages.
