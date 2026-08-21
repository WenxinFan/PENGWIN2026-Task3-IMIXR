# PENGWIN 2026 Task 3: train150-e4 + SA75/LI75/RI50

This repository is the complete research-code release for the submitted
PENGWIN 2026 Task 3 reduction model. It contains the training pipeline,
inference and official-style evaluation code, fixed split manifests, all
intermediate checkpoints needed to resume the documented stages, and the
exported deployment checkpoint.

The PENGWIN datasets are not included. Obtain them from the challenge
organizers and set the two data-root environment variables described below.

## Final submitted method

- AssemblyTransformer coordinate regression followed by rigid SVD/Kabsch pose recovery
- simulation pretraining, full-parameter clinical adaptation and train150 continuation
- fragment-balanced coordinate loss
- translation/quaternion pose consistency with weight `0.2`
- translation-invariant rotation-geometry loss with weight `0.1`
- training input: `1800` points per bone, equal/even sampling
- deployment input: `3000` points per bone, volume/random sampling
- minimum deployment points per fragment: `SA=75`, `LI=75`, `RI=50`
- one forward pass, `max_iters=1`, `tta=1`, seed `42`
- centroid-preserving rotation-vector calibration with `rot_alpha=0.975`

The selected model is `weights/train150_e4.ckpt`; the deployment container
loads `weights/model_state_dict.pt`.

## Repository layout

```text
callbacks/                 Lightning callbacks and reproduction guards
configs/                   Hydra model/data/trainer configs and fixed splits
datasets/                  simulation and clinical OBJ data loaders
models/AssemblyNet/        AssemblyTransformer, losses and pose recovery
submission_runtime/        clean standalone deployment entrypoint
tools/                     export, evaluation and split-preparation utilities
weights/                   all included checkpoints (Git LFS)
docs/ALGORITHM.md          full challenge algorithm description
train.py                   training entrypoint
gc_algorithm.py            production-style inference entrypoint
evaluate.py                official-style local evaluator
Dockerfile                 public-base inference image
Dockerfile.train           training environment image
```

## Environment

The original training runs used Python `3.12.7`, Lightning `2.6.5`, Hydra
`1.3.4`, OmegaConf `2.3.1`, trimesh `4.12.2`, NumPy `1.26.4` and mixed-precision
PyTorch. Flash Attention was explicitly disabled for the selected run.

Create a portable environment:

```bash
conda create -n pengwin-repro python=3.12 -y
conda activate pengwin-repro
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Data layout

Expected simulation layout:

```text
PENGWIN26_task3_simulation_fractures_train/points/
  <case>/<SA|LI|RI>/template.npz
  <case>/<SA|LI|RI>/fractures/*.npz
```

Expected clinical layout:

```text
PENGWIN26_task3_clinical_fractures_train/mesh/
  <case>/peripelvic-fracture-fragments.obj
  <case>/plan_pl_gt.json
```

Set roots before training:

```bash
export PENGWIN_SIM_ROOT=/path/to/PENGWIN26_task3_simulation_fractures_train/points
export PENGWIN_CLINICAL_ROOT=/path/to/PENGWIN26_task3_clinical_fractures_train/mesh
```

Prepare the fixed 20-case selection directory used by automatic official-style
evaluation:

```bash
python tools/prepare_selection_split.py --clinical_root "$PENGWIN_CLINICAL_ROOT"
```

## Training stages

Every continuation uses `model.checkpoint`, meaning weight-only initialization;
it does not resume the preceding optimizer or learning-rate scheduler state.
The matching starting checkpoints are already provided in `weights/`.

### Stage 1: simulation pretraining

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ASSEMBLY_DISABLE_FLASH_ATTN=1 MPLCONFIGDIR=/tmp/mpl-pengwin-sim python train.py --config-name train_A_sim_pretrain_w02
```

The included selected result is `weights/sim_pretrain_e294.ckpt`.

### Stage 2: clinical rotation-stratified adaptation

```bash
CUDA_VISIBLE_DEVICES=0,1 ASSEMBLY_DISABLE_FLASH_ATTN=1 MPLCONFIGDIR=/tmp/mpl-pengwin-rotstrat python train.py --config-name train_fixedparser_sim294_clinical120val30_n1800_equaleven_hardaug_p025_rotstrat_403030_fromsim294
```

This stage starts from `weights/sim_pretrain_e294.ckpt`; the included selected
result is `weights/train120_rotstrat_e169.ckpt`.

### Stage 3: rotation-geometry continuation

```bash
CUDA_VISIBLE_DEVICES=0,1 ASSEMBLY_DISABLE_FLASH_ATTN=1 MPLCONFIGDIR=/tmp/mpl-pengwin-rotgeom python train.py --config-name train_fixedparser_sim294_clinical120val30_n1800_equaleven_hardaug_p025_rotstrat_rotgeom_w01_frome169
```

This stage starts from `weights/train120_rotstrat_e169.ckpt`; the included
selected result is `weights/train120_rotgeom_e19.ckpt`.

### Stage 4: train150 continuation and final checkpoint selection

```bash
CUDA_VISIBLE_DEVICES=0,1 ASSEMBLY_DISABLE_FLASH_ATTN=1 MPLCONFIGDIR=/tmp/mpl-pengwin-train150 python train.py --config-name train_fixedparser_sim294_clinical150_select20hard10_n1800_equaleven_rotgeom_w01_frome19_lr5e6
```

This stage starts from `weights/train120_rotgeom_e19.ckpt`, trains for 15
epochs, and evaluates every fifth epoch on the fixed `val10 + hard10` selection
set. The checkpoint recorded as epoch 4 was selected and is included as
`weights/train150_e4.ckpt`.

## Final inference

Run the clean deployment entrypoint directly:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m submission_runtime.pengwin_task3 --checkpoint weights/model_state_dict.pt --input /path/to/case --output reduction-poses-matrices.json
```

Its baked defaults are exactly `3000/volume/random`, `SA75/LI75/RI50`,
`rot_alpha=0.975`, `max_iters=1`, `tta=1`, seed `42`.

For the full development CLI, the equivalent command is:

```bash
CUDA_VISIBLE_DEVICES=0 ASSEMBLY_DISABLE_FLASH_ATTN=1 python gc_algorithm.py --checkpoint weights/train150_e4.ckpt --input /path/to/case --output reduction-poses-matrices.json --npoints 3000 --allocation volume --min_points_per_fragment 20 --min_points_per_fragment_sa 75 --min_points_per_fragment_li 75 --min_points_per_fragment_ri 50 --surface_sample random --output_type coords --inference_precision fp16 --max_iters 1 --alpha 1.0 --rot_alpha 0.975 --tta 1 --seed 42 --kabsch_robust none
```

## Evaluate the fixed 20-case selection set

```bash
CUDA_VISIBLE_DEVICES=0 ASSEMBLY_DISABLE_FLASH_ATTN=1 python tools/run_official_eval.py --checkpoint weights/train150_e4.ckpt --split_root clinical_valtest_split/mesh_select_val10_hard10_seed42 --run_dir output/train150_e4_sa75_li75_ri50_select20 --summary_csv output/train150_e4_sa75_li75_ri50_summary.csv --epoch 4 --npoints 3000 --allocation volume --min_points_per_fragment 20 --min_points_per_fragment_sa 75 --min_points_per_fragment_li 75 --min_points_per_fragment_ri 50 --surface_sample random --output_type coords --inference_precision fp16 --max_iters 1 --alpha 1.0 --rot_alpha 0.975 --tta 1 --seed 42 --kabsch_robust none
```

Expected aggregate selection metrics are approximately:

| Trans (mm) | Rot (deg) | TRE (mm) | CD (mm) | PA |
|---:|---:|---:|---:|---:|
| 4.154 | 7.817 | 3.274 | 3.8703 | 0.8269 |

## Export the deployment checkpoint

```bash
python tools/export_inference_ckpt.py weights/train150_e4.ckpt -o weights/model_state_dict.pt
```

## Docker inference

```bash
docker build -t pengwin-task3-train150-e4-sa75-li75-ri50:github .
docker run --rm --network none --gpus all -v /path/to/case:/input:ro -v /path/to/output:/output pengwin-task3-train150-e4-sa75-li75-ri50:github
```

The final output is `/output/reduction-poses-matrices.json`.

## GitHub and large files

Run Git LFS before the first commit:

```bash
git lfs install
git lfs track "weights/*.ckpt" "weights/*.pt"
git add -- .gitattributes
```

Do not commit the PENGWIN datasets, generated `output/` directories or Docker
archives. See `WEIGHTS.md` for checkpoint hashes and `NOTICE.md` for provenance.
The completed local validation record is in `RELEASE_VALIDATION.md`.
