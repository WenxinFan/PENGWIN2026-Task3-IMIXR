# Release validation

Validation date: 2026-08-21.

- All Python entrypoints and modules compile successfully.
- All four training-stage Hydra configurations compose successfully.
- Stage checkpoint paths resolve to the bundled `weights/` directory.
- The Stage 4 model loads all 226 checkpoint tensors and contains 35,830,304 parameters.
- SHA256 checks pass for all five bundled weight files.
- The fixed selection manifest materializes 20 clinical cases.
- Native CPU inference on clinical case 001 completes without identity fallback.
- The public-base inference Dockerfile builds successfully.
- Offline Docker inference on case 001 completes without identity fallback.
- Native and Docker outputs contain fragment IDs `1`, `101`, and `201`, each
  with one finite 4x4 homogeneous transformation matrix.
- Native versus Docker maximum absolute matrix difference is
  `3.95774841309e-05`, caused by the different PyTorch CPU runtimes.
- The inference image contains exactly one checkpoint:
  `/opt/app/model_state_dict.pt`.

Run the structural weight check with:

```bash
python tools/verify_release.py
```
