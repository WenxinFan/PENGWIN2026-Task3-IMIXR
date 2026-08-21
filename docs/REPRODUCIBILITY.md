# Reproducibility record

This image is a clean engineering rebuild of the original train150 e4 candidate
with the selected SA75/LI75/RI50 inference sampling policy. It
removes stale application files inherited from an earlier runtime image and
contains exactly one inference checkpoint.

The trained weights are unchanged from the validated train150 e4 package.
Deployment defaults are fixed to:

- 3000 sampled points
- volume-based point allocation
- minimum points per fragment: SA 75, LI 75, RI 50 (the total remains 3000 per bone)
- random surface sampling with seed 42
- one inference iteration
- translation alpha 1.0
- rotation alpha 0.975
- one inference pass (no multi-pass TTA)

Before release, the original and rebuilt containers must be run on the same
input case and their output fragment IDs and 4x4 matrices compared. The release
manifest records the resulting maximum absolute matrix difference.

CUDA is selected only when the bundled PyTorch build contains the visible GPU
architecture. Unsupported newer GPUs fall back to real CPU inference rather
than producing an identity result after a CUDA kernel failure. Supported Grand
Challenge GPU architectures retain the original CUDA numerical path.

## Release verification (2026-08-20)

- Input: clinical selection case 001, three fragments
- Native and offline Docker inference both completed without identity fallback
- Native versus Docker maximum absolute 4x4 matrix difference:
  `3.95774841309e-05` (different bundled PyTorch CPU runtimes)
- Output contract validated: fragment IDs `1`, `101`, and `201`, each with one
  finite 4x4 homogeneous transformation matrix
- Network access was disabled for the Docker release run
- The final filesystem contains one checkpoint:
  `/opt/app/model_state_dict.pt` (143,395,235 bytes)
