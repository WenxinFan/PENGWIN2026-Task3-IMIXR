# Provenance and attribution

This research release is derived from the official PENGWIN 2026 Task 3
Fracture Reduction Baseline. The AssemblyTransformer backbone and the original
coordinate-regression/Kabsch formulation originate from that baseline.

The release adds the clinical training recipe, corrected clinical OBJ fragment
mapping, fragment-balanced and pose/rotation-geometry supervision,
rotation-stratified sampling, hard-pose augmentation, the train150 continuation,
and the SA75/LI75/RI50 deployment sampling policy described in
`docs/ALGORITHM.md`.

The PENGWIN training datasets are not redistributed. Users must obtain them
from the challenge organizers and comply with their terms. This repository does
not introduce a new license for upstream code; preserve and follow the license
and attribution terms of the official baseline and all third-party packages.
