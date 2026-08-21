# PENGWIN 2026 Algorithm Description

## Forum topic title

**[Algorithm Description] Task 3 x Team**

## 1. Task

Task 3: Pelvic Fracture Reduction Planning.

## 2. Team name


## 3. Authors


## 4. Affiliations


## 5. Contact author and email address


## 6. Algorithm name or title

**AssemblyTransformer with Fragment-Balanced Coordinate Regression and Rotation-Geometry Calibration (train150-e4 + SA75/LI75/RI50)**

## 7. Method description

Our method formulates pelvic fracture reduction as point-wise coordinate regression followed by rigid-pose recovery. The input is a clinical OBJ file containing fragments of the sacrum (SA), left ilium (LI), and right ilium (RI). Each mesh fragment is identified from its OBJ object/group ID, and surface points and face normals are sampled from all fragments. The sampled coordinates are centered using the bounding-box midpoint and scaled by the maximum point radius.

Each point is represented by sinusoidal encodings of its 3D coordinate and surface normal, together with a learned fragment-ID embedding. A 12-layer AssemblyTransformer alternates attention within each fragment and attention across all fragments in the same case. It predicts the reduced 3D coordinate of every sampled point. For each fragment, a proper rigid transformation is recovered from the input and predicted coordinates using SVD-based least-squares fitting. All fragment poses are expressed relative to the first sacral fragment.

At inference, 3,000 surface points are sampled per bone using volume-proportional allocation across fragments and deterministic random surface sampling (seed 42). Minimum per-fragment allocations are 75 points for SA, 75 for LI, and 50 for RI; excess points are taken from the largest-volume fragment so the per-bone total remains exactly 3,000. The network is applied once. The recovered rotations are then mildly calibrated by scaling their SO(3) rotation vectors by 0.975 while preserving each transformed fragment centroid. The final output is one 4 x 4 rigid transformation matrix per fragment in `reduction-poses-matrices.json`.

## 8. Main technical contributions and components

We retained the public baseline's AssemblyTransformer backbone and coordinate-to-SVD formulation; these are not claimed as our contributions. Relative to the public baseline, our main changes are:

1. **Full-parameter clinical adaptation:** instead of the baseline's LoRA clinical adaptation, we fine-tune the complete network on the organizer-provided clinical cases and their expert reduction poses.
2. **Fragment- and pose-aware supervision:** we replace the plain point-averaged coordinate MSE with fragment-balanced coordinate regression, add differentiable translation and quaternion pose consistency, and introduce a dense translation-invariant rotation-geometry loss with weight 0.1.
3. **Difficulty-aware clinical training:** rotation-stratified sampling balances easy, medium, and hard cases; hard-pose augmentation increases LI/RI pose variation; and small-fragment-aware point allocation increases the representation of fragments that would otherwise be underweighted.
4. **Expanded clinical refinement with controlled selection:** the final continuation uses 150 clinical cases while retaining a disjoint 20-case set for checkpoint selection under the actual deployment protocol.
5. **Calibrated single-step deployment:** inference is fixed to one pose-update step and uses a selected 3,000-point volume/random sampling protocol with bone-specific minimum fragment allocations (SA75/LI75/RI50), plus centroid-preserving SO(3) rotation calibration (`rot_alpha = 0.975`). No additional geometric reranking or model ensemble is used.

## 9. Complete pipeline

1. Read the clinical OBJ and recover the SA, LI, and RI fragment IDs.
2. Sample 3,000 surface points per bone with SA75/LI75/RI50 minimum fragment allocations and obtain their face normals.
3. Center and scale the complete fractured point cloud.
4. Encode coordinates, normals, and fragment identities.
5. Predict reduced point coordinates with the AssemblyTransformer.
6. Fit one rigid transformation per fragment using SVD.
7. Express all poses relative to the first sacral fragment.
8. Apply the centroid-preserving rotation calibration (`rot_alpha = 0.975`).
9. Transform the poses back to the original physical coordinate system and write the final JSON file.

## 10. External data

No external data were used. Both the simulation data used for pretraining and the clinical cases used for fine-tuning were supplied as part of the PENGWIN 2026 Task 3 training data.

## 11. Externally pretrained models

No externally pretrained model was used. The network was initialized from our own AssemblyTransformer checkpoint trained from scratch on the organizer-provided PENGWIN simulation data (simulation epoch 294), and was then fully fine-tuned on the PENGWIN clinical data. No ImageNet, foundation-model, or third-party anatomical-model weights were used.

## 12. Preprocessing

The OBJ object/group labels are mapped to the three pelvic bones according to the Task 3 fragment-ID convention. Surface points and their corresponding face normals are sampled from every valid fragment. During deployment, a total of 3,000 points is assigned to each bone in proportion to fragment volume, subject to minimum per-fragment allocations of 75 points for SA, 75 for LI, and 50 for RI. The full case is recentered using the midpoint of its axis-aligned bounding box and isotropically rescaled by its maximum point radius. The same normalization is inverted after pose estimation so that all output matrices are in the original OBJ coordinate system.

## 13. Data augmentation

Clinical fine-tuning used regional point dropout with probability 0.1, Gaussian coordinate jitter with standard deviation 0.002 in normalized space, and hard-pose perturbations with probability 0.25 for LI/RI fragments. Hard-pose perturbations used rotations up to 35 degrees and translations up to 30 mm while preserving the anchor fragment. Training sampled 1,800 points per bone using an equal/even base pattern, with a 1.8-fold sampling boost for fragments below the 35th surface-area percentile. Cases were also sampled with easy/medium/hard rotation proportions of 40%/30%/30%, defined from the ground-truth rotation relative to the first sacral fragment.

## 14. Training and validation strategy

Training followed a multi-stage schedule. First, the AssemblyTransformer was pretrained from scratch on the organizer-provided simulation set. It was then fully fine-tuned on clinical OBJ cases using the corrected fragment parser, clinical hard-pose augmentation, and rotation-stratified sampling. A rotation-geometry continuation stage introduced the auxiliary rotation-geometry loss.

For the final stage, the model was initialized from the clinical rotation-geometry epoch-19 checkpoint and trained for 15 additional epochs on 150 clinical cases. These 150 cases comprised the original 120-case training subset, 20 previously selected validation cases, and 10 previously selected hard cases. The remaining 10 validation cases and 10 hard cases were kept out of this final fitting stage and used as a fixed 20-case selection set. Checkpoints were evaluated with the deployment protocol (3,000 points, volume-proportional allocation, random surface sampling, and `rot_alpha = 0.975`), and the checkpoint recorded as epoch 4 was selected.

The final continuation used AdamW with a learning rate of 5e-6, weight decay 0.01, and a OneCycle learning-rate schedule. Training used two GPUs, mixed precision, batch size 2 per GPU, and gradient accumulation over 16 batches. The random seed was 42.

## 15. Loss functions

The total training objective is

$$
\mathcal{L}_{\mathrm{total}}
= \mathcal{L}_{\mathrm{coord}}
+ 0.2\,\mathcal{L}_{\mathrm{pose}}
+ 0.1\,\mathcal{L}_{\mathrm{rot\text{-}geom}}.
$$

Let $K$ be the number of fragments, $N_k$ the number of sampled points in fragment $k$, $\hat{\mathbf{y}}_{k,i}\in\mathbb{R}^3$ the predicted reduced coordinate, and $\mathbf{y}_{k,i}$ its ground-truth coordinate. The fragment-balanced coordinate loss is

$$
\mathcal{L}_{\mathrm{coord}}
= \frac{1}{K}\sum_{k=1}^{K}
\frac{1}{N_k}\sum_{i=1}^{N_k}
\frac{1}{3}\left\|\hat{\mathbf{y}}_{k,i}-\mathbf{y}_{k,i}\right\|_2^2.
$$

Thus, every fragment contributes equally regardless of its number of sampled points. A rigid pose $(\hat{\mathbf{t}}_k,\hat{\mathbf{q}}_k)$ is recovered differentiably from the input and predicted coordinates. The pose-consistency loss is

$$
\mathcal{L}_{\mathrm{pose}}
= \operatorname{SmoothL1}(\hat{\mathbf{t}},\mathbf{t})
+ \frac{1}{K}\sum_{k=1}^{K}
\left(1-\left|\left\langle\hat{\mathbf{q}}_k,\mathbf{q}_k\right\rangle\right|\right),
$$

where $\mathbf{t}_k$ and $\mathbf{q}_k$ are the ground-truth translation and unit quaternion. The absolute quaternion inner product makes the loss invariant to the equivalent representations $\mathbf{q}$ and $-\mathbf{q}$.

For the translation-invariant rotation-geometry loss, let $\mathbf{x}_{k,i}^{c}$ and $\mathbf{y}_{k,i}^{c}$ denote centered input and target points, $\hat{\mathbf{R}}_k$ the rotation recovered from the prediction, and

$$
r_k=\sqrt{\frac{1}{N_k}\sum_{i=1}^{N_k}\left\|\mathbf{x}_{k,i}^{c}\right\|_2^2}.
$$

The rotation-geometry term is

$$
\mathcal{L}_{\mathrm{rot\text{-}geom}}
= \frac{1}{K}\sum_{k=1}^{K}
\operatorname{SmoothL1}\!\left(
\frac{\hat{\mathbf{R}}_k\mathbf{x}_{k}^{c}}{r_k},
\frac{\mathbf{y}_{k}^{c}}{r_k}
\right).
$$

This term removes translation and normalizes fragment scale so that it directly supervises rotational alignment. At most 256 points per fragment are used for differentiable pose recovery and the rotation-geometry term. The relative-rotation loss is disabled in the selected model, i.e., its weight is zero.

## 16. Base network architecture

The base network is the unchanged public PENGWIN Task 3 AssemblyTransformer coordinate-regression architecture; we do not claim an architectural contribution. It contains 12 Transformer layers with embedding dimension 384, 8 attention heads, and no dropout. Each layer applies pre-normalized intra-fragment self-attention, inter-fragment self-attention, and a GEGLU feed-forward block. Coordinates and normals use multi-frequency sinusoidal encodings, and fragment identities use a learned 16-dimensional embedding. A two-layer MLP predicts one reduced 3D coordinate for every input point.

## 17. Ensembling during inference

No model ensemble was used. The submitted method uses one checkpoint and one deterministic inference pass (`tta = 1`, `max_iters = 1`).

## 18. Public repository URL


## 19. References

1. PENGWIN 2026 Challenge: <https://pengwin2026.grand-challenge.org/>
2. PENGWIN 2026 Task 3 dataset: <https://zenodo.org/records/19734383>
3. Vaswani, A. et al. *Attention Is All You Need.* Advances in Neural Information Processing Systems, 2017.
