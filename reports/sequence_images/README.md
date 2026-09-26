# Full-sequence gradient and image reconstruction pilot

Implemented and executed an offline bidirectional LSTM, an iterative conditional temporal diffusion model, and a shared convolutional generator producing two MNIST images per round. Both sequence models can use later releases when reconstructing earlier rounds. No private labels or batch identities are provided to them.

Configuration: C=1.0, B=2, 16 rounds, sigma=[0.05, 0.5]. Per condition: 48 public training sequences, 12 public validation sequences, 8 held-out private sequences. The image generator uses both public noise conditions. These small-noise settings are feasibility tests, not strong-privacy claims.

| Method | σ=0.05 pixel MSE | σ=0.5 pixel MSE |
|---|---:|---:|
| clean oracle | 0.04900 | 0.05282 |
| gaussian current | 0.05288 | 0.06979 |
| gaussian smoother | 0.07273 | 0.07525 |
| bilstm | 0.05474 | 0.07738 |
| diffusion mean | 0.05847 | 0.07981 |
| diffusion draw | 0.06997 | 0.09261 |
| context only | 0.07494 | 0.07579 |
| mean image | 0.06579 | 0.06579 |
| diffusion shuffled sequence | 0.08586 | 0.08265 |
| noisy | 0.05684 | 0.11089 |

Pixel errors optimally match the two outputs to the two target images after inference. All 256 target batches (512 image occurrences) are scored; the gallery uses predeclared examples. The clean oracle passes the true **projected** clipped gradient through the same image generator; it is not an attack.

## Paired observations

- σ=0.05, diffusion-mean gain over context_only: +0.01647 MSE (positive is better; run range +0.00662 to +0.02121).
- σ=0.05, diffusion-mean gain over gaussian_current: -0.00559 MSE (positive is better; run range -0.00631 to -0.00488).
- σ=0.05, diffusion-mean gain over bilstm: -0.00373 MSE (positive is better; run range -0.00622 to -0.00197).
- σ=0.05, diffusion-mean gain over clean_oracle: -0.00947 MSE (positive is better; run range -0.01170 to -0.00711).
- σ=0.5, diffusion-mean gain over context_only: -0.00402 MSE (positive is better; run range -0.00868 to +0.00590).
- σ=0.5, diffusion-mean gain over gaussian_current: -0.01002 MSE (positive is better; run range -0.01309 to -0.00578).
- σ=0.5, diffusion-mean gain over bilstm: -0.00243 MSE (positive is better; run range -0.00490 to -0.00109).
- σ=0.5, diffusion-mean gain over clean_oracle: -0.02699 MSE (positive is better; run range -0.03526 to -0.02339).

## Interpretation limits

- Public classifier held-out accuracy: 82.7%. Reported digit agreement is a diagnostic, not a perfect recognition metric.
- PCA retains 128 directions; all recovered full gradients use public fingerprints outside that space. The image generator is trained on public clean projected gradients, so its response to imperfect estimates may suffer distribution shift.
- The conditional diffusion adapter is a custom temporal Transformer with official Diffusers DDPM/DDIM scheduling and velocity prediction. It is not a reproduction of RAoPT or CSDI. Sampling uses 40 reverse steps and four draws; sample averaging occurs in gradient coordinates before image generation.
- The generator is deterministic and trained with a permutation-invariant pixel loss. Ambiguous gradients can yield blurry conditional-average images. Good-looking outputs alone do not establish recovery of particular private examples.
- There are only 48 public training trajectories per condition. Runs share a role-specific data pool and a public initial CNN; run standard deviations are descriptive, not independent-dataset confidence intervals.
- The conservative privacy bounds are large for these settings and do not use subsampling amplification. Do not extrapolate this pilot to tiny epsilon or claim that added noise increases leakage.
- Paired noisy and clean gradients and original target images are evaluation-only laboratory logs. Saved inference accepts only noisy releases and public fingerprints.

## Artifacts

- `batches_sigma0.05.png` / `.pdf`, `batches_sigma0.5.png` / `.pdf`: fixed image galleries.
- `summary.csv`, `per_run.csv`, `measurements.csv`: all measurements.
- `image_errors.pdf`: vector comparison plot.
- `privacy.csv`: conservative per-run privacy bounds.
- `../../docs/sequence_images_protocol.md`: setup, access assumptions, and reproduction.
- `../../scripts/reconstruct_batches.py`: saved-model inference without target truth.
