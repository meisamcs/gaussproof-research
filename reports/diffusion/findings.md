# Measured diffusion-denoiser pilot

Trained nine upstream Diffusers U-Nets: three seeds × epsilon/sample/velocity objectives. Each model had 2,000 optimizer steps, with the checkpoint chosen on separate calibration data. Evaluated 12,960 method/condition/target measurements. The 24-test suite passed, including exact upstream noise conversion, clipping before averaging, saved-model reload and inference without clean targets.

Example slice: clipping C=0.1, individual gradients B=1. Values are full-vector squared error divided by C² and by the number of checkpoints; lower is better. Requested noise is rounded to a nearby scheduler level and actual values are recorded in CSVs.

| Method | τ≈0.01 | τ≈0.1 | τ≈1 |
| --- | ---: | ---: | ---: |
| Prior only | 0.5462 | 0.5462 | 0.5462 |
| Gaussian shrinkage | 0.3457 | 0.5399 | 0.5461 |
| DDPM noise prediction | 0.3437 | 1.4123 | 59.1627 |
| Direct clean prediction | 0.3574 | 0.5426 | 0.5465 |
| DDPM velocity prediction | 0.3453 | 0.5429 | 0.5461 |
| Velocity DDIM iterative | 0.3729 | 0.7047 | 0.7382 |

For this executed pilot, explicit epsilon prediction has competitive error at the lowest noise in this slice but deteriorates badly at higher noise. Direct and velocity predictions are substantially more stable. At the largest noise they approach the prior-only error rather than demonstrating target-specific recovery. Iterative DDIM does not reliably improve squared-error estimation. The paired CSV includes Gaussian comparisons and intervals; tiny mean differences should not be read as an established neural advantage.

Use the velocity implementation as a stable diffusion-style starting point, while retaining Gaussian shrinkage and prior-only controls. The experiment does not establish that diffusion beats Gaussian shrinkage, that noise can be predicted precisely enough to remove high-noise privacy protection, or that more noise increases vulnerability.

This is a limited laboratory study: fixed public checkpoints, public PCA representation, known label histograms, 30 held-out images per seed, and a small training budget. It does not rule out improvements from richer data, architectures, longer trajectories or better conditioning. The reliable upstream components and tested conversion equations should be distinguished from the experimental gradient representation and fitted model's efficacy.

See [the full protocol](README.md), `summary.csv`, `noise_prediction.csv`, and `paired_vs_gaussian.csv`. Raw per-target records and all trained weights remain in the local run directory; the separate model bundle includes public CNN checkpoints and gradient adapters required for inference.
