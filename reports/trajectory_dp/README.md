# Actual DP-SGD trajectory denoising: measured pilot

Completed: 33 actual CNN training trajectories (2,112 DP-SGD rounds), nine private target trajectories, and six denoisers. Each denoiser trained for 1,200 steps; model selection used separate public calibration runs. The 8-release and full-history observers use identical underlying trajectories.

Target: clean clipped batch sum divided by expected batch size, not an individual record or original image. C=0.1, expected B=32, N=2,000, 64 rounds, delta=1e-5. The full-history observer uses every release available so far, without batch membership information.

| σ | ε upper bound | Diffusion | Diffusion context only | Gaussian current | Gaussian temporal |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.4655 | 0.03109 | 0.04355 | 0.01992 | 0.02859 |
| 4 | 0.1188 | 0.04911 | 0.05029 | 0.03617 | 0.03893 |
| 16 | 0.0245 | 0.06299 | 0.06311 | 0.04915 | 0.05595 |

Errors are full-vector squared error divided by C², averaged over 64 rounds and three target runs. Privacy bounds use Opacus RDP accounting with an expanded order grid, add/remove adjacency and Poisson sampling.

**Finding:** Diffusion does not outperform Gaussian current-observation shrinkage at any tested noise level. The added-current-observation benefit becomes very small at high noise; full history does not rescue diffusion in this pilot.

## Observation-dependent benefit

Positive gains below mean that adding the current release improves the diffusion estimate over its same-network context-only control. They do not establish membership leakage or superiority over Gaussian estimation.

- σ=1: 28.60% mean relative error reduction; absolute gains across three seeds [0.011776, 0.013099].
- σ=4: 2.33% mean relative error reduction; absolute gains across three seeds [0.000924, 0.001575].
- σ=16: 0.19% mean relative error reduction; absolute gains across three seeds [0.000087, 0.000142].

## Utility and limits

- σ=1: final public-test CNN accuracy 80.67% (three-run mean).
- σ=4: final public-test CNN accuracy 78.13% (three-run mean).
- σ=16: final public-test CNN accuracy 43.53% (three-run mean).

This pilot uses a rank-64 public representation. Every learned/Gaussian estimator retains the public fingerprint outside that space; full-dimensional errors and a projection oracle are supplied. The denoiser is an upstream Diffusers U-Net trained with a diffusion corruption schedule and clean-sample prediction, evaluated in one step. It is not an iterative posterior sampler.

Only six public training trajectories and three private evaluation trajectories per noise setting were used. The U-Net treats temporal lags as channels and PCA coefficients as an arbitrary grid; performance is not a bound on other architectures. The Gaussian temporal ridge baseline also has limited training data. The strongest tested comparator must include Gaussian current-observation shrinkage.

Increasing σ changes the learned CNN and the clean gradients as well as the noise. Cross-σ error comparisons are not controlled demonstrations that more noise creates more leakage. Within-condition comparisons and current-release ablations are the appropriate evidence here.

This experiment is causal: estimates at round t cannot use rounds after t. An offline all-round smoothing attack is not evaluated. Private clean-gradient logs and batch sizes remain in ignored local runs, excluded from this report.

See ../../docs/trajectory_dp_protocol.md for reproduction and access assumptions.
