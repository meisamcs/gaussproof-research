# Fingerprint access held fixed: noise versus recoverability

Used 90 real clipped MNIST gradient templates from 3 public CNN seeds and 2 public checkpoints. Access, templates, labels, hypotheses, and 16 releases per checkpoint are fixed across all noise levels. Each template has 20,000 Monte Carlo observations per hypothesis per noise level. Computation uses exact Gaussian projected sufficient statistics, not costly generation of every full noise vector. This is a controlled Gaussian experiment, not dynamic DP-SGD or a new empirical dataset of independent people.

The candidate is a known audit canary: its fingerprint is available under BOTH hypotheses; only whether it contributed is unknown. The background is known/subtracted. H1 assumes presence at every release, with no hidden sampling. This isolates the strongest matched-fingerprint detector for a specified simple Gaussian alternative. Wrong same-label templates and reversed checkpoint order may retain signal because gradients are correlated; they are not automatically chance-level controls. Orthogonal directions have no signal. The energy-only detector's power is computed analytically from a noncentral chi-square distribution.

| σ | ε upper bound | Matched TPR | Optimal TPR | Denoising MSE |
| --- | ---: | ---: | ---: | ---: |
| 0.5 | 364.578 | 0.8213 | 0.8213 | 0.0374 |
| 1 | 118.289 | 0.6794 | 0.6792 | 0.0713 |
| 2 | 43.145 | 0.3647 | 0.3648 | 0.1352 |
| 4 | 17.572 | 0.0974 | 0.0975 | 0.2026 |
| 8 | 7.786 | 0.0335 | 0.0335 | 0.2354 |
| 16 | 3.643 | 0.0184 | 0.0183 | 0.2461 |
| 32 | 1.759 | 0.0136 | 0.0135 | 0.2490 |
| 64 | 0.864 | 0.0117 | 0.0116 | 0.2497 |
| 128 | 0.428 | 0.0108 | 0.0108 | 0.2499 |

Observed FPR is saved separately. Thresholds are analytic null thresholds, never selected on evaluation labels. Each noise level reuses standard-normal draws; different noise conditions are paired, not independent replications. Exact theoretical power is included to distinguish genuine monotonicity from Monte Carlo fluctuations.

## A bound that can actually be proved

For a fixed concatenated fingerprint H, let E=||H||², tau=σC/√R, and μ=√E/tau. The normalized matched score is N(0,1) under absence and N(μ,1) under presence. By the likelihood-ratio test, the optimal power at FPR α is Φ(μ−Φ⁻¹(1−α)).

For α < β, requiring optimal TPR ≤ β is equivalent, in THIS model, to

σ ≥ √(R E) / [ C (Φ⁻¹(1−α) + Φ⁻¹(β)) ].

With α=0.01 and β=0.05, the denominator's quantile factor is approximately 0.6815. Across the tested bank the largest required noise multiplier is **8.3007**. This is an exact model-specific noise floor for the stated detection cap. It is not a universal lower bound for arbitrary DP-SGD attacks, unknown backgrounds, or arbitrary priors.

For equal-prior binary amplitude a∈{0,1}, the Bayes posterior mean minimizes squared error. Its posterior is sigmoid(μZ−μ²/2). ANY estimator's prior-averaged amplitude MSE obeys

MSE ≥ ½ Φ(−μ/2).

Proof: conditional squared-error risk is minimized by p, with risk p(1−p) ≥ ½ min(p,1−p); the optimal equal-prior Gaussian classification error is Φ(−μ/2). Multiply by E/K for the corresponding per-checkpoint trajectory MSE bound. This is a Bayes-risk bound, not a pointwise guarantee for each possible amplitude. At infinite noise it approaches the no-release risk 1/4. The independent-background-free, fixed-template setup is essential.

Increasing noise tightens the reported conservative privacy bound and makes the optimal attack worse. Empirical failure of one algorithm alone would not prove this; the Gaussian likelihood calculation supplies the mathematical statement. A minimum noise requirement corresponds to a maximum allowable privacy-budget bound for fixed sensitivity/releases, not a minimum ε needed for privacy. Epsilon in the table is an upper bound from replace-one Gaussian composition, not an estimate of exact leakage.

The theory is a specialization of Gaussian hypothesis testing; it is not a new GAUSSPROOF theorem. A research contribution would establish comparable guarantees for changing, imperfectly accessible fingerprints and hidden participation, or demonstrate a useful efficient decoder in that setting.

Reference: [Dong, Roth, Su, Gaussian Differential Privacy](https://arxiv.org/abs/1905.02383).

Reproduce after the reconstruction run:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.cache/matplotlib python scripts/fingerprint_noise_sweep.py --source runs/reconstruction --output reports/fingerprint_noise
```
