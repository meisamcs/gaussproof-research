# Positioning GAUSSPROOF against Nasr et al. (USENIX Security 2023)

Primary source: Milad Nasr et al., [*Tight Auditing of Differentially Private Machine Learning*](https://www.usenix.org/system/files/usenixsecurity23-nasr.pdf), especially Sections 3, 5.1–5.4, and 6.4. This note interprets an **already completed** same-trajectory projection control; it does not claim to reproduce Nasr et al.'s complete audit or report a new training run.

| Aspect | Nasr et al. | Current GAUSSPROOF study |
| --- | --- | --- |
| Access | Every privatized update and model state in the white-box experiments. | Same trajectory interface for its white-box scores. |
| Target | An auditor-inserted gradient canary, or an inserted input-space canary whose gradient is recomputed; the strongest input-space canary is crafted to be distinguishable from the background. | A fixed, uncrafted held-out MNIST image and label, excluded from public fitting data; the auditor cannot modify the candidate or inject a chosen gradient. |
| Inclusion | A canary sampling rate `q_c`, which may differ from ordinary-record sampling; `q_c=1` isolates the noise mechanism. | A hidden Bernoulli participation schedule at controlled `q=0.5, 0.1, 0.02, 0.004`; high `q` is a deliberately favorable audit setting. |
| Primitive | Dot product of the privatized update and canary gradient. | The same raw dot-product control, plus a public-background residual and a Bernoulli-mixture Gaussian likelihood score. |
| Result | A mechanism-specific empirical privacy lower bound using f-DP/PLD and error-rate inference. | Holdout membership AUC, calibrated operating points, and closed-gallery retrieval. No empirical epsilon lower bound. |

For a known fixed Gaussian mean shift `s_t`, zero background, and fixed `||s_t||`, the per-round log-likelihood ratio is an affine function of Nasr et al.'s dot product. Our `q`-aware score applies `log(1-q+q*exp(llr_t))` to model the hidden inclusion indicator. These are standard Gaussian hypothesis tests, not a newly discovered projection attack. Our public-background approximation does not make the score an exact privacy-loss variable for full DP-SGD training.

The closest **same-observation empirical control** is `trajectory_raw_alignment` in `reports/trajectory_endpoint_replication/{original_lr,high_utility}/summary.csv`. It computes the mean of `sum_t <h_t,y_t>` over the trajectory. The method name in the CSV is retained for reproducibility; the paper calls it a *Nasr-style projection control*. It does not reproduce Nasr et al.'s canary crafting, threshold-selection, f-DP/PLD conversion, or empirical epsilon guarantee.

At `q=0.5`, `T=128`, and noise multiplier `sigma=4`, the completed 80-identity holdout gives:

| Learning rate | Public accuracy | GAUSSPROOF `q`-aware AUC | Nasr-style projection AUC | Final-model LiRA AUC |
| --- | ---: | ---: | ---: | ---: |
| 0.05 | 0.244 | 0.810 | 0.784 | 0.693 |
| 0.005 (exploratory) | 0.662 | 0.818 | 0.831 | 0.830 |

For the higher-utility condition, the identity-paired bootstrap interval for `q`-aware minus projection AUC is `-0.013 [-0.051, 0.025]` (`reports/trajectory_endpoint_replication/high_utility/paired_differences.csv`). The score has **no demonstrated advantage** over the established projection primitive. At ordinary-record proxy `q=0.004`, the separate 20-identity experiment finds trajectory AUC 0.503 at 128 releases; that experiment did not include the 80-identity projection/strong-endpoint replication.

## What a full matched Nasr audit would require

Fix **one** neighboring pair `D` and `D∪{z}` before evaluation, including the same candidate, initialization rule, trainer, clipping, sampling rate, noise multiplier, and released interface. Run independent H0 and H1 trials with fresh batch and Gaussian randomness. Compare Nasr's projection statistic and the `q`-aware statistic on those **same** releases. Preselect thresholds on separate calibration trials, estimate false-positive and false-negative probabilities on untouched trials, and derive a confidence-qualified lower bound using a mechanism-appropriate f-DP/PLD procedure. Repeat separately for an ordinary fixed input canary and a crafted input canary, because the latter grants the auditor extra power. Report the actual sampler and accounting upper bound. Cross-identity AUCs from the present experiment are not repeated trials on a fixed neighboring pair and must never be converted to an empirical epsilon lower bound.
