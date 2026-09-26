# Measured reconstruction results

Executed 3 seeds, 3 noise levels, 3 release counts, and known/unknown batch backgrounds. Saved 12,150 gradient measurements, 1,620 identity measurements, and 54 image measurements. These counts include repeated methods and conditions, not independent samples.

Gradient error per checkpoint, known background, R=64:

| Noise σ | Prior only | Gaussian prior | Neural prior |
| --- | ---: | ---: | ---: |
| 0.25 | 0.3647 | 0.1865 | 0.2018 |
| 1 | 0.3647 | 0.2708 | 0.2459 |
| 4 | 0.3647 | 0.3527 | 0.3453 |

At σ=1, the neural estimator improves over Gaussian shrinkage by 0.0249 squared-error units (paired hierarchical 95% bootstrap interval 0.0189–0.0312). This is a pilot result with only three independently trained models. The neural method is not uniformly best across noise conditions.

Pixel reconstruction at σ=1, R=64; lower MSE is better:

| Method | Pixel MSE |
| --- | ---: |
| mean | 0.1826 |
| prior_only | 0.1315 |
| gaussian | 0.1015 |
| neural | 0.1026 |
| public_mean_image | 0.0609 |
| clean_oracle | 0.1125 |

The tested subset contains digits [0, 1, 2], with 3 fixed images per seed. It is not a balanced ten-digit reconstruction evaluation. Learned-prior inversion does not beat the public class-mean image on average. Gradient denoising therefore does **not** establish successful target-specific image reconstruction here. The clean-gradient oracle provides clean attack input, not a globally optimal inversion: several optimizations fail, so its average error is not a theoretical lower bound. All failures remain in the results and gallery.

The known-background release budget at this displayed condition has a conservative replace-one composition bound ε=364.58 at δ=1e-05. The experiment does not establish reconstruction under a strong small-ε privacy guarantee. Repeating releases spends privacy budget; larger σ alone does not define the overall privacy level.

The supported message is that a transferable public gradient prior can improve estimation of unseen gradients beyond prior-only guessing in this controlled channel. The evidence does not support “excess noise backfires,” a successful novel-image reconstruction claim, or superiority over the matched exact-bank likelihood under identical information. The original MIA pilot is preserved separately.

Next research gates: calibrate a more reliable inversion procedure on public holdout images, lock it before a new unseen evaluation, expand pixel tests across all digits, and then test changing DP-SGD checkpoints and hidden participation under an explicit release/privacy budget. Treat these as new experiments rather than tuning on the nine reported targets.

---

# Controlled gradient reconstruction experiment

These are measured MNIST reconstructions from repeated Gaussian releases at **fixed public CNN checkpoints**, not a DP-SGD training run. Labels are known. Public model training, prior fitting, calibration, targets, and distractors use disjoint records. Targets are balanced across labels when the configured count permits it; inversion uses the first fixed targets, never selection on success.

`gradient_summary.csv` reports per-checkpoint squared gradient error, normalized error, cosine similarity, and paired gain over prior-only estimates. `identity_summary.csv` evaluates an exact target-containing bank separately. `image_summary.csv` and `image_paired_summary.csv` report pixel MSE/PSNR and paired improvement over the public class-mean image. Lower errors and higher gains are better. Neural means a public-data latent denoiser; Gaussian means empirical low-rank covariance shrinkage. Neither is the original sparse MIA decoder.

Intervals are 95% hierarchical percentile bootstrap intervals, resampling seeds then target records. Three seeds and nine inverted images support only pilot conclusions. Conditions reuse paired records and noise; they are not independent replications. Plots use absolute errors, not only percentage reductions. Shuffled releases retain labels in the full configured run. Unknown backgrounds are actual sampled public gradients, but covariance-based estimators approximate their distribution diagonally in the learned basis. Stale priors repeat the first checkpoint's fingerprints.

Image inversion receives the estimated gradient, known label, and public checkpoints. All estimators use the same random image initializations, inversion steps, and restarts. The clean-gradient oracle checks inversion capacity. Public mean images and prior-only gradients check plausible class reconstructions without any target release. Gallery originals are public MNIST examples used as experimental targets.

More releases incur more privacy cost. Raw records include a conservative replace-one Gaussian-composition epsilon for K×R releases; no subsampling amplification is claimed. Large noise cannot improve the optimal attack when all other information is fixed. A learned prior beating noisy averaging is evidence of denoising, not a violation of DP, and does not establish superiority over an optimal likelihood attack with the same prior. Exact-bank retrieval is identification, not unseen-image reconstruction.

The RAoPT paper motivates learning a reconstruction prior from public clean/privatized pairs; its empirical GPS results do not establish a result for gradients: https://arxiv.org/pdf/2210.09375 . The reconstruction-likelihood comparison is informed by https://arxiv.org/pdf/2302.07225 . Full dynamic DP-SGD, hidden labels/participation, architecture transfer, and formal privacy auditing remain outside this experiment.
