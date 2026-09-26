# Learned denoising of canary trajectories

Fourfold identity cross-fitting compares a BiLSTM and conditional temporal
diffusion model with analytic fingerprint scores on four unseen MNIST canaries.
All methods use the same 16-round white-box trajectory information.

| sigma | sum LLR | mixture LLR | BiLSTM | schedule diffusion | final release |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.929 | 0.903 | 0.900 | 0.944 | 0.644 |
| 1 | 0.884 | 0.857 | 0.864 | 0.852 | 0.559 |
| 4 | 0.609 | 0.610 | 0.610 | 0.566 | 0.520 |

The diffusion-minus-sum-LLR AUC difference is +0.015 at sigma 0.25, with a
paired 95% bootstrap interval of -0.003 to +0.036. At sigma 1 the difference is
-0.032, and at sigma 4 it is -0.043. Thus the small low-noise gain is not yet
resolved against the strongest analytic baseline, and diffusion does not recover
the high-noise signal. The BiLSTM never materially improves on the analytic
score.

The learned models reconstruct or classify a known canary's inclusion trajectory.
They do not reconstruct unknown gradients or MNIST images. The audit is
white-box, the privacy bounds are weak, and increasing noise reduces every
method toward chance.

Cross-fitted learned scores have fold-dependent scales, so their pooled
low-FPR thresholds are exploratory. The primary result is holdout AUC with the
paired AUC-difference analysis.

See the [protocol](../../docs/canary_learned_protocol.md),
[summary CSV](summary.csv), [paired AUC differences](paired_auc_differences.csv),
[comparison plot](learned_comparison.png), and
[paired uncertainty plot](auc_difference.png).
