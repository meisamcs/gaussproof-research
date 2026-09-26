# Unseen-canary fingerprint calibration

This experiment asks whether a fingerprint detector calibrated on some known
canaries transfers to different canaries. It is the cleanest next check after
the paired-seed audit because it tests calibration generalization without
claiming unknown-image reconstruction.

## Design

The run uses one public pretrained `SmallCNN`, 16 update rounds, batch size 8,
clip norm `C=1`, learning rate 0.05, and noise multipliers
`sigma ∈ {0.25, 1, 4}`. The MNIST rows are split by row ID into pretraining,
public background, calibration canaries, holdout canaries, population, and test
roles. The four calibration identities and four holdout identities are disjoint.

For each sigma, a positive trajectory includes a given canary independently
with probability 0.5 at each round. A negative trajectory never includes that
canary. All other batch records come from the population role. Positive and
negative trajectories use paired seeds, so the hidden batch and Gaussian-noise
draws are matched as closely as the membership intervention permits.

At each pre-update checkpoint the auditor forms the canary's clipped gradient
`h_t` and a public background mean `b_t`. The released observation is

\[
 y_t = b_t + \frac{1}{B}(h_t-b_t) I_t + \eta_t,
 \qquad \eta_t\sim N(0,(\sigma C/B)^2 I).
\]

The primary score is the sum over rounds of the Gaussian matched score

\[
 \ell_t = \frac{(y_t-b_t)^T(h_t-b_t)}{\tau^2}
          - \frac{\|h_t-b_t\|^2}{2\tau^2},
 \qquad \tau=\sigma C/B.
\]

Thresholds are selected using calibration canaries only, at nominal 1% and 5%
false-positive rates. The reported holdout metrics then use the four unseen
canaries. `last_llr` is a final-release control; `max_llr` is a scan-over-time
control. `sequence_projection` is a projection-only control.

## Results

The holdout contains 48 positive and 48 negative trajectories per sigma. The
small sample gives coarse low-FPR resolution, so the table is descriptive.

| sigma | sequence AUC | final-release AUC | sequence TPR at calibrated 5% | realized holdout FPR |
|---:|---:|---:|---:|---:|
| 0.25 | 0.929 | 0.644 | 0.563 | 0.021 |
| 1 | 0.884 | 0.559 | 0.375 | 0.042 |
| 4 | 0.609 | 0.520 | 0.167 | 0.083 |

The sequence score transfers from calibration identities to unseen identities
at weak and moderate noise, while the final release is close to chance. At
`sigma=4`, transfer is weak and approaches chance. This is evidence for a
persistent, calibratable known-fingerprint signal, not evidence that a
diffusion model can reconstruct an arbitrary private image.

The accounting in this laboratory uses conservative replace-one zCDP without
sampling amplification. The corresponding epsilon upper bounds are 665.6,
70.4, and 11.6 for sigma 0.25, 1, and 4 at delta `1e-5`; these are weak privacy
settings. A stronger DP claim requires a separately audited subsampled
accountant and a larger holdout experiment.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=.cache/matplotlib \
python -m gaussproof.canary_holdout --data /path/to/mnist_train.csv \
  --config configs/canary_holdout.json --output runs/canary_holdout
```

The package exports aggregate CSVs, a vector plot, split identities, and a
completion manifest. Private trajectory arrays remain local and are not part of
the report export.
