# Exact high-noise trajectory-length audit

This experiment tests the most promising GAUSSPROOF mechanism after learned
diffusion failed to improve the high-noise detector: repeated fingerprint
evidence can accumulate across many noisy DP-SGD releases.

## Design

The audit fixes the noise multiplier at `sigma=4`, clip norm at `C=1`, batch
size at 8, and the known-canary inclusion probability at 0.5. It uses the four
unseen holdout canaries from the earlier calibration experiment, with 12 paired
positive and negative trajectories per identity. Each CNN follows one exact
128-round evolving DP-SGD trajectory. Scores are evaluated on nested prefixes
of 16, 32, 64, and 128 releases, so every longer comparison contains the same
earlier observations and continues from the corresponding noisy model state.

The auditor receives the noisy gradient, the canary's clipped gradient at the
current checkpoint, and a public background estimate. It never receives the
hidden inclusion schedule. The primary score is the cumulative Gaussian matched
fingerprint statistic. A Bernoulli-mixture likelihood uses the known 0.5
inclusion probability as a second analytic control.

Uncertainty is estimated with 2,000 paired cluster-bootstrap repetitions that
resample canary identities and positive/negative sequence indices together.
Only four identities are available, so the intervals are descriptive rather
than population-level guarantees.

## Results

| releases | sum-LLR AUC | paired gain from 16 | 95% interval for gain | epsilon upper bound |
|---:|---:|---:|---:|---:|
| 16 | 0.609 | 0.000 | [0.000, 0.000] | 11.60 |
| 32 | 0.675 | +0.067 | [+0.010, +0.124] | 17.57 |
| 64 | 0.768 | +0.159 | [+0.094, +0.218] | 27.19 |
| 128 | 0.819 | +0.211 | [+0.144, +0.272] | 43.14 |

The 16-round prefix exactly reproduces the earlier holdout AUC. The longer
prefixes show that a fingerprint too weak for reliable detection in one short
high-noise window becomes much more visible after repeated releases.

For approximately independent per-round evidence with mean gap `Delta` and
variance `v`, summing `T` matched scores produces mean gap `T*Delta` and
standard deviation proportional to `sqrt(T*v)`. Separation therefore grows as
`sqrt(T)`. The matched filter runs in polynomial time, roughly `O(Td)` for `T`
releases and `d` observed gradient coordinates; diffusion is not required for
this effect.

This does not mean that increasing noise makes a fixed mechanism less private.
The conservative replace-one zCDP cost grows linearly with the number of
releases, and the table reports the corresponding no-amplification epsilon
bound. The result shows why a noise multiplier cannot be interpreted without
trajectory length and composition. It supports a repeated-release warning, not
a universal lower bound on sigma.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=.cache/matplotlib \
python -m gaussproof.canary_long_trajectory \
  --source runs/canary_holdout --data /path/to/mnist_train.csv \
  --config configs/canary_long_trajectory.json \
  --output runs/canary_long_sigma4
```

The output contains per-trajectory prefix scores, cluster-bootstrap summaries,
privacy composition, a completion manifest, and PDF/PNG figures.
