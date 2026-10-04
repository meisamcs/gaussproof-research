# Matched noise-multiplier and clipping sweep

This study extends the fixed-natural-pair audit in
[`reports/nasr_lira_integration`](../reports/nasr_lira_integration/README.md).
It asks how a candidate-conditioned trajectory score behaves as the DP-SGD
noise multiplier and clipping norm change, while also measuring final-model
utility and two same-pair controls. It is a **single fixed-neighbor audit**
under a Bernoulli privileged slot, not an ordinary uniform-minibatch or
population-average membership benchmark.

The public MNIST checkpoint, common population (2,000 records), background
estimator (64 public records), candidate (test record 46993), and replacement
record (test record 24898) are fixed before the sweep. At each round, the
special slot is used with probability `q=0.5`; it contains the candidate in
world H1 and the replacement record in world H0. If unused, all eight batch
records come from the common population. If used, seven come from that
population. H0 and H1 use the same random stream within a pair of trials,
and the same pair seeds are reused across grid cells. All released gradients
are computed from the **actual evolving model state**, with global per-record
clipping before Gaussian noise is added.

The prespecified grid is `C = {0.1, 1, 4}` and `sigma = {2, 4, 8, 16, 32}`.
An additional low-epsilon tail at `C=1, sigma={64,128}` reaches conservative
128-step upper bounds of about 1.76 and 0.86. It is analyzed separately
because the full clipping grid would be costly and the expected signal at
these multipliers is small.
An additional weak-clipping control at `C=16` uses the five main noise
multipliers. Its initial public-gradient sample has no clipped records,
whereas the `C=0.1` and `C=1` groups begin with nearly all records clipped.
This control is analyzed separately from the main grid.
Because very high noise can destroy the CNN before privacy is interpretable,
three further controls use larger batches at the same `q=0.5`: `B=32,
C=1, sigma=32`, `B=64, C=1, sigma=64`, and `B=128, C=1, sigma=128`.
This reduces the absolute
noise on the averaged update while also diluting the privileged record's
contribution. The conservative record-level bound below is unchanged because
both the sensitivity and Gaussian noise scale as `1/B`. These are utility
checks, not points in the prespecified `B=8` clip/noise grid.
The learning rate is `0.005`, and each 128-step run also supplies its 64-step
prefix. Noise on the averaged gradient has standard deviation
`sigma*C/8`. The no-amplification replace-one bound is

`rho = 2*T/sigma^2; epsilon_upper = rho + 2*sqrt(rho*log(1/delta))`

at `delta=1e-5`. It is deliberately conservative and independent of `C`:
the maximum replace-one sensitivity `2*C` and the Gaussian noise standard
deviation both scale with `C`. The realized attack signal and model utility
may change with `C`, because clipping changes the update direction and norm.
The study does not claim that larger noise worsens formal privacy.

| `sigma` | Upper `epsilon`, 64 updates | Upper `epsilon`, 128 updates |
|---:|---:|---:|
| 2 | 70.39 | 118.29 |
| 4 | 27.19 | 43.14 |
| 8 | 11.60 | 17.57 |
| 16 | 5.30 | 7.79 |
| 32 | 2.52 | 3.64 |
| 64 (tail only) | 1.23 | 1.76 |
| 128 (tail only) | 0.61 | 0.86 |

For every cell, 30 paired seeds per world calibrate a threshold at nominal
5% FPR; 80 new paired seeds per world evaluate it. No holdout world labels
are used to choose scores or thresholds. The primary GAUSSPROOF score is a
sum of per-round Bernoulli-mixture log density ratios computed from
recalculated, clipped gradients of both natural records and a public
background mean. Random minibatches make this an approximate likelihood,
but it is a valid prespecified test statistic. Controls are raw cumulative
gradient alignment (Nasr-style same-access projection), the final noisy
update, and the final CNN checkpoint's candidate-minus-replacement
correct-label log probability. The final-checkpoint control sees only the
endpoint, not intermediate releases. It is **not** LiRA, which would require
fresh matched OUT reference models for each grid cell.

Every row reports tie-aware held-out AUC, a 95% interval centered on the
observed statistic using the paired-seed bootstrap standard error,
TPR and FPR at the calibration threshold, public-query accuracy on 256
separate MNIST records, actual batch clipping fraction, and positive-world
appearances. A one-event empirical epsilon lower bound uses a conservative
lower binomial confidence bound for TPR and upper bound for FPR. Both
per-comparison and Bonferroni familywise bounds are exported. These are
lower bounds for **this pair and specified sampler**; they cannot be read as
worst-case DP guarantees or direct comparisons with Nasr et al.'s tight
two-run f-DP auditing procedure. Bootstrap AUC gaps versus the final
checkpoint resample whole H0/H1 seed pairs.
An aggregate paired-comparison table separately evaluates GAUSSPROOF minus
raw alignment, adjacent noise levels, and the larger-batch controls using
the same held-out seeds. Its nominal 95% intervals are descriptive across
many comparisons; no positive cell is treated as a discovery merely because
its unadjusted interval excludes zero.

Run after generating the source checkpoint as in
[`canary_holdout_protocol.md`](canary_holdout_protocol.md):

```bash
bash scripts/run_clip_noise_sweep.sh /path/to/mnist_train.csv runs/canary_holdout
```

The runner executes every condition, validates and merges the aggregate
reports, computes paired score comparisons, and renders the vector/raster
figures. It checks the MNIST digest against the public checkpoint metadata
and skips completed conditions when resumed. The paired seed-level score
rows stay under ignored `runs/`.

The run is resumable after interruption with the same config and output
paths. Raw paired scores and MNIST remain under ignored `runs/`; the report
contains aggregate CSVs, provenance, and publication-resolution PDF/PNG
figures. Each completed cell is appended to the report so intermediate
progress can be inspected without using incomplete cells as final evidence.
