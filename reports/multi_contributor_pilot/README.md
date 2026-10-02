# Multi-contributor fingerprint denoising on actual DP-SGD

The first [equal-access gallery pilot](../whitebox_gallery_pilot/README.md)
found that per-round sparse decoding loses to an analytic likelihood when
at most one gallery record can contribute. This follow-up tests the sparse
decoder's intended structural niche: **several known candidate records may
contribute to the same noisy CNN update**, while their identities and the
ordinary private batch are hidden from every estimator.

Twenty-four newly trained evolving MNIST CNN trajectories were evaluated:
four public-calibration and eight held-out training runs at each of noise
multipliers sigma=1 and sigma=4. Each runs 128 steps with clipping norm C=1,
batch size 8, learning rate 0.005, and eight known unmodified candidate
images (one each from digits 0–7). Each candidate joins each round with
probability q=0.2, so the observed mean is 1.633 gallery contributors per
round. The remaining batch places are uniformly sampled from a private
2,000-record population. All methods see **the same** current noisy update,
checkpoint-derived clipped gallery gradients, 64-image public background,
and mechanism parameters. The actual active subset and Gaussian noise are
never attacker inputs.

The primary usability measure is full-vector squared error estimating the
clean *clipped batch-average gradient*, divided by C². A second measure is
candidate-round support AUC: whether each of the eight known records was
included in each round. The comparators are public prior, centered
alignment, independent q-aware matched scores, and an exact enumeration of
all `2^8` candidate subsets under the declared Gaussian **surrogate** with
public-mean background. This posterior is exact for that surrogate, not for
the actual random private minibatch. It is a stronger fair comparator than
raw noisy updates. Sparse penalties were selected by gradient error on the
four disjoint calibration runs at each sigma; held-out outcomes did not
select a penalty.

| sigma | Public-prior gradient error | Exact-surrogate posterior error | Calibrated sparse error | Exact support AUC | Sparse support AUC | Public-query CNN accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.12670 | 0.10816 | **0.10441** | 0.709 | 0.686 | 60.9% |
| 4 | 0.12644 | **0.12516** | 0.13359 | 0.552 | 0.527 | 60.4% |

At sigma=1, sparse decoding reduces held-out clean-gradient error by
0.00375 relative to the exact-surrogate posterior (3.47% of its error),
with an eight-run paired bootstrap 95% interval [0.00196, 0.00517]. It
also reduces error by 0.02229 versus the public prior. Yet its support AUC
is **2.26 points lower** than the exact-surrogate posterior, interval
[-2.71, -1.88] points. Thus the supported, narrow result is better
*aggregate-gradient estimation*, not better identification of contributing
records. At sigma=4, sparse decoding is worse than both the exact posterior
and the public prior on gradient error, and its support AUC is weaker. The
multi-contributor structure does not rescue a high-noise claim.

The [paper-ready figure](multi_contributor.pdf) and
[PNG preview](multi_contributor.png) show both endpoints. Exact means and
run-clustered intervals are in [summary_with_ci.csv](summary_with_ci.csv);
[paired_differences.csv](paired_differences.csv) contains the prespecified
method contrasts. Four calibration runs selected penalty factors 0.25 at
sigma=1 and 1.0 at sigma=4 from [0, 0.25, 0.5, 1.0] times a public
noise-and-bank scale. The comparison uses the chosen factor even when an
unselected factor is numerically better on evaluation.

This is a pilot with only eight independent held-out training runs per noise
condition, one warm-start model, one fixed eight-image gallery, and one
private-population split. Candidate-round observations within a run are
correlated; intervals resample **runs**, not individual rounds or events.
The public background is a 64-image mean; the true ordinary minibatch is
random, so the Gaussian surrogate's assumptions are imperfect. The sigma=1
advantage should be replicated across new galleries, backgrounds and
architectures before it is a general method claim. The model's utility is
about 60%, and q=0.2 is far higher than ordinary B/N sampling. A loose
no-amplification replace-one upper bound at delta=1e-5 is about 364.6 for
sigma=1 and 43.14 for sigma=4; these are not small-epsilon settings.

The task does not reconstruct any image or an unclipped individual gradient.
It tests whether a public fingerprint bank can help estimate the clean
aggregate update when several of its entries are present. The observed
moderate-noise result is a plausible usability niche for GAUSSPROOF's
sparse representation; the high-noise negative result sharply limits it.

## Reproduce

Use the same public checkpoint and dataset preparation described in the
[gallery report](../whitebox_gallery_pilot/README.md), then run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-gaussproof \
python -m gaussproof.multi_contributor \
  --source runs/canary_holdout --data /path/to/mnist_train.csv \
  --config configs/multi_contributor_pilot.json \
  --output runs/multi_contributor_pilot \
  --report reports/multi_contributor_pilot

OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-gaussproof \
python -m scripts.summarize_multi_contributor \
  --run runs/multi_contributor_pilot --report reports/multi_contributor_pilot
```

Raw candidate-round scores, split identities, and trajectories remain in
ignored `runs/`. The repository report contains only aggregate results,
configuration and [provenance](provenance.json).
