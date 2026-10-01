# Equal-access GAUSSPROOF gallery and gradient pilot

This pilot asks whether the **bounded sparse trajectory decoder** adds value
when all methods see the same real DP-SGD releases, CNN checkpoints, 32
candidate MNIST images, and 64 public background images. It tests both
closed-gallery linkage and estimation of the *clean clipped batch-average
gradient*. It does **not** compare with final-model LiRA/RMIA or claim
unknown-image reconstruction.

The answer for the prespecified single-eligible-candidate model is negative
for the sparse decoder. A q-aware analytic trajectory score and simple
centered alignment achieve 20% exact top-1 recovery at 128 releases in both
balanced and long-tail sampling, versus 7.5% for the sparse decoder selected
on disjoint calibration identities. Random top-1 for the 32-image gallery is
3.125%. The paired sparse-minus-mixture difference is -12.5 percentage
points, with identity-clustered 95% bootstrap interval [-27.5, 0] in balanced
sampling and [-25, 0] in long-tail sampling. On same-digit comparisons, the
mixture score reaches 55% and the sparse decoder 25%/35%, respectively. The
same-class random reference is about 31.7% for these held-out identities.

| Private sampling, matched public background | T | Alignment top-1 | q-aware mixture top-1 | Sparse top-1 | Mixture same-class top-1 | Sparse same-class top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced | 64 | 15.0% | 15.0% | 10.0% | 45.0% | 32.5% |
| Balanced | 128 | 20.0% | 20.0% | 7.5% | 55.0% | 25.0% |
| Long-tail | 64 | 15.0% | 15.0% | 12.5% | 42.5% | 42.5% |
| Long-tail | 128 | 20.0% | 20.0% | 7.5% | 55.0% | 35.0% |

The [figure](equal_access_gallery.pdf), [aggregate metrics](summary.csv),
and [paired differences](paired_differences.csv) give full intervals and
precise values. A balanced public background on long-tail private sampling
reduces the mixture's exact top-1 at T=128 from 20% to 5%. The sparse score
reaches 10% under that mismatch, but its +5-point paired gain over the
mixture has interval [-5, +15] points; it is not evidence of robust sparse
superiority. See the mismatch rows in the CSV.

## Clean-gradient estimation

All errors are mean full-vector squared error divided by C², evaluated against
the actual clean clipped **batch mean** before Gaussian noise. At 128 releases,
the public-background mean alone has error 0.13319 in the balanced condition
and 0.12233 in the long-tail condition. The online q-aware posterior mean,
which uses the same gallery and trajectory through the current round, lowers
those errors to 0.13072 (1.85% reduction) and 0.11900 (2.72%). The sparse
decoder **raises** error to 0.13613 and 0.12492. Identity-clustered paired
intervals for the posterior's absolute MSE reduction are [0.00206, 0.00293]
and [0.00275, 0.00388]. The raw noisy release has error around 3300 because
Gaussian noise spans all 13,242 CNN coordinates; beating raw release alone
would be a trivial denoising claim. The public background and analytic
posterior are the relevant baselines.

These gains concern a clean batch mean. They do not recover an unclipped
individual gradient or target pixels. Even an evaluator-only oracle that
knows the candidate and its inclusion schedule retains substantial error
from the unknown ordinary minibatch, as shown in `summary.csv`.

## Protocol and access

The public initial CNN was trained for 100 ordinary SGD steps on 3,000
disjoint MNIST images. Each of **110 new target CNN trajectories** then
trained for 128 actual clipped, noisy DP-SGD steps with C=1, batch size 8,
sigma=4, and learning rate 0.005. The target canary is one of 32 unmodified
held-out images and joins with q=0.5 each round; other batch records come
from a private 2,000-record population. The attacker sees the 32 candidate
images, public background, every pre-update checkpoint and noisy release,
and the mechanism parameters. It does not see hidden batch membership,
sampled noise, or clean gradients. The background-distribution match is an
explicit assumption, with a balanced-background mismatch control.

Ten identities (one per digit) with one repetition each chose the sparse
penalty from [0, .25, .5, 1] times a noise-and-bank scale and selected the
strongest of three same-access baseline scores by gallery MRR. **Twenty
other identities** (two per digit) with two repetitions each are held out
for evaluation. The selected sparse penalty is zero; the selected baseline
is the q-aware mixture score. Selection occurred before analyzing those
held-out identities. Each distribution also has five candidate-absent
trajectories. Their maximum-score ranges overlap the candidate-present
ranges, so this small null control does not justify a detection threshold;
see [null_controls.csv](null_controls.csv).

Mean public-query CNN accuracy at T=128 is 63.6% for balanced and 59.4% for
long-tail sampling. The query images are disjoint from both the candidate
gallery and training population. Under a fixed batch-selection rule, a loose
replace-one Gaussian composition calculation would give epsilon <= 43.14
at delta=1e-5 for T=128, sigma=4. It is **not** a low-epsilon experiment.
The synthetic class-weighted sampler depends on the fixed private population
labels, so this number is **not** presented as a general DP guarantee for
the full distribution-generation procedure. No outcome here contradicts DP.

The experiment is deliberately narrow. One eligible candidate per round is
well matched to the mixture score and may be unfavorable to a sparse decoder
designed for multiple simultaneous candidate contributors. A decisive
sparse-method test must use that multi-contributor model, preserve identical
access, and compare against a same-model Bayesian or regularized baseline.
The held-out set has only 20 independent target identities, so intervals
remain wide for top-1 recovery.

## Reproduce

The MNIST CSV must contain original 0–255 pixels and match dataset SHA256
`fb60bc58af4dac3554e394af262b3184479833d3cc540ff8783f274b73492d5d`.
Generate the public warm-start checkpoint and splits with
`python -m gaussproof.canary_holdout --data /path/to/mnist_train.csv
--config configs/canary_holdout.json --output runs/canary_holdout`, then run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-gaussproof \
python -m gaussproof.whitebox_gallery \
  --source runs/canary_holdout --data /path/to/mnist_train.csv \
  --config configs/whitebox_gallery_pilot.json \
  --output runs/whitebox_gallery_pilot \
  --report reports/whitebox_gallery_pilot

OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-gaussproof \
python -m scripts.summarize_whitebox_gallery \
  --run runs/whitebox_gallery_pilot --report reports/whitebox_gallery_pilot
```

The individual trajectories, split identities, and score rows remain in
ignored `runs/`. Only aggregates, configuration, provenance, and vector/PNG
figures are in this report. See [provenance.json](provenance.json) and the
[scope and comparison plan](../../docs/gaussproof_scope_and_equal_access_plan.md).
