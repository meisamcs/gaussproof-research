# Repeated-client participation in the DP-FTRLM pilot

**The setting that tests the trajectory hypothesis is repeated participation.**
Here \(q=B/N\) is a client's probability of being selected in each round,
where \(B\) of \(N\) cohort clients are sampled without replacement. With
\(B=8\) and 64 rounds, \(q=.02,.1,.5\) correspond to cohorts of
\(N=400,80,16\) and to 1.28, 6.4, and 32 *expected* appearances. The older
[one-insertion experiment](../dp_ftrl_hybrid_uniform/README.md) is a useful
control, but it cannot establish the high-\(q\) repeated-fingerprint claim.

**Finding.** At tree-noise multiplier \(\sigma=4\), the public GAUSSPROOF
sparse-trajectory score rises from AUC .504 at \(q=.02\) to .598 at
\(q=.5\). A fixed 50:50 combination with calibrated endpoint RMIA rises from
.505 to .645; at \(q=.5\), RMIA alone is .6195. This is evidence that a
repeated client can leave a membership signal despite high *per-release*
noise in this eight-client stress setting. It is **not** evidence that
increasing noise improves an attack or that the run has a small user-level
\(\varepsilon\). At \(q=.5\), a client participates about 32 times.

The mechanism is a source-aligned NumPy implementation of momentum DP-FTRL
with tree-aggregated noise and a pooled softmax classifier. It is not the
official TensorFlow Federated CNN. The DP-FTRL [paper](https://proceedings.mlr.press/v139/kairouz21b/kairouz21b.pdf)
describes correlated tree noise and privacy analysis without relying on
sampling amplification; the [authors' repository](https://github.com/google-research/DP-FTRL)
points to separate federated code. This experiment changes client sampling
and reuses clients, so its one-pass privacy bound is **inapplicable**. We do
not report an \(\varepsilon\) for these repeated-client runs. This report
also does not contain a matched DP-SGD execution.
Under this replace-one adjacency, two clipped clients can differ by as much
as \(2C/B\), while each tree node uses Gaussian standard deviation
\(\sigma C/B\). Thus the nominal \(\sigma=4\) is only a noise-to-worst-case-
replacement-sensitivity ratio of 2. It must not be presented as a
\(\sigma=4\) add/remove guarantee.

## Attack comparison

The table shows mean held-out **client-membership AUC across three seeds** at
\(\sigma=4\). Each run has 40 calibration and 80 independent evaluation
client identities. The [vector figure](replicated/auc_by_participation.pdf)
includes \(\sigma=2\) and seed min–max ranges; the [full table](replicated/summary_over_seeds.csv)
and [individual seed AUCs](replicated/seed_auc.csv) give exact values.

| q | Expected appearances | GAUSSPROOF | RERO-style | LiRA | LiRA + GP | RMIA | RMIA + GP | Model accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| .02 | 1.28 | .504 | .507 | .505 | .505 | .505 | .505 | .249 |
| .1 | 6.4 | .526 | .532 | .524 | .534 | .523 | .533 | .250 |
| .5 | 32 | .598 | .642 | .620 | .638 | .619 | .645 | .245 |

At \(q=.5,\sigma=4\), RMIA + GP gains .025 AUC over RMIA alone on average
and the gain is positive in all three seeds. The individual paired-bootstrap
gain intervals, however, include zero in two seeds; see [paired gains](replicated/paired_seed_gains.csv).
LiRA + GP gains .019 on average but is negative in one seed. The simple
RERO-style trajectory score is stronger than GAUSSPROOF alone and nearly
matches the best combination. At \(\sigma=2,q=.5\), LiRA and RMIA alone
already reach about .698 and .695, while the 50:50 combinations are slightly
worse; the added trajectory score is not uniformly beneficial.

The [calibrated 5% FPR columns](replicated/summary_over_seeds.csv) must be
read with their **achieved held-out FPR**. For example, at \(q=.5,\sigma=4\),
RMIA + GP has mean TPR .233 but mean achieved FPR .092. The 40 calibration
negatives give a coarse threshold, so we do not claim .233 TPR at a true 5%
FPR. A 1% FPR operating point is not resolvable with this calibration set.

## Mechanism and access

Every round draws \(B\) clients from a cohort of \(N=B/q\). In the absence
world, all \(N\) are ordinary background clients. In the presence world,
the candidate replaces one designated cohort client, so the candidate is
selected with probability \(q\). Both worlds have exactly \(B\) real clients
per round, unlike the older one-empty-slot design. We pair the round schedule
and tree-noise draws across worlds. The negative and positive target models
remain separate adaptive training runs; no private background update is given
to a public attack. [Observed appearance counts](replicated/participation_summary.csv)
confirm the sampling rate and record zero-appearance trials.

LiRA and RMIA see the final trained model, the candidate's 16 labeled EMNIST
records, 32 independently trained OUT reference models with the same
\(B\)-of-\(N\) sampling rule, and disjoint population data. Their client-level
score variants are chosen on calibration identities only. GAUSSPROOF and
RERO-style attacks additionally see all 64 global checkpoints, public
mechanism parameters, and a disjoint 12-client fingerprint bank. They do not
see private client updates, the selected rounds, or tree-noise draws. GP is
the mean bounded nonnegative sparse coefficient of the candidate fingerprint
over checkpoint-derived noisy releases. The fixed hybrid standardizes the
endpoint and GP scores using calibration identities in both worlds, then
adds them equally. A separate [calibration-selected fusion](replicated/fusion_selections.csv)
is included in the full CSVs but is not used for the headline table. Its
weight is selected from \(\{0,.25,.5,.75,1\}\) without held-out labels.

The hidden-background, known-participation-round likelihood score is an
**informed diagnostic**, not a public attack: its AUC is .998 at
\(q=.5,\sigma=4\). That large gap from the public attacks quantifies how
much the inaccessible background and schedule matter. It does not certify
that GP can recover them.

## Larger-batch robustness

To check whether the result survives greater model utility, we reran
\(q=.5,\sigma=4,T=64\) with \(B=32,N=64\) across the same three seeds.
Mean accuracy rises from .245 to .486. The [large-batch results](replicated/large_batch_summary.csv)
show GP AUC .535, RERO .548, LiRA .608, RMIA .613, LiRA + GP .585, and
RMIA + GP .587. **The hybrid advantage disappears and reverses** in this
better-utility setting. A separate [one-seed, 128-round check](replicated/long_batch_exploratory.csv)
with \(B=32,q=.5,\sigma=4\) reaches accuracy .561; GP is .554, RMIA .642,
and RMIA + GP .614. That last check is exploratory and not a replicated
estimate. These controls rule out a general claim that high \(q\) and high
noise make GAUSSPROOF superior to endpoint attacks.

The defensible scope is a **small-cohort, repeated-client stress test** in
which a public trajectory score sometimes adds to endpoint LiRA/RMIA. At
fixed \(B\), changing \(q\) changes cohort size \(N\), so the sweep is a
comparison of sampling regimes, not an isolated causal effect of \(q\).
Model utility at \(B=8,\sigma=4\) is too low for a practical deployed-model
claim. The results concern client membership, not reconstruction of an
unknown image.

## Reproduce

The dataset stays local; published files are aggregate tables, metadata, and
figures. With the FedJAX federated EMNIST digits-only training SQLite file at
a private path, run from the repository root:

```bash
python -m unittest discover -s experiments/dp_ftrl_independent -p 'test_*.py' -v
for seed in 20261011 20261013 20261017; do
  for q in 0.02 0.1 0.5; do
    OPENBLAS_NUM_THREADS=1 python experiments/dp_ftrl_independent/run_distributions.py \
      --sqlite /private/path/federated_emnist_digitsonly_train.sqlite \
      --output "reports/dp_ftrl_repeated_q/seed_${seed}_q${q}" \
      --modes natural_writer --rounds 64 --position 32 \
      --participation-q "$q" --clients-per-round 8 \
      --calibration-identities 40 --holdout-identities 80 \
      --public-bank 12 --public-pool 512 --population-clients 64 \
      --background-pool 500 --reference-models 32 --examples 16 \
      --client-example-selection uniform --pixel-transform ink \
      --reserve-utility-clients 1124 --utility-seed 20260929 \
      --sigma 2 4 --clip 1 --client-lr 1 --server-lr 0.2 \
      --momentum 0.9 --bootstrap 400 --seed "$seed" \
      --calibrate-endpoint --fuse-trajectory
  done
  OPENBLAS_NUM_THREADS=1 python experiments/dp_ftrl_independent/run_distributions.py \
    --sqlite /private/path/federated_emnist_digitsonly_train.sqlite \
    --output "reports/dp_ftrl_repeated_q/large_batch_seed_${seed}" \
    --modes natural_writer --rounds 64 --position 32 \
    --participation-q 0.5 --clients-per-round 32 \
    --calibration-identities 40 --holdout-identities 80 \
    --public-bank 12 --public-pool 512 --population-clients 64 \
    --background-pool 500 --reference-models 32 --examples 16 \
    --client-example-selection uniform --pixel-transform ink \
    --reserve-utility-clients 1124 --utility-seed 20260929 \
    --sigma 4 --clip 1 --client-lr 1 --server-lr 0.2 \
    --momentum 0.9 --bootstrap 400 --seed "$seed" \
    --calibrate-endpoint --fuse-trajectory
done
```

The exploratory 128-round command has the same large-batch settings, with
`--rounds 128 --position 64 --seed 20261011` and output
`reports/dp_ftrl_repeated_q/long_batch_seed_20261011`. Then run:

```bash
python experiments/dp_ftrl_independent/summarize_repeated_q.py \
  --root reports/dp_ftrl_repeated_q \
  --output reports/dp_ftrl_repeated_q/replicated
```

The runner refuses to overwrite populated directories. The implementation
is in [`run_distributions.py`](../../experiments/dp_ftrl_independent/run_distributions.py),
[`run_pilot.py`](../../experiments/dp_ftrl_independent/run_pilot.py),
[`endpoint_baselines.py`](../../experiments/dp_ftrl_independent/endpoint_baselines.py),
[`hybrid_fusion.py`](../../experiments/dp_ftrl_independent/hybrid_fusion.py),
and [`summarize_repeated_q.py`](../../experiments/dp_ftrl_independent/summarize_repeated_q.py).
