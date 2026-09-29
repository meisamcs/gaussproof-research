# Distribution and endpoint-baseline extension

This is a controlled, source-aligned NumPy DP-FTRLM **pilot**, not the authors'
TensorFlow Federated application. It evaluates whether known-client membership
signal survives changes to the client data distribution and whether a trajectory
score beats credible final-model attacks. The target still uses a 7×7-pooled
softmax model, one client step, the official server momentum recurrence, and
efficient-tree noise. Client insertion is exactly once. A server-log observer
sees all checkpoints; endpoint baselines see only the final checkpoint.

## Distributions and separation

All images are from genuine federated EMNIST digits. The original writer
partition is `natural_writer`. `iid_mixed` uniformly reassigns those images
into virtual 16-image clients. `label_sorted` sorts images by digit before
making virtual 16-image clients, producing extreme class skew. The latter two
are deliberately artificial distribution shifts; their virtual clients must
not be described as real EMNIST writers. Candidate, public/reference,
population, and target-background roles are split by writer **before** any
within-role reassignment. Each image is used once within its role, so neither
reference models nor population examples contain a candidate record.

All methods attack the **same paired presence/absence worlds** per candidate.
The worlds share a background-client schedule and tree-noise draws. Candidate
identity, model architecture, optimizer, and public bank are fixed. Candidate
participation time is hidden from public attacks. Only the informed likelihood
uses the true background and insertion time. The cross-identity AUC discards a
candidate's own positive/negative pair, which would otherwise inflate AUC.

## Endpoint baselines

The record-level offline RMIA score is the [ICML 2024 paper's](https://proceedings.mlr.press/v235/zarifzadeh24a.html)
population dominance test and offline probability approximation. Eight OUT
reference models are trained with the same fixed-slot DP-FTRLM recurrence,
using only public clients. Both the candidate and independent population have
true-label probabilities from the target and every reference model. We report
the [authors' repository](https://github.com/privacytrustlab/ml_privacy_meter/tree/d32734161a3395211fe5f3cd461932290b1fafbe/research/2024_rmia)
defaults `a=1, gamma=2`, plus the GAUSSPROOF benchmark's pre-existing fixed
`a=0.5, gamma=1`. Equality counts in the pairwise comparison (`>=`). If all
record scores tie, AUC is 0.5 and the method has no discriminating power here.

Offline fixed-variance LiRA follows the [authors' original code](https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/plot.py):
transform each true-label probability by its logit, take each record's median
OUT reference signal, fit one global OUT standard deviation, and use negative
Gaussian log-density under OUT as the membership score. The original plotting
code passes the negative of its raw log-density score to the ROC sweep; our
positive-membership orientation applies that sign directly. We omit its
constant normalizer, which cannot affect ranking. There are no IN references
in this offline setting. The global variance is estimated solely from the OUT
reference predictions, including the candidate and disjoint population data.

RMIA and LiRA are **record-level attacks in the original papers**. Here their
record scores are averaged within the known client's bag to obtain a client
membership score. That is an explicit client-level adaptation. It is not an
exact reproduction of the original papers' model, dataset, attack unit, or
number of reference models. The references are trained on the same pilot
mechanism as the target; they are not an oracle for private other-client data.

We additionally evaluated the exact scoring **variants** used by
[Pierre-Joly/Membership-Inference-Attacks at `9182ed8`](https://github.com/Pierre-Joly/Membership-Inference-Attacks/tree/9182ed809d9fa3d5141d50816b7e83a06590371b).
Its offline RMIA uses the same `a=0.5` correction for the candidate but the
uncorrected mean OUT probability for population records. Its offline LiRA uses
raw correct-class logits, a per-record OUT mean/sample standard deviation,
and the OUT Gaussian CDF. These are separately named `pierre_rmia_a05_g1`
and `pierre_lira_cdf` in the CSV. They should not be confused with the
original-paper variants above. We downloaded the pinned repository and ran
both scoring classes on synthetic PyTorch models: both produced finite scores,
and its RMIA reached a score of 1 for a constructed dominance fixture. The
repository does **not** include the `data/pub.pt`, `data/priv_out.pt`, or
`weights/01_MIA_67.pt` named by its default configuration, so its full command
cannot be reproduced as published. We use its score equations as an extra
cross-check and baseline, without importing its absent training artifacts.
The optional [`verify_pierre_joly.py`](verify_pierre_joly.py) loads that pinned
source, checks SHA-256 hashes of both attack files, runs the original attack
classes on synthetic PyTorch models, and asserts record-score parity with
our adaptations. It requires PyTorch, SciPy, `tqdm`, and `python-dotenv`:

```bash
python experiments/dp_ftrl_independent/verify_pierre_joly.py \
  --source /private/path/Membership-Inference-Attacks-9182ed809d9fa3d5141d50816b7e83a06590371b
```

The matched comparisons are: RERO alignment, GAUSSPROOF bounded sparse
decoder (max and mean scores), public covariance-aware GLS, final-model
true-label log probability, endpoint offline RMIA, endpoint offline LiRA, and
an **informed diagnostic**. No per-client scores or models are published.

## Reproduction

Use Python 3.10+, NumPy, SciPy, Matplotlib, and `msgpack==1.1.1`. Download the
FedJAX federated EMNIST digits SQLite file as described in [README.md](README.md).
From the repository root:

```bash
python -m unittest discover -s experiments/dp_ftrl_independent -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 python experiments/dp_ftrl_independent/run_distributions.py \
  --sqlite /private/path/federated_emnist_digitsonly_train.sqlite \
  --output reports/dp_ftrl_distributions/new_run \
  --rounds 32 --position 16 --clients-per-round 8 \
  --calibration-identities 40 --holdout-identities 80 \
  --public-pool 256 --public-bank 12 --population-clients 20 \
  --background-pool 500 --reference-models 8 \
  --sigma 1 4 --bootstrap 400 --seed 20260929
```

The output refuses to overwrite a nonempty directory. It contains aggregate
`summary.csv`, paired AUC comparisons, metadata including the data SHA-256,
and PDF/PNG AUC figures. The 40 calibration identities cannot resolve an
empirical 1% FPR threshold, so those cells are `nan`; 5% is calibrated, and
its actual held-out FPR is reported. Bootstrap intervals resample independent
client identities, keeping presence/absence pairs together. They describe
candidate variability under **one** target-training seed, not variation across
independent training seeds. Model utility and a conservative one-pass privacy
upper bound are in the CSV. This bound is not a guarantee for repeated clients.

The [measured extension](../../reports/dp_ftrl_distributions/README.md) repeats
the selected eight-client `sigma=4` and two-client `sigma=0.25` stress settings
for three predeclared independent seeds each. To reproduce the seed aggregate
after running the six conditions, use
[`summarize_distributions.py`](summarize_distributions.py); its `CONDITIONS`
mapping fixes the expected output directory names. It checks all three seeds,
the matching settings, and the dataset digest before producing a mean and
range across seeds. The two-client, low-noise condition is a separate
weak-privacy stress test, not evidence of high-noise backfire.

For the exact replicated inputs in the report, run the following from the
repository root after downloading the SQLite file. The first seed also has
additional exploratory noise levels in its individual run directory; the
aggregator selects `sigma=4` and `sigma=0.25` respectively.

```bash
for seed in 20260929 20261001 20261003; do
  eight="eight_clients_seed_${seed}"
  two="two_clients_seed_${seed}"
  if [ "$seed" = 20260929 ]; then
    eight=eight_clients
    two=two_clients_stress
  fi
  OPENBLAS_NUM_THREADS=1 python experiments/dp_ftrl_independent/run_distributions.py \
    --sqlite /private/path/federated_emnist_digitsonly_train.sqlite \
    --output "reports/dp_ftrl_distributions/$eight" \
    --clients-per-round 8 --sigma 4 --seed "$seed"
  OPENBLAS_NUM_THREADS=1 python experiments/dp_ftrl_independent/run_distributions.py \
    --sqlite /private/path/federated_emnist_digitsonly_train.sqlite \
    --output "reports/dp_ftrl_distributions/$two" \
    --clients-per-round 2 --sigma 0.25 --seed "$seed"
done
python experiments/dp_ftrl_independent/summarize_distributions.py \
  --root reports/dp_ftrl_distributions \
  --output reports/dp_ftrl_distributions/replicated
```

The experiment is a mechanism-level benchmark. A decisive deployed-system
claim requires the pinned official TFF CNN, validated privacy accountant,
larger independent replication, and a plausible source of public client data.
