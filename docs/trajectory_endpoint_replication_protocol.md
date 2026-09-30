# Paired trajectory and strong-endpoint DP-SGD replication

This experiment tests whether full-trajectory fingerprint evidence adds
discrimination beyond strong attacks that receive only the final model. It
extends the existing exact evolving MNIST CNN canary audit; it does not change
the original trained public initialization or private population.

## Mechanism and split

- DP-SGD: clipping norm 1, batch size 8, Gaussian noise multiplier 4, with
  64- and 128-release prefixes. The original learning rate is 0.05. A
  follow-up **utility-rescue** condition uses learning rate 0.005, chosen
  after a small screen on the 20 calibration identities and public utility.
  This condition is exploratory; the 80 evaluation identities were never
  used to choose the rate.
- Candidate eligibility: independent per-round probabilities `q=0.5` and
  `q=0.1`. The candidate replaces one ordinary batch record when selected.
  Presence and absence runs share initialization, batch/noise seeds, and the
  same candidate. They diverge adaptively after inclusion.
- One hundred stratified candidate identities, ten per digit, from the source
  test role. The first two per digit are attack-calibration identities. The
  remaining **80 per condition are the only evaluation identities**. They are
  disjoint from the previous 20-identity sensitivity pilot's identities.
- Thirty-two independently trained OUT reference models use the same public
  initialization, optimizer, noise level, and number of updates. Their
  2,000-record training pool is selected from MNIST records excluded from all
  source-run roles. The 128 public RMIA population records are separate from
  both that pool and all source-run roles.

Every method scores the exact same paired target runs. The q-aware Gaussian
mixture likelihood, the q-oblivious sum of round likelihoods, and a raw
candidate-gradient alignment see all intermediate noisy updates and
pre-update checkpoints. The raw alignment is a simple **RERO-style access
control**, not a reproduction of the informed reconstruction algorithm in
Hayes et al. LiRA and RMIA see only the final checkpoint, candidate record,
OUT reference-model predictions, and the disjoint public RMIA population.

Offline LiRA variants are a fixed-variance Gaussian OUT log-density on
correct-class probability logits and a one-sided standardized OUT-logit
score. The fixed variance is pooled from **within-record** OUT-reference
deviations; it is never estimated from evaluation targets. Offline RMIA uses
the ratio and population dominance event of Zarifzadeh et al., with
predeclared `a in {0,0.5,1}` and `gamma in {1,1.05}`. Variant selection
uses only the 20 calibration identities; the selection itself is frozen
before evaluation on the remaining 80. Fixed LiRA and RMIA variants are also
reported separately.

The primary quantity is tie-aware held-out AUC and its **paired
identity-bootstrap** difference from the strongest calibration-selected
endpoint attack. Empirical TPR at 5% FPR is descriptive. A threshold
calibrated on only 20 negative identities has a 5% FPR step, so its achieved
holdout FPR must be reported alongside TPR; 1% calibrated FPR is not resolved.
Public-query accuracy and realized canary appearances accompany the attack
results. The conservative no-amplification replace-one bound is reported
separately; a noise multiplier of four is not a small-epsilon claim.

## Reproduction

First create the public checkpoint and fixed splits using
`docs/canary_holdout_protocol.md` and `configs/canary_holdout.json`. The
resulting `runs/canary_holdout` directory is the `--source` below.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-gaussproof \
  python -m gaussproof.trajectory_endpoint_replication \
  --source /path/to/previous/canary_holdout/run \
  --data /path/to/mnist_train.csv \
  --config configs/trajectory_endpoint_replication.json \
  --output runs/trajectory_endpoint_replication
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-gaussproof \
  python -m gaussproof.trajectory_endpoint_replication \
  --source /path/to/previous/canary_holdout/run \
  --data /path/to/mnist_train.csv \
  --config configs/trajectory_endpoint_high_utility.json \
  --output runs/trajectory_endpoint_high_utility
MPLCONFIGDIR=/tmp/mpl-gaussproof python -m scripts.summarize_trajectory_endpoint_replication \
  --run runs/trajectory_endpoint_replication \
  --output reports/trajectory_endpoint_replication
MPLCONFIGDIR=/tmp/mpl-gaussproof python -m scripts.hybrid_trajectory_endpoint \
  --run runs/trajectory_endpoint_replication \
  --output reports/trajectory_endpoint_replication/hybrid.csv
```

The runner checks the MNIST hash against the source run and caches OUT
reference predictions. Interrupted target runs resume from completed score
rows. The raw score rows and reference predictions stay under ignored
`runs/`; only aggregate CSVs, provenance, and PDF/PNG figures are exported.
