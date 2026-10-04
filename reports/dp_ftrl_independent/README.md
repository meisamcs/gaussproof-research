# Independent federated DP-FTRLM checkpoint pilot: aggregate results

This is a **negative exploratory result** for one-time client participation.
It uses genuine writer-partitioned EMNIST digits and a source-aligned NumPy
implementation of the official DP-FTRLM server recurrence and efficient-tree
noise, but a lightweight 7×7 softmax client model. It is **not** a run of the
official TensorFlow Federated EMNIST training driver or proof of a deployed
system's privacy. See the [protocol and executable code](../../experiments/dp_ftrl_independent/README.md).

Each run has 128 global checkpoints, 8 client slots per round, a candidate
client inserted exactly once, 40 calibration writers, 100 held-out writers,
and a disjoint public fingerprint bank. Paired presence/absence worlds use
the same background clients and tree noise. The public observer knows the
candidate and the checkpoints, but neither the other clients' updates nor
the insertion round. Cross-identity AUC excludes each client's own paired
positive/negative comparison. The informed likelihood is a stronger diagnostic
with exact other-client background and insertion round.

| Candidate insertion | Tree noise multiplier | Sparse max | Sparse mean | Public GLS max | Final-model loss | Informed LLR |
|---|---:|---:|---:|---:|---:|---:|
| Round 65 | 1 | 0.500 | 0.504 | 0.503 | 0.501 | 0.938 |
| Round 65 | 4 | 0.500 | 0.504 | 0.503 | 0.501 | 0.651 |
| Round 65 | 8 | 0.500 | 0.503 | 0.502 | 0.500 | 0.580 |
| Round 1 | 1 | 0.500 | 0.503 | 0.510 | 0.500 | 0.958 |
| Round 1 | 4 | 0.500 | 0.503 | 0.507 | 0.500 | 0.669 |
| Round 1 | 8 | 0.500 | 0.505 | 0.507 | 0.500 | 0.587 |

The sparse `max` coefficient often saturates at 1 in both presence and
absence worlds. At sigma 4, round 65, its AUC is exactly 0.5. The public
sparse `mean` and covariance-aware GLS are also near chance. Their held-out
TPR and FPR at a threshold calibrated for 5% FPR are both 0.02, so there is
no useful low-FPR public signal. The informed LLR reaches TPR 0.18 at FPR
0.05, showing that one participation can leave a signal **when the exact
private background is known**. This access gap is the main finding.

A paired bank-size ablation at sigma 4 uses 12 versus 64 disjoint public
writers with identical candidate/background/noise draws. Public GLS AUC
changes from 0.503 to 0.507; sparse max stays at 0.500. This modest gain
does not close the background-knowledge gap. No claim is made that raising
noise increases leakage. Informed AUC falls as sigma increases.

The [aggregate summary](aggregate_summary.csv) contains every method, noise
condition, accuracy, calibrated TPR/FPR, descriptive advantage, bootstrap
interval, and conservative one-pass epsilon upper bound. The
[paired comparisons](aggregate_paired_comparisons.csv) and
[run metadata](published_metadata.json) retain the comparison definitions and
configuration. [Late-insertion](auc_vs_noise_late.pdf) and
[early-insertion](auc_vs_noise_early.pdf) plots are vector PDFs; PNG versions
are alongside them. The first 40 calibration negatives cannot resolve a
1% FPR threshold, so the 1% fields are `nan`. No raw EMNIST data, client
identities, per-client scores, checkpoints, or private trajectories are here.

The next decisive check is the official pinned TFF application with a
realistic public-background model and more than one random seed. Its legacy
TFF 0.20 dependency set could not run on the local Apple Silicon Python 3.11
host because `jaxlib~=0.1.76` has no compatible wheel. Until that check,
this is a mechanism-level pilot and a threat-model warning, not evidence
that GAUSSPROOF breaks federated DP-FTRLM.
