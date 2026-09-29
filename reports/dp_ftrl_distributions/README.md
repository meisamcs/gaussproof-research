# Client-distribution and endpoint-baseline pilot

This report extends the [independent DP-FTRLM pilot](../../experiments/dp_ftrl_independent/README.md)
with three client-data arrangements and final-checkpoint RMIA/LiRA baselines.
It is **not** a run of the official TensorFlow Federated application. The
candidate client participates exactly once; the server-log observer sees the
whole checkpoint sequence and knows the candidate's 16 EMNIST images, while
the endpoint attacks see only the final model and those images. Candidate
records are absent from the eight OUT reference models and population set.

## What works, and what does not

The high-noise setting (eight clients per round, `sigma=4`, 32 rounds) does
**not** show useful public-information GAUSSPROOF leakage. Across three
independent target-training seeds, its max-sparse AUC is exactly 0.500 on
natural EMNIST writers, IID-mixed virtual clients, and label-sorted virtual
clients. RERO max ranges from 0.502 to 0.504 in mean AUC; the original-paper
endpoint LiRA variant is about 0.503. The pinned Pierre-Joly LiRA **variant**
is slightly above chance (0.532–0.545), but this is a different scoring
formula. The informed likelihood, which knows the private background and
insertion round, reaches 0.656–0.670. That is a diagnostic upper-context
comparison, not an available public attack.

The separate weak-privacy stress condition (two clients per round,
`sigma=0.25`) does show membership signal. The table gives mean AUC over three
seeds; the error bars in the [vector figure](replicated/replicated_auc.pdf)
are the minimum and maximum seed values, not deployment-wide confidence
intervals.

| Condition | Client data | GAUSSPROOF max | RERO max | Final loss | Offline LiRA (original) | Pierre LiRA CDF |
|---|---|---:|---:|---:|---:|---:|
| 8 clients, σ=4 | Natural writer | .500 | .504 | .503 | .503 | .536 |
| 8 clients, σ=4 | IID mixed | .500 | .502 | .503 | .503 | .532 |
| 8 clients, σ=4 | Label sorted | .500 | .503 | .520 | .503 | .545 |
| 2 clients, σ=.25 | Natural writer | .649 | .828 | .534 | .506 | .676 |
| 2 clients, σ=.25 | IID mixed | .666 | .867 | .535 | .556 | .727 |
| 2 clients, σ=.25 | Label sorted | .544 | .584 | .588 | .548 | .655 |

The original-paper offline RMIA with the predeclared `a=.5, gamma=1` has
mean AUC .504–.516 at high noise and .519–.599 in the stress setting. The
paper authors' `a=1, gamma=2` default ties all scores here (AUC .500), and
the Pierre-Joly RMIA variant also ties. These are reported as actual outcomes,
not silently dropped baselines. The [full summary](replicated/seed_summary.csv)
includes every method's mean, standard deviation, seed range, calibrated 5%
TPR, and *achieved* held-out FPR. The [per-seed table](replicated/seed_auc.csv)
and each run's `summary.csv` preserve the individual AUCs and within-seed
client-bootstrap intervals. A 1% FPR threshold cannot be resolved with only
40 calibration clients and is marked `nan` in individual summaries.

The class-skew setting weakens RERO much more than IID mixing, so distribution
is materially relevant in the weak-privacy stress test. It does not rescue
GAUSSPROOF's proposed high-noise advantage. At `sigma=4`, measured model
accuracy is only about 28%, versus ten classes; utility is limited. The
reported conservative one-pass upper bound is ε≈3.13 at δ=10⁻⁵ for `sigma=4`
and ε≈95.0 for `sigma=.25`. These are loose analytical bounds for this
one-time-client pilot, not official TFF accountant results. In particular,
the low-noise stress test should not be presented as a private deployment.

## Implementation validation

The [endpoint baseline code](../../experiments/dp_ftrl_independent/endpoint_baselines.py)
has formula tests for RMIA—including equality ties—and offline LiRA, a
record-partition test, and a check that an OUT reference model follows the
same DP-FTRLM recurrence as the absent-target world. The pinned
[Pierre-Joly repository](https://github.com/Pierre-Joly/Membership-Inference-Attacks/tree/9182ed809d9fa3d5141d50816b7e83a06590371b)
was downloaded independently. Its actual `OfflineRMIA` and `OfflineLiRA`
classes ran on controlled PyTorch fixtures and matched our two separately
named Pierre variants. The [parity harness](../../experiments/dp_ftrl_independent/verify_pierre_joly.py)
checks the pinned source-file hashes and both record-score arrays.

The external repository's default complete run is **not reproducible from its
published checkout alone**: its configuration names `data/pub.pt`,
`data/priv_out.pt`, and `weights/01_MIA_67.pt`, which are absent. Its classes
working on synthetic fixtures confirms their score path, not its unseen
training pipeline. Original-paper RMIA and LiRA remain separate baselines
because the Pierre formulas differ. All endpoint methods here aggregate
record scores to client scores by averaging. That is a client-level
adaptation, not an original-paper reproduction on the same attack unit.

The focused suite passes 13 mechanism/extension tests plus 15 core benchmark
tests. Runs use 40 calibration and 80 held-out client identities per seed and
distribution, eight OUT references, 32 rounds, a shared paired background/noise
schedule, and the same candidate pool for trajectory and endpoint attacks.
The three arrangements all derive from **one EMNIST digits dataset**; IID and
label-sorted clients are virtual, and each seed changes target-training
randomness rather than the underlying dataset. No raw records, identity-level
scores, or models are committed.

This mechanism-level pilot cannot establish the effect on official DP-FTRL,
other datasets, or multiple appearances of one client. The next decisive test
is the pinned official TFF model with an explicit server-log access policy and
the official privacy accountant, followed by repeated-client participation.
