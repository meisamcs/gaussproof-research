# GAUSSPROOF with LiRA-style scoring and a fixed-neighbor audit

This report tests two distinct uses of trajectory fingerprints. First, a
candidate's full-trajectory score is fused with an **offline LiRA-style**
one-sided OUT-logit score from the final model. Second, the trajectory score
serves as the test statistic in a fixed-natural-record privacy audit. The
second study is inspired by the full-update threat model of [Nasr et al.
(2023)](https://www.usenix.org/conference/usenixsecurity23/presentation/nasr);
it does **not** reproduce their tight two-run auditing method or claim an
improved accountant. [LiRA](https://ieeexplore.ieee.org/document/9833649/)
motivates evaluating true-positive rates at low false-positive rates.

## Final-checkpoint fusion

The existing paired MNIST CNN runs use clipping norm 1, batch size 8,
Gaussian noise multiplier 4, 64 or 128 released updates, and candidate
participation probability 0.5 or 0.1. One hundred unseen candidate
identities are stratified across ten digits. The first two per digit (20)
calibrate attack scores; the remaining eight per digit (80) are held out.
Thirty-two OUT reference models use a disjoint public pool. All attacks score
the same H0/H1 runs. The final-checkpoint LiRA-style score sees only the
final model and candidate; GAUSSPROOF also sees intermediate noisy updates
and pre-update checkpoints.

The fusion weight is searched over `0, 0.05, ..., 1` using **only**
calibration identities. The criterion is calibration TPR at a 5% calibration
FPR threshold, then calibration AUC, then smaller trajectory weight. The
operating threshold is also selected on calibration identities and is frozen
before holdout evaluation. The table gives *achieved holdout* rates, not a
claim of 5% FPR on the test population.

| Setting at q=0.5, T=128 | LiRA-style TPR / FPR | Fusion TPR / FPR | Fusion weight | Held-out AUC, LiRA / fusion |
|---|---:|---:|---:|---:|
| LR 0.05, public accuracy 24.4% | 12.5% / 2.5% | 26.3% / 2.5% | 0.95 | 0.693 / 0.809 |
| LR 0.005, public accuracy 66.2% | 56.3% / 6.3% | 56.3% / 6.3% | 0 | 0.830 / 0.830 |

For the original learning rate, the 13.8-point TPR gain has a paired
identity-bootstrap 95% interval of 3.8 to 25.0 points. It occurs in a
low-utility model and is exploratory across multiple inspected conditions.
The higher-utility run chooses LiRA alone, so these results do **not** show a
general complementary gain. At `q=0.1`, neither run shows a reliable
128-update fusion gain. Full rates and intervals are in
[`low_fpr_original_lr.csv`](low_fpr_original_lr.csv) and
[`low_fpr_high_utility.csv`](low_fpr_high_utility.csv).

Only 20 calibration nonmembers permit 5% empirical FPR increments, and 80
holdout nonmembers permit 1.25% increments. These runs cannot support a
calibrated 1% FPR claim. The 5% nominal calibration point sometimes yields
more than 5% FPR on holdout. There is no evidence here that added noise
increases leakage, that GAUSSPROOF reconstructs unknown samples, or that a
DP guarantee is violated.

## Fixed-neighbor audit

The separate audit chooses the first two natural MNIST records in a fixed,
previously held-out test split as a dummy and a target. The two neighboring
datasets differ **only** in that one record slot. In each of 128 steps, the
slot is eligible independently with `q=0.5`; when used, seven other records
come from the same 2,000-record population, and otherwise all eight come
from it. The released update averages clipped per-record gradients and adds
Gaussian noise with multiplier 4. The public CNN initialization and all
other records are the same in both worlds. This is an explicitly specified
privileged-slot sampler, **not ordinary uniform minibatch DP-SGD**. The
attacker knows both natural records, the public background sample, all
pre-update checkpoints, and the noisy releases. They do not see slot
participation outcomes.

The GAUSSPROOF statistic sums an approximate Bernoulli-mixture log-likelihood
ratio based on recomputed clipped gradients of both records. This is a
discriminator, not an exact likelihood for random minibatches. Last-update
and final-checkpoint-only scores are controls. Calibration runs fix each
test's threshold; independent holdout runs estimate TPR and FPR. A
Clopper–Pearson confidence lower bound on TPR and upper bound on FPR imply
the one-event empirical bound

`epsilon >= max(0, log((TPR_lower - delta) / FPR_upper))`.

It applies to this **fixed pair and specified sampler**, at `delta=1e-5`;
it does not estimate worst-case epsilon, assert a DP violation, or extend to
all MNIST records. The conservative no-amplification zCDP accounting bound
is supplied separately in the provenance file. Each reported confidence
bound is marginal to a prespecified score; selecting the best method after
seeing the holdout would require multiplicity correction.

The fixed pair was MNIST test records 24898 (digit 9) and 46993 (digit 4),
chosen before running the audit. Thirty paired trials per world calibrated
the thresholds; 80 independent paired trials per world tested them:

| Score | Held-out AUC | Held-out TPR / FPR | 95% empirical epsilon lower bound |
|---|---:|---:|---:|
| GAUSSPROOF full trajectory | 0.948 | 80.0% / 7.5% | 1.49 |
| Last noisy update | 0.546 | 8.8% / 3.8% | 0 |
| Final checkpoint | 0.944 | 80.0% / 8.8% | 1.40 |

Thus repeated releases are substantially more informative than one isolated
release in this example. The endpoint is **almost as informative as the
trajectory**. The trajectory-minus-endpoint AUC difference is 0.004, with a
paired bootstrap 95% interval from -0.005 to 0.012. This experiment therefore
does **not** establish a distinct trajectory advantage over the final model.
The conservative accounting upper bound is epsilon 43.14 at the stated
delta; the empirical lower bounds are far below it. See
[`fixed_pair_summary.csv`](fixed_pair_summary.csv),
[`fixed_pair_provenance.json`](fixed_pair_provenance.json),
[`fixed_pair_roc.pdf`](fixed_pair_roc.pdf), and
[`lira_fusion_low_fpr.pdf`](lira_fusion_low_fpr.pdf).

## Reproduction

The private MNIST CSV, individual score rows, trained checkpoint, and OUT
reference predictions stay out of Git. Generate the source checkpoint with
`python -m gaussproof.canary_holdout --data /path/to/mnist_train.csv
--config configs/canary_holdout.json --output runs/canary_holdout`. Then run:

```bash
python -m scripts.low_fpr_lira_fusion --run runs/trajectory_endpoint_replication --output reports/nasr_lira_integration/low_fpr_original_lr.csv
python -m scripts.low_fpr_lira_fusion --run runs/trajectory_endpoint_high_utility --output reports/nasr_lira_integration/low_fpr_high_utility.csv
python -m gaussproof.fixed_pair_audit --source runs/canary_holdout --data /path/to/mnist_train.csv --config configs/fixed_pair_audit.json --output runs/fixed_pair_audit --report reports/nasr_lira_integration
python -m scripts.plot_nasr_lira_integration --report reports/nasr_lira_integration --audit-run runs/fixed_pair_audit
```

The original trajectory/endpoint runs are generated as described in
[`../../docs/trajectory_endpoint_replication_protocol.md`](../../docs/trajectory_endpoint_replication_protocol.md).
