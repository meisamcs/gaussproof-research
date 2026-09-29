# DP-FTRL checkpoint audit: measured result

This is a controlled MNIST audit of the tree mechanism and optimizer in the
[DP-FTRL authors' repository](https://github.com/google-research/DP-FTRL), pinned
to commit `513500a8e31e412972a7d457e9c66756e4a48348`. It tests whether a
**once-participating, known candidate** is easier to detect from a sequence of
released model checkpoints than from the final checkpoint. It is not a run of
the authors' federated language model, a reconstruction of an unknown image,
or a violation of the DP guarantee.

The local model is 10-class MNIST softmax regression. We use the official
`FTRLOptimizer` and standard/efficient Gaussian tree classes, 128 rounds,
batch size 64, clipping bound 1, and `alpha=64`. The candidate occupies one
slot in either round 1 or round 65 and never participates again. The absence
world leaves that slot empty. Member/nonmember runs have matched other records,
schedule, and Gaussian noise; the score never receives the noise seed. The
experiment uses 40 separate calibration identities and 60 held-out identities
(three paired repetitions each), stratified by digit. Validation accuracy for
the main condition is 83.17%.

| Condition / score | Held-out AUC | 95% identity-cluster interval |
|---|---:|---:|
| Efficient tree, informed trajectory, 16 checkpoints, sigma 4 | 0.664 | [0.647, 0.681] |
| Efficient tree, informed trajectory, 128 checkpoints, sigma 4 | **0.693** | [0.674, 0.711] |
| Standard tree, informed trajectory, 128 checkpoints, sigma 4 | 0.693 | [0.673, 0.714] |
| Efficient tree, round-65 insertion, 128 checkpoints, sigma 4 | 0.631 | [0.607, 0.654] |
| Efficient tree, informed trajectory, 128 checkpoints, sigma 13.862 | 0.560 | [0.552, 0.568] |
| Last noisy prefix, **oracle** background, sigma 4 | 0.599 | [0.587, 0.611] |
| Genuine final-model candidate loss, sigma 4 | 0.507 | [0.505, 0.509] |
| Full trajectory with disjoint public background estimate, sigma 4 | 0.518 | [0.513, 0.522] |

The informed 128-checkpoint score exceeds its own 16-checkpoint score by
0.028 AUC (paired interval [0.010, 0.047]). Its gap over the oracle last-prefix
score is 0.093 [0.074, 0.113], and its gap over the genuine final-model loss
attack is 0.186 [0.165, 0.207]. The last-prefix score is **not** a genuine
endpoint-only attack: its exact background subtraction requires intermediate
model states. It is included as a matched-access control, while the final-model
loss is the actual endpoint-only baseline. At sigma 4, the informed score's
empirical TPR at 1% and 5% FPR is 4.4% and 21.1%; its max attack advantage is
0.333. Thresholds selected on calibration negatives and tested on held-out
identities give TPR/FPR pairs of 6.7%/2.2% (nominal 1%) and 17.8%/4.4%
(nominal 5%). The 1% operating point is noisy with 120 calibration negatives.

## What access changes

The strongest score knows every other record and its batch slot and recomputes
their clipped gradients from the checkpoints. With just 1 of 63 other batch
records unknown, the 128-checkpoint AUC is 0.673; with 8 unknown it is 0.576;
with 32 unknown it is 0.528; with all 63 unknown and a fixed disjoint 128-image
public estimate it is 0.518. This sensitivity is the main practical limit of
the result. The public estimator is deliberately simple, so its poor result
does not rule out stronger attackers.

A same-digit decoy fingerprint scores 0.598 AUC; the true-minus-decoy score
scores 0.586. Thus part of the apparent true-candidate signal is shared with
same-class records. The identity contrast still exceeds chance in this
setting, but the experiment does not establish reconstruction of a specific
unknown record. [The access and decoy figure](dp_ftrl_access.pdf) makes these
limits visible.

## Privacy accounting and interpretation

For one add/remove participation, we report the conservative standard-tree
zCDP upper bound `rho = ceil(log2(T+1))/(2 sigma^2)`, converted at
`delta=1e-5` as `epsilon <= rho + 2 sqrt(rho log(1/delta))`. The efficient
tree is a post-processing of Gaussian tree nodes. At 128 rounds, sigma 4 has
the bound epsilon <= 3.643. Raising sigma to 13.862 gives epsilon <= 1.000
and lowers the informed AUC to 0.560. More per-node noise therefore **did not
backfire** in this test. Repeated releases can accumulate a weak signal, but
the accountant charges for their shared tree nodes; no formal privacy failure
is claimed. The high-noise model's mean validation accuracy is 71.38%.

The [main vector figure](dp_ftrl_audit.pdf) compares the checkpoint count,
standard and efficient trees, late insertion, high noise, and endpoint scores.
`combined_summary.csv` contains every attack's aggregate AUC, TPR, advantage,
and calibrated FPR across these four conditions. `all_controls_summary.csv`,
`all_controls_comparisons.csv`, and the metadata files preserve the detailed
main-control audit. No image, raw candidate score, or individual identifier is
distributed. Code and commands are in [`experiments/dp_ftrl`](../../experiments/dp_ftrl/README.md).

This is one MNIST split and one model family. A full federated claim would
require actual client updates, changing client populations, heterogeneous
data, unknown participation time, and the official federated training stack.
