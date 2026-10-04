# Matched access and fingerprint robustness on a useful MNIST CNN

This study asks whether a q-aware GAUSSPROOF fingerprint score has an
operational advantage over a simpler linear projection when the attacker sees
the same noisy DP-SGD releases. It also compares with a strong final-model
LiRA-style control. The answer in this test is **no**: the score tracks linear
alignment closely, and the endpoint control performs better.

The evolving CNN uses MNIST, `C=1`, batch size 8, noise multiplier `sigma=4`,
learning rate 0.005, `T=128`, and a known natural candidate eligible with
probability `q=0.5` at every step. The initial checkpoint and 64-record
background sample are public to the trajectory attacker. The target's
participation schedule, sampled private minibatches, and Gaussian draws are
hidden. Twenty stratified natural identities calibrate scores; **80 different
identities** are held out. Presence and absence runs are paired by seed.
Mean final accuracy is 71.8% on 128 disjoint public-query images. The
conservative no-amplification replace-one zCDP upper bound is
`epsilon <= 43.14` at `delta=1e-5`, so this is not a small-epsilon result.

| Attacker view | GAUSSPROOF AUC | Equal-access alignment AUC |
|---|---:|---:|
| Every release, current fingerprint | 0.763 | 0.764 |
| Every fourth release | 0.666 | 0.669 |
| Every sixteenth release | 0.582 | 0.580 |
| All releases, fingerprint held for four rounds | 0.764 | 0.764 |
| All releases, fingerprint held for sixteen rounds | 0.764 | 0.767 |
| All releases, assume noise 3 instead of 4 | 0.760 | 0.764 |
| All releases, assume noise 5 instead of 4 | 0.765 | 0.764 |

The full-access paired AUC gap, GAUSSPROOF minus alignment, is `-0.0011`
with identity-bootstrap 95% interval `[-0.0081, 0.0059]`. No access-stress
condition has a GAUSSPROOF-over-alignment interval excluding zero. Full
results, including pointwise intervals, calibrated TPR **and achieved FPR**,
are in [summary.csv](summary.csv). The figures are
[access_robustness.pdf](access_robustness.pdf) and
[gain_over_alignment.pdf](gain_over_alignment.pdf).

The final-model-only one-sided offline LiRA-style control uses 32 disjoint
OUT reference models with the same initialization and training settings. Its
held-out AUC is `0.799`; the final candidate loss alone is `0.576`.
GAUSSPROOF minus LiRA-style AUC is `-0.0358`, paired 95% interval
`[-0.0648, -0.0068]`. A calibration-selected fusion (25% trajectory weight)
has AUC `0.798` and does not improve on LiRA. See
[strong_endpoint_comparison.csv](strong_endpoint_comparison.csv) and its
[provenance](strong_endpoint_provenance.json). The OUT references came from a
previous matched, disjoint-public-data run; the comparison script verifies
candidate order, noise, learning rate, and checkpoint hash before scoring.

The 5% FPR threshold is calibrated from only 20 negative identities and is
coarse. For example, its GAUSSPROOF holdout FPR is 11.25%, so its holdout TPR
must **not** be reported as TPR at 5% FPR. No 1% FPR claim is supported.
The error bars in the figures are 95% paired-identity bootstrap standard-error
intervals; the seven access restrictions are exploratory, not a
familywise-confirmatory discovery search.

The method tested here is the **q-aware per-round Gaussian-mixture fingerprint
score**, not every sparse dictionary or denoising variant elsewhere in the
repository. The linear comparator projects the same centered noisy updates
onto the same candidate-minus-background clipped fingerprints. Sparse
checkpoint access means one release in each block; stale fingerprints mean
all releases are visible but the public checkpoint-derived fingerprint is
held fixed for a block. In neither case does the attacker see private batch
membership. The apparent resilience to stale fingerprints is shared by
alignment and may reflect the small learning rate; it is not a unique
GAUSSPROOF property. When all exact updates and optimizer settings are
available, intermediate checkpoints can sometimes be recomputed; the stale
condition is therefore an estimator-ablation test, not proof that a real
observer is unable to obtain fresh checkpoints.

## Reproduce

The MNIST CSV and individual labeled score rows remain under ignored `runs/`.
The public report contains only aggregates, metadata, and figures.

```bash
python -m gaussproof.trajectory_access_robustness \
  --source /path/to/canary_holdout \
  --data /path/to/mnist_train.csv \
  --config configs/trajectory_access_robustness.json \
  --output runs/trajectory_access_robustness \
  --report reports/trajectory_access_robustness
python -m scripts.compare_access_lira \
  --run runs/trajectory_access_robustness \
  --references /path/to/trajectory_endpoint_high_utility/out_references.npz \
  --output reports/trajectory_access_robustness
python -m scripts.plot_trajectory_access_robustness \
  --report reports/trajectory_access_robustness
```
