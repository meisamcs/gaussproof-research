# Ordinary-rate participation at a matched conservative privacy bound

This experiment asks whether giving an ordinary-rate MNIST record more chances
to appear makes its gradient fingerprint detectable **when per-release noise
is increased to keep the same conservative total privacy upper bound**. The
answer in this setting is **no**. The candidate's eligibility probability is
`q=0.004`, approximately `B/N=8/2000` for the fixed population and batch.
This is a Bernoulli inclusion sampler with a fixed-size background batch,
not an independently audited production DP-SGD sampler.

Both runs start from the same public CNN and use the same 2,000-record
background population, clipping norm `C=1`, batch size `B=8`, and 100
stratified natural candidate identities. The first two per digit calibrate
scores; the other **80 identities** are held out. Positive and negative
worlds share seeds, but the candidate appears only in the positive world.
The 128-step run uses `(sigma, learning rate)=(4, 0.005)`; the 512-step run
uses `(8, 0.00125)`. The smaller learning rate was fixed in advance so
`T × learning rate` stays constant and model utility remains comparable.
This is a *joint training schedule change*, so the AUC difference is not an
isolated causal effect of the number of releases.

The no-amplification replace-one zCDP upper bound has `rho=2T/sigma²=16`
in both cases, giving `epsilon <= 43.14` at `delta=10^-5`. This matches a
**conservative upper bound**, not the exact privacy loss or a tight
subsampled accountant. It is not a small-epsilon demonstration.

| T | sigma | Expected / observed appearances | Zero-appearance positive runs | Public accuracy | GAUSSPROOF AUC | Alignment AUC | Final-loss AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 0.51 / 0.46 | 58.8% | 69.3% | 0.502 | 0.502 | 0.501 |
| 512 | 8 | 2.05 / 2.24 | 12.5% | 68.5% | 0.505 | 0.503 | 0.501 |

The held-out GAUSSPROOF AUC increase is `+0.0030` with a paired
identity-bootstrap 95% interval `[-0.0082, +0.0143]`. Neither run separates
members from nonmembers meaningfully, despite the much lower fraction of
zero-appearance runs at `T=512`. Both the q-aware Gaussian-mixture score
and an equal-access linear fingerprint alignment see the full noisy
trajectory and all pre-update checkpoints. The endpoint control uses only
the final candidate loss; LiRA and RMIA are **not rerun** in this cell.
The data do not establish an algorithm-specific GAUSSPROOF advantage, an
increasing-noise backfire effect, or a privacy violation.

Exact aggregates and paired intervals are in [summary.csv](summary.csv)
and [paired_differences.csv](paired_differences.csv). The
[vector figure](ordinary_q_budget.pdf) has a matching
[300-dpi preview](ordinary_q_budget.png). Per-run provenance is in
[T128](T128/provenance.json) and [T512](T512/provenance.json), with the
individual condition summaries in those subdirectories. The raw labeled
score rows, candidate IDs, private trajectories, and MNIST CSV remain in
ignored `runs/`, not on GitHub.

The q-aware likelihood uses a public 64-record background-gradient mean
and the known candidate's clipped gradient at each checkpoint. At
`q=0.004`, almost 59% of short positive runs realize no inclusion, so a
short-window membership detector has little evidence to use. Long runs
increase appearances, but doubling `sigma` offsets the added releases in
the conservative composition measure. The model and reference background
are fixed; the reported identity-bootstrap intervals are descriptive for
this MNIST split and training schedule, not population-wide privacy bounds.

## Reproduce

Run the two configurations with `gaussproof.trajectory_access_robustness`
and separate output directories, then compare their ignored raw score rows:

```bash
python -m gaussproof.trajectory_access_robustness \
  --source /path/to/canary_holdout --data /path/to/mnist_train.csv \
  --config configs/ordinary_q_budget_T128.json \
  --output runs/ordinary_q_budget_T128 --report reports/ordinary_q_budget/T128
python -m gaussproof.trajectory_access_robustness \
  --source /path/to/canary_holdout --data /path/to/mnist_train.csv \
  --config configs/ordinary_q_budget_T512.json \
  --output runs/ordinary_q_budget_T512 --report reports/ordinary_q_budget/T512
MPLCONFIGDIR=/tmp/mpl-gaussproof python -m scripts.summarize_ordinary_q_budget \
  --short runs/ordinary_q_budget_T128 --long runs/ordinary_q_budget_T512 \
  --output reports/ordinary_q_budget
```
