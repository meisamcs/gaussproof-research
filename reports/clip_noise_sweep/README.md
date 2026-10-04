# Matched clipping/noise audit on a natural MNIST pair

This report evaluates 25 clipping/noise/batch settings, each at 64 and 128
releases, using an evolving CNN and the fixed-neighbor protocol in
[`docs/clip_noise_sweep_protocol.md`](../../docs/clip_noise_sweep_protocol.md).
The main grid has `C={0.1,1,4}`, `sigma={2,4,8,16,32}`, and batch size 8.
Additional controls use very high noise (`sigma=64,128` at `C=1`), an initially
weak-clipping norm (`C=16`), and larger batches to retain model utility at
`sigma=32,64,128`. The clipped-gradient attacker knows the two candidate
records, the current model state, and the full noisy update trajectory.

## What the results show

* **More noise did not backfire in this experiment.** Across all 36 adjacent
  increases in `sigma` at fixed `C`, batch size, and trajectory length, the
  trajectory AUC decreased. This includes both 64 and 128 releases. Nominal
  paired-bootstrap standard-error intervals for 33 of the 36 drops exclude zero; these
  multiple descriptive comparisons are not a familywise significance claim.
* **Trajectory access can beat a final checkpoint, but its largest gains occur
  with poor model utility.** At `T=128, C=16, sigma=8`, trajectory/final AUC
  is `0.735/0.541` (gain `0.195`, paired 95% interval `[0.143,0.254]`), while
  public-query accuracy is `0.105`. At `C=0.1, sigma=8`, accuracy is `0.757`
  and the AUC gain is only `0.003` (interval `[-0.003,0.008]`).
* **The score is not stronger than a simpler trajectory projection here.**
  The cumulative raw-gradient-alignment control follows GAUSSPROOF almost
  exactly: the largest absolute AUC gap is `0.0036` in 50 comparisons, and
  none of their paired intervals excludes zero. The paired comparison table
  should be used to assess any method-specific claim; there is no
  demonstrated edge in this single-pair sweep.
* **The low-epsilon controls restore utility but not a sizable trajectory
  advantage.** At `C=1, sigma=32, B=32` (conservative `epsilon<=3.64`), the
  CNN reaches `0.694` accuracy and trajectory/final AUC is `0.587/0.581`.
  At `sigma=64, B=64` (`epsilon<=1.76`), accuracy is `0.694` and AUC is
  `0.548/0.543`. At `sigma=128, B=128` (`epsilon<=0.86`), accuracy is
  `0.694` and both attacks reach AUC `0.526`. All three AUC-gain intervals
  include zero. The initial public checkpoint has `0.766` accuracy on the
  same query set.

These findings support a **limited trajectory-observation risk**, not a
claim that increasing Gaussian noise increases membership leakage or that
the current mixture score improves on established gradient alignment.
An attack on a known natural record is also not image reconstruction.

## Main grid at 128 releases

`epsilon` is the conservative no-amplification replace-one *upper* bound at
`delta=10^-5`, not a measured privacy loss or a tight subsampled accountant.
Accuracy uses 256 disjoint public MNIST queries. The full 64- and 128-release
tables, calibrated TPR/FPR, intervals, clipping fractions, and empirical
epsilon lower bounds are in [`summary.csv`](summary.csv).

| Clip C | Noise sigma | epsilon upper | GAUSSPROOF AUC | Raw alignment AUC | Final CNN AUC | Public accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 2 | 118.29 | 1.000 | 1.000 | 1.000 | 0.761 |
| 0.1 | 4 | 43.14 | 0.965 | 0.963 | 0.963 | 0.759 |
| 0.1 | 8 | 17.57 | 0.811 | 0.811 | 0.808 | 0.757 |
| 0.1 | 16 | 7.79 | 0.669 | 0.668 | 0.667 | 0.756 |
| 0.1 | 32 | 3.64 | 0.588 | 0.589 | 0.586 | 0.745 |
| 1 | 2 | 118.29 | 1.000 | 1.000 | 1.000 | 0.737 |
| 1 | 4 | 43.14 | 0.962 | 0.961 | 0.950 | 0.730 |
| 1 | 8 | 17.57 | 0.806 | 0.806 | 0.782 | 0.691 |
| 1 | 16 | 7.79 | 0.657 | 0.658 | 0.643 | 0.519 |
| 1 | 32 | 3.64 | 0.579 | 0.579 | 0.567 | 0.219 |
| 4 | 2 | 118.29 | 1.000 | 1.000 | 0.995 | 0.710 |
| 4 | 4 | 43.14 | 0.951 | 0.952 | 0.893 | 0.595 |
| 4 | 8 | 17.57 | 0.780 | 0.779 | 0.717 | 0.270 |
| 4 | 16 | 7.79 | 0.646 | 0.645 | 0.578 | 0.122 |
| 4 | 32 | 3.64 | 0.576 | 0.576 | 0.517 | 0.107 |

Paper-ready vector figures: [AUC versus epsilon](auc_vs_epsilon_T128.pdf),
[clipping, accuracy, and endpoint gap](clip_noise_tradeoff_T128_with_C16.pdf),
[high-noise extension](low_epsilon_extension_T128.pdf), and
[utility-preserving batch controls](utility_rescue_T128.pdf). Matching
64-release PDF/PNG versions are included. The exact seed-paired method and
noise comparisons are in [`paired_differences.csv`](paired_differences.csv).

The [`low_epsilon_tail.csv`](low_epsilon_tail.csv) extends `C=1, B=8` to
`sigma=64,128`: at `T=128`, trajectory AUC is `0.544,0.522`, respectively,
while accuracy is `0.113,0.105`. The
[`weak_clipping_control.csv`](weak_clipping_control.csv) holds `C=16`; only
`0.5%` of batch gradients clip at `sigma=2` over the run, but noise-induced
model drift increases clipping in later settings. It is therefore a control
for the clipping regime, not a claim that every later gradient remains
unclipped. The [`utility_rescue.csv`](utility_rescue.csv) reports the
larger-batch runs. [`all_cells_summary.csv`](all_cells_summary.csv) combines
every setting and recomputes familywise one-event empirical epsilon lower
bounds across all reported methods and cells.

## Reproducibility and limits

Every physical setting uses 30 paired calibration seeds and 80 independent
paired holdout seeds in each neighboring world. Thresholds are calibrated
before holdout evaluation; bootstrap units are complete H0/H1 seed pairs.
The same public checkpoint, common population of 2,000 examples, background
set of 64, target digit 4, and replacement digit 9 are used throughout.
`q=0.5` is an explicit Bernoulli privileged-slot participation probability,
far above an ordinary record's approximately `8/2000` uniform-minibatch
participation rate. Thus the findings do **not** establish population-wide
MIA performance or leakage under ordinary uniform sampling. They also do
not replicate LiRA's shadow-model calibration or Nasr et al.'s complete
auditing procedure. The raw-alignment score is a same-observation projection
control, not the full Nasr attack.

The candidate and replacement identities are one fixed natural pair;
repetition varies the sampling and Gaussian noise, not the record identity.
The known-pair attacker recomputes per-step clipped gradients and uses a
public background estimate. Reported empirical epsilon lower bounds concern
only this pair and sampler. At the smaller-epsilon settings, the simultaneous
lower bound is zero with this trial count. No result here violates the stated
DP upper bound.

Run the complete study with:

```bash
bash scripts/run_clip_noise_sweep.sh /path/to/mnist_train.csv runs/canary_holdout
```

The script validates the dataset/checkpoint hashes, produces aggregate CSVs
and paired comparisons, and renders each figure as both PDF and 300-dpi PNG.
Individual labeled score rows and MNIST stay under ignored `runs/`.
