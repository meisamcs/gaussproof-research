# One-run white-box audit: per-round sparse pilot

This is the first direct audit test of a GAUSSPROOF scoring patch against the
clipped-gradient dot-product score in Algorithm 3 of Steinke, Nasr, and
Jagielski, *Privacy Auditing with One (1) Training Run*. Each audit randomizes
64 known, unmodified MNIST image canaries independently IN/OUT with fair coins
**before training one CNN**. The auditor sees the same pre-update checkpoints,
noisy updates, known canaries, public 64-image background, and mechanism
parameters for every score. It never sees the inclusion bits while scoring.

The pilot mechanism draws eight ordinary examples from a fixed 2,000-record
population each round and independently adds each IN canary with probability
q=0.1. It divides the clipped gradient sum by eight and adds Gaussian noise.
This **oversamples canaries** relative to ordinary records and is a controlled
auditing mechanism, not a reproduction of uniform Poisson DP-SGD. We test
sigma=1 and 4, C=1, learning rate 0.005, and 128 updates. There are 12 fresh
one-run audits per sigma, each with a new canary set, coins, ordinary draws,
and noise. Repetitions assess variability; an operational audit still needs
only one training run.

All methods make exactly 16 positive and 16 negative guesses, fixed before
seeing the bits. We compare the paper's raw clipped-gradient dot product,
public-background residualized dot product, an independent q-aware Gaussian
mixture, and a bounded nonnegative **per-round** sparse decoder with fixed
penalty factor 0.25. The [raw per-run results](one_run_audit.csv),
[paired differences](paired_differences.csv), [summary with run-bootstrap
intervals](summary_with_ci.csv), and [figure](one_run_audit.pdf) are retained.

At sigma=1 the mean correct guesses out of 32 are 19.08 (paper dot product),
19.17 (Gaussian mixture), and 19.83 (sparse). Sparse minus dot product is
+0.75 guesses with a paired 95% run-bootstrap interval [-1.67, 3.08]; sparse
minus mixture is +0.67 [-1.08, 2.25]. At sigma=4 sparse averages 18.08,
versus 18.75 for the paper score. **This pilot does not establish an audit
improvement.** The initially promising single run was not representative.

For each run, the code also computes a lower bound at delta=1e-5 by inverting
the paper's exact Corollary 5.4 p-value and a looser bound using its 2mδ
slack. The familywise 0.05 error budget is divided among the four scores
within that run. We report all runs and never select the largest lower bound
across runs. Descriptive AUC and run-bootstrap intervals are not themselves
privacy lower bounds. The small 64-canary audit has limited statistical power;
its results cannot support a broad epsilon claim.

## Reproduce

The input CSV must match SHA256
`fb60bc58af4dac3554e394af262b3184479833d3cc540ff8783f274b73492d5d`.
Create the disjoint public warm start with `python -m gaussproof.canary_holdout`
as described in the [gallery report](../whitebox_gallery_pilot/README.md), then:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-gaussproof \
python -m gaussproof.one_run_audit \
  --source runs/canary_holdout --data /path/to/mnist_train.csv \
  --config configs/one_run_audit_pilot.json \
  --output runs/one_run_audit_repeated --report reports/one_run_audit_repeated

OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-gaussproof \
python -m scripts.summarize_one_run_audit --report reports/one_run_audit_repeated
```

The code tests the paper's Appendix D examples for the epsilon inversion.
Candidate identities, hidden bits, and raw trajectories stay in ignored
`runs/`; aggregate outputs and [provenance](provenance.json) are here.
