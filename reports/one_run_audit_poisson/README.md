# Uniform-Poisson one-run audit control

This control tests the proposed GAUSSPROOF scoring patch under **the same
per-round sampling rate for ordinary records and eligible canaries**. It is
closer to Algorithm 2/3 of Steinke, Nasr, and Jagielski than our earlier
canary-oversampling stress test. Each audit independently flips fair IN/OUT
coins for 64 unmodified MNIST canaries before training one evolving CNN.
At each of 512 rounds, all 256 fixed ordinary records and every IN canary
are independently sampled with q=8/256=0.03125. Each selected record's
gradient is clipped at C=1; the clipped sum is divided by the public
expected ordinary batch size eight and receives Gaussian noise with standard
deviation sigma*C/8. The actual batch size is variable. The candidate bits,
sampled batches and Gaussian draws are hidden from all scoring methods.

We trained **eight independent one-run audits per noise multiplier**
sigma∈{1,4}, with new canary sets, bits, batches and noise. All five scores
use the same released trajectory, pre-update checkpoints, canaries, public
64-image background and q/C/sigma. We fixed 16 positive and 16 negative
guesses, sparse penalty, and joint-trajectory iteration count in the config
before these runs. The repeated audits estimate variability; each individual
audit itself requires only one training run.

| sigma | Paper dot product correct / 32 | q-aware Gaussian correct / 32 | Per-round sparse correct / 32 | Joint patch correct / 32 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16.75 | 16.38 | 16.88 | 16.13 |
| 4 | 15.75 | 17.00 | 16.63 | 16.63 |

At sigma=1 the joint patch minus the paper score is -0.63 correct guesses
per run, paired 95% run-bootstrap interval [-2.88, 1.38]; at sigma=4 it is
+0.88 [-1.25, 2.75]. The AUC contrasts also include zero. **No method
produced a positive one-run empirical epsilon lower bound** under the
paper's Corollary 5.4 with delta=1e-5 and within-run Bonferroni correction
across five scores. AUC is around chance, and no GAUSSPROOF audit improvement
is supported in this condition. The mean number of active canaries is only
1.015 per round. Final public-query CNN accuracy is about 62.8% at sigma=1
and 62.4% at sigma=4.

This is a finite pilot, not a proof that trajectory patches can never help:
there are eight independent runs per condition, one warm-start CNN, one
256-record ordinary population, and only 64 canaries. The expected number
of appearances for an IN canary is 16. At a 2,000-record ordinary population
with B=8, q would be 0.004 and signal would be weaker at the same T. The
earlier q=0.1 oversampling result must **not** be presented as ordinary
DP-SGD evidence. Increasing sigma did not increase observed audit power.

The [paper-ready figure](one_run_audit.pdf), [per-run aggregates](one_run_audit.csv),
[run-bootstrap summary](summary_with_ci.csv), [paired contrasts](paired_differences.csv),
and [provenance](provenance.json) give the full result. Raw hidden bits and
scores stay in ignored `runs/`.

## Reproduce

Use the pinned public checkpoint and MNIST CSV described in the
[first one-run pilot](../one_run_audit_repeated/README.md), then:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-gaussproof \
python -m gaussproof.one_run_audit \
  --source runs/canary_holdout --data /path/to/mnist_train.csv \
  --config configs/one_run_audit_poisson_pilot.json \
  --output runs/one_run_audit_poisson --report reports/one_run_audit_poisson

OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-gaussproof \
python -m scripts.summarize_one_run_audit --report reports/one_run_audit_poisson
```
