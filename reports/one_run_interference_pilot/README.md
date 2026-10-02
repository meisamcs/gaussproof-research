# Natural-canary interference pilot (exploratory)

We tested whether the GAUSSPROOF joint trajectory score helps specifically
when natural canary gradients overlap. A public initial CNN checkpoint chose
64 unmodified MNIST canaries from the same two digit classes in every
condition: 32 class-1 and 32 class-7 images. A greedy selection used **only
initial clipped-gradient cosine similarity** to form low- and high-overlap
cohorts before drawing any audit bits or private batches. Their mean
within-class absolute cosine was 0.628 and 0.859, respectively. The
corresponding effective Gram ranks averaged 3.9 and 2.6 for 64 canaries;
the high-overlap cohort may therefore be close to unidentifiable.

The actual mechanism sampled 240 ordinary records and every IN canary by
Poisson at q=0.1, giving an expected ordinary batch of 24. Each of eight
independent one-run CNN audits per condition used 256 updates, C=1, and
sigma in {1,2}. The fixed audit budget was 16 IN and 16 OUT guesses. Every
score received the same checkpoints, noisy releases, public background,
canary bank, and q/C/sigma. The joint and sparse hyperparameters were fixed
before this pilot. Run-bootstrap intervals pair scores within repetitions.

| sigma | overlap | paper dot | q-aware mixture | per-round sparse | joint trajectory |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | low | 14.88 | 16.25 | 16.50 | 17.38 |
| 1 | high | 16.62 | 17.00 | 18.62 | 19.00 |
| 2 | low | 15.00 | 16.75 | 16.50 | 17.12 |
| 2 | high | 16.38 | 16.62 | 18.38 | 19.75 |

These are mean **correct guesses out of 32**; chance is 16. The high-overlap
joint score exceeds the paper dot score by 2.38 guesses at sigma=1 (paired
95% run-bootstrap interval [0.88, 3.88]) and by 3.38 at sigma=2
([2.12, 4.88]). Its lead over the strongest same-access per-round sparse
decoder is only 0.38 [-1.25, 1.88] and 1.38 [-0.38, 3.00]. The critical
high-minus-low interaction against sparse is -0.50 [-3.38, 2.25] at sigma=1
and +0.75 [-2.00, 3.75] at sigma=2. Thus **this pilot does not establish that
interference is where joint decoding helps**. The paper's one-run empirical
epsilon lower bound is zero in nearly every run.

The [plot](interference_audit.pdf), [per-run aggregates](one_run_interference.csv),
[clustered summary](summary_with_ci.csv), [paired contrasts](paired_differences.csv),
and [provenance](provenance.json) are included. Because this eight-run result
was used to choose the next confirmation, it must not be treated as an
independent confirmatory study.

## Reproduce

The input data is `/path/to/mnist_train.csv` with the checksum in the
provenance file. Use the disjoint public checkpoint built by
`python -m gaussproof.canary_holdout`, then run:

```bash
python -m gaussproof.interference_audit \
  --source runs/canary_holdout --data /path/to/mnist_train.csv \
  --config configs/one_run_interference_pilot.json \
  --output runs/one_run_interference_pilot \
  --report reports/one_run_interference_pilot
python -m scripts.summarize_interference_audit \
  --report reports/one_run_interference_pilot
```
