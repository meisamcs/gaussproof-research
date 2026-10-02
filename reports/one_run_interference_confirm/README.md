# Natural-canary interference confirmation (negative result)

This report evaluates the **frozen** follow-up to the exploratory
[eight-run pilot](../one_run_interference_pilot/README.md). The code, two
digit classes, 64 canaries, public-initial-gradient cohort selection,
Poisson q=0.1 sampling for ordinary records and IN canaries, 256 updates,
C=1, sigma=2, and all attack hyperparameters were unchanged. Twenty new
seeds supplied new natural-canary subsets, independent audit bits, private
batches, and Gaussian draws. Each audit trains one evolving CNN. The public
checkpoint and underlying held-out MNIST pool are shared across repetitions,
so this is **not** a cross-architecture or cross-dataset result.

The mean within-class absolute gradient cosine was 0.631 (low cohort) and
0.855 (high cohort), with effective Gram ranks 3.88 and 2.55 among 64
canaries. Thus the manipulation changed overlap, but the high cohort was
also close to collinear.

| cohort | paper dot | centered dot | q-aware mixture | per-round sparse | joint trajectory |
| :--- | ---: | ---: | ---: | ---: | ---: |
| low overlap | 17.25 | 17.50 | 17.60 | 16.30 | 17.05 |
| high overlap | 17.15 | 16.60 | 17.10 | 15.95 | 16.65 |

Entries are mean correct guesses out of the 16-IN/16-OUT fixed budget;
chance is 16. The preregistered primary high-overlap contrast,
joint-minus-per-round-sparse, is only +0.70 guesses, with paired 95%
run-bootstrap interval **[-0.10, 1.55]**. The required high-minus-low
interaction against sparse is -0.05 **[-1.60, 1.50]**. Against the paper
dot score in high overlap, the joint score is -0.50 [-2.20, 1.35]. No
method produced a positive paper-Corollary-5.4 one-run epsilon lower bound
at delta=1e-5 under the fixed within-run familywise correction. AUCs were
0.50–0.54. **The proposed fixed-natural-canary audit advantage did not
replicate.** The pilot's positive contrast must not be promoted into the
paper as a confirmed gain.

The [plot](interference_audit.pdf), [per-run aggregates](one_run_interference.csv),
[clustered summary](summary_with_ci.csv), [paired contrasts](paired_differences.csv),
and [provenance](provenance.json) permit review. The individual audit bits,
canary identities, and raw scores remain in ignored `runs/`.

## Reproduce

Use the same public source and MNIST CSV as the pilot, then:

```bash
python -m gaussproof.interference_audit \
  --source runs/canary_holdout --data /path/to/mnist_train.csv \
  --config configs/one_run_interference_confirm.json \
  --output runs/one_run_interference_confirm \
  --report reports/one_run_interference_confirm
python -m scripts.summarize_interference_audit \
  --report reports/one_run_interference_confirm
```
