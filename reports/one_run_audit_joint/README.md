# Joint-trajectory scoring patch for a one-run audit

This follow-up asks whether **one persistent dataset-level IN/OUT bit per
canary** can be inferred better by jointly explaining all noisy updates than
by scoring each update independently. It is a scoring patch to the one-run
white-box auditor of Steinke, Nasr, and Jagielski, not a new training
algorithm. All scores receive identical checkpoints, released updates,
canaries, public background, clipping/noise/sampling parameters, and fixed
16-positive/16-negative guess budget. The patch uses an eight-iteration,
fixed-damping mean-field approximation: each candidate's trajectory
likelihood subtracts the expected contributions of the *other* candidate
canaries. No hidden inclusion bits enter the patch.

Twenty new one-run MNIST CNN audits were trained for each of sigma=1 and 4;
none are reused from the earlier per-round pilot. The mechanism is the same
**canary-oversampling stress test** as that pilot: 64 image canaries with
independent fair dataset-level IN/OUT coins, q=0.1 conditional per-round
participation, eight fixed ordinary records, C=1, and 128 updates. It does
not reproduce standard uniform DP-SGD sampling. This method was designed
after examining the earlier pilot and then evaluated on fresh runs; it is an
exploratory follow-up.

| sigma | Paper dot product correct / 32 | Independent Gaussian correct / 32 | Per-round sparse correct / 32 | Joint patch correct / 32 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 19.65 | 19.45 | 19.90 | 20.15 |
| 4 | 18.25 | 18.25 | 17.40 | 18.45 |

At sigma=1 the joint patch gains +0.70 correct guesses per run over the
independent Gaussian score, paired run-bootstrap interval [0, 1.40], but
only +0.50 over the paper's dot product, interval [-0.90, 1.80]. Its AUC
gain over the Gaussian score is +0.0232 [0.0100, 0.0363]; the AUC gain over
the paper score is +0.0273 [-0.0027, 0.0556]. At sigma=4 there is no
reliable gain over either strong comparator. The joint score produced a
positive paper-Corollary-5.4 epsilon lower bound in 2/20 sigma=1 runs, equal
to the Gaussian score's 2/20; neither did so at sigma=4. **The primary
auditing advantage over the paper score is unproven.**

For each run the 0.05 familywise error budget is divided across all five
reported scores. We report every run; no largest run or best post hoc guess
count is selected. These per-run lower bounds are not a 95% simultaneous
statement across the 20 research repetitions. The [figure](one_run_audit.pdf),
[run-bootstrap summary](summary_with_ci.csv), [paired contrasts](paired_differences.csv),
[all per-run aggregates](one_run_audit.csv), and [provenance](provenance.json)
make the null result reviewable.

## Reproduce

Use the same pinned public checkpoint and MNIST CSV described in the
[earlier pilot](../one_run_audit_repeated/README.md). Run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-gaussproof \
python -m gaussproof.one_run_audit \
  --source runs/canary_holdout --data /path/to/mnist_train.csv \
  --config configs/one_run_audit_joint_pilot.json \
  --output runs/one_run_audit_joint --report reports/one_run_audit_joint

OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-gaussproof \
python -m scripts.summarize_one_run_audit --report reports/one_run_audit_joint
```

The raw fingerprints, hidden bits and scores remain in ignored `runs/`; only
aggregate outputs and vector/PNG plots are committed.
