# Participation-rate sensitivity and endpoint comparison

## Question

Does the high-noise trajectory signal persist when a candidate participates
less often, and how does it compare with an attack that sees only the final
trained checkpoint?

## Fixed mechanism

- MNIST small CNN and public initialization from `runs/canary_holdout`.
- Clip norm `C=1`, batch size `B=8`, learning rate `0.05`, noise multiplier
  `sigma=4`.
- Prefix lengths `T in {16, 32, 64, 128}` from each exact evolving run.
- Candidate participation `q in {0.5, 0.1, 0.02, 0.004}`.  The last value is
  approximately `B/N=8/2000`.
- Twenty unseen target identities: the first two records of each digit in the
  persisted 1,000-record test role.  This selection is fixed before outcomes
  are computed.  Each identity has three paired positive and negative runs.

The inclusion RNG is separate from batch selection.  A shared seed is used
across q conditions, so every lower-q inclusion schedule is a subset of the
corresponding higher-q schedule.  Positive and negative conditions also share
the seed and Gaussian-noise stream.  Model trajectories still diverge after a
candidate is included, as they must.

## Attacks

1. **Full trajectory:** the q-aware Bernoulli-mixture likelihood
   `sum_t log((1-q)+q exp(ell_t))`, where `ell_t` is the checkpoint-specific
   Gaussian fingerprint LLR.
2. **Final checkpoint:** negative cross-entropy of the known candidate under
   the model after step T.  This score uses the final model and candidate only;
   it does not use a released update, an intermediate checkpoint, or a hidden
   inclusion indicator.

Both attacks are evaluated on exactly the same positive and negative model
runs.  Tie-aware AUC is the primary metric.  Two thousand bootstrap repetitions
resample target identities and paired sequence indices together.

## Interpretation gate

The positive hypothesis makes the candidate eligible independently each round;
it does not condition on at least one realized inclusion.  At `q=0.004` and
`T=128`, the expected number of inclusions is `0.512` and the probability of no
inclusion is `(1-q)^T`, about `0.599`.  Those zero-inclusion positive runs are
observationally identical to their negative counterpart.  A weak low-q AUC is
therefore an expected boundary of this finite observation window, not evidence
that the implementation failed.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=.cache/matplotlib \
  .venv-diffusion/bin/python -u -m gaussproof.canary_q_sensitivity \
  --source runs/canary_holdout \
  --data /path/to/mnist_train.csv \
  --config configs/canary_q_sensitivity.json \
  --output runs/canary_q_sensitivity_paired
```

Completed output directories are never overwritten.
