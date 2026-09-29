# DP-FTRL checkpoint audit

This experiment asks whether a known, once-participating MNIST canary leaves
more detectable evidence in a DP-FTRL model trajectory than in the final model.
It calls the **unmodified** `FTRLOptimizer`, `CummuNoiseTorch`, and
`CummuNoiseEffTorch` from the [authors' public PyTorch repository](https://github.com/google-research/DP-FTRL)
at commit `513500a8e31e412972a7d457e9c66756e4a48348`. It is a controlled
centralized analogue of one-time client participation, **not** a run of the
authors' CNN or TensorFlow Federated Stack Overflow system.

## Why the tree matters

For the standard tree, a clipped record update inserted in round 1 affects
every subsequent released noisy prefix, even though the record participates
once. For the efficient tree, these sums
have a different, correlated noise covariance. The script probes each Gaussian
draw in the official generator with a basis vector and obtains its exact
checkpoint-noise map `A`; it then uses `K = A A^T` in the joint Gaussian
likelihood. Treating all checkpoint noises as independent would count reused
noise incorrectly.

The optimizer uses one parameter tensor for softmax regression. At step `t`,
`FTRLOptimizer` sets `W_t = -(G_t + N_t)/alpha`. Thus the attack reconstructs
the *noisy prefix* from the saved checkpoint as `-alpha W_t`. The code asserts
this identity against the optimizer's internal state but never supplies the
internal state or sampled noise to the attack. Both presence and absence runs
use the same background schedule and noise seed; no attack score sees that
seed. The canary occupies one fixed slot in its insertion round, or a zero
placeholder under absence. It is never reinserted. Background examples are
drawn without replacement within each run.

The strongest attack knows every other training example and its batch slot,
as in an informed-background reconstruction threat model. It computes their
clipped gradients at each observed checkpoint and subtracts their exact
cumulative contribution. A weaker attack estimates those gradients from a
disjoint 128-image public pool. Partial-information controls know all but 1,
8, or 32 of the 63 other examples in every batch. The test optionally uses
another held-out image of the same digit as a decoy fingerprint and evaluates
the true-minus-decoy identity contrast.

For a known insertion round `r`, the Gaussian shift vector has entries
`d_t = 1[t >= r]`. If `R_t` is the background-subtracted noisy prefix,
`h_r` the candidate's clipped update, and each node has noise standard
deviation `sigma*C`, the attack computes

```
LLR = (d^T K^-1 <h_r,R_t>_t - 0.5 (d^T K^-1 d) ||h_r||^2) / (sigma*C)^2.
```

We compare nested prefixes, a final noisy-prefix score with an oracle
background, and a genuine final-checkpoint candidate-loss attack. The oracle
final-prefix comparator is labeled separately because calculating its exact
background requires intermediate checkpoints and other-record knowledge.

## Reproduce

Python 3.10 or newer with PyTorch, NumPy, Matplotlib, and `absl-py` is required.
The MNIST CSV has one digit label and 784 raw 0--255 pixels per row; the file
is not redistributed. Clone the authors' code at the pinned commit:

```bash
git clone https://github.com/google-research/DP-FTRL.git upstream-dp-ftrl
git -C upstream-dp-ftrl checkout 513500a8e31e412972a7d457e9c66756e4a48348
pip install 'absl-py==2.1.0'
DP_FTRL_UPSTREAM=upstream-dp-ftrl python -m unittest experiments/dp_ftrl/test_audit.py -v
```

An example of the main run (use a new output path for each condition):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/dp_ftrl/run_audit.py \
  --data /path/to/mnist_train.csv --upstream upstream-dp-ftrl \
  --output runs/dp_ftrl_efficient_sigma4 --efficient --public-score \
  --rounds 128 --prefixes 16 32 64 128 --position 1 \
  --batch 64 --alpha 64 --clip 1 --sigma 4 --delta 1e-5 \
  --calibration-per-class 4 --holdout-per-class 6 --repetitions 3 \
  --bootstrap 600 --seed 20260930
```

For the stronger-noise condition, use `--sigma 13.862`; for late insertion,
use `--position 65`; for the standard tree, omit `--efficient`.
`--unknown-counts 1 8 32` adds partial-background controls, and
`--decoy-control` adds the same-class decoy. The 40 calibration and 60 holdout
identities are disjoint and stratified by digit, with three paired repetitions
per identity. The 120 calibration negatives make a 1% FPR threshold resolvable.

Each run saves only `summary.csv`, `comparisons.csv`, and `metadata.json`, with
aggregate metrics and no image rows or individual scores. See `reports/dp_ftrl/`
for the checked outputs and vector/PNG figure.

## Interpretation

The standard-tree accounting bound for a once-participating example is
`rho = ceil(log2(T+1)) / (2*sigma^2)` under add/remove adjacency. We report
the conservative conversion `epsilon <= rho + 2 sqrt(rho log(1/delta))` next
to each attack result. The efficient tree is a post-processing of noisy tree
nodes and inherits that bound. The audit is **not** a claimed violation of
DP-FTRL. At fixed privacy budget, the noise must rise when the tree grows.

Public-background inference remains a separate empirical question. A strong
informed-background AUC does not imply an external observer can reproduce it,
and a same-class decoy may expose class-level rather than identity-level
evidence. This experiment does not reconstruct an unknown image or client
dataset.
