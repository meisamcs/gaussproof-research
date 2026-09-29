# Independent federated DP-FTRLM checkpoint pilot

This experiment is an **independent, source-aligned pilot**, not an execution
of the authors' TensorFlow Federated (TFF) training driver. The exact official
TFF 0.20 dependency set could not be installed on the local Apple Silicon
Python 3.11 runtime: `tensorflow-federated==0.20.0` requires
`jaxlib~=0.1.76`, for which pip found no compatible wheel. The original
federated code is pinned to
[`google-research/federated@a5af8c4`](https://github.com/google-research/federated/tree/a5af8c4433c9ee2d2b1565f1bcc68c89e2000a6b/dp_ftrl).
The server recurrence follows its `DPFTRLMServerOptimizer`, including
momentum and the sign of tree noise. Client deltas are clipped to a common
norm, equally averaged, and the node-noise standard deviation is
`clip * sigma / clients_per_round`, matching `run_emnist.py`. The efficient
binary-tree recurrence follows TensorFlow Privacy 0.6.0. Targeted tests check
the official optimizer's noise-free fixture and checkpoint inversion.

## Scientific question and access

Can a server-log observer who has **all global checkpoints** and knows a
candidate client's examples detect whether that client contributed **once**?
The main attacks cannot see other clients' data or updates, the tree-noise
draws, or the insertion round. They receive a disjoint public bank of clients.
The informed Gaussian likelihood is a deliberately stronger diagnostic: it
knows all other participating client updates and the insertion round. The
endpoint baseline uses only the final model and candidate examples. Ordinary
federated clients are not assumed to see the entire checkpoint sequence.

The data are genuine writer-partitioned Federated EMNIST digits, fetched from
[Google's FedJAX archive](https://storage.googleapis.com/gresearch/fedjax/emnist/federated_emnist_digitsonly_train.sqlite).
The SHA-256 of the file used here is
`0540b878da75b6a9a8c14fadae5c4b0a62cf12f8ff5fcf675b6f6c6c774df323`.
Each simulated client uses its first 16 examples, pooled to a 7×7 image and
processed by one local softmax-regression SGD step. This lighter model is a
**substantial deviation** from the official EMNIST CNN and full local epoch.
It allows a controlled mechanism check, but cannot establish the performance
of the official TFF application. This run also uses a zero-update placeholder
for the missing client's eighth slot, retaining a fixed divisor of eight in
both worlds; it represents add/remove adjacency under that fixed-slot
interface. Eight clients are scheduled per round, with seven disjoint
background writers and at most one candidate. Background writers do not
repeat within a 128-round run. The worlds share the background schedule and
all tree-noise draws.

From checkpoints `W_t`, the observer computes

```
v_t = (W_0 - W_t) / server_lr
s_t = v_t - momentum * v_(t-1)
```

The official recurrence makes `s_t` the noisy cumulative average client
gradient; its round-to-round difference is the observed noisy update. A
public-bank estimate of the seven other clients' mean is subtracted before
scoring. RERO-style alignment and GAUSSPROOF's bounded nonnegative sparse
decoder both receive the same observations, candidate fingerprint, and public
bank. Their `max` scores adapt to unknown one-time participation; their
`mean` scores show the original repeated-participation aggregation. The
informed likelihood uses the exact tree covariance, so correlated releases
are not treated as independent.

The covariance-aware `public_gls_max` score uses the same public bank and
candidate, while accounting for the tree-noise correlation. It does not
receive the exact private background or insertion round.

## Reproduce

Python 3.10+ with NumPy, Matplotlib, and `msgpack==1.1.1` suffices. Raw client
data and identity-level scores stay outside the repository. Download the
dataset to a private local path and run:

```bash
python -m pip install 'msgpack==1.1.1'
curl -L -o /tmp/federated_emnist_digitsonly_train.sqlite \
  https://storage.googleapis.com/gresearch/fedjax/emnist/federated_emnist_digitsonly_train.sqlite
python -m unittest discover -s experiments/dp_ftrl_independent -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 python experiments/dp_ftrl_independent/run_pilot.py \
  --sqlite /tmp/federated_emnist_digitsonly_train.sqlite \
  --output reports/dp_ftrl_independent/new_run \
  --rounds 128 --position 64 --clients-per-round 8 \
  --calibration-identities 40 --holdout-identities 100 \
  --sigma 1 4 8 --seed 20261007
```

The run writes aggregate-only `summary.csv`, `paired_comparisons.csv`,
`metadata.json`, and vector/PNG AUC figures. Existing nonempty outputs are
never overwritten. The first 40 writer identities calibrate thresholds;
the next 100 are held out. AUC excludes comparisons of positive and negative
runs of the *same* identity, since such self-pairs introduce a small artificial
lift under paired worlds. The pooled AUC is also exported for transparency.
Intervals resample independent identities, preserving each positive/negative
pair. At 40 calibration negatives, a 1% FPR operating point is unresolved;
those CSV fields are `nan`. The 5% threshold is calibrated on the 40 clients
and its actual held-out FPR is reported.

## Interpretation

The 128-round held-out pilot does **not** find an operational public-information
GAUSSPROOF advantage. With the candidate inserted at round 65, cross-identity
AUC at `sigma=4` is 0.500 for sparse `max`, 0.504 for sparse `mean`,
0.503 for public GLS, 0.501 for final-model loss, and 0.651 for the
informed likelihood. The public attack's calibrated 5% operating point
achieves TPR/FPR 0.02/0.02 for sparse `mean` and public GLS; the informed
reference reaches 0.18/0.05. Early insertion at round 1 gives 0.500,
0.503, 0.507, 0.500, and 0.669 AUC, respectively, in the same order.
The sparse `max` score saturates at high noise, making it uninformative.
Larger disjoint public banks (12 versus 64 writers) slightly improve public
GLS AUC at `sigma=4` (0.503 to 0.507), still far from a useful attack.
All published measurements are aggregate-only; the underlying paired scores
and raw federated data are excluded.

The result is a test of a **one-time known-client membership hypothesis**.
It does not reconstruct unknown client data or violate a DP guarantee. The
conservative one-pass privacy upper bound in the CSV uses
`rho = ceil(log2(T+1))/(2*sigma^2)` and
`epsilon <= rho + 2*sqrt(rho*log(1/delta))`, with `delta=1e-5`.
No bound is claimed for clients who participate repeatedly. Model accuracy
is measured on a disjoint group of genuine EMNIST writers and is modest in
this small-client, high-noise setup. A full conclusion needs the pinned TFF
driver, its CNN, the official accountant, and a stronger public-background
model. A strong informed reference result alone cannot support a practical
GAUSSPROOF claim.
