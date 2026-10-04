# Closed-gallery record-linkage replication

This replication tests whether the high-noise trajectory score identifies an
**already known** MNIST image in a 32-record auxiliary gallery. It does not
generate an unknown image. The exact DP-SGD release uses clipping norm 1,
batch size 8, noise multiplier 4, and the original learning rate 0.05. The
candidate participates with probability 0.5 per round. We evaluated 30
distinct held-out target identities (three per digit), twice each, for each
of four private class distributions: 240 trajectories in total. The 16, 32,
64, and 128 release results are nested prefixes of those trajectories.

At 128 releases, with a public background matched to each private class
distribution, the exact top-1 rates were:

| Private class distribution | Exact top-1 | Same-class top-1 | Exact top-5 |
| --- | ---: | ---: | ---: |
| Balanced | 0.133 | 0.650 | 0.433 |
| Long-tail | 0.150 | 0.600 | 0.483 |
| Candidate class rare | 0.150 | 0.683 | 0.417 |
| Candidate class dominant | 0.133 | 0.600 | 0.367 |

Each cell has 60 trials clustered into 30 identities. The random-gallery
top-1 and top-5 rates are 0.031 and 0.156. A label-only oracle guessing
uniformly within the target's digit has expected exact top-1 of 0.317;
same-class retrieval exceeds that reference in all four matched-background
conditions. The exact top-1 identity-bootstrap intervals remain broad: for
balanced data, [0.033, 0.250]. Top-1 at 128 releases is not uniformly higher
than at 64, so these data do not support a monotone retrieval claim.

The background-mismatch control exposes a class shortcut. On
candidate-dominant private data, a balanced public background produces
open-gallery digit agreement of 1.000 even after the true image is removed,
but same-class top-1 falls to 0.250, below label-only random guessing. The
large digit result is therefore **not** evidence of recovery of the target
image. Mismatch also lowers exact top-1 in the long-tail and rare-class
conditions to 0.067. The plots and full aggregate metrics are in this
directory; row-level identities and trajectories remain in ignored `runs/`.

Reproduce from the source run and MNIST CSV:

```bash
python -m gaussproof.canary_gallery \
  --source /path/to/canary_holdout \
  --data /path/to/mnist_train.csv \
  --config configs/canary_gallery_replication.json \
  --output runs/canary_gallery_replication
```

The source run and dataset are hash-checked by the runner. The source run's
public checkpoint and split roles are reused, with all gallery candidates
drawn from its held-out role.
