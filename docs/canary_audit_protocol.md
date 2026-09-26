# Persistent canary fingerprint audit

This audit tests the narrow hypothesis that a known sample's gradient fingerprint
can be detected by aggregating repeated noisy DP-SGD releases. It is separate from
unknown-image reconstruction and does not use diffusion.

## Mechanism

Start from a public pretrained SmallCNN. At each of 16 rounds, choose a fixed-size
batch of eight examples, clip each per-example gradient to C=1, average, add
Gaussian noise with coordinate standard deviation `sigma*C/B`, and update the CNN.
The known canary is included independently with probability 0.5 in positive
trajectories and never included in negative trajectories. The remaining batch
examples come from a disjoint population. The inclusion schedule is hidden from
the score.

At every pre-update checkpoint the auditor computes the canary's clipped gradient
and a mean clipped gradient over 64 public background examples. These are permitted
white-box audit inputs. The auditor sees the complete released gradient sequence,
but not batch indices, private labels, or the inclusion schedule.

For each release, use the known-fingerprint Gaussian matched score

\[
 \ell_t = \frac{(y_t-b_t)^T(h_t-b_t)}{\tau^2}
          - \frac{\|h_t-b_t\|^2}{2\tau^2},
 \qquad \tau=\sigma C/B,
\]

where `h_t` is the canary fingerprint and `b_t` is the public background mean.
The reported sequence score is `sum_t ell_t`. A single-release control uses only
the final round's score; the max control scans all rounds and therefore has a
multiple-testing advantage. Actual random background batches remain in the
simulation, so the Gaussian score is not an exact likelihood ratio for the full
DP-SGD process.

## Results

The pilot uses 32 positive and 32 negative trajectories per noise setting, with
the same public initial CNN and disjoint role splits. Conservative replace-one
zCDP accounting without sampling amplification gives epsilon upper bounds 665.6,
70.4 and 11.6 for sigma 0.25, 1 and 4 respectively at delta=1e-5. These are not
strong-privacy settings; they establish feasibility only.

| σ | Sequence-sum AUC | Final-release AUC | Max-over-rounds AUC |
|---:|---:|---:|---:|
| 0.25 | 0.946 | 0.612 | 0.787 |
| 1 | 0.864 | 0.621 | 0.744 |
| 4 | 0.646 | 0.531 | 0.563 |

The sequence score is materially better than one release at sigma 0.25 and 1.
At sigma 4 it is close to chance. This demonstrates persistent fingerprint
detectability under these conditions, not recovery of an unknown private image.
The canary is known and its checkpoint gradient is explicitly available; this is
an auditing threat model. A target-specific candidate bank is stronger than a
realistic unknown-sample attacker and must be labeled as such.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=.cache/matplotlib \
python -m gaussproof.canary_audit --data /path/to/mnist_train.csv \
  --config configs/canary_audit.json --output runs/canary_audit
```

Outputs include `sequences.csv`, `rounds.csv`, `summary.csv`, privacy accounting,
and `canary_auc.pdf/png`. The canary index and all splits are persisted. The raw
private trajectory arrays are ignored by the package and are not published.

The next sound step is to calibrate the matched score on held-out canaries and
test unseen canary identities. Only after this audit is reproducible should a
diffusion model be compared against the matched fingerprint statistic.
