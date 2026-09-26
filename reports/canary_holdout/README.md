# Unseen-canary holdout

This report calibrates a known-fingerprint detector on four canary identities
and evaluates it on four disjoint identities. The same public CNN, background,
population role, batch size, clipping, and 16-round trajectory protocol are used
for both splits.

| sigma | sequence AUC | final-release AUC | sequence TPR at calibrated 1% | realized FPR | sequence TPR at calibrated 5% | realized FPR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.929 | 0.644 | 0.313 | 0.000 | 0.563 | 0.021 |
| 1 | 0.884 | 0.559 | 0.167 | 0.021 | 0.375 | 0.042 |
| 4 | 0.609 | 0.520 | 0.063 | 0.000 | 0.167 | 0.083 |

The sequence detector generalizes to unseen canaries at sigma 0.25 and 1,
whereas one final noisy release is nearly uninformative. At sigma 4 the signal
is weak. There are 48 positive and 48 negative holdout trajectories per noise
setting, so the low-FPR estimates have limited resolution.

This is a known-canary white-box audit. The auditor is given each holdout
canary's checkpoint gradient fingerprint and a public background estimate. It
does not establish unknown-sample image reconstruction or a diffusion advantage.
The conservative no-amplification epsilon upper bounds are 665.6, 70.4, and
11.6 for sigma 0.25, 1, and 4 respectively.

See the [protocol](../../docs/canary_holdout_protocol.md),
[summary CSV](summary.csv), [privacy CSV](privacy.csv), and
[holdout plot](holdout_auc.png).
