# Participation-rate sensitivity and endpoint comparison

This report addresses three reviewer concerns in one fixed-design run:
participation probability, final-model-only access, and the number of unseen
target identities.

## Design

- Exact evolving MNIST CNN DP-SGD at `sigma=4`, `C=1`, `B=8`.
- `q = 0.5, 0.1, 0.02, 0.004`; `T = 16, 32, 64, 128`.
- Twenty stratified unseen identities, two per digit, with three paired
  positive and negative sequences per identity.
- 480 full trajectories and 1,920 prefix-level score rows.
- Shared seeds across q, with a separate inclusion RNG, make lower-q schedules
  exact subsets of higher-q schedules.
- 2,000 bootstrap repetitions resample identities and paired sequence indices.

The trajectory attack is the q-aware Bernoulli-mixture LLR. The endpoint
baseline is negative candidate cross-entropy from the model after step T and
uses no intermediate checkpoint or released update.

## Result at 128 releases

| q | Expected inclusions | Trajectory AUC (95% CI) | Endpoint AUC (95% CI) |
|---:|---:|---:|---:|
| 0.5 | 64.0 | 0.805 [0.749, 0.882] | 0.730 [0.686, 0.801] |
| 0.1 | 12.8 | 0.571 [0.547, 0.620] | 0.549 [0.523, 0.593] |
| 0.02 | 2.56 | 0.531 [0.503, 0.568] | 0.517 [0.494, 0.551] |
| 0.004 | 0.512 | 0.503 [0.488, 0.519] | 0.498 [0.480, 0.515] |

The trajectory-minus-endpoint gap at `q=0.5,T=128` is `+0.076`, with paired
95% interval `[-0.001, 0.148]`; the experiment does not establish a statistically
clear interface gap. It does establish the sensitivity boundary: the strong
trajectory signal attenuates quickly as participation becomes less frequent.

At `q=0.004,T=128`, 61.7% of positive trajectories realized zero inclusions,
close to the theoretical 59.9%. Such runs are observationally identical to
their negative counterpart. The chance-level AUC is therefore the expected
result for this finite window, not an implementation failure.

See `summary.csv`, `trajectory_vs_endpoint.csv`, and `inclusion_summary.csv`
for all prefixes and intervals. The complete method is in
`docs/canary_q_sensitivity_protocol.md`.
