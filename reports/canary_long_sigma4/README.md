# High-noise fingerprints over long trajectories

This exact DP-SGD run fixes `sigma=4` and extends the same 96 paired unseen-
canary trajectories from 16 to 128 releases.

| releases | sum-LLR AUC | paired gain from 16 | 95% interval for gain | epsilon upper bound |
|---:|---:|---:|---:|---:|
| 16 | 0.609 | 0.000 | [0.000, 0.000] | 11.60 |
| 32 | 0.675 | +0.067 | [+0.010, +0.124] | 17.57 |
| 64 | 0.768 | +0.159 | [+0.094, +0.218] | 27.19 |
| 128 | 0.819 | +0.211 | [+0.144, +0.272] | 43.14 |

This is the strongest positive result in the current audit: high per-round
Gaussian noise suppresses a short-window attack, but repeated matched
fingerprints accumulate into a strong signal. The effect appears on exact
evolving CNN checkpoints and does not require diffusion.

The privacy cost grows with the number of observations. Therefore this result
supports trajectory-aware accounting and release minimization. It does not show
that increasing sigma at fixed composition increases vulnerability, and it does
not violate differential privacy's post-processing guarantee.

See the [protocol](../../docs/canary_long_trajectory_protocol.md),
[summary CSV](summary.csv), [privacy CSV](privacy.csv), and
[paper-ready plot](long_trajectory_auc.png).
