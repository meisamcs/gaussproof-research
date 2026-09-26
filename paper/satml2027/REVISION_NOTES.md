# Revision after Reza's comments

## What was run

One focused experiment addresses all three requested checks:

- noise multiplier `sigma=4`, clip norm `C=1`, batch size `B=8`;
- `q = 0.5, 0.1, 0.02, 0.004`;
- nested prefixes `T = 16, 32, 64, 128`;
- 20 stratified unseen identities, two per MNIST digit;
- three paired positive/negative trajectories per identity;
- 480 exact evolving trajectories in total;
- 2,000 identity/paired-sequence bootstrap repetitions.

Lower-q conditions share the same random uniforms, batches, and Gaussian-noise
seeds as higher-q conditions. A dedicated inclusion RNG makes every lower-q
schedule an exact subset of the paired higher-q schedule.

## Main result

At `T=128`, the q-aware trajectory AUCs are:

| q | Expected inclusions | Trajectory AUC | Final-checkpoint AUC |
|---:|---:|---:|---:|
| 0.5 | 64.0 | 0.805 | 0.730 |
| 0.1 | 12.8 | 0.571 | 0.549 |
| 0.02 | 2.56 | 0.531 | 0.517 |
| 0.004 | 0.512 | 0.503 | 0.498 |

The final-checkpoint score is the known candidate's negative cross-entropy from
the model after step T. It receives no intermediate model or update.

The q=0.004 condition is chance-level. In that condition, 61.7% of positive
runs realized zero inclusions, close to the theoretical 59.9%. This is the
correct finite-window boundary and prevents extrapolating the q=0.5 finding to
ordinary record sampling.

At q=0.5 and T=128, the trajectory-minus-endpoint gap is +0.076 with paired 95%
interval [-0.001, 0.148]. We therefore do not claim a statistically clear
interface advantage from this pilot, although the trajectory point estimate is
higher.

## Manuscript changes

- Abstract and contribution list now state that the strong result requires
  frequent participation.
- Experimental setup documents the q sweep, 20 identities, nested schedules,
  and endpoint-only score.
- A new results subsection, figure, and table report every q value and the
  final-checkpoint comparison.
- Limitations now report the measured low-q boundary instead of proposing it as
  future work.
- The conclusion no longer implies that ordinary records are vulnerable within
  128 rounds.
- Validation count is updated from 46 to 48 tests.

## Validation

- 48 tests passed.
- 1,920 prefix rows were produced.
- Inclusion counts are monotone across the paired q schedules.
- Negative endpoint scores are identical across paired q conditions.
- Aggregate CSVs and the paper figure are included in this package.
