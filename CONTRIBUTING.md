# Contributing

## Before starting an experiment

1. Write the hypothesis, threat model, access assumptions, fixed conditions, primary
   metric, and stopping rule in a protocol file under `docs/`.
2. Add a committed JSON configuration under `configs/`.
3. Create a branch named for one bounded experiment.
4. Keep target identities disjoint from public training, calibration, and model-selection
   identities.

## Implementation rules

- Preserve paired random streams when comparing conditions.
- Never select thresholds or hyperparameters using evaluation identities.
- Treat the candidate gallery, candidate gradients, intermediate checkpoints, and full
  trajectory as separate access assumptions and label them in every table.
- Keep score orientation explicit: larger scores must always mean more likely membership.
- Add a test when changing splits, sampling, clipping, noise, accounting, metrics, or
  aggregation.
- Write generated state to a new `runs/` directory; do not overwrite completed runs.

## Pull request checklist

- [ ] `python -m gaussproof.cli smoke` passes.
- [ ] The protocol and configuration were committed before result-driven changes.
- [ ] No dataset, checkpoint, trajectory tensor, row identity, credential, or local path
      is committed.
- [ ] Aggregate CSVs include uncertainty and the number of independent identities.
- [ ] Negative and null results are retained.
- [ ] The report states the precise attacker access and privacy accounting method.
- [ ] Figures can be regenerated from checked-in aggregate data.
- [ ] Manuscript claims match the aggregate results and confidence intervals.
