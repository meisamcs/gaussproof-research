# GAUSSPROOF

Research code and aggregate evidence for **candidate-conditioned white-box audits of noisy gradient trajectories**. The central question is when repeated, hidden participation by a *known* record leaves a detectable fingerprint in DP-SGD updates. The repository also contains earlier membership-inference, DP-FTRL, diffusion, and reconstruction studies; those are separately scoped experiments, not evidence for the central claim.

![Held-out fingerprint signal across participation and noise conditions](reports/canary_first_principles_holdout/fingerprint_signal.png)

The plotted noise comparison uses one paired sequence per identity in every arm; the primary `sigma=4` table below uses both saved sequences. See the [report](reports/canary_first_principles_holdout/README.md) for both estimands.

## Result and boundary

An observer knows a candidate record, the clipping and noise parameters, a public background estimate, and each released update and pre-update model state. The observer recomputes the candidate's clipped gradient at each state and sums a Bernoulli-participation Gaussian-mixture score. Private minibatch membership and sampled noise are hidden. This is a white-box audit with stronger access than a final-model-only attack. The likelihood approximation and background assumptions are described in the [system model](docs/system_adversary_model.md) and [research position](docs/first_principles_research_position.md).

| Exact evolving MNIST CNN condition | Held-out result | Interpretation |
| --- | --- | --- |
| `q=0.5`, `sigma=4`, 80 identities | AUC 0.633 at 16 releases; 0.843 at 256 | Frequent participation and longer exposure increase detection. |
| `q=0.1`, `sigma=4`, 80 identities | AUC 0.530 at 16; 0.589 at 256 | The same effect is much weaker at lower participation. |
| `q=0.004`, `sigma=4`, 20 identities | AUC 0.503 at 128 | The ordinary-minibatch proxy is at chance in this window. |
| `q=0.5`, 256 releases, matched 80 identities | AUC 0.910, 0.843, 0.737 at `sigma=2,4,8` | More noise **reduces** detection at fixed exposure. |

The [80-identity report](reports/canary_first_principles_holdout/README.md) gives paired intervals, exact configurations, inclusion counts, and CSVs. The [q-sensitivity report](reports/canary_q_sensitivity/README.md) covers the ordinary-rate proxy. At `q=0.5`, the candidate is expected to participate 128 times in 256 updates; the conservative no-amplification bound is `epsilon <= 70.39` at `delta=1e-5`. This is **not** a small-epsilon result or evidence that noise backfires. Its nominal 5% FPR calibration achieved 10–12% FPR, so low-FPR usability is unproven.

The same-access [Nasr-style gradient projection](reports/trajectory_endpoint_replication/README.md) and final-model LiRA match or exceed the q-aware score when the model has useful accuracy. A [one-run natural-canary confirmation](reports/one_run_interference_confirm/README.md) did **not** establish an audit advantage or a positive empirical epsilon lower bound. GAUSSPROOF here characterizes a repeated-exposure regime; it does not claim a new optimal projection, unknown-image reconstruction, or superiority to a full Nasr audit. AUC is not an empirical DP lower bound.

## Start here

Use Python 3.10+ in a fresh environment. The basic benchmark needs no dataset for its code checks:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -m gaussproof.cli smoke
```

The current suite discovers 77 tests. Without the optional diffusion/trajectory packages, 20 are skipped; CI also exercises the optional profile. The [artifact guide](REPRODUCIBILITY.md) gives the exact data format, rerun commands, compute scope, provenance, and a command to regenerate the headline figure from committed aggregate CSVs. The [student handoff](STUDENT_HANDOFF.md) identifies remaining research gates. Tests do not require MNIST; full experiment runs do.

For a small end-to-end benchmark, [generate the exact MNIST CSV](REPRODUCIBILITY.md#environment-and-inputs) from public IDX files, then run:

```bash
OPENBLAS_NUM_THREADS=1 python -m gaussproof.cli run \
  --config configs/quick.json --data data/mnist_train.csv --output runs/quick
python -m gaussproof.cli report --run runs/quick --output reports/quick
```

Use a **new** output directory for each run. `runs/`, datasets, checkpoints, raw scores, and identity-level files are ignored by Git. The checked-in `reports/` contain aggregate data and figures only. `configs/full.json` is a large research sweep, not a quick-start target.

## Where to look

| Location | Purpose |
| --- | --- |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Environment, data, exact headline rerun, provenance, artifact checks. |
| [docs/first_principles_research_position.md](docs/first_principles_research_position.md) | Main hypothesis, score, assumptions, and current interpretation. |
| [reports/canary_first_principles_holdout/](reports/canary_first_principles_holdout/README.md) | 80-identity trajectory/participation/noise replication and plotted aggregates. |
| [reports/trajectory_endpoint_replication/](reports/trajectory_endpoint_replication/README.md) | Useful-model LiRA/RMIA and equal-access projection comparison. |
| [reports/one_run_interference_confirm/](reports/one_run_interference_confirm/README.md) | Negative one-run audit confirmation. |
| [docs/protocol.md](docs/protocol.md) and [reports/pilot/](reports/pilot/README.md) | Earlier sparse-decoder MIA benchmark; a different experiment from the q-aware audit. |
| [reports/dp_ftrl_repeated_q/](reports/dp_ftrl_repeated_q/README.md) | Exploratory client-level tree-noise stress test, with separate mechanism and access limits. |
| [reports/diffusion/](reports/diffusion/README.md) and [reports/reconstruction/](reports/reconstruction/README.md) | Separate denoising and reconstruction laboratories; neither proves the audit claim. |
| [paper/satml2027/main.tex](paper/satml2027/main.tex) | Current working manuscript source; [build status](paper/satml2027/README.md) notes the checked-in PDF is stale. |

The earlier benchmark's “GAUSSPROOF” implementation is a bounded sparse decoder. The current manuscript's GAUSSPROOF score is a q-aware Gaussian-mixture likelihood over time. Both remain in the code; results and tables must name the specific score. The RERO-style baseline in the original benchmark is mean gradient alignment, **not** a full reproduction of RERO's informed reconstruction attack. Black-box endpoint attacks and white-box trajectory attacks have different access; comparisons must state that difference.

## Research and reuse status

This is an active research artifact, not a released, accepted paper or a certified privacy auditor. The current manuscript PDF predates the latest source and must be rebuilt before circulation. No software license has been selected; public visibility alone does not grant reuse rights. See [CONTRIBUTING.md](CONTRIBUTING.md) for experiment and review rules.
