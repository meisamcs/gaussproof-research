# GAUSSPROOF contributor handoff

Read the [project README](README.md), [artifact guide](REPRODUCIBILITY.md), and [current research position](docs/first_principles_research_position.md) before changing an experiment. The repository is public and contains aggregate results, code, and a working manuscript. Raw identity-level scores, datasets, checkpoints, and private trajectories stay in ignored local `runs/` and `data/` directories.

For a bounded undergraduate contribution, use the [single-question public-background mismatch project](docs/course_project_background_shift.md). It fixes MNIST, one CNN, and one DP-SGD setting while testing whether the audit depends on a representative public background.

## What is established

The primary experiment is a known-candidate white-box audit of exact evolving MNIST CNN trajectories. At `sigma=4`, 80 held-out identities, and `q=0.5`, the q-aware trajectory AUC rises from 0.633 at 16 releases to 0.843 at 256. At `q=0.1`, the same values are 0.530 and 0.589. On matched identities at `q=0.5,T=256`, increasing `sigma` from 2 to 8 lowers AUC from 0.910 to 0.737. The separate `q=0.004,T=128` ordinary-sampling proxy remains at chance (AUC 0.503). The [primary report](reports/canary_first_principles_holdout/README.md) and [q-sensitivity report](reports/canary_q_sensitivity/README.md) give intervals, data, and caveats.

The high-`q` exposure uses many expected inclusions and a large conservative composed privacy bound. It is not evidence that more noise backfires. In a useful-model comparison, final-model LiRA and a same-access Nasr-style gradient projection match the q-aware score; see the [endpoint replication](reports/trajectory_endpoint_replication/README.md). The [natural-canary one-run confirmation](reports/one_run_interference_confirm/README.md) did not establish an advantage over projection or a positive empirical epsilon lower bound. Image-gallery ranking is not unknown-image reconstruction. Do not promote any of those unestablished claims.

The original MIA benchmark's bounded sparse decoder and the later q-aware Gaussian-mixture score have both been called GAUSSPROOF. Name the **specific score** in every table, figure, and pull request. The “RERO-style” code in the original benchmark is mean alignment, not an implementation of RERO's informed reconstruction attack.

## Priority research gates

1. **Make the low-FPR audit credible.** A nominal 5% threshold selected on 20 calibration negatives achieved 10–12% FPR on the 80-identity holdout. Use a larger independent calibration set, prespecify threshold selection, and report achieved FPR with uncertainty. Do not derive an empirical DP lower bound from pooled AUCs.
2. **Test the realistic participation boundary.** Repeated exposure is established for `q=0.5`, weaker for `q=0.1`, and absent in the tested `q=0.004` window. Run actual uniform minibatch sampling at ordinary `B/N` over longer, composition-accounted horizons, using new unseen identities. Preserve negative results.
3. **Match baselines and access.** Compare q-aware likelihood with raw/centered projection on the *same* releases and candidate gradients; include final-model LiRA/RMIA only with their extra reference-model assumptions stated. Report model utility and calibrate variants without evaluation identities.
4. **Generalize carefully.** A second dataset or model family should retain fixed known candidates, independent target identities, paired randomness, and class/identity controls. The existing DP-FTRL client pilot has a different correlated-noise mechanism and cannot be pooled with DP-SGD AUCs.
5. **Finish the manuscript artifact.** The current source is [`paper/satml2027/main.tex`](paper/satml2027/main.tex); its old PDF was removed because it predates the source. Build and inspect a new PDF before circulation, then check every claim against a committed aggregate table.

## Code and review workflow

Run the data-free checks before a change:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -m gaussproof.cli smoke
python scripts/check_public_artifact.py
```

The suite currently discovers 77 tests; the minimal environment skips 20 optional trajectory/diffusion tests. See [CONTRIBUTING.md](CONTRIBUTING.md) for split isolation, provenance, thresholding, score orientation, and pull-request review rules. Record the dataset hash, source revision, config, seed, software versions, exact command, independent identity count, and all negative controls for a new experiment. Use a new branch and output directory; never replace a completed run or tune on held-out identities.

No software license has been selected. Public visibility does not grant reuse rights; ask the repository owner to choose a license before advertising this as a reusable package.
