# Reproducing the GAUSSPROOF artifact

This guide distinguishes three operations: checking the code without data, regenerating a figure from committed aggregate data, and rerunning the exact evolving-model experiment. Run commands from the repository root. The manuscript is a working draft; the current source is [`paper/satml2027/main.tex`](paper/satml2027/main.tex).

## Environment and inputs

- Python 3.10+; the reported runs used PyTorch 2.5.1. [`requirements-tested.txt`](requirements-tested.txt) pins the core verification profile. The minimal `pip install -e .` path supports the core benchmark and data-free checks. Install [`requirements-trajectory.txt`](requirements-trajectory.txt) for optional trajectory/diffusion modules.
- `mnist_train.csv` is **not** in this repository. It contains a digit label followed by 784 raw pixel values (0–255) per row, with no header. The recorded 80-identity runs used dataset SHA-256 `fb60bc58af4dac3554e394af262b3184479833d3cc540ff8783f274b73492d5d`. The exact file can be regenerated from the public [CVDF-hosted original MNIST training IDX files](https://github.com/cvdfoundation/mnist#download) using the converter below. A different digest is a different data input, even if it is also MNIST.
- A CPU can run the experiments; the 80-identity `sigma=4` run generated 960 paired trajectories and its completion manifest reports roughly 1,336 seconds for the resumed invocation. The `sigma=2` and `sigma=8` arms each generated 160 trajectories and report roughly 616 and 645 seconds, respectively, on the original machine. These are observed times, not hardware-independent estimates. The prerequisite public-checkpoint run has additional cost.
- `runs/` is ignored. It can contain row identities, private audit bits, trajectories, models, and scores; keep those files local. Only aggregate reports are committed. New runs must use empty output directories.

Create the exact experiment input (all files stay under ignored `data/`):

```bash
mkdir -p data
curl -fL https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz \
  -o data/train-images-idx3-ubyte.gz
curl -fL https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz \
  -o data/train-labels-idx1-ubyte.gz
python -m scripts.prepare_mnist_csv \
  --images data/train-images-idx3-ubyte.gz \
  --labels data/train-labels-idx1-ubyte.gz \
  --output data/mnist_train.csv \
  --expect-sha256 fb60bc58af4dac3554e394af262b3184479833d3cc540ff8783f274b73492d5d
```

The converter validates the IDX headers, 60,000 image/label pairs, image dimensions, and output digest. This exact download-and-conversion path was checked locally against the saved experiment CSV.

## Fast, data-free verification

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -m gaussproof.cli smoke
python scripts/check_public_artifact.py
```

The smoke command currently discovers 77 tests; a minimal environment skips 20 optional tests. CI tests both the minimal and trajectory dependency profiles. The artifact checker verifies report grids, local documentation links, and referenced manuscript figures. Neither command proves that the reported MNIST values independently reproduce.

The committed [two-panel headline figure](reports/canary_first_principles_holdout/fingerprint_signal.pdf) can be regenerated **without** raw scores or MNIST:

```bash
python -m scripts.plot_committed_fingerprint \
  --report reports/canary_first_principles_holdout \
  --output /tmp/gaussproof-fingerprint.pdf
```

This plots the committed aggregate `participation_auc_matched.csv` and `noise_auc.csv`; it does not recompute bootstrap intervals. The underlying AUCs and intervals were obtained from identity-level scores and 2,000 paired bootstrap draws. Those raw scores are intentionally not in Git.

## Rerun the primary 80-identity experiment

First generate the public initial CNN, fixed split, and background used by the later arms:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m gaussproof.canary_holdout \
  --config configs/canary_holdout.json \
  --data data/mnist_train.csv \
  --output runs/canary_holdout
```

Then run the fixed participation arm and the two matched-noise arms. They must use the **same** source run and dataset. The 80 identities are selected with offset two per digit, excluding the earlier two-per-digit pilot block. The `sigma=4` arm uses two paired sequences per identity; the matched-noise figure uses only its sequence zero so that all three noise arms have the same one-sequence design.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m gaussproof.canary_q_sensitivity \
  --source runs/canary_holdout --data data/mnist_train.csv \
  --config configs/canary_first_principles_holdout.json \
  --output runs/fingerprint_sigma4

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m gaussproof.canary_q_sensitivity \
  --source runs/canary_holdout --data data/mnist_train.csv \
  --config configs/canary_first_principles_sigma2.json \
  --output runs/fingerprint_sigma2

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m gaussproof.canary_q_sensitivity \
  --source runs/canary_holdout --data data/mnist_train.csv \
  --config configs/canary_first_principles_sigma8.json \
  --output runs/fingerprint_sigma8
```

The runner rejects completed output directories. `--resume` is for an interrupted arm and accepts only complete, validated q blocks; it does not make a new design comparable to the published run. The saved completion manifests record the dataset digest, configuration, trajectory count, source-checkpoint digest, and implementation digest. Compare those to the [committed manifests](reports/canary_first_principles_holdout/completion.json) before comparing AUCs.

Recompute paired contrasts and the matched one-sequence figure from the new private score files:

```bash
python -m scripts.summarize_first_principles \
  --scores runs/fingerprint_sigma4/scores.csv \
  --output runs/fingerprint_sigma4/paired_contrasts.csv \
  --bootstrap 2000 --seed 20261013

python -m scripts.plot_first_principles \
  --sigma2 runs/fingerprint_sigma2/scores.csv \
  --sigma4 runs/fingerprint_sigma4/scores.csv \
  --sigma8 runs/fingerprint_sigma8/scores.csv \
  --output runs/fingerprint_figure/fingerprint_signal.pdf \
  --bootstrap 2000 --seed 20261015
```

Compare the generated aggregate files with [`summary.csv`](reports/canary_first_principles_holdout/summary.csv), [`paired_contrasts.csv`](reports/canary_first_principles_holdout/paired_contrasts.csv), [`noise_auc.csv`](reports/canary_first_principles_holdout/noise_auc.csv), and [`noise_contrasts.csv`](reports/canary_first_principles_holdout/noise_contrasts.csv). Use identity-clustered intervals and report achieved FPR alongside any calibrated TPR; the current 20-negative threshold calibration did **not** achieve 5% FPR on the holdout.

## Interpretation and other experiments

The primary test is a **known-candidate, white-box, high-participation** exposure experiment. It does not show that raising noise at fixed exposure increases leakage. Longer transcripts also consume more privacy budget. The conservative `delta=1e-5` bound at `q=0.5, sigma=4, T=256` is `epsilon <= 70.39` without amplification. AUC over different candidate identities is not an empirical DP lower bound for one neighboring-dataset pair.

The [q-sensitivity report](reports/canary_q_sensitivity/README.md) covers `q=0.004` at 128 rounds. The [stronger endpoint comparison](reports/trajectory_endpoint_replication/README.md) includes LiRA, RMIA, and a same-access projection on useful models. The [failed one-run confirmation](reports/one_run_interference_confirm/README.md), [DP-FTRL stress test](reports/dp_ftrl_repeated_q/README.md), [diffusion laboratory](reports/diffusion/README.md), and [reconstruction laboratory](reports/reconstruction/README.md) each have their own protocol and access assumptions. Do not pool their AUCs into one leaderboard.

The paper source needs a standard IEEE LaTeX installation and `latexmk -pdf main.tex`. The previously checked-in PDF was older than the source and was removed to prevent circulation as the current manuscript. See [the paper build note](paper/satml2027/README.md).

## Release verification recorded on 2026-10-04

- The data-free smoke suite completed: **77 discovered, 57 passed, 20 optional skips**. With the pinned `msgpack==1.1.1` dependency, all **23 federated unit tests** passed. The optional diffusion/trajectory suite was not run locally in this pass; CI has a separate dependency profile for it.
- The exact dataset CSV named above was found locally and matched the SHA-256 in the saved completion manifests. Regenerating it from the public MNIST IDX files produced the same digest. The `configs/quick.json` end-to-end benchmark completed two target conditions, generated its full raw run, and exported the aggregate report. This checks the train-to-report path; the quick run is not a scientific replication of the headline condition.
- Recomputing paired contrasts and matched participation/noise summaries from the existing private `scores.csv` files produced **exactly the committed CSV rows** in all four relevant aggregate files. The published figure also regenerated from committed aggregates. This validates the report pipeline but is not a new independent target-training rerun.
- The public artifact checker passed. A current manuscript PDF could not be built in this environment because no project-capable TeX installation is present; compilation and visual page review remain outstanding.
