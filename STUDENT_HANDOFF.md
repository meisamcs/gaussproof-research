# GAUSSPROOF student handoff

This repository contains the working research code, experiment configurations,
tests, aggregate results, paper figures, and current manuscript. It is intended to
let a new contributor reproduce the completed experiments and continue the work
without relying on private checkpoints or a particular machine.

For the immediate SaTML deadline, follow the focused
[`docs/STUDENT_MONDAY_EXECUTION_PLAN.md`](docs/STUDENT_MONDAY_EXECUTION_PLAN.md).

## Scientific question

GAUSSPROOF studies whether a weak record-specific gradient fingerprint that is hard
to detect in one noisy DP-SGD release can accumulate across repeated releases. The
central comparison is between access to the released trajectory and access only to
the final trained model.

The strongest result currently supported by the experiments is narrow:

> At noise multiplier 4, repeated white-box releases can accumulate a known canary's
> weak gradient fingerprint when that canary participates frequently. The effect
> weakens sharply as participation probability decreases and is absent in the tested
> 128-round window at the ordinary-sampling proxy q=0.004.

Do not describe the result as evidence that adding more Gaussian noise makes a record
more vulnerable. The experiments show accumulation across releases, which is already
reflected in privacy composition. They also do not yet establish reconstruction of an
unknown private image.

## Current evidence

- The full smoke suite has 48 tests.
- The exact long-trajectory audit at sigma=4 reaches sequence AUC 0.609, 0.675,
  0.768, and 0.819 at 16, 32, 64, and 128 releases.
- The q-sensitivity study uses 20 unseen identities and three paired repetitions per
  identity. At 128 releases, trajectory AUC is 0.805, 0.571, 0.531, and 0.503 for
  q=0.5, 0.1, 0.02, and 0.004. The corresponding endpoint candidate-loss AUCs are
  0.730, 0.549, 0.517, and 0.498.
- The trajectory-minus-endpoint gap at q=0.5 is +0.076 with paired 95% interval
  [-0.001, 0.148], so the current pilot does not establish a statistically clear
  interface advantage.
- Conditional diffusion did not outperform the analytic sum-LLR detector at high
  noise. Keep this negative result; do not tune against the held-out identities.
- Candidate-gallery linkage is above chance in some high-participation conditions,
  but this is identification against a supplied gallery rather than unknown-image
  synthesis.

The measured reports are under [`reports/`](reports/). The revised manuscript and
compiled PDF are under [`paper/satml2027/`](paper/satml2027/).

## Reproduce the code checks

Use Python 3.10 or newer. The tests do not require MNIST.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m gaussproof.cli smoke
```

The expected result is `48 tests passed`. Optional diffusion tests require:

```bash
pip install -r requirements-diffusion.txt
python -m gaussproof.cli smoke
```

## Data and generated artifacts

The repository does not include MNIST, model checkpoints, raw trajectories, virtual
environments, or identity-level run directories. These are large or sensitive
generated artifacts, not source dependencies.

Obtain `mnist_train.csv` separately. It must contain one digit label followed by 784
pixel values in the original 0--255 scale, with an optional header. Run a quick check:

```bash
OPENBLAS_NUM_THREADS=1 python -m gaussproof.cli run \
  --config configs/quick.json \
  --data /path/to/mnist_train.csv \
  --output runs/quick
```

Every experiment writes to a new directory under `runs/`; a nonempty directory is
never overwritten. Export aggregate-only results with:

```bash
python -m gaussproof.cli report --run runs/quick --output reports/quick
```

Never commit `data/`, `runs/`, `*.pt`, `*.npz`, virtual environments, row-level
membership files, split identities, or access credentials. The root `.gitignore`
enforces the main exclusions.

## Priority work

### 1. Strengthen the ordinary-participation experiment

This is the highest priority. Replace the q=0.004 proxy with the actual uniform
minibatch sampler for N=2000 and B=8. Use nested windows long enough to produce
meaningful expected inclusion counts, such as T=128, 512, and 2048. Increase the
number of unseen identities before increasing repetitions per identity. Preserve
paired random numbers, identity-clustered bootstrap intervals, and the endpoint
comparison. Report privacy accounting alongside every window.

Pre-register the conditions before inspecting results. If detection remains at chance,
that negative result is scientifically important.

### 2. Add a stronger final-model baseline

The current endpoint attack is candidate negative cross-entropy. Add a reference-model
likelihood baseline such as LiRA or the repository's offline RMIA using only the final
checkpoint. Keep its reference-data assumptions explicit. Compare it with the
trajectory detector on identical candidate identities and paired runs.

### 3. Separate identification from reconstruction

For gallery identification, report exact top-1, same-class top-1, candidate-excluded
controls, and background-mismatch controls. For unknown-image reconstruction, train
the prior only on public identities, reconstruct held-out private identities, and
report pixel error plus an identity-aware perceptual measure. Do not call nearest-
candidate retrieval reconstruction.

### 4. Test generalization

After the main ordinary-q experiment is stable, repeat the preregistered comparison on
at least one shifted training distribution and one additional architecture or dataset.
Change one scientific factor at a time and retain fixed held-out identities.

### 5. Update the paper from aggregate results

Regenerate figures from exported CSV files. Every number in the manuscript should be
traceable to a checked-in aggregate table and a documented command. Preserve negative
results and confidence intervals. The current paper source is a working draft rather
than a final accepted claim set.

## Repository map

| Path | Contents |
| --- | --- |
| `gaussproof/` | Core benchmark, attacks, trajectory audits, denoisers, reconstruction |
| `configs/` | Fixed experiment configurations |
| `tests/` | Core and optional diffusion checks |
| `docs/` | Threat models, equations, and experiment protocols |
| `scripts/` | Experiment and summarization entry points |
| `reports/` | Aggregate CSVs, verification notes, and paper-ready figures |
| `paper/satml2027/` | Revised LaTeX manuscript, bibliography, figures, and PDF |

## Contribution discipline

Create a branch per experiment. Record the dataset hash, source revision, configuration,
seed, package versions, and exact command. Add a focused test for any change to sampling,
privacy accounting, score orientation, pairing, or data separation. Before opening a
pull request, run the smoke suite and link the new aggregate report. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the review checklist.

No public software license has been selected. Keep the repository private and use it
for the research collaboration until the project lead chooses a release license.
