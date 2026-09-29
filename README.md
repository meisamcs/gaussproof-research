# GAUSSPROOF

A Python/PyTorch research repository for membership inference, fingerprint detection,
and reconstruction from repeated noisy gradient releases. It trains CNNs on MNIST,
compares black-box and white-box attacks on fixed candidates, and exports reproducible
measurements and paper-ready vector figures.

**Start here:** [STUDENT_HANDOFF.md](STUDENT_HANDOFF.md) records the current evidence,
the exact boundary of the claims, and the prioritized work needed to finish the paper.

**DP-FTRL checkpoint audit:** [experiment and reproduction commands](experiments/dp_ftrl/README.md)
and [measured results with figures](reports/dp_ftrl/README.md) use the authors'
pinned tree-noise implementation. The trajectory signal depends strongly on
knowing the other batch records; this is a controlled audit, not a general
attack on deployed federated DP-FTRL.

**Status:** active research implementation. GAUSSPROOF includes a bounded nonnegative
sparse trajectory decoder, exact known-fingerprint likelihood tests, learned temporal
denoisers, and reconstruction studies. The RERO-style baseline is mean gradient
alignment rather than a claimed reproduction of a particular published implementation.
Results do not assume any attack wins.

**Completed local validation:** 48 tests passed with the trajectory dependencies installed. The [measured pilot report](reports/pilot/README.md)
contains 9 target conditions (3 seeds × 3 noise levels), 18 reference CNNs, and 81
attack/condition measurements. Held-out digit accuracy was 73.4–85.9%. This small pilot
does not show a consistent GAUSSPROOF or hybrid advantage over RMIA. The larger full
configuration is provided but has not been executed. Later focused studies and their
aggregate outputs are included under [`reports/`](reports/).

## Install and run

Python 3.10+ and a CPU are sufficient. GPU execution is not implemented in this version.
`requirements-tested.txt` records the exact versions used for the local verification run.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m gaussproof.cli smoke
OPENBLAS_NUM_THREADS=1 python -m gaussproof.cli run \
  --config configs/quick.json --data /path/to/mnist_train.csv --output runs/quick
```

`mnist_train.csv` must have a digit label followed by 784 pixels in the original 0–255
scale, with an optional header. The dataset is not distributed or uploaded. Missing,
malformed, and already normalized inputs fail explicitly. The file is hashed for provenance.

| Configuration | Purpose |
| --- | --- |
| `configs/quick.json` | Two-condition integration check, deliberately short training |
| `configs/pilot.json` | Three seeds, three noise levels, longer CNN training |
| `configs/full.json` | Larger three-seed noise/clipping/batch sweep; computationally expensive |

Change `--config` and choose a new output directory for each run. Existing nonempty
outputs are never overwritten. The full configuration evaluates 90 independently trained
targets and 360 reference models, each for 800 updates, with 8,000 candidate gradients per
update. Plan for substantial CPU time, RAM and disk; it is **not** an automatic quick run.
Reduce the sweep before scaling up. The implementation retains trajectory dictionaries;
one full-condition dictionary alone is about 3.3 GB (float32), excluding temporary arrays.

## Attacks and access

| Score | Required access | Implementation |
| --- | --- | --- |
| Loss | Final model probabilities and known record label | Negative cross entropy |
| Confidence | Final model probabilities | Maximum class probability |
| Negative entropy | Final model probabilities | Sum of p log p |
| Margin | Final model probabilities | Largest minus second largest probability |
| Shokri-style | Outputs, labels, disjoint shadow data | Pooled class-conditioned logistic attack trained on shadow members/nonmembers |
| Offline RMIA | Outputs, labels, OUT reference models, population data | Population pairwise likelihood-ratio test |
| RERO-style | Gradient releases, pre-update model states, labeled candidates | Mean time-aligned gradient inner product |
| GAUSSPROOF | Same gradient access and candidate dictionary as RERO-style | Bounded nonnegative sparse decoding at every step, then temporal averaging |
| Hybrid | Both output and gradient access plus known calibration members/nonmembers | Logistic combination of RMIA and GAUSSPROOF |

Gradient attacks have stronger access than black-box attacks. Their success is not
evidence of superiority under an equal black-box threat model. Model states are needed
to construct time-varying candidate gradients. Synthetic fingerprints are used only in
the separately labeled validation suite, never in the MNIST benchmark.

## Fairness and training

- Target training, nonmember evaluation, nonmember calibration, shadow pool, and RMIA
  population are disjoint by row ID. Evaluation members and calibration members are
  separate subsets of the training set. All indices and reference splits are saved.
- Every attack in a condition uses the same target and candidate records. Candidate
  order is independently shuffled. Splits, sampling streams and initialization are paired
  across noise/clipping conditions within a seed.
- Shadows match the target architecture, update count, learning rate, batch size,
  clipping and noise multiplier. Their data and random streams are independent.
- Exact per-record gradients are computed over **all** CNN parameters using autograd.
  Global L2 clipping precedes summation. Gaussian noise of standard deviation `sigma*C`
  is added to the sum; the result divided by batch size is used in the parameter update.
- A fixed coordinate subset of the final layer is observed by the trajectory attacks.
  These are coordinates of the real released gradient, with unchanged Gaussian noise
  variance. Candidates use exact gradients clipped by their **full-model** norm.
- GAUSSPROOF and RERO-style receive only dictionaries and releases. They cannot read
  batch participation, the sampled noise, or membership truth.
- RMIA coefficients and sparse penalty are prespecified in configs. Hybrid fitting uses
  one calibration half; all decision thresholds use the other half. Evaluation labels
  are used only to measure performance and run explicitly labeled permutation controls.

See [the protocol](docs/protocol.md) for equations, privacy accounting, and limitations.

## Outputs

```text
run/
  config.json, provenance.json, completion.json
  splits_seed*.json
  metrics.csv                 # 9 attacks per target condition
  utility.csv                 # model accuracy and conservative privacy bound
  validation.csv              # membership permutation and real-dictionary noise controls
  synthetic_controls.csv      # independent controlled sanity experiments
  figures/                    # comparison, blind-spot, hybrid PDF + PNG
  seed*_sigma*_clip*_B*/
    target.pt, reference_*.pt, reference_*_split.json
    trajectory.npz            # actual noisy sums, exact dictionaries, IDs, coordinates
    observations.npz          # model and reference observations
    scores.csv                # row IDs, roles, membership truth, every score
    attack_fits.json           # fitted weights and calibration IDs
    figures/roc.pdf, figures/roc.png
```

Metrics include tie-aware ROC AUC, stratified bootstrap AUC intervals, empirical
TPR at FPR ≤1% and ≤5%, maximum ROC advantage, and independently calibrated balanced
accuracy, advantage, achieved TPR and achieved FPR. Maximum ROC advantage is a
descriptive test-set statistic, not an independently chosen deployment threshold.
At 128 nonmembers, the FPR resolution is 0.78125%; low-FPR pilot results are noisy.
Plots are vector PDFs with embedded fonts plus 300-dpi PNGs. Appearance suitable for a
paper does not make a small pilot a publication-grade statistical evaluation.

Export an aggregate-only report for GitHub (raw data, row-level scores, split IDs,
checkpoints and trajectories stay local):

```bash
python -m gaussproof.cli report --run runs/pilot --output reports/pilot
```

The export includes a human-readable report, `summary.csv`, separate
`blackbox_mia_results.csv` and `trajectory_mia_sweep.csv`, vector figures and a SHA256 manifest.

## Validation

```bash
python -m gaussproof.cli smoke
```

Tests cover exact gradients against individual autograd, global norm clipping, the
identity between recorded releases and actual parameter updates, deterministic training,
split isolation, pixel layout, tie-aware metrics, explicit RMIA calculations, sparse
recovery, permutation equivariance, and synthetic negative controls. GitHub Actions runs
the same suite without MNIST. The benchmark also measures controls for each real run.

Shuffling **membership labels**, not digit labels, is the chance-level check. Training
on shuffled digit labels can increase memorization and is not expected to destroy MIA.
Noise sweeps report negative and nonmonotone empirical findings without forcing a trend.

## Diffusion-style gradient denoising

The [completed pilot findings](reports/diffusion/findings.md) include nine trained
U-Nets and 12,960 measurements. Stable high-noise prediction approached the prior
baseline; this run does not establish a high-noise recovery advantage.

The [diffusion protocol](docs/diffusion_protocol.md) uses pinned Hugging Face
Diffusers U-Net, DDPM and DDIM implementations. Fresh weights are trained on public
gradient coefficients; epsilon, direct-sample and velocity prediction are compared.
The lab separates individual gradients, means of eight clipped gradients, trajectory
context and public-bank context. Full-gradient errors are measured, including the
explicit Gaussian residual treatment outside a 64-coordinate public basis.

```bash
pip install -r requirements-diffusion.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=.cache/matplotlib \
  python scripts/run_diffusion_lab.py --config configs/diffusion.json \
  --source runs/reconstruction --data /path/to/mnist_train.csv --output runs/diffusion
python scripts/summarize_diffusion.py runs/diffusion
python -m gaussproof.cli report --run runs/diffusion --output reports/diffusion
```

An interrupted run can reuse completed saved models with `--resume`. It must retain
its scientific configuration; only CPU thread count may change. The ordinary test
suite skips the three diffusion tests if optional dependencies are absent. CI is
configured to run both core-only and diffusion-enabled environments.

Apply a saved model to noisy observations without providing clean gradients:

```bash
python scripts/denoise_gradients.py --run runs/diffusion --seed 17 \
  --input noisy_gradients.npz --output denoised.npz \
  --clip 0.1 --batch-size 1 --noise-sd 0.1 --prediction v_prediction
```

The input contains `observations` of shape [N,2,D] in physical gradient units and
known `label_histograms` of shape [N,10]. Checkpoints and parameter order must match
the fitted public gradient space. Output contains reconstructed gradients, estimated
noise, and the nearest trained noise level. `--iterative` enables DDIM; it is not
guaranteed to improve estimation error. The provided velocity default is a stable
parameterization, not a claim of superior reconstruction to Gaussian shrinkage.

## Fingerprint detection and noise bounds

The [fixed-access fingerprint experiment](reports/fingerprint_noise/README.md)
uses 90 real gradient templates, nine noise levels, and 32 releases per condition.
Exact Gaussian sufficient-statistic simulation matches the optimal likelihood test.
Increasing noise reduces detection and increases optimal denoising error. The report
includes a proved binary Bayes-risk lower bound and a model-specific minimum noise
level for capping detection power. These results assume a known canary fingerprint,
known/subtracted background, and presence at every release under the alternative;
they are not general dynamic DP-SGD guarantees.

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.cache/matplotlib python scripts/fingerprint_noise_sweep.py \
  --source runs/reconstruction --output reports/fingerprint_noise
```

The [unseen-canary holdout](reports/canary_holdout/README.md) is the stronger
calibration check for the known-fingerprint audit. Four canary identities are
used for threshold calibration and four disjoint identities are held out. The
sequence detector reaches holdout AUC 0.929, 0.884, and 0.609 at sigma 0.25, 1,
and 4, while a final release reaches 0.644, 0.559, and 0.520. At the calibrated
5% operating point, sequence TPR is 0.563, 0.375, and 0.167. These are weak-
privacy, known-canary white-box results; they do not demonstrate unknown-image
reconstruction or a diffusion advantage.

The follow-up [cross-fitted learned comparison](reports/canary_learned/README.md)
trains a BiLSTM and conditional temporal diffusion model on calibration canaries
only. Holdout AUC for diffusion is 0.944, 0.852, and 0.566 at sigma 0.25, 1,
and 4, compared with 0.929, 0.884, and 0.609 for the strongest sum-LLR baseline.
The low-noise AUC difference is +0.015 with paired 95% interval [-0.003, 0.036];
at higher noise diffusion is worse. This pilot therefore does not show that
learned denoising defeats the analytic trajectory detector or reverses the
privacy benefit of increasing noise.

The stronger [exact long-trajectory audit](reports/canary_long_sigma4/README.md)
then fixes sigma 4 and follows the same unseen-canary CNN trajectories for 16,
32, 64, and 128 releases. Sum-LLR AUC rises from 0.609 to 0.675, 0.768, and
0.819. The 128-round paired gain is +0.211 with interval [0.144, 0.272]. This is
the clearest GAUSSPROOF result: weak per-round fingerprints accumulate over
repeated releases. The conservative epsilon bound simultaneously rises from
11.60 to 43.14, so the result argues for trajectory-aware composition and
release minimization rather than claiming that larger noise itself backfires.

The [participation-rate sensitivity study](reports/canary_q_sensitivity/README.md)
tests the same high-noise mechanism at `q = 0.5, 0.1, 0.02, 0.004` and compares
the trajectory likelihood with a true final-checkpoint candidate-loss attack.
It uses 20 stratified unseen identities and three paired repetitions per
identity. At 128 releases, trajectory AUC is 0.805, 0.571, 0.531, and 0.503,
while endpoint AUC is 0.730, 0.549, 0.517, and 0.498. Thus the strong pilot
effect depends on frequent participation: at the ordinary-sampling proxy
`q=0.004`, where only 0.512 inclusions are expected in 128 rounds, neither
interface provides detectable membership evidence.

The [real-canary gallery experiment](reports/canary_gallery/README.md) turns
that high-noise evidence into record linkage across four private label
distributions. At sigma 4 and 128 releases, exact top-1 recovery from a 32-image
gallery is 15–30% with a matched public background, compared with 3.1% at
random. Long-tail and rare-canary data reach 60% same-class top-1 versus a
31.7% class-only baseline, providing preliminary evidence of individual
fingerprint information. A background-mismatch control also shows that class
distribution leakage can mimic reconstruction. The result is exact candidate
re-identification; removing the true image does not establish unknown-image
synthesis.

## Learned-prior gradient and image reconstruction

The separate [reconstruction plan](docs/reconstruction_plan.md) and
[mathematical specification](docs/reconstruction_math.md) test whether public
gradient priors recover unseen target gradients and pixels from Gaussian releases:

```bash
OPENBLAS_NUM_THREADS=1 python -m gaussproof.cli reconstruct \
  --config configs/reconstruction.json --data /path/to/mnist_train.csv \
  --output runs/reconstruction
python scripts/rebuild_reconstruction_report.py runs/reconstruction
python -m gaussproof.cli report --run runs/reconstruction --output reports/reconstruction
```

This controlled experiment uses fixed public CNN checkpoints; the private releases
do not drive model updates. It compares noisy averaging, prior-only prediction,
subspace projection, empirical Gaussian shrinkage, and a public-data neural denoiser.
Stale fingerprints, unknown batch backgrounds, shuffled releases, clean-gradient
image inversion, and public mean images expose important failure modes. An exact
target-containing bank is evaluated separately under its stronger access assumption.
The neural estimator is a new experimental component, not the original sparse MIA
decoder. This experiment does not establish performance on dynamic DP-SGD training.

The [completed reconstruction report](reports/reconstruction/README.md) contains
12,150 gradient measurements, 1,620 identity measurements, and 54 image measurements
across three seeds. At σ=1 and 64 releases per checkpoint, gradient squared error
fell from 0.365 (prior only) to 0.271 (Gaussian prior) and 0.246 (neural prior).
Pixel inversion did not beat the public class-mean image on the nine evaluated
images. These results support gradient denoising in the specified channel, not a
claim that extra Gaussian noise improves optimal leakage or that novel-image
reconstruction succeeds. Repeated releases also increase the privacy budget.

## References

- Shokri et al., [Membership Inference Attacks Against Machine Learning Models](https://arxiv.org/abs/1610.05820), IEEE S&P 2017.
- Zarifzadeh, Liu, Shokri, [Low-Cost High-Power Membership Inference Attacks](https://proceedings.mlr.press/v235/zarifzadeh24a.html), ICML 2024.
- [Authors' RMIA implementation](https://github.com/privacytrustlab/ml_privacy_meter/blob/master/attacks.py): reference for the offline OUT-probability approximation.

This repository contains new implementations. No upstream code was copied. No software
license is selected yet; the owner should choose one before inviting external reuse.

## Actual DP-SGD trajectory observer

The [trajectory protocol](docs/trajectory_dp_protocol.md) implements a causal
observer of evolving MNIST CNN training, with C=0.1, expected B=32 and three
noise multipliers. Public-only diffusion training and calibration are separate
from private target runs. The observer has public fingerprints and noisy updates,
never private batch membership. Both an 8-release window and all available
history are evaluated on identical trajectories. See the [measured report](reports/trajectory_dp/README.md)
for CSV results, privacy accounting, Gaussian baselines, and vector plots.

Install `requirements-trajectory.txt` and follow the protocol to reproduce.

## Offline sequence-to-image reconstruction

The [full-sequence image pilot](reports/sequence_images/README.md) trains a
bidirectional LSTM, an iterative conditional temporal diffusion model, and a
shared generator that outputs two MNIST images per round. Both earlier and later
noisy releases are available; private batch identities and labels are not inputs.
The [protocol](docs/sequence_images_protocol.md) documents the small-batch
feasibility settings and saved-model inference. Low-noise reconstructions improve
on a mean-image control, but Gaussian shrinkage remains best on pixel error.

## Matched sequence-access ablation

The [ablation protocol](docs/sequence_ablation_protocol.md) compares current-only,
causal, full-sequence and shuffled access using identical LSTM architectures and
training budgets, three initialization seeds, and the frozen image generator.
Its [paired report](reports/sequence_ablation/README.md) separates added sequence
information from temporal order and test-time distribution shift.

The matched pilot found no reproducible image-recovery advantage from full ordered
sequences over the current release given the same public checkpoint context.

## Persistent canary audit

The [canary audit](reports/canary_audit/README.md) tests known-fingerprint
detectability directly. At sigma 0.25 and 1, summing matched scores across 16
releases reaches AUC 0.946 and 0.864, while the final release alone reaches
0.612 and 0.621. At sigma 4 the sequence score is 0.646. This supports a
narrow auditing result, not unknown-image reconstruction or a diffusion advantage.
