# Causal observer of actual DP-SGD training

This experiment trains an MNIST CNN with Poisson-sampled, clipped, Gaussian-noisy
updates. The observer estimates the current clean **clipped batch sum divided by
expected batch size**. It does not estimate original unclipped gradients or images.
Denoised estimates never alter CNN training.

## Reproduce

Install the core environment and `requirements-trajectory.txt`, then run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=.cache/matplotlib \
python -m gaussproof.trajectory_dp \
  --config configs/trajectory_dp.json \
  --data /path/to/mnist_train.csv --output runs/trajectory_dp
```

Configuration: C=0.1, expected B=32, N=2,000, 64 steps, sigma=1,4,16,
delta=1e-5. The public initial CNN receives 150 ordinary SGD pretraining steps.
Subsequent training uses learning rate 0.5 and independently includes every
record with probability 32/2000 each round. Noise is added to the clipped sum
with coordinate standard deviation sigma*C, then divided by expected B.
Empty batches still release pure Gaussian noise and update the CNN.

Public pretraining, fingerprint bank, public test set, six public denoiser-training
datasets, two public calibration datasets, and three private target datasets are
mutually disjoint. The same roles and seeds are reused across sigma conditions.
Each dataset has its own actual evolving noisy training trajectory. Three target
runs are independent datasets/noise seeds, but share the trained denoiser and
public initial CNN. These are a small pilot, not a population-level significance study.

## Accessible information

At each pre-update checkpoint the observer computes clipped gradients of 32 known
public examples. Their mean is the public fingerprint/context estimate. The
observer additionally receives the current noisy gradient and the previous seven
noisy releases. Private batch indices, sizes, labels, clean gradients, and noise
realizations are never observer inputs. Checkpoints and public fingerprint access
are explicit white-box assumptions. Target clean gradients are stored locally
solely for scoring. Releasing those ground-truth logs would not be a DP release.

A rank-64 public PCA basis of clean-minus-fingerprint residuals is fit on the six
public training trajectories. The scalar normalization and residual mean also use
only public training data. All learned and Gaussian estimators use this basis and
retain the public fingerprint outside it. Errors are measured on all 13,242 CNN
parameters. The projection oracle exposes the error floor imposed by this choice.
This is not full-dimensional neural reconstruction.

## Diffusion estimator

The upstream Diffusers UNet2DModel and DDPMScheduler implement a **one-step
clean-sample prediction diffusion denoiser**, not an iterative reverse sampler.
Its 64 coefficients occupy an arbitrary 8x8 grid. Inputs are the VP-scaled current
noisy residual, seven past noisy residuals, history-validity masks and a current
observation mask. The latter is dropped during 25% of training examples, yielding
a context-only control. The network never sees future releases.

Training targets and histories are from public runs. Fresh noise is applied only
to the current public clean residual; this is valid because current DP noise is
independent of the pre-update state and sampled batch. We do not synthesize past
noise while retaining an incompatible fixed training trajectory. Half of each
training batch uses the exact evaluation noise level; the rest uses smaller
scheduler levels. Separate networks are trained for each sigma, with 1,200 AdamW
steps. Public calibration reconstruction error selects the checkpoint every 200
steps. No private target outcomes select models or hyperparameters.

## Baselines and privacy

Compare the public fingerprint alone, zero, raw noisy release, analytic Gaussian
current-observation shrinkage, causal ridge Gaussian regression with and without
the current observation, and the diffusion denoiser with and without it. Gaussian
ridge strength is selected on public calibration runs from a fixed grid. These
baselines use identical public coordinates/history. Noisy history and masks form
the causal Gaussian regression features. No clipping-ball projection is applied:
with Poisson sampling, the sum divided by expected B need not have norm <= C.

Opacus RDPAccountant composes the Poisson Gaussian mechanism over all 64 rounds,
under add/remove adjacency, at delta=1e-5. The full-history run expands the RDP order grid through 512 to obtain tighter
high-noise bounds; the original short-window run used the default grid. Both are
valid upper bounds. Values are for each individual run, not joint
publication of multiple runs over the same private data. Public test accuracy is
measured after updates 1,16,32,48,64. These evaluations use a disjoint public set.
Frequent accuracy testing does not expose private training loss.

## Interpretation

A lower error than context-only establishes observation-dependent improvement
within this experimental setup. Diffusion must also beat the Gaussian baselines
to justify added complexity. Higher sigma changes the evolving CNN and therefore
the target gradients, so lower raw MSE across sigma alone is not evidence that
noise increases private leakage. Inspect paired within-condition improvements,
clean-gradient scale, and CNN utility. No membership or original-image leakage
claim follows from average-gradient reconstruction alone.

## Full causal history comparison

`configs/trajectory_dp_full_history.json` extends the history to all previous
releases in the 64-round run. The current-release estimate at round t uses exactly
releases 1 through t; no records are assigned to batches by the attacker. This is
an online causal observer. It does not use future releases to retrospectively
reconstruct earlier rounds. To reuse the identical generated trajectories:

```sh
python -m gaussproof.trajectory_dp \
  --config configs/trajectory_dp_full_history.json \
  --data /path/to/mnist_train.csv --output runs/trajectory_dp_full_history \
  --trajectory-source runs/trajectory_dp
```

The full-history U-Net has 128 input channels: current residual, 63 historical
residuals, 63 validity indicators, and current-observation availability. Historical
positions are ordered by lag. This is a temporal conditional adapter around a
spatial U-Net, not a pretrained temporal diffusion foundation model. With only
384 public training rounds per noise condition, its generalization is limited.
