# Offline sequence-to-batch-image reconstruction

This experiment extends the causal pilot to **offline full-sequence** reconstruction.
Every estimate can depend on all earlier AND later noisy releases. It follows the
sequence-to-sequence idea of Erik Buchholz et al.'s RAoPT, not a reproduction of its
GPS architecture or a claim that GPS results carry over to DP-SGD.

## Mechanism and public training

An MNIST CNN is pretrained for 200 SGD steps on 3,000 public examples. For every
subsequent round, independently draw two distinct examples uniformly from the
role's dataset, compute and individually clip their gradients, average, add
Gaussian noise of standard deviation sigma*C/2, and update the CNN at learning
rate 0.05. The reconstructor never modifies the CNN's updates.

Initial settings: C=1, sigma=0.05 and 0.5, 16 rounds. These are small-batch
feasibility settings, NOT strong-privacy demonstrations. The reported epsilon
upper bound uses replace-one sensitivity 2C, zCDP composition, and no sampling
amplification. It is conservative and applies separately to each run. No low-epsilon
claim follows from these settings. Fixed-size sampling here differs from the
Poisson sampler used by the prior large-batch experiment.

Disjoint MNIST roles: public pretraining (3,000), public fingerprint bank (32),
public attack training (6,000), public calibration (1,500), private target (1,500),
and public classifier testing (1,000). Each noise condition has 48 public training,
12 public calibration and 8 private target sequences. Sequences within a role
reuse that role's pool; test runs are not disjoint private datasets. The same role
splits and paired sampling/noise seeds are reused across noise conditions.

At each released pre-update checkpoint the attacker computes the mean clipped
gradient of its 32 public fingerprint examples. It observes the full noisy update
sequence and these public fingerprints. Private batch membership, labels, image
pixels and clean gradients are never reconstruction inputs. Public training pairs
DO contain the corresponding public batch images for supervised learning.

## Stages

1. Fit a common rank-128 PCA basis to public clean-minus-fingerprint gradients.
   Fit mean/scales only on public attack-training trajectories, across both noise
   conditions. Represent the public fingerprints in the same basis.
2. Train a bidirectional LSTM to estimate the entire clean residual sequence.
   This is noncausal: output at round t may use round t+1 and later.
3. Train a conditional temporal diffusion model using public clean residual
   sequences as the diffusion target and the complete actual noisy sequence as
   fixed conditioning. An all-round Transformer supplies the denoiser. Upstream
   Diffusers DDPMScheduler supplies forward corruption and velocity targets;
   DDIMScheduler supplies 40 reverse steps. Four sampled reconstructions provide
   an approximate posterior mean and a single-draw comparison. The temporal
   adapter is custom, not a pretrained/reproduced CSDI model.
4. Train a shared convolutional image generator on public **clean** projected
   gradients and public checkpoint fingerprints. It outputs two 28x28 images.
   A permutation-invariant pixel loss selects the cheaper of the two assignments.
   No target labels are supplied. Public calibration clean-gradient image error
   selects the generator; the same generator is used for all attack methods.

Diffusion noise and DP noise are different. Actual DP-noisy sequences are
conditioning throughout every reverse step. Independent artificial diffusion
noise corrupts the public clean reconstruction target during training. We never
replace past DP noise while keeping an incompatible future CNN trajectory fixed.

## Comparisons

- Learned public-checkpoint-only prior (no observed gradient coefficients).
- Analytic Gaussian shrinkage using the current release plus checkpoint prior.
- Linear Gaussian/ridge smoother using all releases, tuned on public calibration.
- Bidirectional LSTM.
- Full iterative conditional diffusion: four-draw latent mean and first draw.
- Raw noisy projected gradient passed through the same image generator.
- Clean projected gradient passed through the generator (oracle diagnostic).
- Public mean image, repeated twice (image-prior control).
- Diffusion estimated sequences rotated across test runs (target-dependence check).

All learned/Gaussian gradient estimators retain the public fingerprint outside the
PCA space. Full-dimensional gradient error includes that residual. The clean
oracle is exact only WITHIN the public representation and is not an attack.

Image scoring finds the better two-image assignment using target pixels AFTER
prediction. Report mean squared error, PSNR and digit agreement measured by the
public pretrained classifier. Digit agreement is an imperfect semantic diagnostic,
not evidence that a specific private image was recovered. The classifier's own
held-out accuracy is recorded. Gallery columns use predeclared run/round pairs
(0,0), (0,8), (1,0), (1,8), without choosing visually successful results.

A good-looking digit can be a prior sample. Success requires improvement over
context/prior controls on unseen private images, not just visual plausibility.
A failed clean-gradient oracle diagnoses the representation/image generator,
not a failure caused by DP noise. Eight short target sequences are a feasibility
pilot; neither DP violation nor increasing vulnerability with noise is established.

## Run

Install the existing core requirements and `requirements-trajectory.txt`.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=.cache/matplotlib \
python -m gaussproof.sequence_images --config configs/sequence_images.json \
  --data /path/to/mnist_train.csv --output runs/sequence_images
```

Saved-model inference accepts only noisy releases and accessible fingerprints:

```sh
python scripts/reconstruct_batches.py --run runs/sequence_images \
  --input observations.npz --sigma 0.05 --method diffusion_mean \
  --output reconstructed.npz
```

The input fields `observations` and `public_fingerprints` have shape
[number of sequences, 16 rounds, 13242 CNN parameters]. Gradients must use the
matching CNN architecture/parameter order, normalization, clip and batch size.
Output `images` has shape [sequences, 16, 2, 28, 28]. No clean target is accepted by
the inference API. Weights remain local; MNIST CSV is not included in the package.

References:
- RAoPT: https://arxiv.org/abs/2210.09375
- Conditional time-series diffusion inspiration (imputation, not this task): https://github.com/ermongroup/CSDI
- Upstream schedulers: https://huggingface.co/docs/diffusers/v0.37.0/en/api/schedulers/ddim
