# Reliable diffusion components, experimental gradient adapter

This experiment implements the proposal to learn Gaussian noise and subtract it
from clipped gradients. It uses Hugging Face Diffusers 0.37.0 rather than implementing
DDPM/DDIM samplers or U-Net layers from scratch. Model weights are trained specifically
on public gradient examples; no image-model pretrained weights are used.

## Observation and scaling

At each of two fixed public CNN checkpoints, compute each example's full gradient,
clip it to norm C, and average B of those clipped vectors. Add independent Gaussian
noise of absolute coordinate standard deviation tau to that average. Vary C and tau
independently. In standard sum-noise-then-average DP-SGD notation the equivalent noise
multiplier is sigma = tau B / C. This lab does not feed these releases back into CNN
training and makes no new dynamic DP-SGD privacy claim.

Fit public class means and a 64-dimensional orthonormal PCA basis at each checkpoint.
Clipped gradients are first expressed in units of C. Center using the known batch
label histogram and public class means; divide all retained coordinates by one common
public scalar s. The retained noise remains isotropic, with standard deviation
r = tau/(C s). The orthogonal complement is explicitly handled by Gaussian shrinkage.
Full-vector error therefore includes both the neural latent estimate and that shared
residual baseline; it is not evidence that the network learns every noise coordinate.

Transform an additive-noise observation q = x0 + r epsilon into a DDPM state:

    alpha_bar = 1/(1+r²)
    x_t = sqrt(alpha_bar) q
        = sqrt(alpha_bar) x0 + sqrt(1-alpha_bar) epsilon.

The official DDPM scheduler adds noise and converts predictions back to x0. Its
default image clipping is disabled; gradient coefficients need not lie in [-1,1].
A 200-timestep log-noise schedule covers r from 0.001 to 1000. Evaluation uses the
nearest timestep on a 100-step DDIM grid; actual physical noise is saved and can differ
slightly from the requested value and across seeds because s is fitted independently.

## Predictions and controls

- Epsilon prediction: train the U-Net against the sampled noise.
- Sample prediction: same U-Net predicts clean coefficients directly.
- Velocity prediction: official target v = sqrt(alpha_bar) epsilon - sqrt(1-alpha_bar) x0.
- Iterative reconstruction: official DDIM with eta=0, starting at the observed state,
  for both epsilon and velocity models. This deterministic map is not generally a
  posterior mean; it may increase reconstruction error.

The epsilon model is also evaluated with a single checkpoint, with both checkpoints,
and with both plus a public-bank feature. During training, masks randomly remove one
checkpoint or the bank feature. The bank contains only fitting-set gradients or
fitting-set batch combinations. Its posterior feature is recomputed from the noisy
observation, never from the clean target. Gaussian and public-bank posterior baselines
receive the same known labels and noisy observations.

The U-Net's input is an arbitrary 8×8 layout of PCA coordinates, not an image of a
gradient. This is a small CPU laboratory architecture; convolutional locality in this
layout is not claimed to be optimal. The experiment evaluates transfer to unseen
examples at the same two checkpoints, not to new architectures or unseen checkpoints.

## Leakage prevention and selection

CNN training, prior fitting, calibration, and evaluation use separate row IDs inherited
from the saved public-checkpoint experiment. The basis, class means, covariance, bank,
and gradient normalization use only prior-fitting examples. Calibration selects the
best of the 200-step training checkpoints using recovered-coefficient error. Evaluation
never selects hyperparameters or the best method. Models, split indices, batch indices,
normalization, priors, training curves and source hashes are saved locally.

All architectures start from the same seed within a run. Differences between epsilon,
sample and velocity training include objective weighting across noise, so comparisons
do not isolate architecture effects. Public feature construction always uses two CPU
threads to preserve the first saved basis; small U-Net training uses one thread after
profiling. The first completed epsilon model was trained with two threads and reused;
the original configuration and provenance are preserved in the resume metadata.

## The noise-loss trap

For y = g + tau epsilon, an estimate g_hat = y - tau epsilon_hat satisfies

    ||g_hat-g||² = tau² ||epsilon_hat-epsilon||².

Returning epsilon_hat = (y-public_mean)/tau reconstructs only the public mean, yet
its noise MSE becomes tiny as tau grows. Report full reconstructed-gradient error,
cosine similarity and paired improvement over prior-only and Gaussian estimates.
Noise-prediction MSE is a secondary diagnostic with this trivial baseline included.

Three seeds, 30 held-out images per seed, and 2,000 steps per model are a pilot. Batch
means share held-out constituents; bootstrap intervals for batch outputs are descriptive
and do not establish independent-record statistical certainty. There is no pixel
inversion or claim that increasing independent Gaussian noise creates information.

Official references:
- https://huggingface.co/docs/diffusers/v0.37.0/en/api/schedulers/ddpm
- https://github.com/huggingface/diffusers/blob/v0.37.0/examples/unconditional_image_generation/train_unconditional.py
- https://github.com/huggingface/diffusers/blob/v0.37.0/src/diffusers/schedulers/scheduling_ddim.py
- https://arxiv.org/abs/2202.00512 (diffusion parameterization and sampling stability)
