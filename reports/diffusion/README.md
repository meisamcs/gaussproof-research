# Diffusers gradient-denoising laboratory

The denoisers use **Hugging Face Diffusers 0.37.0 UNet2DModel, DDPMScheduler, and DDIMScheduler**. We train fresh gradient-specific weights; no pretrained image model or claimed pretrained gradient denoiser is used. Upstream sampler equations, epsilon/sample/velocity conversions, and model architecture are reused. Gradient adapters, public PCA, bank features, training loop, and evaluation are project code.

## What was tested

At two fixed public CNN checkpoints, compute exact per-example gradients, clip each gradient BEFORE averaging, then add independent Gaussian noise. Public fitting, calibration and held-out examples are row-disjoint. The lab varies clipping C=0.1/1, averaging B=1/8, and requested absolute noise SD τ=0.01/0.1/1. These are independent lab controls; the standard DP-SGD equivalent multiplier would be τB/C. These releases do not train the target CNN, so this is not a dynamic DP-SGD attack.

Public PCA retains 64 coordinates per checkpoint, laid out arbitrarily as an 8×8 coefficient grid. A shared scalar preserves isotropic noise in these coordinates; we do not whiten different coordinates unequally. Two channels represent two checkpoints. Label histograms, clipping and batch size condition all neural variants. Optional public-bank posterior features use only public fitting examples or public batch combinations, never test gradients. The bank is an empirical prior rather than an exact target-containing bank. Single-checkpoint and trajectory variants share a model trained with channel/context dropout.

DDPM uses a documented log-spaced additive-noise schedule transformed into VP betas. Requested physical noise is mapped to the nearest DDIM-compatible timestep; `actual_noise` records the noise actually tested for each seed. All methods in each condition see the identical Gaussian-corrupted gradients. `epsilon` predicts noise, `sample` predicts clean coefficients directly, and `v_prediction` uses the upstream velocity parameterization. The three objectives imply different weighting across noise levels. Calibration selects model checkpoints by recovered-coefficient MSE with trajectory/bank context, never by test performance or noise loss. DDIM uses eta=0 and starts from the observed noisy state; it is not claimed to be a posterior-mean estimator.

Residual coordinates outside PCA use the same analytic Gaussian shrinkage for Gaussian, bank and neural methods. Thus any full-vector gains combine learned latent denoising with this explicit residual baseline. This is not a full-dimensional learned noise predictor. Raw noisy, zero-gradient, prior-only, Gaussian and public-bank baselines are included. `squared_l2` is physical gradient error; `clip_normalized_squared_l2` divides by C². These should not be confused with per-coordinate MSE.

## Reading the evidence

`summary.csv` reports full-vector gradient error, cosine similarity and paired gain over prior-only prediction. `noise_prediction.csv` separately reports coefficient error and the algebraically implied noise error: gradient error = noise variance × noise-prediction error. A predictor returning the prior mean can have tiny noise error at large noise. Only lower recovered-gradient error than prior-only/Gaussian baselines supports useful denoising.

Intervals resample training seeds and then evaluation outputs. Individual targets are 30 held-out images per seed. Batch means reuse these images across sampled batches, so the intervals for B=8 are descriptive and do not account fully for shared constituents. Three seeds and 2,000 training steps are pilot evidence, not a comprehensive convergence study. No test-set tuning, image reconstruction, or claim that added noise creates information is involved.

Reproduce in an environment with the normal project dependencies plus `requirements-diffusion.txt`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MPLCONFIGDIR=.cache/matplotlib python scripts/run_diffusion_lab.py --config configs/diffusion.json --source runs/reconstruction --data /path/to/mnist_train.csv --output runs/diffusion
```

References: [official DDPM scheduler](https://huggingface.co/docs/diffusers/v0.37.0/en/api/schedulers/ddpm), [official training example](https://github.com/huggingface/diffusers/blob/v0.37.0/examples/unconditional_image_generation/train_unconditional.py).
