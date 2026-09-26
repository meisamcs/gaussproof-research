"""Measured denoising error and noise-prediction diagnostic; no image inversion."""
from pathlib import Path
import hashlib
import shutil
import numpy as np
from .io import write_csv,write_json
from .reconstruction_report import aggregate
from .plots import style,save,plt
from matplotlib.ticker import NullLocator


def report(output,cfg,rows):
    output=Path(output)
    metrics=['squared_l2','clip_normalized_squared_l2','cosine','gain_over_prior','actual_noise','noise_to_clip_ratio']
    summary=aggregate(rows,['clip','batch_size','requested_noise','method'],metrics,300)
    write_csv(output/'summary.csv',summary)
    neural=[v for v in rows if v['latent_x0_mse']!='']
    diagnostic=aggregate(neural,['clip','batch_size','requested_noise','method'],
        ['latent_x0_mse','implied_latent_noise_mse','trivial_zero_residual_noise_mse'],300)
    write_csv(output/'noise_prediction.csv',diagnostic)
    style();names={'prior_only':'Prior only','gaussian':'Gaussian shrinkage','public_bank':'Public bank posterior',
        'epsilon_trajectory_bank':'DDPM noise prediction','sample_trajectory_bank':'Direct clean prediction',
        'v_prediction_trajectory_bank':'DDPM velocity prediction','epsilon_ddim':'Noise DDIM (iterative)',
        'v_prediction_ddim':'Velocity DDIM (iterative)','epsilon_single':'Noise: single checkpoint',
        'epsilon_trajectory':'Noise: trajectory','noisy':'Noisy observation'}
    for b in cfg['batch_sizes']:
        fig,axes=plt.subplots(1,len(cfg['clips']),figsize=(5*len(cfg['clips']),3.8),squeeze=False,layout='constrained')
        for ax,c in zip(axes[0],cfg['clips']):
            for method in ['prior_only','gaussian','public_bank','epsilon_trajectory_bank','sample_trajectory_bank','v_prediction_trajectory_bank']:
                selected=sorted([v for v in summary if v['clip']==c and v['batch_size']==b and v['method']==method],key=lambda v:v['requested_noise'])
                ax.plot([v['requested_noise'] for v in selected],[v['clip_normalized_squared_l2'] for v in selected],'o-',markersize=3,label=names[method])
            ax.set(xscale='log',yscale='log',xlabel='Requested absolute noise SD τ',ylabel='Full gradient squared error / C² / checkpoint',title=f'Clipping C={c}, batch average B={b}')
            ax.set_xticks(cfg['absolute_noise'],[f'{v:g}' for v in cfg['absolute_noise']]);ax.xaxis.set_minor_locator(NullLocator());ax.grid(alpha=.15)
        axes[0,0].legend(fontsize=6);save(fig,output/f'denoising_B{b}')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    c=cfg['clips'][0];b=1
    for method in ['prior_only','gaussian','epsilon_single','epsilon_trajectory','epsilon_trajectory_bank','epsilon_ddim','v_prediction_trajectory_bank','v_prediction_ddim']:
        selected=sorted([v for v in summary if v['clip']==c and v['batch_size']==b and v['method']==method],key=lambda v:v['requested_noise'])
        axes[0].plot([v['requested_noise'] for v in selected],[v['clip_normalized_squared_l2'] for v in selected],'o-',markersize=3,label=names[method])
    for method in ['epsilon_trajectory_bank','sample_trajectory_bank','v_prediction_trajectory_bank']:
        selected=sorted([v for v in diagnostic if v['clip']==c and v['batch_size']==b and v['method']==method],key=lambda v:v['requested_noise'])
        axes[1].plot([v['requested_noise'] for v in selected],[v['implied_latent_noise_mse'] for v in selected],'o-',label=names[method])
    selected=sorted([v for v in diagnostic if v['clip']==c and v['batch_size']==b and v['method']=='epsilon_trajectory_bank'],key=lambda v:v['requested_noise'])
    axes[1].plot([v['requested_noise'] for v in selected],[v['trivial_zero_residual_noise_mse'] for v in selected],'k:',label='Predict zero clean residual')
    for ax in axes:
        ax.set(xscale='log',yscale='log',xlabel='Requested absolute noise SD τ');ax.set_xticks(cfg['absolute_noise'],[f'{v:g}' for v in cfg['absolute_noise']]);ax.xaxis.set_minor_locator(NullLocator());ax.grid(alpha=.15);ax.legend(fontsize=6)
    axes[0].set_ylabel('Full gradient squared error / C² / checkpoint');axes[1].set_ylabel('Latent noise-prediction MSE')
    fig.suptitle(f'Conditioning / iterative reconstruction and noise-loss trap · C={c}, B=1');save(fig,output/'conditioning_and_noise_loss')
    (output/'README.md').write_text('''# Diffusers gradient-denoising laboratory

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
''')


def export(source,destination):
    source=Path(source);destination=Path(destination)
    if destination.exists() and any(destination.iterdir()):raise FileExistsError('Report destination must be empty')
    destination.mkdir(parents=True)
    for p in source.iterdir():
        if p.is_file() and (p.suffix in ['.pdf','.png','.json','.md'] or p.name in ['summary.csv','noise_prediction.csv','training.csv','spaces.csv','paired_vs_gaussian.csv']):
            shutil.copy2(p,destination/p.name)
    write_json(destination/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.iterdir()) if p.is_file()})
