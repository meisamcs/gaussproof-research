"""Apply a trained laboratory model without access to clean target gradients.

Input NPZ: observations [N,2,D] (physical units), label_histograms [N,10].
Use the same parameter ordering and public checkpoints as the fitted gradient space.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diffusers import UNet2DModel,DDPMScheduler,DDIMScheduler
from gaussproof.diffusion_denoiser import DiffusionDenoiser
from gaussproof.diffusion_lab import GradientSpace


def denoise(run,seed,observations,histograms,clip,batch_size,noise_sd,prediction='v_prediction',iterative=False):
    run=Path(run);cfg=json.loads((run/'config.json').read_text());folder=run/f'seed{seed}'
    torch.set_num_threads(cfg['threads'])
    if clip not in cfg['clips'] or batch_size not in cfg['batch_sizes'] or noise_sd<=0:
        raise ValueError('Use trained clipping/batch settings and positive noise SD')
    observations=np.asarray(observations,dtype=np.float32);histograms=np.asarray(histograms,dtype=np.float32)
    arrays=np.load(folder/'gradient_space.npz')
    space=GradientSpace.__new__(GradientSpace);space.cfg=cfg;space.k=2
    space.basis=arrays['basis'];space.scale=float(arrays['scale']);space.dimension=space.basis.shape[1]
    space.means={c:arrays[f'means_{c}'] for c in cfg['clips']}
    space.floor={c:arrays[f'floor_{c}'] for c in cfg['clips']}
    space.cov={c:arrays[f'cov_{c}'] for c in cfg['clips']}
    if observations.ndim!=3 or observations.shape[1:]!=(2,space.dimension) or not np.isfinite(observations).all():
        raise ValueError('Expected finite physical gradient observations [N,2,D]')
    n=len(observations)
    if histograms.shape!=(n,10) or not np.isfinite(histograms).all() or (histograms<0).any() or not np.allclose(histograms.sum(1),1,atol=1e-5):
        raise ValueError('Known label histograms must be nonnegative and sum to one')
    model=DiffusionDenoiser(cfg,prediction)
    scheduler_config=json.loads((folder/prediction/'scheduler/scheduler_config.json').read_text())
    if scheduler_config['prediction_type']!=prediction:raise ValueError('Prediction type does not match checkpoint')
    model.model=UNet2DModel.from_pretrained(folder/prediction/'unet',low_cpu_mem_usage=False).eval()
    model.schedule=DDPMScheduler.from_pretrained(folder/prediction/'scheduler')
    model.ddim=DDIMScheduler.from_config(model.schedule.config)
    model.ddim.set_timesteps(cfg['inference_steps'])
    ratio=noise_sd/(clip*space.scale)
    valid=model.ddim.timesteps
    limits=((1-model.schedule.alphas_cumprod[valid])/model.schedule.alphas_cumprod[valid]).sqrt()
    if ratio<float(limits.min()) or ratio>float(limits.max()):raise ValueError('Noise is outside the trained scheduler range')
    t,nominal=model.timestep(ratio)
    center=space.center(histograms,clip);unit=observations/clip
    z=space.encode(unit-center)/space.scale
    xt=torch.from_numpy(z.reshape(-1,2,8,8))*model.schedule.alphas_cumprod[t].sqrt()
    bank=torch.from_numpy(arrays[f'bank_{clip}_{batch_size}']).unsqueeze(0).expand(n,-1,-1,-1,-1)
    fitted=model.predict(xt,t,torch.from_numpy(histograms),torch.full((n,),clip),
                        torch.full((n,),float(batch_size)),bank,iterative=iterative).numpy().reshape(-1,2,64)
    reconstructed=clip*(center+space.decode(fitted*space.scale)+space.outside(unit-center,clip,batch_size,noise_sd/clip))
    return reconstructed,dict(prediction_type=prediction,iterative=iterative,timestep=t,
        supplied_noise_sd=noise_sd,nearest_trained_noise_sd=nominal*space.scale*clip,
        note='Noise is approximated by the nearest scheduler level; full-vector complement uses Gaussian shrinkage.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--clip',type=float,required=True);p.add_argument('--batch-size',type=int,required=True)
    p.add_argument('--noise-sd',type=float,required=True);p.add_argument('--prediction',default='v_prediction',choices=['epsilon','sample','v_prediction'])
    p.add_argument('--iterative',action='store_true');a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError('Output already exists')
    data=np.load(a.input)
    result,metadata=denoise(a.run,a.seed,data['observations'],data['label_histograms'],a.clip,a.batch_size,a.noise_sd,a.prediction,a.iterative)
    np.savez_compressed(a.output,reconstructed_gradients=result,estimated_noise=(data['observations']-result)/a.noise_sd,
                        metadata=json.dumps(metadata))
    print(json.dumps(metadata,indent=2))
