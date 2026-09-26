"""Generate two images per round from a noisy sequence; no private labels required."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from diffusers import DDPMScheduler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gaussproof.sequence_images import SequenceLSTM,TemporalDiffusion,BatchGenerator,diffusion_sample,gaussian_predict


def reconstruct(run,observations,fingerprints,sigma,method='diffusion_mean',seed=123):
    run=Path(run);cfg=json.loads((run/'config.json').read_text());space=dict(np.load(run/'public_space.npz'))
    if sigma not in cfg['sigmas']:raise ValueError('Noise level must match a trained condition')
    basis=space['basis'];obs=np.asarray(observations,dtype=np.float32);fp=np.asarray(fingerprints,dtype=np.float32)
    if obs.shape!=fp.shape or obs.ndim!=3 or obs.shape[1:]!=(cfg['steps'],len(basis)) or not np.isfinite(obs).all() or not np.isfinite(fp).all():
        raise ValueError('Provide finite matching [sequences, rounds, CNN parameters] arrays')
    d=dict(y=torch.tensor(((obs-fp)@basis-space['mean'])/space['scale']),context=torch.tensor((fp@basis-space['context_mean'])/space['context_scale']))
    folder=run/f'sigma{sigma:g}'
    with torch.no_grad():
        if method=='bilstm':
            model=SequenceLSTM(cfg['rank']);model.load_state_dict(torch.load(folder/'bilstm.pt',weights_only=True));model.eval();z=model(d['y'],d['context'])
        elif method.startswith('diffusion'):
            model=TemporalDiffusion(cfg['rank'],cfg['steps']);model.load_state_dict(torch.load(folder/'temporal_diffusion.pt',weights_only=True));schedule=DDPMScheduler.from_pretrained(folder/'scheduler')
            samples=diffusion_sample(model,schedule,d,cfg,seed);z=samples[0] if method=='diffusion_draw' else samples.mean(0)
        elif method in ['context_only','gaussian_current','gaussian_smoother']:
            fit=dict(np.load(folder/'gaussian.npz'));z=gaussian_predict(fit,d)[method]
        else:raise ValueError('Unsupported reconstruction method')
        generator=BatchGenerator(cfg['rank']);generator.load_state_dict(torch.load(run/'batch_generator.pt',weights_only=True));generator.eval()
        images=generator(z,d['context']).numpy()
    gradients=fp+(z.numpy()*space['scale']+space['mean'])@basis.T
    return images,gradients

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--sigma',required=True,type=float);p.add_argument('--method',default='diffusion_mean',choices=['bilstm','diffusion_mean','diffusion_draw','context_only','gaussian_current','gaussian_smoother']);a=p.parse_args();torch.set_num_threads(1)
    data=np.load(a.input);images,gradients=reconstruct(a.run,data['observations'],data['public_fingerprints'],a.sigma,a.method)
    np.savez_compressed(a.output,images=images,reconstructed_gradients=gradients)
    print(f'Saved {images.shape[0]} sequences, {images.shape[1]} rounds, two generated images per round to {a.output}')
