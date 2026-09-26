"""Laboratory evaluation of upstream diffusion denoisers on held-out gradients."""
import hashlib
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
import diffusers
from .data import load_mnist,digest
from .models import initialize,per_record_gradients
from .diffusion_denoiser import DiffusionDenoiser,bank_estimate
from .io import write_csv,write_json


class GradientSpace:
    def __init__(self,public,labels,cfg,seed):
        self.cfg=cfg;self.labels=np.asarray(labels);self.raw=public;self.k=public.shape[1]
        if self.k!=2 or cfg['rank']!=64:raise ValueError('This adapter uses two checkpoints and 64 PCA coordinates each')
        self.dimension=public.shape[2];self.units={};self.means={};residual={}
        for c in cfg['clips']:
            self.units[c]=self.unit(public,c)
            self.means[c]=np.stack([self.units[c][self.labels==i].mean(0) for i in range(10)])
            residual[c]=self.units[c]-self.means[c][self.labels]
        torch.manual_seed(seed);self.basis=[]
        for checkpoint in range(2):
            g=torch.from_numpy(np.concatenate([residual[c][:,checkpoint] for c in cfg['clips']]))
            _,_,v=torch.pca_lowrank(g,q=72,center=False,niter=3)
            self.basis.append(v[:,:64].numpy())
        self.basis=np.stack(self.basis)
        allz=np.concatenate([self.encode(residual[c]) for c in cfg['clips']])
        self.scale=float(np.sqrt(np.mean(allz**2)))
        self.training_z={c:self.encode(residual[c])/self.scale for c in cfg['clips']}
        self.cov={};self.floor={}
        for c in cfg['clips']:
            z=self.encode(residual[c]);flat=z.reshape(len(z),-1)
            self.cov[c]=flat.T@flat/(len(flat)-10)
            outside=residual[c]-self.decode(z)
            self.floor[c]=np.sum(outside**2,axis=(0,2))/((len(flat)-10)*(self.dimension-64))
        rng=np.random.default_rng(seed+1000);self.banks={}
        for c in cfg['clips']:
            for b in cfg['batch_sizes']:
                rng=np.random.default_rng(seed+1000+b)
                ids=self.draw_ids(len(public),b,cfg['bank_size'],rng)
                clean,hist=self.samples(self.raw,self.labels,c,b,ids)
                center=self.center(hist,c);z=self.encode(clean-center)/self.scale
                self.banks[c,b]=torch.from_numpy(z.reshape(-1,2,8,8))

    @staticmethod
    def draw_ids(n,b,count,rng):
        return np.stack([rng.choice(n,b,replace=False) for _ in range(count)])

    @staticmethod
    def unit(raw,c):
        norms=np.linalg.norm(raw,axis=2,keepdims=True)
        return raw*np.minimum(1,c/np.maximum(norms,1e-12))/c

    def center(self,hist,c):return np.einsum('nc,ckd->nkd',hist,self.means[c])
    def encode(self,g):return np.einsum('nkd,kdr->nkr',g,self.basis,optimize=True)
    def decode(self,z):return np.einsum('nkr,kdr->nkd',z,self.basis,optimize=True)

    def samples(self,raw,labels,c,b,ids):
        clean=(self.units[c] if raw is self.raw else self.unit(raw,c))[ids].mean(1)
        hist=np.eye(10,dtype=np.float32)[labels[ids]].mean(1)
        return clean.astype(np.float32),hist

    def sampler(self,seed):
        rng=np.random.default_rng(seed)
        def sample(n):
            c=float(rng.choice(self.cfg['clips']));b=int(rng.choice(self.cfg['batch_sizes']))
            ids=self.draw_ids(len(self.raw),b,n,rng)
            hist=np.eye(10,dtype=np.float32)[self.labels[ids]].mean(1)
            z=self.training_z[c][ids].mean(1)
            return (torch.from_numpy(z.reshape(-1,2,8,8)),torch.from_numpy(hist),
                    torch.full((n,),c),torch.full((n,),float(b)),self.banks[c,b].unsqueeze(0).expand(n,-1,-1,-1,-1))
        return sample

    def gaussian(self,observation,hist,c,b,eta):
        center=self.center(hist,c);res=observation-center;z=self.encode(res);flat=z.reshape(len(z),-1)
        covariance=self.cov[c]/b
        filt=np.linalg.solve(covariance+eta**2*np.eye(128),covariance)
        fitted=(flat@filt).reshape(-1,2,64)
        return center+self.decode(fitted)+self.outside(res,c,b,eta)

    def outside(self,res,c,b,eta):
        floor=self.floor[c]/b;shrink=floor/(floor+eta**2)
        return (res-self.decode(self.encode(res)))*shrink[None,:,None]

    def save(self,path):
        values=dict(basis=self.basis,scale=self.scale)
        for c in self.cfg['clips']:
            values[f'means_{c}']=self.means[c];values[f'cov_{c}']=self.cov[c];values[f'floor_{c}']=self.floor[c]
            for b in self.cfg['batch_sizes']:values[f'bank_{c}_{b}']=self.banks[c,b].numpy()
        np.savez_compressed(path,**values)


def calibration_set(space,raw,labels,model,cfg,seed):
    rng=np.random.default_rng(seed);result=[]
    for c in cfg['clips']:
        for b in cfg['batch_sizes']:
            ids=space.draw_ids(len(raw),b,cfg['calibration_batches'],rng)
            clean,hist=space.samples(raw,labels,c,b,ids)
            z=torch.from_numpy((space.encode(clean-space.center(hist,c))/space.scale).reshape(-1,2,8,8))
            for tau in cfg['absolute_noise']:
                t,_=model.timestep(tau/c/space.scale)
                noise=torch.from_numpy(rng.standard_normal(z.shape).astype(np.float32))
                xt=model.schedule.add_noise(z,noise,torch.full((len(z),),t))
                result.append((xt,t,z,torch.from_numpy(hist),torch.full((len(z),),c),
                               torch.full((len(z),),float(b)),space.banks[c,b].unsqueeze(0).expand(len(z),-1,-1,-1,-1)))
    return result


def run(cfg,source,data,output,log=print,resume=False):
    source=Path(source);output=Path(output)
    previous=None
    if output.exists() and any(output.iterdir()):
        if not resume:raise FileExistsError('Run directory must be empty; use explicit --resume for an interrupted run')
        if (output/'completion.json').exists():raise FileExistsError('Completed runs cannot be resumed')
        previous=json.loads((output/'config.json').read_text())
        if {k:v for k,v in previous.items() if k!='threads'}!={k:v for k,v in cfg.items() if k!='threads'}:
            raise ValueError('Resume requires the original scientific configuration; only CPU threads may change')
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(cfg['threads']);torch.use_deterministic_algorithms(True)
    started=time.time();x,y=load_mnist(data);rows=[];history=[];spaces=[]
    write_json(output/'config.json',cfg)
    old_provenance=json.loads((output/'provenance.json').read_text()) if previous else None
    if old_provenance and old_provenance['dataset_sha256']!=digest(data):raise ValueError('Dataset changed since interrupted run')
    write_json(output/'provenance.json',dict(diffusers=diffusers.__version__,torch=torch.__version__,
        numpy=np.__version__,python=platform.python_version(),dataset_sha256=digest(data),
        source_reconstruction_provenance=json.loads((source/'provenance.json').read_text()),
        mechanism='Independent Gaussian corruption of individual or averaged clipped gradients at two fixed public checkpoints',
        representations='64 public PCA coefficients per checkpoint, common scalar normalization; residual coordinates use Gaussian shrinkage',
        labels='Known label histogram given equally to all estimators',
        bank='256 public prior samples or public batch combinations per clipping/batch condition; test records excluded',
        iterative='DDIM eta=0 reconstruction starting from the observed diffusion state; not guaranteed posterior mean',
        resumed_from_config=previous,previous_provenance=old_provenance,
        public_gradient_and_pca_threads=2,unet_training_threads=cfg['threads'],
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('diffusion_denoiser.py')]}))
    if previous and (output/'training.csv').exists():
        with (output/'training.csv').open() as f:
            for item in csv.DictReader(f):
                history.append(dict(seed=int(item['seed']),prediction_type=item['prediction_type'],
                    step=int(item['step']),training_loss=float(item['training_loss']),calibration_x0_mse=float(item['calibration_x0_mse'])))
    for seed in cfg['seeds']:
        # Keep feature construction identical to the original run when tuning U-Net CPU threads.
        torch.set_num_threads(2)
        folder=output/f'seed{seed}';folder.mkdir(exist_ok=True);src=source/f'seed{seed}'
        roles=json.loads((src/'splits.json').read_text())
        write_json(folder/'splits.json',{k:roles[k] for k in ['model_train','prior','calibration','target']})
        chosen=[set(roles[k]) for k in ['model_train','prior','calibration','target']]
        if any(chosen[i]&chosen[j] for i in range(4) for j in range(i)):raise ValueError('Data roles overlap')
        models=[]
        for j in range(2):
            model=initialize(seed);model.load_state_dict(torch.load(src/f'public_checkpoint_{j}.pt',weights_only=True));models.append(model.eval())
        log(f'[diffusion seed {seed}] compute exact public, calibration and held-out gradients',flush=True)
        raw={role:np.stack([per_record_gradients(model,x[roles[role]],y[roles[role]]).numpy() for model in models],axis=1)
             for role in ['prior','calibration','target']}
        labels={role:y[roles[role]].numpy() for role in raw}
        space=GradientSpace(raw['prior'],labels['prior'],cfg,seed);space.save(folder/'gradient_space.npz')
        torch.set_num_threads(cfg['threads'])
        spaces.append(dict(seed=seed,normalization_scale=space.scale,gradient_dimension=space.dimension,
            pca_coordinates_per_checkpoint=64,public_examples=len(raw['prior']),heldout_examples=len(raw['target'])))
        trained={}
        for prediction in cfg['prediction_types']:
            denoiser=DiffusionDenoiser(cfg,prediction,seed+10)
            if resume and (folder/prediction/'scheduler/scheduler_config.json').exists() and any(v['seed']==seed and v['prediction_type']==prediction for v in history):
                from diffusers import UNet2DModel
                denoiser.model=UNet2DModel.from_pretrained(folder/prediction/'unet',low_cpu_mem_usage=False).eval()
                trained[prediction]=denoiser
                log(f'[diffusion seed {seed}] reuse completed {prediction} model',flush=True)
                continue
            calibration=calibration_set(space,raw['calibration'],labels['calibration'],denoiser,cfg,seed+20)
            training=denoiser.train(space.sampler(seed+30),calibration,seed=seed+40,log=log)
            for item in training:history.append(dict(seed=seed,prediction_type=prediction,**item))
            denoiser.save(folder/prediction);trained[prediction]=denoiser
            write_csv(output/'training.csv',history)
        reference=trained['epsilon'];rng=np.random.default_rng(seed+50)
        for c in cfg['clips']:
            for b in cfg['batch_sizes']:
                rng=np.random.default_rng(seed+50+b)
                ids=(np.arange(len(raw['target']))[:,None] if b==1 else
                     space.draw_ids(len(raw['target']),b,cfg['evaluation_batches'],rng))
                clean,hist=space.samples(raw['target'],labels['target'],c,b,ids);n=len(clean)
                write_json(folder/f'evaluation_batches_C{c}_B{b}.json',ids.tolist())
                center=space.center(hist,c);clean_z=space.encode(clean-center)/space.scale
                noise=rng.standard_normal(clean.shape).astype(np.float32)
                for requested_tau in cfg['absolute_noise']:
                    t,ratio=reference.timestep(requested_tau/c/space.scale);eta=ratio*space.scale;tau=eta*c
                    observation=clean+eta*noise
                    observed_z=space.encode(observation-center)/space.scale
                    a=reference.schedule.alphas_cumprod[t]
                    xt=torch.from_numpy(observed_z.reshape(-1,2,8,8))*a.sqrt()
                    bank=space.banks[c,b].unsqueeze(0).expand(n,-1,-1,-1,-1)
                    args=(xt,t,torch.from_numpy(hist),torch.full((n,),c),torch.full((n,),float(b)),bank)
                    outside=space.outside(observation-center,c,b,eta)
                    estimates={'zero':np.zeros_like(clean),'prior_only':center,'noisy':observation,
                               'gaussian':space.gaussian(observation,hist,c,b,eta)}
                    bank_z=bank_estimate(torch.from_numpy(observed_z.reshape(-1,2,8,8)),bank,torch.full((n,),ratio**2),torch.ones(n,2))
                    estimates['public_bank']=center+space.decode(bank_z.numpy().reshape(-1,2,64)*space.scale)+outside
                    latent_predictions={}
                    for prediction,model in trained.items():
                        modes=['single','trajectory','trajectory_bank'] if prediction=='epsilon' else ['trajectory_bank']
                        for mode in modes:
                            key=f'{prediction}_{mode}';pred=model.predict(*args,mode=mode).numpy().reshape(-1,2,64)
                            latent_predictions[key]=pred;estimates[key]=center+space.decode(pred*space.scale)+outside
                        # Iterative reverse diffusion uses identical input; no target selection.
                        if prediction in ['epsilon','v_prediction']:
                            key=f'{prediction}_ddim';pred=model.predict(*args,mode='trajectory_bank',iterative=True).numpy().reshape(-1,2,64)
                            latent_predictions[key]=pred;estimates[key]=center+space.decode(pred*space.scale)+outside
                    prior_error=np.mean(np.sum((center-clean)**2,axis=2),axis=1)
                    for method,estimate in estimates.items():
                        error=np.mean(np.sum((estimate-clean)**2,axis=2),axis=1)
                        cosine=np.sum(estimate*clean,axis=(1,2))/(np.linalg.norm(estimate.reshape(n,-1),axis=1)*np.linalg.norm(clean.reshape(n,-1),axis=1)+1e-12)
                        latent_error=None
                        if method in latent_predictions:latent_error=np.mean((latent_predictions[method]-clean_z)**2,axis=(1,2))
                        for i in range(n):
                            rows.append(dict(seed=seed,clip=c,batch_size=b,requested_noise=requested_tau,actual_noise=tau,
                                noise_to_clip_ratio=eta,equivalent_dp_noise_multiplier=eta*b,timestep=t,
                                method=method,evaluation_index=i,squared_l2=float(error[i]*c*c),
                                clip_normalized_squared_l2=float(error[i]),cosine=float(cosine[i]),
                                gain_over_prior=float(prior_error[i]-error[i]),
                                latent_x0_mse='' if latent_error is None else float(latent_error[i]),
                                implied_latent_noise_mse='' if latent_error is None else float(latent_error[i]/ratio**2),
                                trivial_zero_residual_noise_mse=float(np.mean(clean_z[i]**2)/ratio**2)))
                    log(f'[diffusion seed {seed}] C={c}, B={b}, noise≈{tau:.4g} evaluated',flush=True)
        write_csv(output/'records.csv',rows);write_csv(output/'spaces.csv',spaces)
    from .diffusion_report import report
    report(output,cfg,rows)
    write_json(output/'completion.json',dict(status='complete',elapsed_seconds=time.time()-started,
        trained_unets=len(cfg['seeds'])*len(cfg['prediction_types']),measurements=len(rows),
        test_images_per_seed=30,model_family='Diffusers UNet2DModel, randomly initialized and trained on public gradient coefficients'))
    log(f'Diffusion laboratory complete: {output}',flush=True)
