"""Causal denoising audit of evolving Poisson DP-SGD trajectories.

Clean private gradients are evaluation ground truth only. Each observer receives
public fingerprints at released checkpoints and the current/past noisy updates.
"""
import argparse, copy, csv, json, time
from pathlib import Path
import numpy as np
import torch
from diffusers import UNet2DModel, DDPMScheduler
from opacus.accountants import RDPAccountant
from .models import initialize, per_record_gradients, clip_gradients
from .data import load_mnist, digest


def write_csv(path, rows):
    with open(path, 'w') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def epsilon(cfg, sigma):
    acc=RDPAccountant()
    for _ in range(cfg['steps']):
        acc.step(noise_multiplier=sigma, sample_rate=cfg['batch_size']/cfg['population'])
    return acc.get_epsilon(cfg['delta'], alphas=[1+x/10 for x in range(1,100)]+list(range(12,513)))


def apply_update(model, update, lr):
    with torch.no_grad():
        offset=0
        for p in model.parameters():
            p.add_(update[offset:offset+p.numel()].reshape_as(p), alpha=-lr)
            offset+=p.numel()


def trajectory(initial, x, labels, ids, bank, test, cfg, sigma, seed):
    model=copy.deepcopy(initial); rng=np.random.default_rng(seed)
    nrng=torch.Generator().manual_seed(seed+100000)
    clean=[]; observed=[]; context=[]; utility=[]; sizes=[]
    q=cfg['batch_size']/len(ids); c=cfg['clip']; b=cfg['batch_size']
    d=sum(p.numel() for p in model.parameters())
    for t in range(cfg['steps']):
        # This public fingerprint is computed at the PRE-update checkpoint.
        f=clip_gradients(per_record_gradients(model,x[bank],labels[bank]),c).mean(0)
        selected=ids[rng.random(len(ids))<q]
        if len(selected):
            g=clip_gradients(per_record_gradients(model,x[selected],labels[selected]),c).sum(0)/b
        else:g=torch.zeros(d)
        y=g+torch.randn(d,generator=nrng)*(sigma*c/b)
        clean.append(g.numpy()); observed.append(y.numpy()); context.append(f.numpy()); sizes.append(len(selected))
        apply_update(model,y,0.5)
        if t==0 or (t+1)%16==0:
            with torch.no_grad():
                logits=model(x[test]); utility.append({'step':t+1,'accuracy':float((logits.argmax(1)==labels[test]).float().mean()),'loss':float(torch.nn.functional.cross_entropy(logits,labels[test]))})
    return dict(clean=np.stack(clean),observed=np.stack(observed),context=np.stack(context),batch_sizes=np.array(sizes)),utility,model


def histories(a, window):
    # Seven past observations only; zero padding and explicit validity channels.
    out=np.zeros((len(a),window-1,a.shape[1]),np.float32); valid=np.zeros((len(a),window-1),np.float32)
    for t in range(len(a)):
        k=min(t,window-1)
        if k:out[t,-k:]=a[t-k:t]; valid[t,-k:]=1
    return out,valid


def encode_runs(runs,basis,mean,scale,cfg):
    values=[]
    for run in runs:
        z=((run['clean']-run['context'])@basis-mean)/scale
        y=((run['observed']-run['context'])@basis-mean)/scale
        h,v=histories(y,cfg['window'])
        values.append(dict(z=z,y=y,h=h,v=v,run=run))
    return values


def arrays(items):
    return {k:torch.tensor(np.concatenate([a[k] for a in items]),dtype=torch.float32) for k in ['z','y','h','v']}


def net_input(xt,h,v,mask):
    n=len(xt); side=8
    return torch.cat([xt.reshape(n,1,side,side)*mask[:,None,None,None],h.reshape(n,-1,side,side),v[:,:,None,None].expand(-1,-1,side,side),mask[:,None,None,None].expand(-1,1,side,side)],1)


@torch.no_grad()
def neural_predict(net,schedule,data,ratio,mask=1):
    net.eval(); out=[]; t=len(schedule.alphas_cumprod)-1; a=schedule.alphas_cumprod[t]
    for start in range(0,len(data['y']),64):
        y=data['y'][start:start+64]; h=data['h'][start:start+64]; v=data['v'][start:start+64]
        pred=net(net_input(y*a.sqrt(),h,v,torch.full((len(y),),float(mask))),t).sample
        # Official sample-prediction conversion. No clipping of latent coefficients.
        xt=(y*a.sqrt()).reshape(-1,1,8,8)
        out.append(schedule.step(pred,t,xt).pred_original_sample.flatten(1))
    return torch.cat(out).numpy()


def train_denoiser(train,cal,ratio,cfg,folder,seed):
    torch.manual_seed(seed)
    net=UNet2DModel(sample_size=8,in_channels=2*cfg['window'],out_channels=1,layers_per_block=1,
        block_out_channels=(16,32),down_block_types=('DownBlock2D','DownBlock2D'),
        up_block_types=('UpBlock2D','UpBlock2D'),norm_num_groups=8,add_attention=False)
    ratios=np.geomspace(min(0.001,ratio/100),ratio,100); ab=1/(1+ratios**2)
    schedule=DDPMScheduler(num_train_timesteps=100,trained_betas=1-ab/np.r_[1.,ab[:-1]],prediction_type='sample',clip_sample=False)
    opt=torch.optim.AdamW(net.parameters(),lr=0.0005)
    gen=torch.Generator().manual_seed(seed+1); best=float('inf'); saved=None; logs=[]
    for step in range(cfg['denoiser_steps']):
        net.train(); ix=torch.randint(len(train['z']),(cfg['train_batch'],),generator=gen)
        z=train['z'][ix].reshape(-1,1,8,8)
        t=torch.randint(100,(len(ix),),generator=gen); t[:len(ix)//2]=99
        noise=torch.randn(z.shape,generator=gen); xt=schedule.add_noise(z,noise,t)
        mask=(torch.rand(len(ix),generator=gen)>.25).float()
        pred=net(net_input(xt,train['h'][ix],train['v'][ix],mask),t).sample
        loss=(pred-z).square().mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1); opt.step()
        if (step+1)%200==0:
            mse=float(np.mean((neural_predict(net,schedule,cal,ratio)-cal['z'].numpy())**2))
            logs.append(dict(step=step+1,loss=float(loss.detach()),calibration_mse=mse))
            if mse<best:best=mse;saved=copy.deepcopy(net.state_dict())
            print(f'  diffusion {step+1}: calibration MSE {mse:.4f}',flush=True)
    net.load_state_dict(saved); net.save_pretrained(folder/'unet'); schedule.save_pretrained(folder/'scheduler')
    write_csv(folder/'training.csv',logs)
    return net,schedule


def gaussian_fit(train,cal,ratio):
    z=train['z'].numpy(); y=train['y'].numpy(); h=train['h'].numpy().reshape(len(z),-1); v=train['v'].numpy()
    zmean=z.mean(0); cov=np.cov(z,rowvar=False); gain=np.linalg.solve(cov+ratio**2*np.eye(len(zmean)),cov)
    # Causal linear Gaussian regression, ridge chosen on separate public runs.
    fits={}
    for use_current in [False,True]:
        design=np.c_[h,v,y] if use_current else np.c_[h,v]
        cm=design.mean(0); zm=z.mean(0); a=design-cm
        calx=np.c_[cal['h'].numpy().reshape(len(cal['z']),-1),cal['v'].numpy()]
        if use_current:calx=np.c_[calx,cal['y'].numpy()]
        best=None
        for ridge in [.01,.1,1,10,100]:
            w=a.T@np.linalg.solve(a@a.T+len(a)*ridge*np.eye(len(a)),z-zm)
            error=np.mean(((calx-cm)@w+zm-cal['z'].numpy())**2)
            if best is None or error<best[0]:best=(error,cm,w,zm,ridge)
        fits[use_current]=best
    return zmean,gain,fits


def evaluate(items,basis,mean,scale,net,schedule,gaussian,ratio,cfg,sigma):
    rows=[]; zmean,gain,fits=gaussian
    for seed,item in enumerate(items):
        data=arrays([item]); y=item['y']; h=item['h'].reshape(len(y),-1); v=item['v']; r=item['run']
        latent={'gaussian_current':(y-zmean)@gain+zmean,
                'diffusion_temporal':neural_predict(net,schedule,data,ratio),
                'diffusion_context_only':neural_predict(net,schedule,data,ratio,mask=0)}
        for current,name in [(False,'gaussian_context_only'),(True,'gaussian_temporal')]:
            _,cm,w,zm,_=fits[current]; design=np.c_[h,v,y] if current else np.c_[h,v]
            latent[name]=(design-cm)@w+zm
        predictions={name:r['context']+(z*scale+mean)@basis.T for name,z in latent.items()}
        predictions.update(public_fingerprint=r['context'],noisy=r['observed'],zero=np.zeros_like(r['clean']))
        # Complement outside public PCA uses context only for every learned/Gaussian method.
        # A clean projection oracle measures the representation ceiling, not an attack.
        predictions['projection_oracle']=r['context']+((r['clean']-r['context'])@basis)@basis.T
        for method,est in predictions.items():
            err=((est-r['clean'])**2).sum(1)/cfg['clip']**2
            cos=(est*r['clean']).sum(1)/(np.linalg.norm(est,axis=1)*np.linalg.norm(r['clean'],axis=1)+1e-12)
            for t in range(len(err)):
                rows.append(dict(sigma=sigma,seed=seed,step=t+1,method=method,normalized_squared_error=float(err[t]),cosine=float(cos[t])))
    return rows


def report(rows,privacy,utility,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=[]; seedrows=[]
    for sigma in sorted(set(r['sigma'] for r in rows)):
        for method in sorted(set(r['method'] for r in rows)):
            subset=[r for r in rows if r['sigma']==sigma and r['method']==method]
            seedmeans=[np.mean([r['normalized_squared_error'] for r in subset if r['seed']==s]) for s in sorted(set(r['seed'] for r in subset))]
            summary.append(dict(sigma=sigma,method=method,error=float(np.mean(seedmeans)),seed_sd=float(np.std(seedmeans,ddof=1)),cosine=float(np.mean([r['cosine'] for r in subset]))))
            for s,value in enumerate(seedmeans):seedrows.append(dict(sigma=sigma,method=method,seed=s,error=value))
    write_csv(out/'summary.csv',summary);write_csv(out/'per_seed.csv',seedrows);write_csv(out/'measurements.csv',rows);write_csv(out/'privacy.csv',privacy);write_csv(out/'utility.csv',utility)
    methods=['public_fingerprint','gaussian_current','gaussian_temporal','gaussian_context_only','diffusion_temporal','diffusion_context_only','projection_oracle']
    fig,ax=plt.subplots(figsize=(8,4.8))
    for method in methods:
        rr=[r for r in summary if r['method']==method]
        ax.plot([r['sigma'] for r in rr],[r['error'] for r in rr],marker='o',label=method.replace('_',' '))
    ax.set(xscale='log',xlabel='DP-SGD noise multiplier σ',ylabel='Full-gradient squared error / C²',title='Evolving DP-SGD: causal clipped-average recovery')
    ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'recovery.{ext}',dpi=220)
    plt.close(fig)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/trajectory_dp.json');parser.add_argument('--data',required=True);parser.add_argument('--output',default='runs/trajectory_dp');parser.add_argument('--trajectory-source');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'completion.json').exists():raise RuntimeError('Choose a new output directory for a completed experiment')
    torch.set_num_threads(1);start=time.time();x,y=load_mnist(args.data)
    source=Path(args.trajectory_source) if args.trajectory_source else out
    if source!=out:
        old=json.loads((source/'config.json').read_text())
        assert all(old[k]==v for k,v in cfg.items() if k not in ['window','denoiser_steps']), 'Trajectory configuration mismatch'
        assert json.loads((source/'completion.json').read_text())['dataset_sha256']==digest(args.data), 'Dataset mismatch'
    rng=np.random.default_rng(cfg['seed']);perm=rng.permutation(len(x));cursor=0;splits={}
    roles={'pretrain':2000,'fingerprints':cfg['fingerprints'],'test':2000}
    for name,size in roles.items():splits[name]=perm[cursor:cursor+size];cursor+=size
    for role,count in [('train',cfg['public_train_runs']),('cal',cfg['public_calibration_runs']),('target',cfg['target_runs'])]:
        for k in range(count):splits[f'{role}{k}']=perm[cursor:cursor+cfg['population']];cursor+=cfg['population']
    assert cursor<=len(x)
    (out/'splits.json').write_text(json.dumps({k:v.tolist() for k,v in splits.items()}));(out/'config.json').write_text(json.dumps(cfg,indent=2))
    initial=initialize(cfg['seed']);opt=torch.optim.SGD(initial.parameters(),lr=.1)
    for _ in range(cfg['pretrain_steps']):
        ids=rng.choice(splits['pretrain'],64,replace=False);loss=torch.nn.functional.cross_entropy(initial(x[ids]),y[ids]);opt.zero_grad();loss.backward();opt.step()
    torch.save(initial.state_dict(),out/'public_initial_cnn.pt')
    allrows=[];privacy=[];utility=[]
    for sigma in cfg['sigmas']:
        folder=out/f'sigma{sigma:g}';folder.mkdir(exist_ok=True);runs={}
        privacy.append(dict(sigma=sigma,clip=cfg['clip'],batch_size=cfg['batch_size'],population=cfg['population'],steps=cfg['steps'],delta=cfg['delta'],epsilon=epsilon(cfg,sigma),accountant='Opacus RDP, Poisson, add/remove'))
        for role,count in [('train',cfg['public_train_runs']),('cal',cfg['public_calibration_runs']),('target',cfg['target_runs'])]:
            runs[role]=[]
            for k in range(count):
                print(f'sigma={sigma} {role}{k}: actual DP-SGD trajectory',flush=True)
                path=source/f'sigma{sigma:g}'/f'{role}{k}.npz'
                if path.exists():
                    data=dict(np.load(path));u=json.loads(path.with_suffix('.json').read_text())
                else:
                    data,u,model=trajectory(initial,x,y,splits[f'{role}{k}'],splits['fingerprints'],splits['test'],cfg,sigma,cfg['seed']+1000*['train','cal','target'].index(role)+k)
                    np.savez_compressed(path,**data);path.with_suffix('.json').write_text(json.dumps(u));torch.save(model.state_dict(),folder/f'{role}{k}_final_cnn.pt')
                runs[role].append(data)
                utility.extend(dict(sigma=sigma,role=role,seed=k,**r) for r in u)
        residual=torch.tensor(np.concatenate([r['clean']-r['context'] for r in runs['train']]))
        torch.manual_seed(cfg['seed']);_,_,basis=torch.pca_lowrank(residual,q=cfg['rank']+8,center=True,niter=3);basis=basis[:,:cfg['rank']].numpy()
        coeff=residual.numpy()@basis;mean=coeff.mean(0);scale=float(np.sqrt(np.mean((coeff-mean)**2)))
        encoded={role:encode_runs(rr,basis,mean,scale,cfg) for role,rr in runs.items()}
        train=arrays(encoded['train']);cal=arrays(encoded['cal']);ratio=sigma*cfg['clip']/cfg['batch_size']/scale
        np.savez_compressed(folder/'public_space.npz',basis=basis,mean=mean,scale=scale)
        net,schedule=train_denoiser(train,cal,ratio,cfg,folder,cfg['seed'])
        gf=gaussian_fit(train,cal,ratio)
        np.savez_compressed(folder/'gaussian.npz',mean=gf[0],gain=gf[1],context_center=gf[2][False][1],context_weights=gf[2][False][2],context_target_mean=gf[2][False][3],temporal_center=gf[2][True][1],temporal_weights=gf[2][True][2],temporal_target_mean=gf[2][True][3])
        allrows.extend(evaluate(encoded['target'],basis,mean,scale,net,schedule,gf,ratio,cfg,sigma))
        report(allrows,privacy,utility,out)
    metadata=dict(trajectory_source=str(source.resolve()),seconds=time.time()-start,dataset_sha256=digest(args.data),torch=torch.__version__,configuration=cfg)
    (out/'completion.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata),flush=True)

if __name__=='__main__':main()
