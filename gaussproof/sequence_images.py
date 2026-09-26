"""Offline BiLSTM/conditional diffusion reconstruction and unordered MNIST batches.

This is an explicitly small-batch feasibility audit. Public supervised training is
separate from private evaluation. No private batch labels or indices are inputs.
"""
import argparse,copy,csv,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from diffusers import DDPMScheduler, DDIMScheduler
from .models import initialize,per_record_gradients,clip_gradients,conservative_epsilon
from .data import load_mnist,digest
from .trajectory_dp import apply_update,write_csv


def batch_mse(pred,target,reduction=True):
    """Permutation invariant two-image squared error; assignment for scoring only."""
    direct=(pred-target).square().mean(dim=(1,2,3))
    swapped=(pred-target.flip(1)).square().mean(dim=(1,2,3))
    losses=torch.minimum(direct,swapped)
    return losses.mean() if reduction else losses


class BatchGenerator(nn.Module):
    def __init__(self,rank):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(2*rank,256),nn.SiLU(),nn.Linear(256,2*32*7*7),nn.SiLU())
        self.decoder=nn.Sequential(nn.ConvTranspose2d(32,16,4,2,1),nn.SiLU(),nn.ConvTranspose2d(16,1,4,2,1),nn.Sigmoid())
    def forward(self,z,context):
        shape=z.shape[:-1]
        h=self.embed(torch.cat([z,context],-1)).reshape(-1,32,7,7)
        return self.decoder(h).reshape(*shape,2,28,28)


class SequenceLSTM(nn.Module):
    def __init__(self,rank):
        super().__init__();self.embed=nn.Linear(2*rank,128)
        self.lstm=nn.LSTM(128,96,batch_first=True,bidirectional=True)
        self.output=nn.Linear(192,rank)
    def forward(self,y,context):
        h=self.embed(torch.cat([y,context],-1)).tanh()
        return self.output(self.lstm(h)[0])


class TemporalDiffusion(nn.Module):
    """Conditional all-round attention denoiser, custom adapter, upstream scheduler."""
    def __init__(self,rank,steps):
        super().__init__();self.input=nn.Linear(3*rank,128)
        self.time=nn.Embedding(100,128);self.position=nn.Parameter(torch.randn(1,steps,128)*.01)
        layer=nn.TransformerEncoderLayer(128,4,256,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.output=nn.Linear(128,rank)
    def forward(self,xk,k,y,context):
        h=self.input(torch.cat([xk,y,context],-1))+self.position+self.time(k)[:,None]
        return self.output(self.temporal(h))


def generate_run(initial,x,labels,ids,bank,cfg,sigma,seed):
    model=copy.deepcopy(initial);rng=np.random.default_rng(seed);trng=torch.Generator().manual_seed(seed+111)
    clean=[];noisy=[];context=[];images=[];truth=[];selected_ids=[]
    for t in range(cfg['steps']):
        chosen=rng.choice(ids,cfg['batch_size'],replace=False)
        f=clip_gradients(per_record_gradients(model,x[bank],labels[bank]),cfg['clip']).mean(0)
        g=clip_gradients(per_record_gradients(model,x[chosen],labels[chosen]),cfg['clip']).mean(0)
        y=g+torch.randn(g.shape,generator=trng)*sigma*cfg['clip']/cfg['batch_size']
        clean.append(g.numpy());noisy.append(y.numpy());context.append(f.numpy())
        images.append(x[chosen,0].numpy());truth.append(labels[chosen].numpy());selected_ids.append(chosen)
        apply_update(model,y,cfg['learning_rate'])
    return dict(clean=np.stack(clean),noisy=np.stack(noisy),context=np.stack(context),images=np.stack(images),labels=np.stack(truth),ids=np.stack(selected_ids))


def make_data(runs,basis,mean,scale,context_mean,context_scale):
    return dict(z=torch.tensor(np.stack([(r['clean']-r['context'])@basis for r in runs])-mean)/scale,
        y=torch.tensor(np.stack([(r['noisy']-r['context'])@basis for r in runs])-mean)/scale,
        context=torch.tensor(np.stack([r['context']@basis for r in runs])-context_mean)/context_scale,
        images=torch.tensor(np.stack([r['images'] for r in runs])),runs=runs)


def gaussian_model(train,cal,ratio):
    z=train['z'].numpy();y=train['y'].numpy();context=train['context'].numpy()
    # Public-checkpoint-only linear predictor, selected on calibration sequences.
    x=context.reshape(-1,context.shape[-1]);target=z.reshape(-1,z.shape[-1]);xm=x.mean(0);zm=target.mean(0)
    def ridge_fit(a,b,penalty):
        if a.shape[1]<=a.shape[0]:return np.linalg.solve(a.T@a+len(a)*penalty*np.eye(a.shape[1]),a.T@b)
        return a.T@np.linalg.solve(a@a.T+len(a)*penalty*np.eye(len(a)),b)
    best=None
    for ridge in [.01,.1,1,10,100]:
        w=ridge_fit(x-xm,target-zm,ridge)
        error=np.mean(((cal['context'].numpy()-xm)@w+zm-cal['z'].numpy())**2)
        if best is None or error<best[0]:best=(error,w)
    w=best[1];prior=lambda c:(c-xm)@w+zm
    res=z-prior(context);cov=np.cov(res.reshape(-1,res.shape[-1]),rowvar=False)
    gain=np.linalg.solve(cov+ratio**2*np.eye(len(cov)),cov)
    # Offline linear smoother: pooled position-centered sequence features, all rounds.
    features=np.concatenate([y,context],-1).reshape(len(y),-1)
    targets=z.reshape(len(z),-1);fm=features.mean(0);tm=targets.mean(0)
    calx=np.concatenate([cal['y'].numpy(),cal['context'].numpy()],-1).reshape(len(cal['y']),-1)
    best=None
    for ridge in [.01,.1,1,10,100,1000]:
        sw=ridge_fit(features-fm,targets-tm,ridge)
        error=np.mean(((calx-fm)@sw+tm-cal['z'].numpy().reshape(len(calx),-1))**2)
        if best is None or error<best[0]:best=(error,sw)
    return dict(xm=xm,zm=zm,w=w,gain=gain,fm=fm,tm=tm,sw=best[1])


def gaussian_predict(fit,data):
    y=data['y'].numpy();c=data['context'].numpy();p=(c-fit['xm'])@fit['w']+fit['zm']
    current=p+(y-p)@fit['gain']
    f=np.concatenate([y,c],-1).reshape(len(y),-1)
    smooth=((f-fit['fm'])@fit['sw']+fit['tm']).reshape(y.shape)
    return {'context_only':torch.tensor(p,dtype=torch.float32),'gaussian_current':torch.tensor(current,dtype=torch.float32),'gaussian_smoother':torch.tensor(smooth,dtype=torch.float32)}


def fit_lstm(train,cal,cfg,seed,folder):
    torch.manual_seed(seed);net=SequenceLSTM(cfg['rank']);opt=torch.optim.AdamW(net.parameters(),lr=.001)
    best=float('inf');state=None;log=[]
    for step in range(cfg['lstm_steps']):
        ix=torch.randint(len(train['z']),(16,));net.train();pred=net(train['y'][ix],train['context'][ix]);loss=(pred-train['z'][ix]).square().mean()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
        if (step+1)%200==0:
            net.eval()
            with torch.no_grad():error=float((net(cal['y'],cal['context'])-cal['z']).square().mean())
            if error<best:best=error;state=copy.deepcopy(net.state_dict())
            log.append(dict(step=step+1,train_loss=float(loss.detach()),calibration_mse=error));print('  BiLSTM',log[-1],flush=True)
    net.load_state_dict(state);net.eval();torch.save(net.state_dict(),folder/'bilstm.pt');write_csv(folder/'bilstm_training.csv',log)
    return net


@torch.no_grad()
def diffusion_sample(net,schedule,data,cfg,seed,draws=None):
    net.eval();scheduler=DDIMScheduler.from_config(schedule.config);scheduler.set_timesteps(cfg['diffusion_sampling_steps'])
    generator=torch.Generator().manual_seed(seed);samples=[]
    for _ in range(draws or cfg['posterior_draws']):
        xk=torch.randn(data['y'].shape,generator=generator)
        for t in scheduler.timesteps:
            pred=net(xk,torch.full((len(xk),),int(t),dtype=torch.long),data['y'],data['context'])
            xk=scheduler.step(pred,int(t),xk,eta=0).prev_sample
        samples.append(xk)
    return torch.stack(samples)


def fit_diffusion(train,cal,cfg,seed,folder):
    torch.manual_seed(seed);net=TemporalDiffusion(cfg['rank'],cfg['steps'])
    schedule=DDPMScheduler(num_train_timesteps=cfg['diffusion_train_timesteps'],beta_schedule='squaredcos_cap_v2',prediction_type='v_prediction',clip_sample=False)
    opt=torch.optim.AdamW(net.parameters(),lr=.0005);best=float('inf');state=None;logs=[]
    for step in range(cfg['diffusion_steps']):
        net.train();ix=torch.randint(len(train['z']),(16,));z=train['z'][ix];noise=torch.randn_like(z)
        k=torch.randint(cfg['diffusion_train_timesteps'],(len(ix),));xk=schedule.add_noise(z,noise,k)
        pred=net(xk,k,train['y'][ix],train['context'][ix]);loss=(pred-schedule.get_velocity(z,noise,k)).square().mean()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
        if (step+1)%400==0:
            # Full reverse sampling, held-out public sequences, fixed random seed.
            recovered=diffusion_sample(net,schedule,cal,cfg,seed+100,draws=2).mean(0)
            error=float((recovered-cal['z']).square().mean())
            if error<best:best=error;state=copy.deepcopy(net.state_dict())
            logs.append(dict(step=step+1,loss=float(loss.detach()),calibration_sample_mse=error));print('  diffusion',logs[-1],flush=True)
    net.load_state_dict(state);net.eval();torch.save(net.state_dict(),folder/'temporal_diffusion.pt');schedule.save_pretrained(folder/'scheduler');write_csv(folder/'diffusion_training.csv',logs)
    return net,schedule


def fit_generator(training,calibration,cfg,folder):
    torch.manual_seed(cfg['seed']+7);net=BatchGenerator(cfg['rank']);opt=torch.optim.AdamW(net.parameters(),lr=.0005)
    z=torch.cat([a['z'].flatten(0,1) for a in training]);c=torch.cat([a['context'].flatten(0,1) for a in training]);images=torch.cat([a['images'].flatten(0,1) for a in training])
    cz=torch.cat([a['z'].flatten(0,1) for a in calibration]);cc=torch.cat([a['context'].flatten(0,1) for a in calibration]);ci=torch.cat([a['images'].flatten(0,1) for a in calibration])
    best=float('inf');state=None;logs=[]
    for step in range(cfg['generator_steps']):
        net.train();ix=torch.randint(len(z),(32,))
        pred=net(z[ix],c[ix]);loss=batch_mse(pred,images[ix]);opt.zero_grad();loss.backward();opt.step()
        if (step+1)%250==0:
            net.eval()
            with torch.no_grad():error=float(torch.cat([batch_mse(net(cz[a:a+64],cc[a:a+64]),ci[a:a+64],False) for a in range(0,len(cz),64)]).mean())
            if error<best:best=error;state=copy.deepcopy(net.state_dict())
            logs.append(dict(step=step+1,train_mse=float(loss.detach()),calibration_mse=error));print('  image generator',logs[-1],flush=True)
    net.load_state_dict(state);net.eval();torch.save(net.state_dict(),folder/'batch_generator.pt');write_csv(folder/'generator_training.csv',logs)
    return net


def image_score(pred,target,labels,classifier):
    direct=(pred-target).square().mean((2,3)).sum(1);swapped=(pred.flip(1)-target).square().mean((2,3)).sum(1)
    aligned=pred.clone();swap=swapped<direct;aligned[swap]=pred[swap].flip(1)
    with torch.no_grad():classes=classifier(aligned.reshape(-1,1,28,28)).argmax(1).reshape(-1,2)
    return ((aligned-target).square().mean((1,2,3))).numpy(),(classes==labels).float().mean(1).numpy(),aligned


def evaluate(data,estimates,generator,basis,mean,scale,cfg,classifier,public_mean,sigma,out):
    n,t,_=data['z'].shape;predictions={};rows=[]
    with torch.no_grad():
        for method,z in estimates.items():
            im=torch.cat([generator(z.flatten(0,1)[a:a+64],data['context'].flatten(0,1)[a:a+64]) for a in range(0,n*t,64)])
            predictions[method]=im
    predictions['mean_image']=public_mean[None,None].expand(n*t,2,-1,-1).clone()
    # Shuffling estimates across complete held-out sequences tests target dependence.
    # This ablation uses no extra truth and the same legitimate checkpoint context.
    shifted=estimates['diffusion_mean'].roll(1,0)
    with torch.no_grad():predictions['diffusion_shuffled_sequence']=generator(shifted.flatten(0,1),data['context'].flatten(0,1))
    target=data['images'].flatten(0,1);labels=torch.tensor(np.stack([r['labels'] for r in data['runs']])).flatten(0,1)
    galleries={}
    for method,pred in predictions.items():
        mse,acc,aligned=image_score(pred,target,labels,classifier);galleries[method]=aligned.numpy().reshape(n,t,2,28,28)
        if method in estimates:
            # Full-vector recovery, including all discarded PCA components.
            reconstructed=np.stack([r['context'] for r in data['runs']])+(estimates[method].numpy()*scale+mean)@basis.T
            gradient_error=np.sum((reconstructed-np.stack([r['clean'] for r in data['runs']]))**2,axis=-1)/cfg['clip']**2
        else:gradient_error=np.full((n,t),np.nan)
        for i in range(n):
            for j in range(t):rows.append(dict(sigma=sigma,run=i,round=j,method=method,pixel_mse=float(mse[i*t+j]),psnr_db=float(-10*np.log10(max(mse[i*t+j],1e-12))),digit_accuracy=float(acc[i*t+j]),gradient_error=float(gradient_error[i,j])))
    # Synthetic recovery images only; gallery includes held-out truth for laboratory inspection.
    np.savez_compressed(out/'reconstructed_batches.npz',truth=data['images'].numpy(),**galleries)
    return rows,galleries


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/sequence_images.json');parser.add_argument('--data',required=True);parser.add_argument('--output',default='runs/sequence_images');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'completion.json').exists():raise RuntimeError('Completed run: choose another output folder')
    assert cfg['batch_size']==2 and cfg['diffusion_train_timesteps']==100
    torch.set_num_threads(1);started=time.time();x,labels=load_mnist(args.data);rng=np.random.default_rng(cfg['seed']);perm=rng.permutation(len(x));cursor=0;splits={}
    for role,size in [('pretrain',3000),('bank',32),('train',6000),('cal',1500),('target',1500),('test',1000)]:splits[role]=perm[cursor:cursor+size];cursor+=size
    (out/'config.json').write_text(json.dumps(cfg,indent=2));(out/'splits.json').write_text(json.dumps({k:v.tolist() for k,v in splits.items()}))
    initial=initialize(cfg['seed']);opt=torch.optim.SGD(initial.parameters(),lr=.1)
    for _ in range(cfg['pretrain_steps']):
        ix=rng.choice(splits['pretrain'],64,False);loss=nn.functional.cross_entropy(initial(x[ix]),labels[ix]);opt.zero_grad();loss.backward();opt.step()
    initial.eval();torch.save(initial.state_dict(),out/'public_initial_cnn.pt')
    with torch.no_grad():accuracy=float((initial(x[splits['test']]).argmax(1)==labels[splits['test']]).float().mean())
    raw={};public=[];privacy=[]
    for sigma in cfg['sigmas']:
        folder=out/f'sigma{sigma:g}';folder.mkdir(exist_ok=True);raw[sigma]={}
        privacy.append(dict(sigma=sigma,clip=cfg['clip'],batch_size=2,steps=cfg['steps'],delta=cfg['delta'],epsilon_upper=conservative_epsilon(cfg['steps'],sigma,cfg['delta']),accountant='replace-one zCDP; no sampling amplification; per run'))
        for role,count in [('train',cfg['public_train_runs']),('cal',cfg['public_calibration_runs']),('target',cfg['target_runs'])]:
            raw[sigma][role]=[]
            for i in range(count):
                path=folder/f'{role}{i}.npz'
                if path.exists():r=dict(np.load(path))
                else:
                    r=generate_run(initial,x,labels,splits[role],splits['bank'],cfg,sigma,cfg['seed']+10000*['train','cal','target'].index(role)+i)
                    np.savez_compressed(path,**r)
                raw[sigma][role].append(r)
                if (i+1)%8==0:print(f'sigma {sigma}: generated {role} {i+1}/{count}',flush=True)
            if role=='train':public.extend(raw[sigma][role])
    residual=torch.tensor(np.concatenate([r['clean']-r['context'] for r in public]));torch.manual_seed(cfg['seed'])
    _,_,v=torch.pca_lowrank(residual,q=cfg['rank']+8,niter=3);basis=v[:,:cfg['rank']].numpy()
    z=residual.numpy()@basis;mean=z.mean(0);scale=float(z.std());context=np.concatenate([r['context'] for r in public])@basis;cm=context.mean(0);cs=float(context.std())
    np.savez_compressed(out/'public_space.npz',basis=basis,mean=mean,scale=scale,context_mean=cm,context_scale=cs)
    data={s:{role:make_data(rr,basis,mean,scale,cm,cs) for role,rr in group.items()} for s,group in raw.items()}
    decoder=fit_generator([data[s]['train'] for s in cfg['sigmas']],[data[s]['cal'] for s in cfg['sigmas']],cfg,out)
    rows=[]
    for sigma in cfg['sigmas']:
        print(f'Reconstructing complete sequences, sigma={sigma}',flush=True);folder=out/f'sigma{sigma:g}';d=data[sigma]
        fit=gaussian_model(d['train'],d['cal'],sigma*cfg['clip']/2/scale);np.savez_compressed(folder/'gaussian.npz',**fit)
        lstm=fit_lstm(d['train'],d['cal'],cfg,cfg['seed'],folder)
        diffusion,schedule=fit_diffusion(d['train'],d['cal'],cfg,cfg['seed']+1,folder)
        estimates=gaussian_predict(fit,d['target'])
        with torch.no_grad():estimates['bilstm']=lstm(d['target']['y'],d['target']['context'])
        samples=diffusion_sample(diffusion,schedule,d['target'],cfg,cfg['seed']+200)
        estimates.update(diffusion_mean=samples.mean(0),diffusion_draw=samples[0],noisy=d['target']['y'],clean_oracle=d['target']['z'])
        np.savez_compressed(folder/'estimated_latents.npz',**{k:v.numpy() for k,v in estimates.items()})
        rr,_=evaluate(d['target'],estimates,decoder,basis,mean,scale,cfg,initial,x[splits['pretrain'],0].mean(0),sigma,folder);rows.extend(rr)
        write_csv(out/'measurements.csv',rows)
    write_csv(out/'privacy.csv',privacy)
    info=dict(seconds=time.time()-started,dataset_sha256=digest(args.data),public_classifier_accuracy=accuracy,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch=torch.__version__)
    (out/'completion.json').write_text(json.dumps(info,indent=2));print(info,flush=True)

if __name__=='__main__':main()
