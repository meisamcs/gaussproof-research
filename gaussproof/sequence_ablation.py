"""Matched access ablations: identical BiLSTM, current public context, fixed decoder."""
import argparse,copy,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .sequence_images import BatchGenerator,batch_mse,make_data
from .trajectory_dp import write_csv

MODES=('current','past','full','shuffled')


def probe_inputs(observations,contexts,sequence,rounds,mode,generator=None):
    """Observations: raw projected noisy gradients in public normalized units.

    Contexts are the CURRENT checkpoint projected fingerprints in the same units.
    Other checkpoint fingerprints are never used by the model.
    """
    b=len(sequence);length=observations.shape[1]
    tokens=observations[sequence].clone();center=contexts[sequence,rounds]
    positions=torch.arange(length)[None].expand(b,-1)
    if mode=='shuffled':
        # Each row has a permutation that fixes its queried current round.
        for row,t in enumerate(rounds.tolist()):
            other=torch.cat([torch.arange(t),torch.arange(t+1,length)])
            order=other[torch.randperm(length-1,generator=generator)]
            tokens[row,other]=tokens[row,order]
    available=(positions==rounds[:,None]) if mode=='current' else (positions<=rounds[:,None]) if mode=='past' else torch.ones(b,length,dtype=torch.bool)
    relative=(positions-rounds[:,None]).float()/length
    # The same current checkpoint fingerprint is provided in every access setting.
    residual=(tokens-center[:,None])*available[:,:,None]
    return torch.cat([residual,center[:,None].expand(-1,length,-1),available[:,:,None].float(),relative[:,:,None]],-1)


class MatchedReconstructor(nn.Module):
    def __init__(self,rank):
        super().__init__();self.embed=nn.Linear(2*rank+2,128)
        self.lstm=nn.LSTM(128,96,batch_first=True,bidirectional=True)
        self.head=nn.Linear(192,rank)
    def forward(self,features,rounds):
        encoded=self.lstm(self.embed(features).tanh())[0]
        return self.head(encoded[torch.arange(len(rounds)),rounds])


def prepare(runs,space):
    basis=space['basis'];scale=float(space['scale']);mean=space['mean']
    # mean subtraction belongs to observation only, so observation-context equals
    # the original public residual normalization at the queried current round.
    obs=(np.stack([r['noisy'] for r in runs])@basis-mean)/scale
    context=(np.stack([r['context'] for r in runs])@basis)/scale
    z=(np.stack([r['clean']-r['context'] for r in runs])@basis-mean)/scale
    generator_context=(np.stack([r['context'] for r in runs])@basis-space['context_mean'])/space['context_scale']
    return dict(observations=torch.tensor(obs),contexts=torch.tensor(context),z=torch.tensor(z),generator_context=torch.tensor(generator_context),images=torch.tensor(np.stack([r['images'] for r in runs])),runs=runs)


@torch.no_grad()
def predict(net,data,mode,seed):
    net.eval();n,length,_=data['z'].shape
    sequence=torch.arange(n).repeat_interleave(length);rounds=torch.arange(length).repeat(n)
    rng=torch.Generator().manual_seed(seed);results=[]
    for start in range(0,len(sequence),64):
        s=sequence[start:start+64];t=rounds[start:start+64]
        f=probe_inputs(data['observations'],data['contexts'],s,t,mode,rng)
        results.append(net(f,t))
    return torch.cat(results).reshape(n,length,-1)


def train(train,cal,mode,cfg,seed,folder):
    torch.manual_seed(seed);net=MatchedReconstructor(cfg['rank'])
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    # Separate streams ensure the exact same probes across all access variants.
    probe_rng=torch.Generator().manual_seed(seed+1);shuffle_rng=torch.Generator().manual_seed(seed+2)
    best=float('inf');state=None;history=[];n,length,_=train['z'].shape
    for step in range(cfg['training_steps']):
        net.train();seq=torch.randint(n,(cfg['train_batch'],),generator=probe_rng)
        t=torch.randint(length,(cfg['train_batch'],),generator=probe_rng)
        f=probe_inputs(train['observations'],train['contexts'],seq,t,mode,shuffle_rng)
        loss=(net(f,t)-train['z'][seq,t]).square().mean()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
        if (step+1)%200==0:
            error=float((predict(net,cal,mode,999)-cal['z']).square().mean())
            if error<best:best=error;state=copy.deepcopy(net.state_dict());beststep=step+1
            history.append(dict(step=step+1,training_mse=float(loss.detach()),calibration_mse=error))
    net.load_state_dict(state);net.eval();torch.save(net.state_dict(),folder/f'{mode}_seed{seed}.pt')
    write_csv(folder/f'{mode}_seed{seed}_training.csv',history)
    print(f'  {mode} seed={seed}: selected step {beststep}, calibration={best:.5f}',flush=True)
    return net,beststep


def score(est,data,decoder,space,clip):
    n,length,_=est.shape
    with torch.no_grad():
        flat=est.flatten(0,1);context=data['generator_context'].flatten(0,1)
        images=torch.cat([decoder(flat[i:i+64],context[i:i+64]) for i in range(0,len(flat),64)])
        errors=batch_mse(images,data['images'].flatten(0,1),False).reshape(n,length).numpy()
    full=np.stack([r['context'] for r in data['runs']])+(est.numpy()*space['scale']+space['mean'])@space['basis'].T
    grad=((full-np.stack([r['clean'] for r in data['runs']]))**2).sum(-1)/clip**2
    return errors,grad,images.reshape(n,length,2,28,28).numpy()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='runs/sequence_images');p.add_argument('--output',default='runs/sequence_ablation');p.add_argument('--config',default='configs/sequence_ablation.json');a=p.parse_args()
    torch.set_num_threads(1);source=Path(a.source);out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    if (out/'completion.json').exists():raise RuntimeError('Choose a fresh output directory')
    assert (source/'completion.json').exists();cfg=json.loads(Path(a.config).read_text());original=json.loads((source/'config.json').read_text())
    cfg.update(rank=original['rank'],sigmas=original['sigmas'],steps=original['steps'],clip=original['clip'])
    (out/'config.json').write_text(json.dumps(cfg,indent=2));space=dict(np.load(source/'public_space.npz'))
    decoder=BatchGenerator(cfg['rank']);decoder.load_state_dict(torch.load(source/'batch_generator.pt',weights_only=True));decoder.eval()
    rows=[];selection=[];start=time.time()
    for sigma in cfg['sigmas']:
        folder=out/f'sigma{sigma:g}';folder.mkdir(exist_ok=True);data={}
        for role,count in [('train',original['public_train_runs']),('cal',original['public_calibration_runs']),('target',original['target_runs'])]:
            runs=[dict(np.load(source/f'sigma{sigma:g}'/f'{role}{i}.npz')) for i in range(count)]
            data[role]=prepare(runs,space)
        for seed in cfg['seeds']:
            for mode in MODES:
                print(f'sigma={sigma}, fitting {mode}, seed={seed}',flush=True)
                net,selected=train(data['train'],data['cal'],mode,cfg,seed,folder)
                selection.append(dict(sigma=sigma,seed=seed,mode=mode,selected_step=selected,parameters=sum(p.numel() for p in net.parameters())))
                variants=[(mode,mode)]
                if mode=='full':variants.append(('full_test_shuffled','shuffled'))
                for label,inputmode in variants:
                    # Multiple permutations reduce chance effects in shuffle comparisons.
                    draws=cfg['shuffle_repeats'] if inputmode=='shuffled' else 1
                    pixel=[];gradient=[]
                    for repeat in range(draws):
                        est=predict(net,data['target'],inputmode,7000+repeat)
                        pe,ge,images=score(est,data['target'],decoder,space,cfg['clip']);pixel.append(pe);gradient.append(ge)
                        if repeat==0:np.savez_compressed(folder/f'{label}_seed{seed}_predictions.npz',images=images,latents=est.numpy())
                    pe=np.mean(pixel,axis=0);ge=np.mean(gradient,axis=0)
                    for i in range(pe.shape[0]):
                        for t in range(pe.shape[1]):rows.append(dict(sigma=sigma,seed=seed,run=i,round=t,mode=label,pixel_mse=float(pe[i,t]),gradient_error=float(ge[i,t])))
                write_csv(out/'measurements.csv',rows)
    write_csv(out/'model_selection.csv',selection)
    info=dict(seconds=time.time()-start,source=str(source.resolve()),source_completion=json.loads((source/'completion.json').read_text()),generator_sha256=hashlib.sha256((source/'batch_generator.pt').read_bytes()).hexdigest(),representation_sha256=hashlib.sha256((source/'public_space.npz').read_bytes()).hexdigest(),implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'completion.json').write_text(json.dumps(info,indent=2));print(info,flush=True)

if __name__=='__main__':main()
