"""Public-data gradient priors. No target gradients enter fitting methods."""
import copy
import numpy as np
import torch
from torch import nn


class GradientPrior:
    def fit(self, gradients, labels, rank=32, seed=0):
        g=torch.as_tensor(gradients,dtype=torch.float32)
        labels=torch.as_tensor(labels,dtype=torch.long)
        if g.ndim!=2 or len(g)!=len(labels) or not torch.isfinite(g).all():
            raise ValueError('Expected finite public gradients and matching labels')
        if set(labels.tolist())!=set(range(10)):
            raise ValueError('Prior-fitting data must cover all ten known classes')
        self.means=torch.stack([g[labels==k].mean(0) for k in range(10)])
        residual=g-self.means[labels]
        rank=min(rank,len(g)-10,g.shape[1])
        if rank<1:raise ValueError('Insufficient public samples for a prior')
        torch.manual_seed(seed)
        _,singular,basis=torch.pca_lowrank(residual,q=min(rank+8,len(g)-1,g.shape[1]),center=False,niter=3)
        self.basis=basis[:,:rank].contiguous()
        self.variance=(singular[:rank]**2/max(1,len(g)-10)).clamp_min(1e-10)
        coordinates=residual@self.basis
        discarded=(residual.square().sum()-coordinates.square().sum()).clamp_min(0)
        self.floor=float(discarded/max(1,(len(g)-10)*(g.shape[1]-rank)))
        global_residual=g-g.mean(0)
        bgz=global_residual@self.basis
        self.background_variance=bgz.var(0,unbiased=True)
        self.background_floor=float((global_residual.square().sum()-bgz.square().sum()).clamp_min(0)/
                                    max(1,(len(g)-1)*(g.shape[1]-rank)))
        self.public_mean=g.mean(0)
        self.network=None
        return self

    def noise(self, variance, background_factor=0):
        v=torch.as_tensor(variance,dtype=torch.float32)
        f=torch.as_tensor(background_factor,dtype=torch.float32)
        return v[...,None]+f[...,None]*self.background_variance, v+f*self.background_floor

    def estimate(self, observation, labels, variance, background_factor=0, method='gaussian'):
        y=torch.as_tensor(observation,dtype=torch.float32)
        labels=torch.as_tensor(labels,dtype=torch.long)
        mean=self.means[labels]
        if method=='prior_only':return mean.numpy().copy()
        residual=y-mean
        z=residual@self.basis
        nv,outside_nv=self.noise(variance,background_factor)
        if method=='projection':return (mean+z@self.basis.T).numpy()
        if method=='gaussian':
            fitted=z*self.variance/(self.variance+nv)
        elif method=='neural':
            if self.network is None:raise ValueError('Train the denoiser first')
            with torch.no_grad():fitted=self.network(self.encode(z,labels,nv))*self.variance.sqrt()
        else:raise ValueError(f'Unknown estimator {method}')
        outside=residual-z@self.basis.T
        shrink=self.floor/(self.floor+outside_nv+1e-20)
        return (mean+fitted@self.basis.T+outside*shrink[...,None]).numpy()

    def encode(self,z,labels,nv):
        nv=torch.broadcast_to(nv,z.shape)
        return torch.cat([z/(self.variance+nv).sqrt(),
                          torch.log1p(nv/self.variance).clamp(max=25)/10,
                          nn.functional.one_hot(labels,10).float()],1)

    def train_denoiser(self, public, public_labels, calibration, calibration_labels,
                       steps=1200, seed=0, clip=1., max_repeats=64, batch_size=8):
        torch.manual_seed(seed)
        rng=torch.Generator().manual_seed(seed+1)
        public=torch.as_tensor(public,dtype=torch.float32)
        public_labels=torch.as_tensor(public_labels,dtype=torch.long)
        calibration=torch.as_tensor(calibration,dtype=torch.float32)
        calibration_labels=torch.as_tensor(calibration_labels,dtype=torch.long)
        z=(public-self.means[public_labels])@self.basis
        cz=(calibration-self.means[calibration_labels])@self.basis
        rank=self.basis.shape[1]
        self.network=nn.Sequential(nn.Linear(2*rank+10,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU(),nn.Linear(96,rank))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        opt=torch.optim.AdamW(self.network.parameters(),lr=.002,weight_decay=.0001)
        # Calibration noise is generated independently once and reused for model selection.
        cnv,_=self.noise(torch.logspace(-3,1,len(cz))*clip**2,0)
        cy=cz+torch.randn(cz.shape,generator=rng)*cnv.sqrt()
        best=float('inf');best_state=None;history=[]
        for step in range(steps):
            ids=torch.randint(len(z),(128,),generator=rng)
            clean=z[ids];labels=public_labels[ids]
            sigma=torch.exp(torch.empty(128).uniform_(np.log(.125),np.log(8),generator=rng))*clip
            repeat=torch.exp(torch.empty(128).uniform_(0,np.log(max_repeats),generator=rng))
            background=(torch.rand(128,generator=rng)>.5).float()*(batch_size-1)/repeat
            nv,_=self.noise(sigma.square()/repeat,background)
            noisy=clean+torch.randn(clean.shape,generator=rng)*nv.sqrt()
            estimate=self.network(self.encode(noisy,labels,nv))*self.variance.sqrt()
            loss=(estimate-clean).square().mean()/self.variance.mean()
            opt.zero_grad();loss.backward();opt.step()
            if (step+1)%100==0 or step+1==steps:
                with torch.no_grad():
                    pred=self.network(self.encode(cy,calibration_labels,cnv))*self.variance.sqrt()
                    error=float((pred-cz).square().sum(1).mean())
                history.append(dict(step=step+1,training_loss=float(loss),calibration_error=error))
                if error<best:best=error;best_state=copy.deepcopy(self.network.state_dict())
        self.network.load_state_dict(best_state);self.network.eval()
        return history

    def save(self,path):
        torch.save(dict(means=self.means,basis=self.basis,variance=self.variance,floor=self.floor,
                        background_variance=self.background_variance,background_floor=self.background_floor,
                        public_mean=self.public_mean,network=None if self.network is None else self.network.state_dict()),path)


def exact_bank(observation, bank, labels, bank_labels, variance):
    """Uniform within-class prior. Correct norm term; target image is in this bank.

    Exact for isotropic Gaussian observations of a fixed concatenated signal.
    Returns identity indices and posterior mean, never used by learned-prior fitting.
    """
    y=np.asarray(observation,dtype=np.float64);b=np.asarray(bank,dtype=np.float64)
    labels=np.asarray(labels);bl=np.asarray(bank_labels)
    if variance<=0:raise ValueError('Positive noise variance required')
    alignment=y@b.T
    loglike=(alignment-.5*np.sum(b*b,axis=1)[None,:])/variance
    valid=labels[:,None]==bl[None,:]
    if not valid.any(1).all():raise ValueError('No candidate matches a known target label')
    loglike=np.where(valid,loglike,-np.inf)
    alignment=np.where(valid,alignment,-np.inf)
    weights=np.exp(loglike-loglike.max(1,keepdims=True));weights/=weights.sum(1,keepdims=True)
    return dict(alignment=alignment.argmax(1),likelihood=loglike.argmax(1),posterior=weights@b)
