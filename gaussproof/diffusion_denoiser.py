"""Gradient adapter around upstream Diffusers U-Net/DDPM/DDIM implementations.

Diffusers is optional: install requirements-diffusion.txt to use this experiment.
The 8x8 layout is an arbitrary arrangement of 64 public PCA coefficients, not pixels.
"""
import copy
import numpy as np
import torch
from diffusers import UNet2DModel,DDPMScheduler,DDIMScheduler


def scheduler(prediction_type='epsilon',steps=200):
    # VE-equivalent noise ratios mapped to DDPM's VP parameterization.
    ratios=np.geomspace(.001,1000,steps)
    abar=1/(1+ratios**2)
    betas=1-abar/np.r_[1.,abar[:-1]]
    return DDPMScheduler(num_train_timesteps=steps,trained_betas=betas,
                         prediction_type=prediction_type,clip_sample=False)


def make_unet(seed=0):
    torch.manual_seed(seed)
    return UNet2DModel(sample_size=8,in_channels=19,out_channels=2,layers_per_block=1,
        block_out_channels=(16,32),down_block_types=('DownBlock2D','DownBlock2D'),
        up_block_types=('UpBlock2D','UpBlock2D'),norm_num_groups=8,add_attention=False)


def bank_estimate(observation,bank,variance,mask):
    """Public empirical posterior feature; targets are never inserted in the bank."""
    flat=observation.flatten(1);b=bank.flatten(2);weights=mask.repeat_interleave(64,dim=1)
    distance=((flat[:,None]-b).square()*weights[:,None]).sum(2)
    prob=torch.softmax(-distance/(2*variance[:,None].clamp_min(1e-12)),1)
    return (prob[:,:,None]*b).sum(1).reshape(-1,2,8,8)


def features(xt,mask,histogram,clip,batch_size,bank_context,use_bank):
    n=len(xt)
    # 2 noisy channels + 2 masks + 10 label fractions + C + B + 2 bank channels + flag.
    def expand(v):return v[:,:,None,None].expand(-1,-1,8,8)
    return torch.cat([xt*mask[:,:,None,None],expand(mask),expand(histogram),
        expand(torch.log10(clip)[:,None]),expand(torch.log2(batch_size)[:,None]/3),
        bank_context*use_bank[:,None,None,None],expand(use_bank[:,None])],1)


def reconstruct_x0(schedule,model_output,timestep,xt):
    # Upstream implementation supplies epsilon/sample conversion, with gradient-safe clipping OFF.
    return schedule.step(model_output,int(timestep),xt,generator=torch.Generator().manual_seed(0)).pred_original_sample


class DiffusionDenoiser:
    def __init__(self,cfg,prediction_type='epsilon',seed=0):
        self.cfg=cfg;self.prediction_type=prediction_type
        self.model=make_unet(seed)
        self.schedule=scheduler(prediction_type,cfg['diffusion_steps'])
        self.ddim=DDIMScheduler.from_config(self.schedule.config)
        self.ddim.set_timesteps(cfg['inference_steps'])

    def timestep(self,noise_ratio):
        valid=self.ddim.timesteps
        ratios=((1-self.schedule.alphas_cumprod[valid])/self.schedule.alphas_cumprod[valid]).sqrt()
        index=torch.argmin((ratios.log()-np.log(noise_ratio)).abs())
        t=int(valid[index]);return t,float(ratios[index])

    def train(self,sampler,calibration,steps=None,seed=0,log=print):
        steps=steps or self.cfg['training_steps'];rng=torch.Generator().manual_seed(seed)
        opt=torch.optim.AdamW(self.model.parameters(),lr=self.cfg['learning_rate'],weight_decay=1e-4)
        best=float('inf');state=None;history=[]
        for step in range(steps):
            self.model.train()
            clean,hist,clips,batches,bank=sampler(self.cfg['train_batch'])
            t=torch.randint(self.cfg['diffusion_steps'],(len(clean),),generator=rng)
            noise=torch.randn(clean.shape,generator=rng)
            xt=self.schedule.add_noise(clean,noise,t)
            abar=self.schedule.alphas_cumprod[t]
            mask=torch.ones(len(clean),2)
            drop=torch.rand(len(clean),generator=rng)<.5
            channel=torch.randint(2,(len(clean),),generator=rng);mask[drop,channel[drop]]=0
            use_bank=(torch.rand(len(clean),generator=rng)<.5).float()
            context=bank_estimate(xt/abar.sqrt()[:,None,None,None],bank,(1-abar)/abar,mask)
            pred=self.model(features(xt,mask,hist,clips,batches,context,use_bank),t).sample
            target=(noise if self.prediction_type=='epsilon' else clean if self.prediction_type=='sample'
                    else self.schedule.get_velocity(clean,noise,t))
            loss=((pred-target).square()*mask[:,:,None,None]).sum()/(mask.sum()*64)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.);opt.step()
            if (step+1)%200==0 or step+1==steps:
                error=self.calibrate(calibration)
                history.append(dict(step=step+1,training_loss=float(loss.detach()),calibration_x0_mse=error))
                if error<best:best=error;state=copy.deepcopy(self.model.state_dict())
                log(f'  {self.prediction_type}: step {step+1}/{steps}, calibration x0 error {error:.5f}',flush=True)
        self.model.load_state_dict(state);self.model.eval()
        return history

    @torch.no_grad()
    def calibrate(self,calibration):
        self.model.eval();errors=[]
        for xt,t,clean,hist,clips,batches,bank in calibration:
            mask=torch.ones(len(xt),2);a=self.schedule.alphas_cumprod[t]
            context=bank_estimate(xt/a.sqrt(),bank,torch.full((len(xt),),float((1-a)/a)),mask)
            pred=self.model(features(xt,mask,hist,clips,batches,context,torch.ones(len(xt))),t).sample
            x0=reconstruct_x0(self.schedule,pred,t,xt)
            errors.append(float((x0-clean).square().mean()))
        return float(np.mean(errors))

    @torch.no_grad()
    def predict(self,xt,t,hist,clips,batches,bank,mode='trajectory_bank',iterative=False):
        self.model.eval();n=len(xt);a=self.schedule.alphas_cumprod[t]
        masks=[torch.ones(n,2)] if mode!='single' else [torch.tensor([[1.,0.]]).expand(n,-1),torch.tensor([[0.,1.]]).expand(n,-1)]
        use_bank=torch.full((n,),float(mode=='trajectory_bank'))
        combined=torch.zeros_like(xt)
        for mask in masks:
            context=bank_estimate(xt/a.sqrt(),bank,torch.full((n,),float((1-a)/a)),mask)
            current=xt.clone()
            times=[int(v) for v in self.ddim.timesteps if int(v)<=t] if iterative else [t]
            for time in times:
                pred=self.model(features(current,mask,hist,clips,batches,context,use_bank),time).sample
                if iterative:
                    result=self.ddim.step(pred,time,current,eta=0.);current=result.prev_sample
                    x0=result.pred_original_sample
                else:x0=reconstruct_x0(self.schedule,pred,time,current)
            combined+=x0*mask[:,:,None,None]
        return combined

    def save(self,folder):
        self.model.save_pretrained(folder/'unet',safe_serialization=True)
        self.schedule.save_pretrained(folder/'scheduler')
