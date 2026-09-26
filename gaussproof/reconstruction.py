"""Controlled Gaussian gradient-release reconstruction experiments (not DP-SGD)."""
from pathlib import Path
import copy
import hashlib
import itertools
import json
import platform
import time
import numpy as np
import torch
from .data import load_mnist,digest
from .models import initialize,per_record_gradients,clip_gradients,predict,conservative_epsilon
from .reconstruction_prior import GradientPrior,exact_bank
from .inversion import invert,image_metrics
from .io import write_json,write_csv


def split_roles(n,cfg,seed,labels=None):
    counts={k:cfg[k+'_size'] for k in ['model_train','prior','calibration','target','distractor']}
    if any(type(v)!=int or v<=0 for v in counts.values()) or sum(counts.values())>n:
        raise ValueError('Invalid reconstruction split sizes')
    perm=np.random.default_rng(seed).permutation(n);result={};cursor=0
    for k,count in counts.items():
        if k=='target' and labels is not None and count>=20 and count%10==0:
            remaining=perm[cursor:];lab=np.asarray(labels)
            chosen=np.concatenate([remaining[lab[remaining]==c][:count//10] for c in range(10)])
            if len(chosen)!=count:raise ValueError('Insufficient samples for balanced targets')
            # Interleave classes so the inversion subset is not all one digit.
            chosen=chosen.reshape(10,-1).T.ravel()
            perm=np.r_[perm[:cursor],chosen,remaining[~np.isin(remaining,chosen)]]
        result[k]=perm[cursor:cursor+count];cursor+=count
    return result


def public_checkpoints(x,y,ids,steps,seed,batch_size=64,lr=.15):
    model=initialize(seed)
    opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9)
    rng=np.random.default_rng(seed+1);models=[]
    for step in range(1,max(steps)+1):
        batch=rng.choice(ids,batch_size,replace=False)
        loss=torch.nn.functional.cross_entropy(model(x[batch]),y[batch])
        opt.zero_grad();loss.backward();opt.step()
        if step in steps:models.append(copy.deepcopy(model).eval())
    return models


def gradient_sequences(models,x,y,ids,clip):
    return np.concatenate([clip_gradients(per_record_gradients(model,x[ids],y[ids]),clip).numpy() for model in models],axis=1)


def gradient_metrics(estimate,truth,checkpoints):
    error=np.sum((estimate-truth)**2,axis=1)/checkpoints
    energy=np.sum(truth**2,axis=1)/checkpoints
    cosine=np.sum(estimate*truth,axis=1)/(np.linalg.norm(estimate,axis=1)*np.linalg.norm(truth,axis=1)+1e-12)
    return dict(squared_l2=error,normalized_squared_l2=error/np.maximum(energy,1e-12),cosine=cosine)


def run_reconstruction(cfg,data_path,output,log=print):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Run directory must be empty')
    for key in ['sigmas','repeats','seeds','checkpoint_steps']:
        if not cfg[key] or len(set(cfg[key]))!=len(cfg[key]):raise ValueError(f'Invalid {key}')
    if min(cfg['sigmas'])<=0 or min(cfg['repeats'])<1 or cfg['clip']<=0:raise ValueError('Invalid noise/release counts')
    if cfg['inversion_sigma'] not in cfg['sigmas'] or cfg['inversion_repeats'] not in cfg['repeats']:
        raise ValueError('Inversion condition must be part of sweep')
    if cfg['checkpoint_steps']!=sorted(cfg['checkpoint_steps']):raise ValueError('Checkpoint steps must increase')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(cfg['threads']);torch.use_deterministic_algorithms(True)
    start=time.time();x,y=load_mnist(data_path)
    source=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):source.update(p.name.encode());source.update(p.read_bytes())
    write_json(output/'config.json',cfg)
    write_json(output/'provenance.json',dict(dataset_sha256=digest(data_path),source_sha256=source.hexdigest(),
        python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
        mechanism='Gaussian clipped-gradient releases at fixed public checkpoints; NOT private training',
        target_label='known to all methods',target_image='withheld from learned estimators and inversion',
        adjacency='replace-one target record; fixed public background',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    gradient_rows=[];identity_rows=[];image_rows=[];utility=[];gallery=[]
    k=len(cfg['checkpoint_steps']);maxr=max(cfg['repeats']);clip=cfg['clip']
    for seed in cfg['seeds']:
        folder=output/f'seed{seed}';folder.mkdir()
        roles=split_roles(len(y),cfg,seed,y.numpy())
        write_json(folder/'splits.json',{a:b.tolist() for a,b in roles.items()})
        log(f'[reconstruction seed {seed}] train public CNN and compute exact gradients',flush=True)
        models=public_checkpoints(x,y,roles['model_train'],cfg['checkpoint_steps'],seed,cfg['public_batch_size'],cfg['public_lr'])
        for j,model in enumerate(models):
            torch.save(model.state_dict(),folder/f'public_checkpoint_{j}.pt')
            accuracy=float((predict(model,x,roles['calibration']).argmax(1)==y[roles['calibration']].numpy()).mean())
            utility.append(dict(seed=seed,checkpoint_step=cfg['checkpoint_steps'][j],public_model_accuracy=accuracy))
        vectors={role:gradient_sequences(models,x,y,ids,clip) for role,ids in roles.items() if role!='model_train'}
        labels={role:y[ids].numpy() for role,ids in roles.items()}
        log(f'[reconstruction seed {seed}] fit public covariance and neural denoiser',flush=True)
        prior=GradientPrior().fit(vectors['prior'],labels['prior'],cfg['rank'],seed+100)
        history=prior.train_denoiser(vectors['prior'],labels['prior'],vectors['calibration'],labels['calibration'],
            steps=cfg['denoiser_steps'],seed=seed+200,clip=clip,max_repeats=maxr,batch_size=cfg['batch_size'])
        write_csv(folder/'denoiser_training.csv',history);prior.save(folder/'learned_prior.pt')
        d=vectors['target'].shape[1];block=d//k
        stale_public=np.tile(vectors['prior'][:,:block],(1,k))
        stale=GradientPrior().fit(stale_public,labels['prior'],cfg['rank'],seed+100)
        stale.save(folder/'stale_prior.pt')
        truth=vectors['target'];target_labels=labels['target'];nt=len(truth)
        bank=np.concatenate([truth,vectors['distractor']]);banklabels=np.r_[target_labels,labels['distractor']]
        rng=np.random.default_rng(seed+300)
        # Couple noise across sigma and release counts. Mean sufficient for this fixed-signal channel.
        sums=np.zeros_like(truth);noise_means={}
        for r in range(1,maxr+1):
            sums+=rng.standard_normal(truth.shape).astype(np.float32)
            if r in cfg['repeats']:noise_means[r]=sums.copy()/r
        # Unknown background: exact with-replacement sampling from public records,
        # aggregated efficiently by counts; actual selected IDs are hidden from estimators.
        background_draws=rng.integers(len(vectors['prior']),size=(nt,maxr*(cfg['batch_size']-1)))
        background_means={}
        for r in cfg['repeats']:
            counts=np.stack([np.bincount(ids[:r*(cfg['batch_size']-1)],minlength=len(vectors['prior'])) for ids in background_draws]).astype(np.float32)/r
            background_means[r]=counts@vectors['prior']-(cfg['batch_size']-1)*vectors['prior'].mean(0)
        # Shift observations between same-label targets; singleton labels receive another
        # target observation (label mismatch is explicitly saved for interpretation).
        permutation=np.arange(nt)
        for label in np.unique(target_labels):
            group=np.flatnonzero(target_labels==label)
            permutation[group]=np.roll(group,1) if len(group)>1 else (group+1)%nt
        np.savez_compressed(folder/'ground_truth.npz',gradients=truth,labels=target_labels,ids=roles['target'])
        inversion_estimates=None
        for sigma,r,background in itertools.product(cfg['sigmas'],cfg['repeats'],['known','unknown']):
            variance=(sigma*clip)**2/r
            factor=(cfg['batch_size']-1)/r if background=='unknown' else 0.
            observation=truth+sigma*clip*noise_means[r]
            if background=='unknown':observation=observation+background_means[r]
            estimates={'mean':observation,'prior_only':prior.estimate(observation,target_labels,variance,method='prior_only')}
            for method in ['projection','gaussian','neural']:
                estimates[method]=prior.estimate(observation,target_labels,variance,factor,method)
            estimates['stale_gaussian']=stale.estimate(observation,target_labels,variance,factor)
            estimates['permuted_release']=prior.estimate(observation[permutation],target_labels,variance,factor)
            if background=='known':
                discrete=exact_bank(observation,bank,target_labels,banklabels,variance)
                estimates['exact_bank_posterior']=discrete['posterior']
                for method in ['alignment','likelihood']:
                    for i in range(nt):
                        eligible=int(np.sum(banklabels==target_labels[i]))
                        identity_rows.append(dict(seed=seed,sigma=sigma,repeats=r,method=method,target_index=i,
                            success=int(discrete[method][i]==i),candidate_count=eligible,chance=1/eligible))
            prior_errors=gradient_metrics(estimates['prior_only'],truth,k)['squared_l2']
            for method,estimate in estimates.items():
                values=gradient_metrics(estimate,truth,k)
                for i in range(nt):
                    gradient_rows.append(dict(seed=seed,sigma=sigma,repeats=r,background=background,method=method,
                        target_index=i,label=int(target_labels[i]),access='target_containing_bank' if method=='exact_bank_posterior' else 'public_prior',
                        squared_l2=float(values['squared_l2'][i]),normalized_squared_l2=float(values['normalized_squared_l2'][i]),
                        cosine=float(values['cosine'][i]),gain_over_prior=float(prior_errors[i]-values['squared_l2'][i]),
                        permutation_same_label=int(target_labels[permutation[i]]==target_labels[i]),
                        conservative_epsilon=conservative_epsilon(k*r,sigma,cfg['delta'])))
            if background=='known' and sigma==cfg['inversion_sigma'] and r==cfg['inversion_repeats']:
                inversion_estimates={m:np.asarray(estimates[m],dtype=np.float32) for m in ['mean','prior_only','gaussian','neural']}
                inversion_estimates['clean_oracle']=truth
                np.savez_compressed(folder/'inversion_inputs.npz',**inversion_estimates)
        write_csv(output/'gradient_records.csv',gradient_rows);write_csv(output/'identity_records.csv',identity_rows)
        # Fixed first targets, never selected on reconstruction success; ground truth is
        # accessed only after invert() returns for metrics and the comparison gallery.
        for i in range(min(cfg['inversion_targets'],nt)):
            images={}
            for method,estimates in inversion_estimates.items():
                log(f'[reconstruction seed {seed}] invert target {i+1}/{cfg["inversion_targets"]}, {method}',flush=True)
                image,objective=invert(models,estimates[i],target_labels[i],clip,cfg['inversion_steps'],cfg['inversion_restarts'],seed+400+i*10)
                images[method]=image
                metrics=image_metrics(image,x[roles['target'][i],0].numpy())
                image_rows.append(dict(seed=seed,target_index=i,label=int(target_labels[i]),method=method,
                    sigma=cfg['inversion_sigma'],repeats=cfg['inversion_repeats'],gradient_objective=objective,**metrics))
            # Public class-mean image baseline does not receive the release or gradient.
            public_class=roles['prior'][labels['prior']==target_labels[i]]
            images['public_mean_image']=x[public_class,0].mean(0).numpy()
            image_rows.append(dict(seed=seed,target_index=i,label=int(target_labels[i]),method='public_mean_image',
                sigma=cfg['inversion_sigma'],repeats=cfg['inversion_repeats'],gradient_objective='',
                **image_metrics(images['public_mean_image'],x[roles['target'][i],0].numpy())))
            gallery.append(dict(seed=seed,target_index=i,label=int(target_labels[i]),original=x[roles['target'][i],0].numpy(),**images))
        write_csv(output/'image_records.csv',image_rows);write_csv(output/'utility.csv',utility)
        log(f'[reconstruction seed {seed}] finished',flush=True)
    np.savez_compressed(output/'gallery.npz',**{f'{row["seed"]}_{row["target_index"]}_{key}':value for row in gallery for key,value in row.items() if isinstance(value,np.ndarray)})
    from .reconstruction_report import finish_report
    finish_report(output,cfg,gradient_rows,identity_rows,image_rows,gallery)
    write_json(output/'completion.json',dict(status='complete',elapsed_seconds=time.time()-start,
               gradient_measurements=len(gradient_rows),identity_measurements=len(identity_rows),image_measurements=len(image_rows)))
    log(f'Reconstruction complete in {time.time()-start:.1f}s: {output}',flush=True)
    return gradient_rows
