"""Aggregate reconstruction evidence, including paired prior-only comparisons."""
from collections import defaultdict
from pathlib import Path
import hashlib
import shutil
import numpy as np
from .io import write_csv,write_json
from .plots import style,save,plt
from matplotlib.ticker import NullLocator


def noise_ticks(ax,cfg):
    ax.set_xticks(cfg['sigmas'],[f'{s:g}' for s in cfg['sigmas']])
    ax.xaxis.set_minor_locator(NullLocator())


def interval(rows,metric,repeats=500,seed=123):
    """Hierarchical percentile bootstrap: seeds, then records within each seed."""
    groups=defaultdict(list)
    for row in rows:groups[row['seed']].append(float(row[metric]))
    arrays=[np.asarray(v) for v in groups.values()]
    rng=np.random.default_rng(seed);boot=[]
    for _ in range(repeats):
        sampled=[arrays[i] for i in rng.integers(len(arrays),size=len(arrays))]
        boot.append(np.mean([rng.choice(a,len(a),replace=True).mean() for a in sampled]))
    return float(np.mean([a.mean() for a in arrays])),*map(float,np.quantile(boot,[.025,.975]))


def aggregate(rows,keys,metrics,repeats):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in keys)].append(row)
    result=[]
    for key,values in sorted(groups.items()):
        row=dict(zip(keys,key));row.update(n=len(values),seeds=len({v['seed'] for v in values}))
        for metric in metrics:
            mean,lo,hi=interval(values,metric,repeats)
            row.update({metric:mean,metric+'_lo':lo,metric+'_hi':hi})
        result.append(row)
    return result


def finish_report(output,cfg,gradients,identities,images,gallery):
    output=Path(output);figs=output/'figures';style()
    gr=aggregate(gradients,['sigma','repeats','background','method'],
                 ['squared_l2','normalized_squared_l2','cosine','gain_over_prior'],cfg['bootstrap'])
    ir=aggregate(identities,['sigma','repeats','method'],['success','chance'],cfg['bootstrap'])
    im=aggregate(images,['method'],['pixel_mse','psnr_db'],cfg['bootstrap'])
    write_csv(output/'gradient_summary.csv',gr);write_csv(output/'identity_summary.csv',ir)
    write_csv(output/'image_summary.csv',im)
    # Paired target-level image differences, preserving common initialization.
    baseline={(r['seed'],r['target_index']):r['pixel_mse'] for r in images if r['method']=='public_mean_image'}
    paired=[dict(r,gain_over_public_image=baseline[r['seed'],r['target_index']]-r['pixel_mse']) for r in images]
    write_csv(output/'image_paired_summary.csv',aggregate(paired,['method'],['gain_over_public_image'],cfg['bootstrap']))
    names={'mean':'Noisy mean','prior_only':'Prior only','projection':'Subspace projection',
           'gaussian':'Gaussian prior','neural':'Neural prior','stale_gaussian':'Stale prior',
           'permuted_release':'Shuffled release','exact_bank_posterior':'Exact-bank posterior',
           'public_mean_image':'Public mean image','clean_oracle':'Clean-gradient oracle'}
    for background in ['known','unknown']:
        fig,axes=plt.subplots(1,len(cfg['repeats']),figsize=(4*len(cfg['repeats']),3.6),squeeze=False,layout='constrained')
        for ax,r in zip(axes[0],cfg['repeats']):
            for method in ['mean','prior_only','projection','gaussian','neural']:
                rows=sorted([v for v in gr if v['background']==background and v['repeats']==r and v['method']==method],key=lambda v:v['sigma'])
                xx=[v['sigma'] for v in rows];yy=[v['squared_l2'] for v in rows]
                ax.plot(xx,yy,'o-',label=names[method],markersize=3)
                ax.fill_between(xx,[v['squared_l2_lo'] for v in rows],[v['squared_l2_hi'] for v in rows],alpha=.10)
            ax.set(xscale='log',yscale='log',xlabel='Noise multiplier σ',ylabel='Squared gradient error / checkpoint',title=f'{r} releases / checkpoint')
            noise_ticks(ax,cfg)
            ax.grid(alpha=.15)
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'{background.capitalize()} batch background · public-prior access',fontsize=11)
        save(fig,figs/f'gradient_error_{background}')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    r=cfg['inversion_repeats']
    for ax,bg in zip(axes,['known','unknown']):
        for method in ['gaussian','neural','stale_gaussian','permuted_release']:
            rows=sorted([v for v in gr if v['background']==bg and v['repeats']==r and v['method']==method],key=lambda v:v['sigma'])
            ax.plot([v['sigma'] for v in rows],[v['gain_over_prior'] for v in rows],'o-',label=names[method])
        ax.axhline(0,color='.4',ls=':');ax.set(xscale='log',xlabel='Noise multiplier σ',ylabel='Prior-only error − estimator error',title=f'{bg.capitalize()} background, R={r}')
        noise_ticks(ax,cfg)
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=7);save(fig,figs/'prior_gain_controls')
    fig,axes=plt.subplots(1,len(cfg['repeats']),figsize=(4*len(cfg['repeats']),3.4),squeeze=False,layout='constrained')
    for ax,r in zip(axes[0],cfg['repeats']):
        for method in ['alignment','likelihood']:
            rows=sorted([v for v in ir if v['repeats']==r and v['method']==method],key=lambda v:v['sigma'])
            ax.plot([v['sigma'] for v in rows],[v['success'] for v in rows],'o-',label=method)
            ax.fill_between([v['sigma'] for v in rows],[v['success_lo'] for v in rows],[v['success_hi'] for v in rows],alpha=.12)
        ax.axhline(np.mean([v['chance'] for v in ir if v['repeats']==r]),color='.4',ls=':',label='Chance')
        ax.set(xscale='log',xlabel='Noise multiplier σ',ylabel='Correct identity fraction',ylim=(0,1.03),title=f'R={r}');ax.grid(alpha=.15)
        noise_ticks(ax,cfg)
    axes[0,0].legend(fontsize=7);fig.suptitle('Target-containing bank · separate, stronger access assumption')
    save(fig,figs/'exact_bank_identification')
    fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained')
    order=['mean','prior_only','gaussian','neural','public_mean_image','clean_oracle']
    rows=[next(v for v in im if v['method']==m) for m in order]
    yy=np.array([v['pixel_mse'] for v in rows]);lo=np.array([v['pixel_mse_lo'] for v in rows]);hi=np.array([v['pixel_mse_hi'] for v in rows])
    ax.bar(range(len(rows)),yy,yerr=np.maximum(0,np.array([yy-lo,hi-yy])),capsize=3)
    ax.set(xticks=range(len(rows)),xticklabels=[names[m] for m in order],ylabel='Pixel mean squared error',
           title=f'Fixed inversion subset · σ={cfg["inversion_sigma"]}, R={cfg["inversion_repeats"]}')
    ax.tick_params(axis='x',rotation=25);save(fig,figs/'image_error')
    columns=['original','mean','prior_only','gaussian','neural','public_mean_image','clean_oracle']
    fig,axes=plt.subplots(len(gallery),len(columns),figsize=(10,1.35*len(gallery)),squeeze=False,layout='constrained')
    for i,row in enumerate(gallery):
        for j,key in enumerate(columns):
            ax=axes[i,j];ax.imshow(row[key],cmap='gray',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title('Original' if key=='original' else names[key],fontsize=8)
            if j==0:ax.set_ylabel(f'Seed {row["seed"]}\ntarget {row["target_index"]}',fontsize=7)
    save(fig,figs/'reconstruction_gallery')
    (output/'README.md').write_text('''# Controlled gradient reconstruction experiment

These are measured MNIST reconstructions from repeated Gaussian releases at **fixed public CNN checkpoints**, not a DP-SGD training run. Labels are known. Public model training, prior fitting, calibration, targets, and distractors use disjoint records. Targets are balanced across labels when the configured count permits it; inversion uses the first fixed targets, never selection on success.

`gradient_summary.csv` reports per-checkpoint squared gradient error, normalized error, cosine similarity, and paired gain over prior-only estimates. `identity_summary.csv` evaluates an exact target-containing bank separately. `image_summary.csv` and `image_paired_summary.csv` report pixel MSE/PSNR and paired improvement over the public class-mean image. Lower errors and higher gains are better. Neural means a public-data latent denoiser; Gaussian means empirical low-rank covariance shrinkage. Neither is the original sparse MIA decoder.

Intervals are 95% hierarchical percentile bootstrap intervals, resampling seeds then target records. Three seeds and nine inverted images support only pilot conclusions. Conditions reuse paired records and noise; they are not independent replications. Plots use absolute errors, not only percentage reductions. Shuffled releases retain labels in the full configured run. Unknown backgrounds are actual sampled public gradients, but covariance-based estimators approximate their distribution diagonally in the learned basis. Stale priors repeat the first checkpoint's fingerprints.

Image inversion receives the estimated gradient, known label, and public checkpoints. All estimators use the same random image initializations, inversion steps, and restarts. The clean-gradient oracle checks inversion capacity. Public mean images and prior-only gradients check plausible class reconstructions without any target release. Gallery originals are public MNIST examples used as experimental targets.

More releases incur more privacy cost. Raw records include a conservative replace-one Gaussian-composition epsilon for K×R releases; no subsampling amplification is claimed. Large noise cannot improve the optimal attack when all other information is fixed. A learned prior beating noisy averaging is evidence of denoising, not a violation of DP, and does not establish superiority over an optimal likelihood attack with the same prior. Exact-bank retrieval is identification, not unseen-image reconstruction.

The RAoPT paper motivates learning a reconstruction prior from public clean/privatized pairs; its empirical GPS results do not establish a result for gradients: https://arxiv.org/pdf/2210.09375 . The reconstruction-likelihood comparison is informed by https://arxiv.org/pdf/2302.07225 . Full dynamic DP-SGD, hidden labels/participation, architecture transfer, and formal privacy auditing remain outside this experiment.
''')


def export_report(source,destination):
    source=Path(source);destination=Path(destination)
    if destination.exists() and any(destination.iterdir()):raise FileExistsError('Report directory must be empty')
    destination.mkdir(parents=True,exist_ok=True)
    for name in ['config.json','provenance.json','completion.json','utility.csv','gradient_summary.csv',
                 'identity_summary.csv','image_summary.csv','image_paired_summary.csv','README.md']:
        shutil.copy2(source/name,destination/name)
    shutil.copytree(source/'figures',destination/'figures')
    paired=source/'neural_gaussian_paired_summary.csv'
    if paired.exists():shutil.copy2(paired,destination/paired.name)
    write_json(destination/'manifest.json',{str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(destination.rglob('*')) if p.is_file()})
