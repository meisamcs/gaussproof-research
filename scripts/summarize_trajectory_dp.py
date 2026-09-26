"""Export completed causal-observer comparisons without private ground-truth logs."""
import csv,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
out=root/'reports/trajectory_dp';out.mkdir(exist_ok=True)
read=lambda p:list(csv.DictReader(p.open()))
allsummary=[];paired=[];utilities=[]
for window,name in [(8,'trajectory_dp'),(64,'trajectory_dp_full_history')]:
    source=root/'runs'/name
    assert (source/'completion.json').exists()
    dest=out/f'window{window}';dest.mkdir(exist_ok=True)
    for filename in ['summary.csv','per_seed.csv','measurements.csv','privacy.csv','utility.csv','config.json','completion.json','recovery.pdf','recovery.png']:
        shutil.copy2(source/filename,dest/filename)
    summary=read(source/'summary.csv');seeds=read(source/'per_seed.csv');utility=read(source/'utility.csv')
    allsummary.extend(dict(window=window,**r) for r in summary)
    utilities.extend(dict(window=window,**r) for r in utility if r['role']=='target' and int(r['step'])==64)
    for sigma in [1.,4.,16.]:
        for comparator in ['diffusion_context_only','gaussian_context_only','gaussian_current','gaussian_temporal']:
            gains=[];relative=[]
            for seed in range(3):
                match=lambda method:float(next(r['error'] for r in seeds if float(r['sigma'])==sigma and int(r['seed'])==seed and r['method']==method))
                a=match('diffusion_temporal');b=match(comparator);gains.append(b-a);relative.append((b-a)/b*100)
            paired.append(dict(window=window,sigma=sigma,comparator=comparator,mean_error_gain=np.mean(gains),min_seed_gain=min(gains),max_seed_gain=max(gains),mean_percent_gain=np.mean(relative)))

def write(name,rows):
    with (out/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write('window_comparison.csv',allsummary);write('paired_gains.csv',paired);write('target_final_utility.csv',utilities)
fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
methods=['diffusion_temporal','diffusion_context_only','gaussian_current','gaussian_temporal','public_fingerprint']
for ax,window in zip(axes,[8,64]):
    for method in methods:
        rr=[r for r in allsummary if r['window']==window and r['method']==method]
        ax.plot([float(r['sigma']) for r in rr],[float(r['error']) for r in rr],marker='o',label=method.replace('_',' '))
    ax.set(xscale='log',xlabel='DP-SGD noise multiplier σ',title=f'{window}-release causal window');ax.grid(alpha=.2)
axes[0].set_ylabel('Full-gradient squared error / C² (lower is better)')
axes[1].legend(fontsize=8);fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(out/f'window_comparison.{ext}',dpi=220)
plt.close(fig)

full=[r for r in allsummary if r['window']==64]
privacy=read(out/'window64/privacy.csv')
lines=['# Actual DP-SGD trajectory denoising: measured pilot','',
'Completed: 33 actual CNN training trajectories (2,112 DP-SGD rounds), nine private target trajectories, and six denoisers. Each denoiser trained for 1,200 steps; model selection used separate public calibration runs. The 8-release and full-history observers use identical underlying trajectories.','',
'Target: clean clipped batch sum divided by expected batch size, not an individual record or original image. C=0.1, expected B=32, N=2,000, 64 rounds, delta=1e-5. The full-history observer uses every release available so far, without batch membership information.','',
'| σ | ε upper bound | Diffusion | Diffusion context only | Gaussian current | Gaussian temporal |','|---:|---:|---:|---:|---:|---:|']
for sigma in [1.,4.,16.]:
    val=lambda method:float(next(r['error'] for r in full if float(r['sigma'])==sigma and r['method']==method))
    eps=float(next(r['epsilon'] for r in privacy if float(r['sigma'])==sigma))
    lines.append(f'| {sigma:g} | {eps:.4f} | {val("diffusion_temporal"):.5f} | {val("diffusion_context_only"):.5f} | {val("gaussian_current"):.5f} | {val("gaussian_temporal"):.5f} |')
lines+=['','Errors are full-vector squared error divided by C², averaged over 64 rounds and three target runs. Privacy bounds use Opacus RDP accounting with an expanded order grid, add/remove adjacency and Poisson sampling.','',
'**Finding:** Diffusion does not outperform Gaussian current-observation shrinkage at any tested noise level. The added-current-observation benefit becomes very small at high noise; full history does not rescue diffusion in this pilot.','',
'## Observation-dependent benefit','',
'Positive gains below mean that adding the current release improves the diffusion estimate over its same-network context-only control. They do not establish membership leakage or superiority over Gaussian estimation.','']
for sigma in [1.,4.,16.]:
    r=next(r for r in paired if r['window']==64 and r['sigma']==sigma and r['comparator']=='diffusion_context_only')
    lines.append(f'- σ={sigma:g}: {r["mean_percent_gain"]:.2f}% mean relative error reduction; absolute gains across three seeds [{r["min_seed_gain"]:.6f}, {r["max_seed_gain"]:.6f}].')
lines+=['','## Utility and limits','']
for sigma in [1.,4.,16.]:
    u=[float(r['accuracy']) for r in utilities if r['window']==64 and float(r['sigma'])==sigma]
    lines.append(f'- σ={sigma:g}: final public-test CNN accuracy {np.mean(u)*100:.2f}% (three-run mean).')
lines+=['','This pilot uses a rank-64 public representation. Every learned/Gaussian estimator retains the public fingerprint outside that space; full-dimensional errors and a projection oracle are supplied. The denoiser is an upstream Diffusers U-Net trained with a diffusion corruption schedule and clean-sample prediction, evaluated in one step. It is not an iterative posterior sampler.','',
'Only six public training trajectories and three private evaluation trajectories per noise setting were used. The U-Net treats temporal lags as channels and PCA coefficients as an arbitrary grid; performance is not a bound on other architectures. The Gaussian temporal ridge baseline also has limited training data. The strongest tested comparator must include Gaussian current-observation shrinkage.','',
'Increasing σ changes the learned CNN and the clean gradients as well as the noise. Cross-σ error comparisons are not controlled demonstrations that more noise creates more leakage. Within-condition comparisons and current-release ablations are the appropriate evidence here.','',
'This experiment is causal: estimates at round t cannot use rounds after t. An offline all-round smoothing attack is not evaluated. Private clean-gradient logs and batch sizes remain in ignored local runs, excluded from this report.','',
'See ../../docs/trajectory_dp_protocol.md for reproduction and access assumptions.']
(out/'README.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))

# Preserve source and dependency identity without distributing private run logs.
import hashlib, platform, diffusers, opacus, torch
manifest={}
for rel in ['gaussproof/trajectory_dp.py','scripts/summarize_trajectory_dp.py','configs/trajectory_dp.json','configs/trajectory_dp_full_history.json','docs/trajectory_dp_protocol.md']:
    manifest[rel]=hashlib.sha256((root/rel).read_bytes()).hexdigest()
(out/'provenance.json').write_text(json.dumps(dict(source_sha256=manifest,python=platform.python_version(),torch=torch.__version__,diffusers=diffusers.__version__,opacus=opacus.__version__),indent=2))
fig,axes=plt.subplots(1,2,figsize=(10,4.2))
for window in [8,64]:
    rr=[r for r in paired if r['window']==window and r['comparator']=='diffusion_context_only']
    axes[0].plot([r['sigma'] for r in rr],[r['mean_percent_gain'] for r in rr],marker='o',label=f'{window}-release window')
axes[0].set(xscale='log',xlabel='Noise multiplier σ',ylabel='Error reduction from current release (%)',title='Does the current observation add information?')
axes[0].axhline(0,color='gray',linewidth=.8);axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
u=read(out/'window64/utility.csv')
for sigma in [1.,4.,16.]:
    rr=[r for r in u if r['role']=='target' and float(r['sigma'])==sigma]
    steps=sorted(set(int(r['step']) for r in rr))
    axes[1].plot(steps,[np.mean([float(r['accuracy'])*100 for r in rr if int(r['step'])==t]) for t in steps],marker='o',label=f'σ={sigma:g}')
axes[1].set(xlabel='DP-SGD round',ylabel='Public-test accuracy (%)',title='CNN utility along actual noisy training');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(out/f'observation_gain_and_utility.{ext}',dpi=220)
plt.close(fig)
