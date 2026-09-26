"""Paired reporting for the matched sequence-access experiment."""
import csv,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];source=root/'runs/sequence_ablation';out=root/'reports/sequence_ablation';out.mkdir(exist_ok=True,parents=True)
assert (source/'completion.json').exists()
cfg=json.loads((source/'config.json').read_text());rows=list(csv.DictReader((source/'measurements.csv').open()))
modes=['current','past','full','shuffled','full_test_shuffled'];metrics=['pixel_mse','gradient_error'];seeds=cfg['seeds'];nruns=8
summary=[];runrows=[];seedrows=[];contrasts=[];arrays={}
for sigma in cfg['sigmas']:
 for mode in modes:
  for metric in metrics:
   a=np.empty((len(seeds),nruns))
   for si,seed in enumerate(seeds):
    for run in range(nruns):
     a[si,run]=np.mean([float(r[metric]) for r in rows if float(r['sigma'])==sigma and int(r['seed'])==seed and int(r['run'])==run and r['mode']==mode])
     runrows.append(dict(sigma=sigma,mode=mode,metric=metric,seed=seed,run=run,value=a[si,run]))
    seedrows.append(dict(sigma=sigma,mode=mode,metric=metric,seed=seed,value=a[si].mean()))
   arrays[sigma,mode,metric]=a
   summary.append(dict(sigma=sigma,mode=mode,metric=metric,mean=a.mean(),training_seed_sd=a.mean(1).std(ddof=1)))
 for comparator,method in [('current','past'),('current','full'),('current','shuffled'),('shuffled','full'),('full_test_shuffled','full')]:
  for metric in metrics:
   baseline=arrays[sigma,comparator,metric];estimate=arrays[sigma,method,metric];difference=baseline-estimate
   rng=np.random.default_rng(426);boot=[]
   for _ in range(2000):
    ss=rng.integers(len(seeds),size=len(seeds));rr=rng.integers(nruns,size=nruns);boot.append(difference[np.ix_(ss,rr)].mean())
   lo,hi=np.quantile(boot,[.025,.975]);seedgains=difference.mean(1)
   contrasts.append(dict(sigma=sigma,metric=metric,baseline=comparator,method=method,absolute_gain=difference.mean(),percent_gain=100*difference.mean()/baseline.mean(),bootstrap_low=lo,bootstrap_high=hi,min_seed_gain=seedgains.min(),max_seed_gain=seedgains.max(),positive_training_seeds=int((seedgains>0).sum())))

def write(name,data):
 with (out/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write('summary.csv',summary);write('per_seed_run.csv',runrows);write('per_training_seed.csv',seedrows);write('paired_contrasts.csv',contrasts)
for name in ['measurements.csv','model_selection.csv','config.json','completion.json']:shutil.copy2(source/name,out/name)
fig,axes=plt.subplots(2,2,figsize=(11,7))
labels=['Current only','Past + current','Full sequence','Shuffled trained','Full, shuffled test']
for row,sigma in enumerate(cfg['sigmas']):
 for col,metric in enumerate(metrics):
  ax=axes[row,col];means=[arrays[sigma,m,metric].mean() for m in modes]
  ax.bar(np.arange(5),means,color=['#7b8794','#4c78a8','#e28c35','#65a783','#b487ba'],alpha=.75)
  for si in range(len(seeds)):ax.scatter(np.arange(5)+(si-1)*.1,[arrays[sigma,m,metric][si].mean() for m in modes],s=18,color='black',zorder=3)
  ax.axhline(means[0],color='gray',linestyle='--',linewidth=.8)
  ax.set_xticks(np.arange(5),labels,rotation=25,ha='right',fontsize=8);ax.set_title(f'σ={sigma:g}');ax.set_ylabel('Matched image MSE' if metric=='pixel_mse' else 'Full gradient squared error / C²');ax.grid(axis='y',alpha=.15)
fig.suptitle('Matched architecture and training budget; dots show three initialization seeds',fontsize=12);fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(out/f'access_comparison.{ext}',dpi=220)
plt.close(fig)
lines=['# Does sequence access improve reconstruction?','',
'Completed a matched LSTM ablation on the existing MNIST DP-SGD trajectories. All modes use identical architecture, initialization seeds, training probes, 1,200 optimizer updates, and public calibration selection. The public representation and image generator are frozen. Only observation access/order changes.','',
'Each mode receives the same current-checkpoint fingerprint; other checkpoint fingerprints are excluded even from preprocessing. This isolates the added value of surrounding noisy releases given the current public context.','',
'24 models trained: four access modes × three initialization seeds × two noise settings. Evaluation uses the same eight private 16-round trajectories per setting. Shuffled errors average five permutations, preserving the current release.','',
'| Access setting | Image MSE, σ=0.05 | Image MSE, σ=0.5 |','|---|---:|---:|']
for mode,label in zip(modes,labels):lines.append(f'| {label} | {arrays[.05,mode,"pixel_mse"].mean():.5f} | {arrays[.5,mode,"pixel_mse"].mean():.5f} |')
lines+=['','Lower error is better. Dots in the figure show means for three independently initialized reconstructors, not three independent datasets.','',
'## Primary paired comparison: full sequence versus current release','']
for sigma in cfg['sigmas']:
 for metric in metrics:
  r=next(r for r in contrasts if r['sigma']==sigma and r['metric']==metric and r['baseline']=='current' and r['method']=='full')
  lines.append(f'- σ={sigma:g}, {metric}: full-sequence improvement {r["percent_gain"]:+.2f}%; absolute gain {r["absolute_gain"]:+.6f}, descriptive 95% bootstrap interval [{r["bootstrap_low"]:+.6f}, {r["bootstrap_high"]:+.6f}]. Positive in {r["positive_training_seeds"]}/3 training seeds.')
lines+=['','## Does order matter?','']
for sigma in cfg['sigmas']:
 for baseline in ['shuffled','full_test_shuffled']:
  r=next(r for r in contrasts if r['sigma']==sigma and r['metric']=='pixel_mse' and r['baseline']==baseline and r['method']=='full')
  lines.append(f'- σ={sigma:g}: ordered full-model image improvement over {baseline}: {r["percent_gain"]:+.2f}%; positive in {r["positive_training_seeds"]}/3 seeds. Absolute-gain interval [{r["bootstrap_low"]:+.6f}, {r["bootstrap_high"]:+.6f}].')
lines+=['','## Limits','',
'- These are the earlier C=1, B=2, sigma=0.05/0.5 feasibility conditions, not tiny-epsilon demonstrations.',
'- The current fingerprint can already contain information from past model updates. Conclusions concern additional explicit sequence access given that context.',
'- Shuffling preserves other observations but destroys their temporal positions. The separately trained shuffled model and full-model test-time shuffle diagnose different effects; the latter can suffer distribution shift.',
'- All eight target runs reuse one private pool; all reconstructors share the same public representation and decoder. Crossed seed/run bootstrap intervals are descriptive, not population-level guarantees.',
'- Image errors use the same fixed generator and optimal two-image assignment. Gradient errors include discarded representation components. More information need not help a finite trained estimator.',
'- No private outcomes were used for model selection. No new DP-SGD trajectories were generated for this comparison.','',
'See ../../docs/sequence_ablation_protocol.md for exact masks, normalization, shuffle construction, and reproduction.']
(out/'README.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

# Exploratory per-round curves: primary conclusions remain the paired whole-run means.
roundrows=[]
for sigma in cfg['sigmas']:
 for mode in modes:
  for t in range(cfg['steps']):
   rr=[r for r in rows if float(r['sigma'])==sigma and r['mode']==mode and int(r['round'])==t]
   roundrows.append(dict(sigma=sigma,mode=mode,round=t+1,**{metric:np.mean([float(r[metric]) for r in rr]) for metric in metrics}))
write('per_round.csv',roundrows)
import hashlib
(out/'source_manifest.json').write_text(json.dumps({p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['gaussproof/sequence_ablation.py','scripts/summarize_sequence_ablation.py','configs/sequence_ablation.json','docs/sequence_ablation_protocol.md']},indent=2))
