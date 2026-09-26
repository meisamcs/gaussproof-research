"""Aggregate the offline batch reconstruction experiment and fixed image galleries."""
import csv,json,shutil,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];source=root/'runs/sequence_images';out=root/'reports/sequence_images';out.mkdir(parents=True,exist_ok=True)
info=json.loads((source/'completion.json').read_text());cfg=json.loads((source/'config.json').read_text())
rows=list(csv.DictReader((source/'measurements.csv').open()));methods=sorted(set(r['method'] for r in rows));summary=[];perrun=[]
for sigma in cfg['sigmas']:
 for method in methods:
  part=[r for r in rows if float(r['sigma'])==sigma and r['method']==method]
  for run in range(cfg['target_runs']):
   rr=[r for r in part if int(r['run'])==run]
   perrun.append(dict(sigma=sigma,method=method,run=run,**{k:float(np.mean([float(r[k]) for r in rr])) for k in ['pixel_mse','psnr_db','digit_accuracy','gradient_error']}))
  rr=[r for r in perrun if r['sigma']==sigma and r['method']==method]
  summary.append(dict(sigma=sigma,method=method,**{k:float(np.mean([r[k] for r in rr])) for k in ['pixel_mse','psnr_db','digit_accuracy','gradient_error']},pixel_mse_seed_sd=float(np.std([r['pixel_mse'] for r in rr],ddof=1))))

def write(name,data):
 with (out/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write('summary.csv',summary);write('per_run.csv',perrun)
for name in ['measurements.csv','privacy.csv','config.json','completion.json','generator_training.csv']:
 shutil.copy2(source/name,out/name)
for sigma in cfg['sigmas']:
 for name in ['bilstm_training.csv','diffusion_training.csv']:
  shutil.copy2(source/f'sigma{sigma:g}'/name,out/f'sigma{sigma:g}_{name}')
 keys=['truth','clean_oracle','gaussian_current','bilstm','diffusion_mean','diffusion_draw','context_only','mean_image']
 names=['True batch','Clean-gradient oracle','Gaussian shrinkage','Bidirectional LSTM','Diffusion: latent mean','Diffusion: one draw','Checkpoint context only','Public mean image']
 data=np.load(source/f'sigma{sigma:g}'/'reconstructed_batches.npz')
 pairs=[(0,0),(0,8),(1,0),(1,8)]
 fig,axes=plt.subplots(len(keys),8,figsize=(10.5,9.2))
 for row,(key,label) in enumerate(zip(keys,names)):
  for col,(run,t) in enumerate(pairs):
   for b in range(2):
    ax=axes[row,2*col+b];ax.imshow(data[key][run,t,b],cmap='gray',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
    if row==0:ax.set_title(f'Run {run}, round {t+1}\nimage {b+1}',fontsize=8)
    if col==0 and b==0:ax.set_ylabel(label,rotation=0,ha='right',va='center',fontsize=9,labelpad=10)
 fig.suptitle(f'Unordered MNIST batch reconstruction: C=1, B=2, σ={sigma:g}\nFixed examples; output order matched to truth only for display/scoring',fontsize=12)
 fig.subplots_adjust(left=.23,right=.99,top=.9,bottom=.02,hspace=.1,wspace=.08)
 for ext in ['png','pdf']:fig.savefig(out/f'batches_sigma{sigma:g}.{ext}',dpi=200)
 plt.close(fig)

selected=['clean_oracle','gaussian_current','gaussian_smoother','bilstm','diffusion_mean','diffusion_draw','context_only','mean_image']
fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
for ax,sigma in zip(axes,cfg['sigmas']):
 rr=[next(r for r in summary if r['sigma']==sigma and r['method']==m) for m in selected]
 ax.bar(range(len(rr)),[r['pixel_mse'] for r in rr],yerr=[r['pixel_mse_seed_sd'] for r in rr],capsize=2)
 ax.set_xticks(range(len(rr)),[m.replace('_',' ') for m in selected],rotation=45,ha='right',fontsize=8)
 ax.set_title(f'σ={sigma:g}');ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Matched batch pixel MSE (lower is better)')
fig.suptitle('Held-out reconstruction: mean ± run standard deviation');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'image_errors.{ext}',dpi=220)
plt.close(fig)
lines=['# Full-sequence gradient and image reconstruction pilot','',
'Implemented and executed an offline bidirectional LSTM, an iterative conditional temporal diffusion model, and a shared convolutional generator producing two MNIST images per round. Both sequence models can use later releases when reconstructing earlier rounds. No private labels or batch identities are provided to them.','',
f'Configuration: C={cfg["clip"]}, B={cfg["batch_size"]}, {cfg["steps"]} rounds, sigma={cfg["sigmas"]}. Per condition: {cfg["public_train_runs"]} public training sequences, {cfg["public_calibration_runs"]} public validation sequences, {cfg["target_runs"]} held-out private sequences. The image generator uses both public noise conditions. These small-noise settings are feasibility tests, not strong-privacy claims.','',
'| Method | σ=0.05 pixel MSE | σ=0.5 pixel MSE |','|---|---:|---:|']
for method in selected+['diffusion_shuffled_sequence','noisy']:
 values=[next(r['pixel_mse'] for r in summary if r['sigma']==s and r['method']==method) for s in cfg['sigmas']]
 lines.append(f'| {method.replace("_"," ")} | {values[0]:.5f} | {values[1]:.5f} |')
lines+=['','Pixel errors optimally match the two outputs to the two target images after inference. All 256 target batches (512 image occurrences) are scored; the gallery uses predeclared examples. The clean oracle passes the true **projected** clipped gradient through the same image generator; it is not an attack.','',
'## Paired observations','']
for sigma in cfg['sigmas']:
 for comparator in ['context_only','gaussian_current','bilstm','clean_oracle']:
  gains=[]
  for run in range(cfg['target_runs']):
   val=lambda method:next(r['pixel_mse'] for r in perrun if r['sigma']==sigma and r['run']==run and r['method']==method)
   gains.append(val(comparator)-val('diffusion_mean'))
  lines.append(f'- σ={sigma:g}, diffusion-mean gain over {comparator}: {np.mean(gains):+.5f} MSE (positive is better; run range {min(gains):+.5f} to {max(gains):+.5f}).')
lines+=['','## Interpretation limits','',
f'- Public classifier held-out accuracy: {info["public_classifier_accuracy"]*100:.1f}%. Reported digit agreement is a diagnostic, not a perfect recognition metric.',
'- PCA retains 128 directions; all recovered full gradients use public fingerprints outside that space. The image generator is trained on public clean projected gradients, so its response to imperfect estimates may suffer distribution shift.',
'- The conditional diffusion adapter is a custom temporal Transformer with official Diffusers DDPM/DDIM scheduling and velocity prediction. It is not a reproduction of RAoPT or CSDI. Sampling uses 40 reverse steps and four draws; sample averaging occurs in gradient coordinates before image generation.',
'- The generator is deterministic and trained with a permutation-invariant pixel loss. Ambiguous gradients can yield blurry conditional-average images. Good-looking outputs alone do not establish recovery of particular private examples.',
'- There are only 48 public training trajectories per condition. Runs share a role-specific data pool and a public initial CNN; run standard deviations are descriptive, not independent-dataset confidence intervals.',
'- The conservative privacy bounds are large for these settings and do not use subsampling amplification. Do not extrapolate this pilot to tiny epsilon or claim that added noise increases leakage.',
'- Paired noisy and clean gradients and original target images are evaluation-only laboratory logs. Saved inference accepts only noisy releases and public fingerprints.','',
'## Artifacts','',
'- `batches_sigma0.05.png` / `.pdf`, `batches_sigma0.5.png` / `.pdf`: fixed image galleries.',
'- `summary.csv`, `per_run.csv`, `measurements.csv`: all measurements.',
'- `image_errors.pdf`: vector comparison plot.',
'- `privacy.csv`: conservative per-run privacy bounds.',
'- `../../docs/sequence_images_protocol.md`: setup, access assumptions, and reproduction.',
'- `../../scripts/reconstruct_batches.py`: saved-model inference without target truth.']
(out/'README.md').write_text('\n'.join(lines)+'\n')
manifest={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['gaussproof/sequence_images.py','scripts/reconstruct_batches.py','scripts/summarize_sequence_images.py','configs/sequence_images.json','docs/sequence_images_protocol.md']}
(out/'source_manifest.json').write_text(json.dumps(manifest,indent=2));print('\n'.join(lines))

# Compact preview uses the first two predeclared gallery batches, without selection.
preview=np.load(source/'sigma0.05'/'reconstructed_batches.npz')
fig,axes=plt.subplots(4,4,figsize=(6.2,5.6))
for row,(key,label) in enumerate([('truth','True batch'),('bilstm','BiLSTM'),('diffusion_mean','Diffusion'),('gaussian_current','Gaussian')]):
 for col,(t,b) in enumerate([(0,0),(0,1),(8,0),(8,1)]):
  ax=axes[row,col];ax.imshow(preview[key][0,t,b],cmap='gray',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([])
  for spine in ax.spines.values():spine.set_visible(False)
  if col==0:ax.set_ylabel(label,rotation=0,ha='right',va='center',labelpad=12)
  if row==0:ax.set_title(f'Round {t+1}, image {b+1}',fontsize=8)
fig.suptitle('First fixed examples: C=1, B=2, σ=0.05',fontsize=12)
fig.subplots_adjust(left=.2,right=.99,bottom=.02,top=.89,wspace=.07,hspace=.08)
fig.savefig(out/'preview.png',dpi=180);plt.close(fig)
