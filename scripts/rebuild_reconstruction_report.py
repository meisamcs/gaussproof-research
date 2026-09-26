"""Rebuild figures and interpretation from saved measurements; never reruns attacks.

Usage: python scripts/rebuild_reconstruction_report.py runs/reconstruction
Then export to an empty report directory with the regular report command.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gaussproof.reconstruction_report import finish_report,aggregate
from gaussproof.io import write_csv,write_json
from gaussproof.models import conservative_epsilon


def read(path):
    rows=[]
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            for k,v in row.items():
                try:row[k]=float(v)
                except ValueError:pass
            rows.append(row)
    return rows


def rebuild(folder):
    folder=Path(folder);cfg=json.loads((folder/'config.json').read_text())
    gradients=read(folder/'gradient_records.csv');identities=read(folder/'identity_records.csv')
    images=read(folder/'image_records.csv');saved=np.load(folder/'gallery.npz');gallery=[]
    for seed in cfg['seeds']:
        for i in range(min(cfg['inversion_targets'],cfg['target_size'])):
            prefix=f'{seed}_{i}_'
            row={key[len(prefix):]:saved[key] for key in saved.files if key.startswith(prefix)}
            row.update(seed=seed,target_index=i)
            gallery.append(row)
    finish_report(folder,cfg,gradients,identities,images,gallery)
    # Direct paired comparisons against Gaussian shrinkage, beyond prior-only gains.
    keys=['seed','sigma','repeats','background','target_index']
    baseline={tuple(v[k] for k in keys):v['squared_l2'] for v in gradients if v['method']=='gaussian'}
    pairs=[dict(v,gain_over_gaussian=baseline[tuple(v[k] for k in keys)]-v['squared_l2'])
           for v in gradients if v['method']=='neural']
    paired=aggregate(pairs,['sigma','repeats','background'],['gain_over_gaussian'],cfg['bootstrap'])
    write_csv(folder/'neural_gaussian_paired_summary.csv',paired)
    gr=read(folder/'gradient_summary.csv');im=read(folder/'image_summary.csv')
    r=cfg['inversion_repeats'];sigma=cfg['inversion_sigma']
    selected=[v for v in gr if v['repeats']==r and v['background']=='known']
    table='| Noise σ | Prior only | Gaussian prior | Neural prior |\n| --- | ---: | ---: | ---: |\n'
    for s in cfg['sigmas']:
        values=[next(v['squared_l2'] for v in selected if v['sigma']==s and v['method']==m)
                for m in ['prior_only','gaussian','neural']]
        table+=f'| {s:g} | '+ ' | '.join(f'{v:.4f}' for v in values)+' |\n'
    pixels='| Method | Pixel MSE |\n| --- | ---: |\n'
    for m in ['mean','prior_only','gaussian','neural','public_mean_image','clean_oracle']:
        pixels+=f"| {m} | {next(v['pixel_mse'] for v in im if v['method']==m):.4f} |\n"
    comparison=next(v for v in paired if v['sigma']==sigma and v['repeats']==r and v['background']=='known')
    eps=conservative_epsilon(len(cfg['checkpoint_steps'])*r,sigma,cfg['delta'])
    labels=sorted({int(v['label']) for v in images})
    findings=f'''# Measured reconstruction results

Executed {len(cfg['seeds'])} seeds, {len(cfg['sigmas'])} noise levels, {len(cfg['repeats'])} release counts, and known/unknown batch backgrounds. Saved {len(gradients):,} gradient measurements, {len(identities):,} identity measurements, and {len(images):,} image measurements. These counts include repeated methods and conditions, not independent samples.

Gradient error per checkpoint, known background, R={r}:

{table}
At σ={sigma:g}, the neural estimator improves over Gaussian shrinkage by {comparison['gain_over_gaussian']:.4f} squared-error units (paired hierarchical 95% bootstrap interval {comparison['gain_over_gaussian_lo']:.4f}–{comparison['gain_over_gaussian_hi']:.4f}). This is a pilot result with only three independently trained models. The neural method is not uniformly best across noise conditions.

Pixel reconstruction at σ={sigma:g}, R={r}; lower MSE is better:

{pixels}
The tested subset contains digits {labels}, with {cfg['inversion_targets']} fixed images per seed. It is not a balanced ten-digit reconstruction evaluation. Learned-prior inversion does not beat the public class-mean image on average. Gradient denoising therefore does **not** establish successful target-specific image reconstruction here. The clean-gradient oracle provides clean attack input, not a globally optimal inversion: several optimizations fail, so its average error is not a theoretical lower bound. All failures remain in the results and gallery.

The known-background release budget at this displayed condition has a conservative replace-one composition bound ε={eps:.2f} at δ={cfg['delta']:g}. The experiment does not establish reconstruction under a strong small-ε privacy guarantee. Repeating releases spends privacy budget; larger σ alone does not define the overall privacy level.

The supported message is that a transferable public gradient prior can improve estimation of unseen gradients beyond prior-only guessing in this controlled channel. The evidence does not support “excess noise backfires,” a successful novel-image reconstruction claim, or superiority over the matched exact-bank likelihood under identical information. The original MIA pilot is preserved separately.

Next research gates: calibrate a more reliable inversion procedure on public holdout images, lock it before a new unseen evaluation, expand pixel tests across all digits, and then test changing DP-SGD checkpoints and hidden participation under an explicit release/privacy budget. Treat these as new experiments rather than tuning on the nine reported targets.

---

'''
    readme=folder/'README.md';readme.write_text(findings+readme.read_text())
    provenance=json.loads((folder/'provenance.json').read_text())
    h=hashlib.sha256()
    for p in sorted((Path(__file__).resolve().parents[1]/'gaussproof').glob('*.py')):
        h.update(p.name.encode());h.update(p.read_bytes())
    provenance['report_source_sha256']=h.hexdigest()
    provenance['report_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance['report_note']='Figures regenerated from unchanged measurements after correcting noise-axis tick labels. source_sha256 identifies the measurement implementation.'
    write_json(folder/'provenance.json',provenance)


if __name__=='__main__':rebuild(sys.argv[1])
