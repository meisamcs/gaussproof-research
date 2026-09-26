"""Add paired baseline comparisons and a numerical table to a completed lab run."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gaussproof.reconstruction_report import aggregate
from gaussproof.io import write_csv


def read(path):
    result=[]
    for row in csv.DictReader(Path(path).open()):
        for k,v in row.items():
            try:row[k]=float(v)
            except ValueError:pass
        result.append(row)
    return result


if __name__=='__main__':
    root=Path(sys.argv[1]);cfg=json.loads((root/'config.json').read_text())
    records=read(root/'records.csv');summary=read(root/'summary.csv')
    keys=['seed','clip','batch_size','requested_noise','evaluation_index']
    baseline={tuple(r[k] for k in keys):r['clip_normalized_squared_l2'] for r in records if r['method']=='gaussian'}
    paired=[dict(r,gain_over_gaussian=baseline[tuple(r[k] for k in keys)]-r['clip_normalized_squared_l2'])
            for r in records if r['method'] not in ['gaussian','noisy']]
    write_csv(root/'paired_vs_gaussian.csv',aggregate(paired,['clip','batch_size','requested_noise','method'],['gain_over_gaussian'],300))
    c=cfg['clips'][0];b=1
    table='| Method | '+ ' | '.join(f'τ≈{t:g}' for t in cfg['absolute_noise'])+' |\n'
    table+='| --- | '+' | '.join('---:' for t in cfg['absolute_noise'])+' |\n'
    names={'prior_only':'Prior only','gaussian':'Gaussian shrinkage','epsilon_trajectory_bank':'DDPM noise prediction',
           'sample_trajectory_bank':'Direct clean prediction','v_prediction_trajectory_bank':'DDPM velocity prediction','v_prediction_ddim':'Velocity DDIM iterative'}
    for method,name in names.items():
        vals=[next(r['clip_normalized_squared_l2'] for r in summary if r['method']==method and r['clip']==c and r['batch_size']==b and r['requested_noise']==t) for t in cfg['absolute_noise']]
        table+='| '+name+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |\n'
    (root/'findings.md').write_text(f'''# Measured diffusion-denoiser pilot

Trained nine upstream Diffusers U-Nets: three seeds × epsilon/sample/velocity objectives. Each model had 2,000 optimizer steps, with the checkpoint chosen on separate calibration data. Evaluated 12,960 method/condition/target measurements. The 24-test suite passed, including exact upstream noise conversion, clipping before averaging, saved-model reload and inference without clean targets.

Example slice: clipping C={c}, individual gradients B=1. Values are full-vector squared error divided by C² and by the number of checkpoints; lower is better. Requested noise is rounded to a nearby scheduler level and actual values are recorded in CSVs.

{table}
For this executed pilot, explicit epsilon prediction has competitive error at the lowest noise in this slice but deteriorates badly at higher noise. Direct and velocity predictions are substantially more stable. At the largest noise they approach the prior-only error rather than demonstrating target-specific recovery. Iterative DDIM does not reliably improve squared-error estimation. The paired CSV includes Gaussian comparisons and intervals; tiny mean differences should not be read as an established neural advantage.

Use the velocity implementation as a stable diffusion-style starting point, while retaining Gaussian shrinkage and prior-only controls. The experiment does not establish that diffusion beats Gaussian shrinkage, that noise can be predicted precisely enough to remove high-noise privacy protection, or that more noise increases vulnerability.

This is a limited laboratory study: fixed public checkpoints, public PCA representation, known label histograms, 30 held-out images per seed, and a small training budget. It does not rule out improvements from richer data, architectures, longer trajectories or better conditioning. The reliable upstream components and tested conversion equations should be distinguished from the experimental gradient representation and fitted model's efficacy.

See [the full protocol](README.md), `summary.csv`, `noise_prediction.csv`, and `paired_vs_gaussian.csv`. Raw per-target records and all trained weights remain in the local run directory; the separate model bundle includes public CNN checkpoints and gradient adapters required for inference.
''')
