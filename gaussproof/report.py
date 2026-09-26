"""Export only aggregate results and figures for repository publication."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from .data import digest
from .io import write_json, write_csv


def export_report(source, destination):
    source,destination=Path(source),Path(destination)
    if not (source/'completion.json').is_file():
        raise ValueError('Only completed runs may be exported')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Report destination must be empty')
    destination.mkdir(parents=True,exist_ok=True)
    for name in ['config.json','provenance.json','completion.json','metrics.csv','utility.csv',
                 'validation.csv','synthetic_controls.csv']:
        if name.endswith('.csv'):
            with (source/name).open() as f:
                write_csv(destination/name,list(csv.DictReader(f)))
        else:
            shutil.copy2(source/name,destination/name)
    shutil.copytree(source/'figures',destination/'figures')
    for folder in sorted(source.glob('seed*_sigma*_clip*_B*')):
        for ext in ['pdf','png']:
            shutil.copy2(folder/'figures'/f'roc.{ext}',destination/'figures'/f'{folder.name}_roc.{ext}')
    with (source/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
    with (source/'utility.csv').open() as f:utility=list(csv.DictReader(f))
    cfg=json.loads((source/'config.json').read_text())
    write_csv(destination/'blackbox_mia_results.csv',[r for r in rows if r['threat_model']=='black_box'])
    write_csv(destination/'trajectory_mia_sweep.csv',[r for r in rows if r['threat_model']!='black_box'])
    summary=[]
    groups=sorted({(float(r['sigma']),float(r['clip']),int(r['batch_size']),r['attack']) for r in rows})
    for sigma,clip,batch,attack in groups:
        records=[r for r in rows if (float(r['sigma']),float(r['clip']),int(r['batch_size']),r['attack'])==(sigma,clip,batch,attack)]
        row=dict(sigma=sigma,clip=clip,batch_size=batch,attack=attack,n_seeds=len(records))
        for metric in ['auc','tpr_at_1pct_fpr','tpr_at_5pct_fpr','max_advantage','balanced_accuracy']:
            vals=np.array([float(r[metric]) for r in records])
            row[metric+'_mean']=float(vals.mean())
            row[metric+'_sd']=float(vals.std(ddof=1)) if len(vals)>1 else 0.
        summary.append(row)
    write_csv(destination/'summary.csv',summary)
    accuracy=[float(r['heldout_accuracy']) for r in utility]
    lines=['# Measured benchmark report','',
           f"Completed {len(utility)} target conditions with {len(cfg['seeds'])} seeds; {len(rows)} attack/condition measurements.",
           f"Held-out classification accuracy ranged from {min(accuracy):.1%} to {max(accuracy):.1%}.",
           f"Each condition uses {cfg['eval_size']} evaluation members and {cfg['eval_size']} nonmembers; FPR resolution is {1/cfg['eval_size']:.3%}.",'',
           'These are measured results, including negative findings. A small sample pilot does not establish superiority at low FPR.',
           'Different access assumptions apply to black-box, gradient and hybrid attacks. All metrics use the same target/candidates within a condition.','',
           '| σ | C | B | Attack | AUC mean ± seed SD | TPR@1% mean |',
           '| --- | --- | --- | --- | --- | --- |']
    for r in summary:
        lines.append(f"| {r['sigma']:g} | {r['clip']:g} | {r['batch_size']} | {r['attack']} | {r['auc_mean']:.3f} ± {r['auc_sd']:.3f} | {r['tpr_at_1pct_fpr_mean']:.3f} |")
    lines += ['','![Attack comparison](figures/attack_comparison.png)','',
              '![Hybrid noise sweep](figures/hybrid_noise_sweep.png)','',
              'See `summary.csv`, `metrics.csv`, `utility.csv`, and validation tables for the complete measured results.',
              'Only aggregate statistics and figures are exported. Raw sample scores, split IDs, checkpoints and trajectories remain local.',
              'The SHA256 manifest below covers the exported files, not the raw local run.','']
    (destination/'README.md').write_text('\n'.join(lines))
    write_json(destination/'manifest.json',{str(p.relative_to(destination)):digest(p) for p in sorted(destination.rglob('*')) if p.is_file()})
    return summary
