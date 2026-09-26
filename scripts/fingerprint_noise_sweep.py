"""Fixed-access noise sweep on real MNIST gradient templates, no pixel inversion.

Simulates exact low-dimensional Gaussian sufficient statistics, not full DP-SGD.
The projected-statistic simulation is distributionally equivalent to full isotropic
Gaussian releases for each detector. Cross-method noises need not be jointly coupled;
no paired cross-method confidence intervals are claimed.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import norm,chi2,ncx2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gaussproof.fingerprint_bounds import detection_power,minimum_sigma,posterior_probability,binary_mse_lower_bound
from gaussproof.models import conservative_epsilon
from gaussproof.io import write_csv,write_json
from gaussproof.plots import style,save,plt
from matplotlib.ticker import NullLocator


def run(source,output,trials=20000):
    source=Path(source);output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Output must be empty')
    output.mkdir(parents=True)
    cfg=json.loads((source/'config.json').read_text());k=len(cfg['checkpoint_steps']);clip=cfg['clip']
    repeats=16;sigmas=[.5,1.,2.,4.,8.,16.,32.,64.,128.];alpha=.01;beta=.05
    rows=[];bounds=[];hashes={};dimension=None
    for seed in cfg['seeds']:
        path=source/f'seed{seed}'/'ground_truth.npz';hashes[str(path.relative_to(source))]=hashlib.sha256(path.read_bytes()).hexdigest()
        data=np.load(path);templates=data['gradients'].astype(float);labels=data['labels']
        dimension=templates.shape[1];rng=np.random.default_rng(seed+700)
        for i,h in enumerate(templates):
            energy=float(h@h);length=np.sqrt(energy)
            if length<=1e-10:raise ValueError('Zero template requires a separate no-signal case')
            same=np.flatnonzero(labels==labels[i]);wrong=same[(np.flatnonzero(same==i)[0]+1)%len(same)]
            hwrong=templates[wrong];hswap=h.reshape(k,-1)[::-1].reshape(-1)
            corr_wrong=float(h@hwrong/(length*np.linalg.norm(hwrong)))
            corr_swap=float(h@hswap/(length*np.linalg.norm(hswap)))
            # Shared standard-normal draws across sigma reduce Monte Carlo trend noise.
            z0=rng.standard_normal(trials);z1=rng.standard_normal(trials)
            zw0=rng.standard_normal(trials);zw1=rng.standard_normal(trials)
            zs0=rng.standard_normal(trials);zs1=rng.standard_normal(trials)
            zo0=rng.standard_normal(trials);zo1=rng.standard_normal(trials)
            bounds.append(dict(seed=seed,template_index=i,energy=energy,
                minimum_sigma_for_tpr_5pct_at_fpr_1pct=float(minimum_sigma(energy,repeats,clip,alpha,beta))))
            for sigma in sigmas:
                tau=sigma*clip/np.sqrt(repeats);mu=length/tau
                score1=z1+mu;threshold=norm.isf(alpha)
                p0=posterior_probability(z0,mu);p1=posterior_probability(score1,mu)
                mse=float(.5*(np.mean(p0*p0)+np.mean((1-p1)**2)))
                row=dict(seed=seed,template_index=i,sigma=sigma,repeats_per_checkpoint=repeats,
                    total_releases=k*repeats,template_energy=energy,mu=mu,
                    epsilon_upper_bound=conservative_epsilon(k*repeats,sigma,1e-5),
                    matched_tpr=float(np.mean(score1>threshold)),matched_fpr=float(np.mean(z0>threshold)),
                    optimal_tpr=float(detection_power(mu,alpha)),
                    wrong_same_label_tpr=float(np.mean(zw1+mu*corr_wrong>threshold)),
                    wrong_same_label_fpr=float(np.mean(zw0>threshold)),
                    reversed_checkpoints_tpr=float(np.mean(zs1+mu*corr_swap>threshold)),
                    reversed_checkpoints_fpr=float(np.mean(zs0>threshold)),
                    orthogonal_tpr=float(np.mean(zo1>threshold)),orthogonal_fpr=float(np.mean(zo0>threshold)),
                    energy_only_tpr=float(ncx2.sf(chi2.isf(alpha,dimension),dimension,mu*mu)),
                    amplitude_mse=mse,amplitude_mse_lower_bound=float(binary_mse_lower_bound(mu)),
                    prior_only_amplitude_mse=.25,trajectory_mse_per_checkpoint=mse*energy/k,
                    prior_only_trajectory_mse_per_checkpoint=.25*energy/k,
                    raw_mean_expected_mse_per_checkpoint=dimension*tau*tau/k)
                rows.append(row)
        print(f'Completed fingerprint sweep seed {seed}',flush=True)
    write_csv(output/'per_template.csv',rows);write_csv(output/'noise_thresholds.csv',bounds)
    summary=[]
    excluded={'seed','template_index','sigma','repeats_per_checkpoint','total_releases'}
    for sigma in sigmas:
        selected=[v for v in rows if v['sigma']==sigma]
        record=dict(sigma=sigma,templates=len(selected),trials_per_hypothesis_per_template=trials)
        for name in rows[0]:
            if name not in excluded:record[name]=float(np.mean([v[name] for v in selected]))
        summary.append(record)
    write_csv(output/'summary.csv',summary)
    write_json(output/'provenance.json',dict(source_hashes=hashes,seed_offset=700,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        theory_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'gaussproof/fingerprint_bounds.py').read_bytes()).hexdigest(),
        sigmas=sigmas,repeats_per_checkpoint=repeats,checkpoints=k,dimension=dimension,trials=trials,
        observation='Exact projected Gaussian statistics of fixed public-checkpoint templates; no private training',
        hypotheses='H0: zero target contribution, H1: known template present in every release; background known/subtracted',
        prior='equal probability absence/presence',delta=1e-5,
        epsilon='Conservative replace-one full-vector bound using sensitivity 2C, not exact epsilon'))
    style();fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for name,label,ls in [('matched_tpr','Matched fingerprint','-'),('optimal_tpr','Exact optimal power',':'),
        ('wrong_same_label_tpr','Wrong same-label fingerprint','--'),('reversed_checkpoints_tpr','Reversed checkpoints','--'),
        ('energy_only_tpr','Energy only (analytic)','-'),('orthogonal_tpr','Orthogonal control',':')]:
        axes[0].plot(sigmas,[v[name] for v in summary],ls,marker='.',label=label)
    axes[0].axhline(alpha,color='.5',ls=':');axes[0].set(ylabel='Detection probability at 1% FPR',ylim=(0,1.02))
    axes[0].legend(fontsize=6)
    for name,label in [('amplitude_mse','Optimal posterior denoiser (MC)'),
                       ('amplitude_mse_lower_bound','Proved lower bound, any estimator'),
                       ('prior_only_amplitude_mse','No-release prior only')]:
        axes[1].plot(sigmas,[v[name] for v in summary],'o-',markersize=3,label=label)
    axes[1].set(ylabel='Mean squared fingerprint-amplitude error',ylim=(0,.27));axes[1].legend(fontsize=6)
    axes[2].plot(sigmas,[v['epsilon_upper_bound'] for v in summary],'o-');axes[2].set(yscale='log',ylabel='Conservative ε bound (δ=10⁻⁵)')
    for ax in axes:
        ax.set(xscale='log',xlabel='Noise multiplier σ (release count fixed)');ax.set_xticks([.5,2,8,32,128],['0.5','2','8','32','128'])
        ax.xaxis.set_minor_locator(NullLocator());ax.grid(alpha=.15)
    fig.suptitle(f'Accessible fingerprints: more noise reduces detection and worsens optimal denoising · {k*repeats} releases')
    save(fig,output/'fingerprint_noise_tradeoff')
    table='| σ | ε upper bound | Matched TPR | Optimal TPR | Denoising MSE |\n| --- | ---: | ---: | ---: | ---: |\n'
    for v in summary:table+=f"| {v['sigma']:g} | {v['epsilon_upper_bound']:.3f} | {v['matched_tpr']:.4f} | {v['optimal_tpr']:.4f} | {v['amplitude_mse']:.4f} |\n"
    maxbound=max(v['minimum_sigma_for_tpr_5pct_at_fpr_1pct'] for v in bounds)
    (output/'README.md').write_text(f'''# Fingerprint access held fixed: noise versus recoverability

Used {len(bounds)} real clipped MNIST gradient templates from {len(cfg['seeds'])} public CNN seeds and {k} public checkpoints. Access, templates, labels, hypotheses, and {repeats} releases per checkpoint are fixed across all noise levels. Each template has {trials:,} Monte Carlo observations per hypothesis per noise level. Computation uses exact Gaussian projected sufficient statistics, not costly generation of every full noise vector. This is a controlled Gaussian experiment, not dynamic DP-SGD or a new empirical dataset of independent people.

The candidate is a known audit canary: its fingerprint is available under BOTH hypotheses; only whether it contributed is unknown. The background is known/subtracted. H1 assumes presence at every release, with no hidden sampling. This isolates the strongest matched-fingerprint detector for a specified simple Gaussian alternative. Wrong same-label templates and reversed checkpoint order may retain signal because gradients are correlated; they are not automatically chance-level controls. Orthogonal directions have no signal. The energy-only detector's power is computed analytically from a noncentral chi-square distribution.

{table}
Observed FPR is saved separately. Thresholds are analytic null thresholds, never selected on evaluation labels. Each noise level reuses standard-normal draws; different noise conditions are paired, not independent replications. Exact theoretical power is included to distinguish genuine monotonicity from Monte Carlo fluctuations.

## A bound that can actually be proved

For a fixed concatenated fingerprint H, let E=||H||², tau=σC/√R, and μ=√E/tau. The normalized matched score is N(0,1) under absence and N(μ,1) under presence. By the likelihood-ratio test, the optimal power at FPR α is Φ(μ−Φ⁻¹(1−α)).

For α < β, requiring optimal TPR ≤ β is equivalent, in THIS model, to

σ ≥ √(R E) / [ C (Φ⁻¹(1−α) + Φ⁻¹(β)) ].

With α=0.01 and β=0.05, the denominator's quantile factor is approximately 0.6815. Across the tested bank the largest required noise multiplier is **{maxbound:.4f}**. This is an exact model-specific noise floor for the stated detection cap. It is not a universal lower bound for arbitrary DP-SGD attacks, unknown backgrounds, or arbitrary priors.

For equal-prior binary amplitude a∈{{0,1}}, the Bayes posterior mean minimizes squared error. Its posterior is sigmoid(μZ−μ²/2). ANY estimator's prior-averaged amplitude MSE obeys

MSE ≥ ½ Φ(−μ/2).

Proof: conditional squared-error risk is minimized by p, with risk p(1−p) ≥ ½ min(p,1−p); the optimal equal-prior Gaussian classification error is Φ(−μ/2). Multiply by E/K for the corresponding per-checkpoint trajectory MSE bound. This is a Bayes-risk bound, not a pointwise guarantee for each possible amplitude. At infinite noise it approaches the no-release risk 1/4. The independent-background-free, fixed-template setup is essential.

Increasing noise tightens the reported conservative privacy bound and makes the optimal attack worse. Empirical failure of one algorithm alone would not prove this; the Gaussian likelihood calculation supplies the mathematical statement. A minimum noise requirement corresponds to a maximum allowable privacy-budget bound for fixed sensitivity/releases, not a minimum ε needed for privacy. Epsilon in the table is an upper bound from replace-one Gaussian composition, not an estimate of exact leakage.

The theory is a specialization of Gaussian hypothesis testing; it is not a new GAUSSPROOF theorem. A research contribution would establish comparable guarantees for changing, imperfectly accessible fingerprints and hidden participation, or demonstrate a useful efficient decoder in that setting.

Reference: [Dong, Roth, Su, Gaussian Differential Privacy](https://arxiv.org/abs/1905.02383).

Reproduce after the reconstruction run:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.cache/matplotlib python scripts/fingerprint_noise_sweep.py --source runs/reconstruction --output reports/fingerprint_noise
```
''')
    write_json(output/'completion.json',dict(status='complete',templates=len(bounds),conditions=len(summary),
        simulated_hypothesis_observations=len(rows)*2*trials,matched_false_alarm_max_deviation=max(abs(v['matched_fpr']-alpha) for v in summary),
        max_mean_power_error=max(abs(v['matched_tpr']-v['optimal_tpr']) for v in summary)))
    write_json(output/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()})
    print(table,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--trials',type=int,default=20000);a=p.parse_args();run(a.source,a.output,a.trials)
