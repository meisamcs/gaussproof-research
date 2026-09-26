"""Vector PDF plus 300-dpi PNG; plots use measured results only."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .metrics import roc

COLORS = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#AA4499', '#882255']


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42, 'ps.fonttype':42,
                         'axes.prop_cycle':plt.cycler(color=COLORS), 'savefig.bbox':'tight'})


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=300)
    plt.close(fig)


def roc_plot(scores, labels, path, title):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained')
    for name, values in scores.items():
        f, t, _ = roc(values, labels)
        axes[0].step(f, t, where='post', label=name)
        axes[1].step(f, t, where='post', label=name)
    for ax in axes:
        ax.plot([0,1], [0,1], ':', color='0.5', lw=1)
        ax.set(xlabel='False-positive rate', ylabel='True-positive rate', ylim=(0,1))
        ax.grid(alpha=.15)
    axes[0].set_xlim(0,1)
    axes[1].set_xlim(0,.05)
    axes[1].set_title('Low-FPR detail (empirical steps)')
    axes[0].legend(fontsize=7, loc='lower right')
    fig.suptitle(title, fontsize=10)
    save(fig, path)


def summary_plots(rows, directory):
    style()
    directory = Path(directory)
    attacks = list(dict.fromkeys(r['attack'] for r in rows))
    conditions = sorted({(r['sigma'],r['clip'],r['batch_size']) for r in rows})
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax, metric, label in zip(axes, ['auc','tpr_at_1pct_fpr'], ['ROC AUC','TPR at FPR ≤ 1%']):
        # Plot each condition separately; never average different privacy regimes.
        width = .8/len(conditions)
        upper = .1
        for i, condition in enumerate(conditions):
            vals = [[r[metric] for r in rows if r['attack']==a and
                     (r['sigma'],r['clip'],r['batch_size'])==condition] for a in attacks]
            means = [np.mean(v) for v in vals]
            sd = [np.std(v,ddof=1) if len(v)>1 else 0 for v in vals]
            upper = max(upper, max(m+s for m,s in zip(means,sd))*1.2)
            sig,c,b = condition
            ax.bar(np.arange(len(attacks))+(i-(len(conditions)-1)/2)*width, means,
                   width, yerr=sd, capsize=2, label=f'σ={sig:g}, C={c:g}, B={b}')
        ax.set(xticks=np.arange(len(attacks)), xticklabels=attacks, ylabel=label, ylim=(0,1.05))
        ax.tick_params(axis='x', rotation=55)
        ax.grid(axis='y',alpha=.15)
        if metric == 'tpr_at_1pct_fpr': ax.set_ylim(0, min(1.05,upper))
    axes[0].legend(fontsize=6)
    fig.suptitle('Measured attack performance; error bars = SD across seeds',fontsize=10)
    save(fig,directory/'attack_comparison')
    for batch in sorted({r['batch_size'] for r in rows}):
        sigmas = sorted({r['sigma'] for r in rows if r['batch_size']==batch})
        clips = sorted({r['clip'] for r in rows if r['batch_size']==batch})
        fig, axes = plt.subplots(1,2,figsize=(8,3.6),layout='constrained')
        for ax, metric, label in zip(axes,['auc','tpr_at_1pct_fpr'],['Δ AUC','Δ TPR at 1% FPR']):
            grid = np.full((len(clips),len(sigmas)),np.nan)
            for i,c in enumerate(clips):
                for j,s in enumerate(sigmas):
                    r = [v for v in rows if v['batch_size']==batch and v['sigma']==s and v['clip']==c]
                    gp = [v[metric] for v in r if v['attack']=='gaussproof']
                    re = [v[metric] for v in r if v['attack']=='rero']
                    if gp and re: grid[i,j] = np.mean(gp)-np.mean(re)
            lim = max(.05, float(np.nanmax(abs(grid))))
            im = ax.imshow(grid,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim)
            ax.set(xticks=np.arange(len(sigmas)),xticklabels=sigmas,yticks=np.arange(len(clips)),
                   yticklabels=clips,xlabel='Noise multiplier σ',ylabel='Clipping norm C',title=label)
            for (i,j),v in np.ndenumerate(grid):
                if np.isfinite(v): ax.text(j,i,f'{v:+.3f}',ha='center',va='center',fontsize=8)
            fig.colorbar(im,ax=ax,shrink=.8)
        fig.suptitle(f'GAUSSPROOF minus RERO-style alignment · batch {batch}',fontsize=10)
        save(fig,directory/f'blindspot_B_{batch}')
    fig,ax=plt.subplots(figsize=(6.5,3.8),layout='constrained')
    for attack in ['rmia','gaussproof','hybrid']:
        for clip,batch in sorted({(r['clip'],r['batch_size']) for r in rows}):
            ss=sorted({r['sigma'] for r in rows if r['clip']==clip and r['batch_size']==batch})
            means=[np.mean([r['auc'] for r in rows if r['attack']==attack and r['sigma']==s and
                           r['clip']==clip and r['batch_size']==batch]) for s in ss]
            ax.plot(ss,means,'o-',label=f'{attack}, C={clip:g}, B={batch}')
    ax.set(xlabel='Noise multiplier σ',ylabel='ROC AUC',ylim=(0,1))
    ax.axhline(.5,color='0.5',ls=':'); ax.legend(fontsize=7); ax.grid(alpha=.15)
    save(fig,directory/'hybrid_noise_sweep')
