"""Paper-ready figures for the matched clip/noise sweep."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with Path(path).open(newline="") as stream:
        return [{k: (v if k == "method" else float(v)) for k, v in row.items()}
                for row in csv.DictReader(stream)]


def save(fig, path):
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def auc_figure(rows, report, steps, suffix=""):
    clips = sorted({r["clip"] for r in rows})
    methods = [
        ("trajectory_mixture", "GAUSSPROOF", "#bc6c25"),
        ("raw_alignment", "Gradient alignment", "#7a6191"),
        ("endpoint_logprob_difference", "Final checkpoint", "#345c7d"),
        ("last_update", "Last update", "#85964b"),
    ]
    fig, axes = plt.subplots(1, len(clips), figsize=(4.35*len(clips), 3.55),
                             sharey=True, squeeze=False)
    for ax, clip in zip(axes[0], clips):
        for method, label, color in methods:
            part = sorted((r for r in rows if r["clip"] == clip and
                           r["steps"] == steps and r["method"] == method),
                          key=lambda r:r["epsilon_upper_no_amplification"])
            x = np.asarray([r["epsilon_upper_no_amplification"] for r in part])
            y = np.asarray([r["auc"] for r in part])
            lo = np.asarray([r["auc_ci_low"] for r in part])
            hi = np.asarray([r["auc_ci_high"] for r in part])
            ax.plot(x,y,marker="o",lw=1.6,color=color,label=label)
            ax.fill_between(x,lo,hi,color=color,alpha=.11)
        ax.set_xscale("log")
        ax.set_ylim(.4,1.01)
        ax.axhline(.5,color="#777",ls="--",lw=.8)
        ax.grid(alpha=.15)
        ax.set_title(f"Clip $C={clip:g}$")
        ax.set_xlabel(r"Conservative $\epsilon$ upper bound ($\delta=10^{-5}$)")
    axes[0,0].set_ylabel("Held-out AUC")
    axes[0,-1].legend(frameon=False,fontsize=7,loc="lower right")
    fig.suptitle(f"Fixed natural pair, q=0.5, T={steps}; 80 held-out pairs",
                 fontsize=11)
    save(fig,report/f"auc_vs_epsilon_T{steps}{suffix}")


def tradeoff_figure(rows, report, steps, suffix=""):
    clips = sorted({r["clip"] for r in rows})
    sigmas = sorted({r["sigma"] for r in rows})
    lookup = {(r["clip"],r["sigma"],r["method"]):r for r in rows
              if r["steps"] == steps}
    diff = np.asarray([[lookup[c,s,"trajectory_mixture"]["auc"]-
                        lookup[c,s,"endpoint_logprob_difference"]["auc"]
                        for s in sigmas] for c in clips])
    utility = np.asarray([[lookup[c,s,"trajectory_mixture"]["mean_public_accuracy"]
                           for s in sigmas] for c in clips])
    clipped = np.asarray([[lookup[c,s,"trajectory_mixture"]["mean_clipped_fraction"]
                           for s in sigmas] for c in clips])
    fig, axes = plt.subplots(1,3,figsize=(10.6,3.35))
    grids = [(diff,"Trajectory minus endpoint AUC","coolwarm",-.2,.2),
             (utility,"Public accuracy","viridis",0.,1.),
             (clipped,"Fraction of batch gradients clipped","magma",0.,1.)]
    for ax,(array,title,cmap,lo,hi) in zip(axes,grids):
        im=ax.imshow(array,cmap=cmap,vmin=lo,vmax=hi,aspect="auto")
        fmt = ".3f" if title.startswith("Trajectory") else ".2f"
        for i in range(len(clips)):
            for j in range(len(sigmas)):
                red, green, blue, _ = im.cmap(im.norm(array[i,j]))
                luminance = .2126*red + .7152*green + .0722*blue
                ax.text(j,i,format(array[i,j],fmt),ha="center",va="center",
                        fontsize=8,color="black" if luminance>.5 else "white")
        ax.set_xticks(range(len(sigmas)),[f"{s:g}" for s in sigmas])
        ax.set_yticks(range(len(clips)),[f"{c:g}" for c in clips])
        ax.set_xlabel("Noise multiplier $\\sigma$")
        ax.set_ylabel("Clip $C$")
        ax.set_title(title,fontsize=9)
        fig.colorbar(im,ax=ax,fraction=.047,pad=.03)
    fig.suptitle(f"Matched clip/noise grid at T={steps}",fontsize=11)
    save(fig,report/f"clip_noise_tradeoff_T{steps}{suffix}")


def low_epsilon_figure(rows, tail, report, steps):
    series = [r for r in rows + tail if r["clip"] == 1. and r["steps"] == steps]
    fig, ax = plt.subplots(figsize=(5.5,3.8))
    for method,label,color in (
        ("trajectory_mixture","GAUSSPROOF","#bc6c25"),
        ("raw_alignment","Gradient alignment","#7a6191"),
        ("endpoint_logprob_difference","Final checkpoint","#345c7d"),
    ):
        part = sorted((r for r in series if r["method"] == method),
                      key=lambda r:r["epsilon_upper_no_amplification"])
        x = np.asarray([r["epsilon_upper_no_amplification"] for r in part])
        y = np.asarray([r["auc"] for r in part])
        ax.plot(x,y,marker="o",lw=1.7,color=color,label=label)
        ax.fill_between(x,[r["auc_ci_low"] for r in part],
                        [r["auc_ci_high"] for r in part],color=color,alpha=.12)
    ax.set_xscale("log")
    ax.set_ylim(.4,1.01)
    ax.axhline(.5,color="#777",ls="--",lw=.8)
    ax.grid(alpha=.15)
    ax.set_xlabel(r"Conservative $\epsilon$ upper bound ($\delta=10^{-5}$)")
    ax.set_ylabel("Held-out AUC")
    ax.set_title(f"Clip $C=1$, {steps} releases; high-noise extension")
    ax.legend(frameon=False,fontsize=8)
    save(fig,report/f"low_epsilon_extension_T{steps}")


def utility_rescue_figure(rows, tail, rescue, report, steps):
    fig, axes = plt.subplots(2, 3, figsize=(11.7, 5.7),
                             gridspec_kw={"height_ratios": [1.6, 1]},
                             sharex="col")
    for column, sigma in enumerate((32., 64., 128.)):
        baseline = [r for r in rows + tail if r["clip"] == 1. and
                    r["sigma"] == sigma and r["steps"] == steps]
        extra = [r for r in rescue if r["clip"] == 1. and
                 r["sigma"] == sigma and r["steps"] == steps]
        if not baseline or not extra:
            raise ValueError(f"Missing utility control for sigma={sigma:g}, T={steps}")
        batch = int(extra[0]["batch_size"])
        groups = (baseline, extra)
        for offset, method, label, color in (
            (-.22,"trajectory_mixture","GAUSSPROOF","#bc6c25"),
            (0.,"raw_alignment","Gradient alignment","#7a6191"),
            (.22,"endpoint_logprob_difference","Final checkpoint","#345c7d")):
            chosen = [next(r for r in group if r["method"] == method)
                      for group in groups]
            values = np.asarray([r["auc"] for r in chosen])
            errors = np.asarray([[max(0.,r["auc"]-r["auc_ci_low"]) for r in chosen],
                                 [max(0.,r["auc_ci_high"]-r["auc"]) for r in chosen]])
            axes[0,column].bar(np.arange(2)+offset,values,width=.2,
                               color=color,label=label)
            axes[0,column].errorbar(np.arange(2)+offset,values,yerr=errors,
                                    fmt="none",ecolor="#333",elinewidth=.8,capsize=2)
        accuracy = [next(r for r in group if r["method"] ==
                         "trajectory_mixture")["mean_public_accuracy"]
                    for group in groups]
        axes[1,column].bar(range(2),accuracy,width=.55,color="#4a7c59")
        for i,value in enumerate(accuracy):
            axes[1,column].text(i,value+.025,f"{value:.2f}",ha="center",fontsize=8)
        axes[0,column].set_title(
            f"$\\sigma={sigma:g}$; conservative $\\epsilon$≤"
            f"{baseline[0]['epsilon_upper_no_amplification']:.2f}")
        axes[0,column].set_ylim(.4,1.03)
        axes[0,column].axhline(.5,color="#777",ls="--",lw=.7)
        axes[0,column].grid(axis="y",alpha=.15)
        axes[1,column].set_ylim(0,1.05)
        axes[1,column].set_xticks(range(2),["B=8",f"B={batch}"])
        axes[1,column].grid(axis="y",alpha=.15)
    axes[0,0].set_ylabel("Held-out AUC")
    axes[1,0].set_ylabel("Public accuracy")
    axes[0,2].legend(frameon=False,fontsize=7,loc="upper right")
    fig.suptitle(f"Larger batches preserve CNN utility at T={steps}",fontsize=11)
    save(fig,report/f"utility_rescue_T{steps}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--report",required=True)
    args=parser.parse_args()
    report=Path(args.report)
    rows=read(report/"summary.csv")
    tail=read(report/"low_epsilon_tail.csv") if (report/"low_epsilon_tail.csv").exists() else []
    weak=read(report/"weak_clipping_control.csv") if (report/"weak_clipping_control.csv").exists() else []
    rescue=read(report/"utility_rescue.csv") if (report/"utility_rescue.csv").exists() else []
    for steps in sorted({int(r["steps"]) for r in rows}):
        auc_figure(rows,report,steps)
        tradeoff_figure(rows,report,steps)
        if weak:
            auc_figure(rows+weak,report,steps,"_with_C16")
            tradeoff_figure(rows+weak,report,steps,"_with_C16")
        if tail:
            low_epsilon_figure(rows,tail,report,steps)
        if rescue:
            utility_rescue_figure(rows,tail,rescue,report,steps)


if __name__=="__main__":
    main()
