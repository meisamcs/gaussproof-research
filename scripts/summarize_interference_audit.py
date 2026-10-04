"""Paired, run-clustered analysis of the fixed-natural-canary overlap test."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gaussproof.multi_contributor import write_csv


METHODS = ("nasr_dot", "centered_dot", "gaussian_mixture", "sparse",
           "joint_trajectory")
COHORTS = ("low", "high")


def analyze(rows, bootstrap=5000, seed=217):
    runs = sorted({int(row["repetition"]) for row in rows})
    sigmas = sorted({float(row["sigma"]) for row in rows})
    index = {(int(r["repetition"]), r["cohort"], float(r["sigma"]), r["method"]): r
             for r in rows}
    for repetition in runs:
        for cohort in COHORTS:
            for sigma in sigmas:
                for method in METHODS:
                    if (repetition, cohort, sigma, method) not in index:
                        raise ValueError("Incomplete paired audit grid")
    rng = np.random.default_rng(seed)
    resample = rng.integers(len(runs), size=(bootstrap, len(runs)))
    def values(cohort, sigma, method, metric):
        return np.array([float(index[rep, cohort, sigma, method][metric])
                         for rep in runs])
    def summarize(value):
        boot = value[resample].mean(axis=1)
        lo, hi = np.quantile(boot, [.025, .975])
        return float(value.mean()), float(lo), float(hi)
    summary, paired = [], []
    for sigma in sigmas:
        for cohort in COHORTS:
            for method in METHODS:
                for metric in ("correct", "auc", "epsilon_lower_paper",
                               "within_abs_cosine", "all_abs_cosine", "effective_rank"):
                    mean, lo, hi = summarize(values(cohort, sigma, method, metric))
                    summary.append(dict(sigma=sigma, cohort=cohort, method=method,
                                        metric=metric, mean=mean, ci_low=lo, ci_high=hi,
                                        runs=len(runs)))
            for reference in METHODS[:-1]:
                for metric in ("correct", "auc", "epsilon_lower_paper"):
                    delta = values(cohort, sigma, "joint_trajectory", metric) - values(
                        cohort, sigma, reference, metric)
                    mean, lo, hi = summarize(delta)
                    paired.append(dict(sigma=sigma, cohort=cohort, comparison=f"joint-{reference}",
                                       metric=metric, mean_gain=mean, ci_low=lo, ci_high=hi,
                                       runs=len(runs)))
        for reference in METHODS[:-1]:
            for metric in ("correct", "auc"):
                high = values("high", sigma, "joint_trajectory", metric) - values(
                    "high", sigma, reference, metric)
                low = values("low", sigma, "joint_trajectory", metric) - values(
                    "low", sigma, reference, metric)
                mean, lo, hi = summarize(high - low)
                paired.append(dict(sigma=sigma, cohort="high-minus-low",
                                   comparison=f"joint-{reference}", metric=metric,
                                   mean_gain=mean, ci_low=lo, ci_high=hi, runs=len(runs)))
    return summary, paired


def plot(summary, paired, report):
    sigmas = sorted({r["sigma"] for r in summary})
    colors = {"nasr_dot": "#65717a", "centered_dot": "#5a9c70",
              "gaussian_mixture": "#217c92", "sparse": "#de873c",
              "joint_trajectory": "#a24d7c"}
    names = {"nasr_dot": "Clipped-gradient dot product",
             "centered_dot": "Background-centered dot product",
             "gaussian_mixture": "q-aware mixture",
             "sparse": "Per-round sparse decoder",
             "joint_trajectory": "GAUSSPROOF joint score"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
    for axis, cohort in zip(axes, COHORTS):
        for method in METHODS:
            cells = [next(r for r in summary if r["sigma"] == sigma and
                      r["cohort"] == cohort and r["method"] == method and
                      r["metric"] == "correct") for sigma in sigmas]
            x = np.arange(len(sigmas)) + (METHODS.index(method)-2)*.17
            means = np.array([r["mean"] for r in cells])
            error = np.array([[r["mean"]-r["ci_low"] for r in cells],
                              [r["ci_high"]-r["mean"] for r in cells]])
            axis.bar(x, means, .16, yerr=error, capsize=2.5,
                     color=colors[method], label=names[method])
        overlap = next(r["mean"] for r in summary if r["cohort"] == cohort and
                       r["metric"] == "within_abs_cosine")
        axis.set_title(f"{cohort.capitalize()} overlap | mean within-class |cos|={overlap:.2f}")
        axis.set_xticks(np.arange(len(sigmas)), [f"σ={s:g}" for s in sigmas])
        axis.axhline(16, color="black", ls="--", lw=1, alpha=.6)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Correct of 32 fixed audit guesses")
    axes[0].set_ylim(0, 32)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(.5, -.07))
    fig.tight_layout(rect=(0, .15, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(report / f"interference_audit.{extension}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = Path(args.report)
    with (report / "one_run_interference.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary, paired = analyze(rows)
    write_csv(report / "summary_with_ci.csv", summary)
    write_csv(report / "paired_differences.csv", paired)
    plot(summary, paired, report)


if __name__ == "__main__":
    main()
