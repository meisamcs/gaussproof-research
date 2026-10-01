"""Run-clustered descriptive analysis of repeated one-run audits."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gaussproof.multi_contributor import write_csv
from gaussproof.one_run_audit import paper_epsilon_lower


METHODS = ("nasr_dot", "centered_dot", "gaussian_mixture", "sparse",
           "joint_trajectory")


def analyze(rows, repetitions=5000, seed=2207):
    for row in rows:
        if not row.get("epsilon_lower_paper"):
            row["epsilon_lower_paper"] = paper_epsilon_lower(
                int(row["correct"]), int(row["guesses"]), int(row["canaries"]),
                float(row["delta"]), float(row["per_test_alpha"]))
    rng = np.random.default_rng(seed)
    summary, paired = [], []
    present = {row["method"] for row in rows}
    if not {"nasr_dot", "gaussian_mixture", "sparse"} <= present:
        raise ValueError("Required audit comparators missing")
    methods = tuple(method for method in METHODS if method in present)
    for sigma in sorted({float(row["sigma"]) for row in rows}):
        group = [r for r in rows if float(r["sigma"]) == sigma]
        runs = sorted({int(r["repetition"]) for r in group})
        if len(group) != len(runs) * len(methods):
            raise ValueError("Incomplete method/run grid")
        index = rng.integers(len(runs), size=(repetitions, len(runs)))
        def vector(method, key):
            by_run = {int(r["repetition"]): float(r[key]) for r in group
                      if r["method"] == method}
            if set(by_run) != set(runs):
                raise ValueError("Incomplete audit method")
            return np.asarray([by_run[run] for run in runs])
        for method in methods:
            eps = vector(method, "epsilon_lower_paper")
            for metric in ("correct", "guess_accuracy", "auc", "epsilon_lower_paper",
                           "epsilon_lower_conservative"):
                values = vector(method, metric)
                low, high = np.quantile(values[index].mean(axis=1), [.025, .975])
                summary.append(dict(sigma=sigma, method=method, metric=metric,
                    mean=float(values.mean()), ci_low=float(low), ci_high=float(high),
                    runs=len(runs)))
            summary.append(dict(sigma=sigma, method=method,
                metric="runs_with_positive_paper_bound", mean=float(np.count_nonzero(eps)),
                ci_low=float("nan"), ci_high=float("nan"), runs=len(runs)))
        contenders = (("joint_trajectory", ("nasr_dot", "gaussian_mixture", "sparse"))
                      if "joint_trajectory" in methods else
                      ("sparse", ("nasr_dot", "gaussian_mixture")))
        for reference in contenders[1]:
            for metric in ("correct", "auc"):
                difference = vector(contenders[0], metric) - vector(reference, metric)
                low, high = np.quantile(difference[index].mean(axis=1), [.025, .975])
                paired.append(dict(sigma=sigma, metric=metric, first=contenders[0],
                    second=reference, mean_gain=float(difference.mean()),
                    ci_low=float(low), ci_high=float(high), runs=len(runs)))
    return summary, paired


def plot(summary, report):
    sigmas = sorted({r["sigma"] for r in summary})
    methods = tuple(method for method in METHODS if any(
        row["method"] == method for row in summary))
    colors = dict(nasr_dot="#666666", centered_dot="#65a7a2",
                  gaussian_mixture="#1d7790", sparse="#dc7339",
                  joint_trajectory="#8a53ad")
    labels = dict(nasr_dot="Nasr clipped-gradient dot product",
                  centered_dot="Public-background residual",
                  gaussian_mixture="q-aware Gaussian mixture",
                  sparse="Per-round sparse decoder",
                  joint_trajectory="Joint trajectory patch")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for axis, metric, title in zip(axes, ("correct", "auc"),
                                   ("Correct guesses of 32 fixed decisions", "Canary IN/OUT AUC")):
        width = .8 / len(methods)
        for j, method in enumerate(methods):
            cells = [next(r for r in summary if r["sigma"] == sigma
                          and r["method"] == method and r["metric"] == metric)
                     for sigma in sigmas]
            values = [r["mean"] for r in cells]
            errors = [[r["mean"] - r["ci_low"] for r in cells],
                      [r["ci_high"] - r["mean"] for r in cells]]
            axis.bar(np.arange(len(sigmas)) + (j - (len(methods)-1)/2) * width,
                     values, width,
                     yerr=errors, capsize=2.5, color=colors[method], label=labels[method])
        axis.axhline(16 if metric == "correct" else .5, ls="--", lw=1,
                     color="black", alpha=.7)
        axis.set_xticks(np.arange(len(sigmas)), [f"σ={sigma:g}" for sigma in sigmas])
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylim(0, 32)
    axes[1].set_ylim(0, 1)
    handles, text = axes[0].get_legend_handles_labels()
    fig.legend(handles, text, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(.5, -.02))
    fig.tight_layout(rect=(0, .09, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(report / f"one_run_audit.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = Path(args.report)
    with (report / "one_run_audit.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary, paired = analyze(rows)
    write_csv(report / "one_run_audit.csv", rows)
    write_csv(report / "summary_with_ci.csv", summary)
    write_csv(report / "paired_differences.csv", paired)
    plot(summary, report)
    (report / "analysis.json").write_text(json.dumps(dict(
        bootstrap_repetitions=5000, clustered_unit="one independent training run",
        primary_score=("joint_trajectory" if any(r["method"] == "joint_trajectory"
                                             for r in rows) else "sparse"),
        references=["nasr_dot", "gaussian_mixture", "sparse"],
        audit_inference="paper Corollary 5.4 per-run lower bound, Bonferroni across all reported methods only; no best-run selection",
        confidence="descriptive 95% run-bootstrap intervals; not per-run DP lower bounds"), indent=2))
    print(f"summarized {len(rows)} method-run rows", flush=True)


if __name__ == "__main__":
    main()
