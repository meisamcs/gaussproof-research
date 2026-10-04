"""Render aggregate LiRA fusion and fixed-pair audit figures."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gaussproof.metrics import roc


def read(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, output):
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix("." + suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fusion_plot(report):
    data = [read(report / f"low_fpr_{setting}.csv")
            for setting in ("original_lr", "high_utility")]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True)
    colors = {"LiRA": "#345c7d", "fusion": "#bc6c25"}
    for ax, setting, rows in zip(axes, ("Original LR 0.05", "Higher utility LR 0.005"), data):
        selected = {r["method"]: r for r in rows if r["q"] == "0.5" and
                    r["steps"] == "128" and r["method"] in colors}
        for x, name in enumerate(colors):
            row = selected[name]
            y = float(row["calibrated_holdout_tpr"])
            ax.bar(x, y, width=.58, color=colors[name])
            ax.text(x, y + .012, f"FPR {float(row['calibrated_holdout_fpr']):.1%}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(2), ["LiRA-style", "Fusion"])
        ax.set_ylim(0, .75)
        ax.set_title(setting)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Held-out TPR at calibrated threshold")
    fig.suptitle("MNIST, q=0.5, 128 released updates, 5% calibration FPR", fontsize=11)
    save(fig, report / "lira_fusion_low_fpr")


def audit_plot(report, raw_path):
    raw = read(raw_path)
    test = [r for r in raw if r["role"] == "holdout"]
    y = np.asarray([int(r["world"]) for r in test])
    summary = {r["method"]: r for r in read(report / "fixed_pair_summary.csv")}
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for name, label, color in (
        ("trajectory_mixture", "GAUSSPROOF trajectory", "#ba571f"),
        ("last_update", "Last update", "#557591"),
        ("endpoint_logprob_difference", "Final checkpoint", "#66784d"),
    ):
        scores = np.asarray([float(r[name]) for r in test])
        fpr, tpr, _ = roc(scores, y)
        ax.step(fpr, tpr, where="post", color=color, lw=1.8,
                label=f"{label} (AUC {float(summary[name]['auc']):.3f})")
        ax.scatter([float(summary[name]["holdout_fpr"])],
                   [float(summary[name]["holdout_tpr"])], color=color, s=28, zorder=3)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777", lw=.9)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate",
           xlim=(0, 1), ylim=(0, 1),
           title="Fixed natural MNIST neighbor pair")
    ax.grid(alpha=.15)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    save(fig, report / "fixed_pair_roc")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--audit-run", required=True)
    args = parser.parse_args()
    report = Path(args.report)
    fusion_plot(report)
    audit_plot(report, Path(args.audit_run) / "score_rows.csv")


if __name__ == "__main__":
    main()
