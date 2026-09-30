"""Render the matched access-stress experiment as publication-ready figures."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = Path(args.report)
    with (report / "summary.csv").open(newline="") as stream:
        rows = [{k: (v if k in ("scenario", "method") else float(v))
                 for k, v in row.items()} for row in csv.DictReader(stream)]
    order = ["full", "every_4_releases", "every_16_releases",
             "stale_4_rounds", "stale_16_rounds",
             "assume_sigma_3", "assume_sigma_5"]
    labels = ["All updates", "Every 4th", "Every 16th", "Stale 4",
              "Stale 16", "Assume σ=3", "Assume σ=5"]
    lookup = {(r["scenario"], r["method"]): r for r in rows}
    if set(order) != {r["scenario"] for r in rows}:
        raise ValueError("Expected the predeclared seven access conditions")
    x = np.arange(len(order))
    colors = {"gaussproof": "#155e75", "alignment": "#bb6527"}
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    for method, offset, title in (("gaussproof", -.15, "GAUSSPROOF mixture"),
                                  ("alignment", .15, "Linear alignment")):
        selected = [lookup[name, method] for name in order]
        values = np.asarray([r["auc"] for r in selected])
        below = values - np.asarray([r["auc_ci_low"] for r in selected])
        above = np.asarray([r["auc_ci_high"] for r in selected]) - values
        ax.errorbar(x + offset, values, yerr=[below, above], fmt="o", markersize=6,
                    capsize=2, linewidth=1.5, color=colors[method], label=title)
    endpoint = lookup["full", "endpoint_loss"]["auc"]
    ax.axhline(endpoint, color="#777777", linestyle=":", linewidth=1.25,
               label=f"Final loss ({endpoint:.3f})")
    strong_path = report / "strong_endpoint_comparison.csv"
    if strong_path.exists():
        with strong_path.open(newline="") as stream:
            strong = list(csv.DictReader(stream))
        lira = float(next(row["auc"] for row in strong if row["method"] == "lira"))
        ax.axhline(lira, color="#5d4b8c", linestyle="--", linewidth=1.5,
                   label=f"Endpoint LiRA ({lira:.3f})")
    ax.axhline(.5, color="#999999", linewidth=.8)
    ax.set_xticks(x, labels, rotation=24, ha="right")
    ax.set_ylabel("Held-out membership AUC")
    ax.set_ylim(.45, 1)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.24))
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(report / f"access_robustness.{extension}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    selected = [lookup[name, "gaussproof"] for name in order]
    gain = np.asarray([r["gain_vs_alignment"] for r in selected])
    below = gain - np.asarray([r["gain_ci_low"] for r in selected])
    above = np.asarray([r["gain_ci_high"] for r in selected]) - gain
    ax.errorbar(x, gain, yerr=[below, above], fmt="o", color=colors["gaussproof"],
                capsize=2, linewidth=1.5)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_xticks(x, labels, rotation=24, ha="right")
    ax.set_ylabel("GAUSSPROOF − alignment AUC")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(report / f"gain_over_alignment.{extension}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
