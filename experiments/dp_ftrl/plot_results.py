"""Create a vector figure from aggregate DP-FTRL audit outputs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_run(path):
    with (path / "summary.csv").open(newline="") as stream:
        rows = {row["attack"]: row for row in csv.DictReader(stream)}
    info = json.loads((path / "metadata.json").read_text())
    return rows, info


def bounds(row):
    return float(row["auc"]), float(row["auc_ci_low"]), float(row["auc_ci_high"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--efficient", type=Path, required=True)
    p.add_argument("--standard", type=Path, required=True)
    p.add_argument("--high-noise", type=Path, required=True)
    p.add_argument("--late", type=Path, required=True)
    p.add_argument("--access", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    runs = {name: read_run(getattr(args, name.replace("-", "_")))
            for name in ("efficient", "standard", "high-noise", "late")}
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.15), constrained_layout=True)
    ax = axes[0]
    rows, info = runs["efficient"]
    prefixes = [16, 32, 64, 128]
    for stem, label, color, marker in (
        ("tree_", "Informed tree likelihood", "#1B596F", "o"),
        ("public_tree_", "Public background", "#AA5D32", "s"),
    ):
        estimates = [bounds(rows[f"{stem}{t}"]) for t in prefixes]
        y = [v[0] for v in estimates]
        errors = [[v[0] - v[1] for v in estimates],
                  [v[2] - v[0] for v in estimates]]
        ax.errorbar(prefixes, y, yerr=errors, marker=marker, color=color,
                    capsize=2, linewidth=1.6, label=label)
    ax.axhline(float(rows["oracle_final_prefix"]["auc"]), linestyle="--",
               color="#747C83", linewidth=1.1, label="Final noisy prefix (oracle background)")
    ax.axhline(float(rows["endpoint_loss"]["auc"]), linestyle=":",
               color="#8F3D58", linewidth=1.3, label="Final-model loss")
    ax.axhline(0.5, color="#BBBBBB", linewidth=.8)
    ax.set(xlabel="Released model checkpoints", ylabel="Membership AUC",
           title="One-time canary, efficient tree, $\\sigma=4$",
           xticks=prefixes, ylim=(0.47, 0.83))
    ax.legend(loc="upper left", fontsize=7.0, frameon=False)

    ax = axes[1]
    cases = [
        ("standard", "Standard tree\n$\\sigma=4$", "#406B90"),
        ("efficient", "Efficient tree\n$\\sigma=4$", "#1B596F"),
        ("late", "Late insertion\n$\\sigma=4$", "#716AA2"),
        ("high-noise", "Efficient tree\n$\\epsilon\\leq1$", "#AA5D32"),
    ]
    for j, (key, label, color) in enumerate(cases):
        run_rows, _ = runs[key]
        value, lo, hi = bounds(run_rows["tree_128"])
        ax.bar(j, value - .5, bottom=.5, width=.68, color=color, alpha=.86)
        ax.errorbar(j, value, yerr=[[value - lo], [hi - value]],
                    color="#1B2530", capsize=3, fmt="none", linewidth=1)
    ax.axhline(.5, color="#555555", linewidth=.8)
    ax.set(xticks=range(len(cases)), xticklabels=[case[1] for case in cases],
           ylabel="Membership AUC", title="Full trajectory under matched access",
           ylim=(.47, .83))
    fig.savefig(args.output / "dp_ftrl_audit.pdf", bbox_inches="tight")
    fig.savefig(args.output / "dp_ftrl_audit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if args.access:
        access_rows, access_info = read_run(args.access)
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.1), constrained_layout=True)
        ax = axes[0]
        series = [(0, "tree_128"), (1, "unknown1_tree_128"),
                  (8, "unknown8_tree_128"), (32, "unknown32_tree_128"),
                  (63, "public_tree_128")]
        estimates = [bounds(access_rows[key]) for _, key in series]
        x = list(range(len(series)))
        y = [v[0] for v in estimates]
        errors = [[v[0] - v[1] for v in estimates],
                  [v[2] - v[0] for v in estimates]]
        ax.errorbar(x, y, yerr=errors, marker="o", color="#1B596F",
                    linewidth=1.7, capsize=3)
        ax.axhline(.5, color="#888888", linewidth=.8)
        ax.set(xticks=x, xticklabels=[str(count) for count, _ in series],
               xlabel="Other batch records unknown to auditor (of 63)",
               ylabel="Membership AUC", ylim=(.48, .76),
               title="Background knowledge determines attack power")
        ax = axes[1]
        identity = [("tree_128", "True canary", "#1B596F"),
                    ("same_class_decoy_tree_128", "Same-digit decoy", "#A9693E"),
                    ("identity_contrast_tree_128", "True minus decoy", "#716AA2")]
        for j, (key, label, color) in enumerate(identity):
            value, lo, hi = bounds(access_rows[key])
            ax.bar(j, value - .5, bottom=.5, color=color, alpha=.86, width=.65)
            ax.errorbar(j, value, yerr=[[value - lo], [hi - value]],
                        fmt="none", color="#1B2530", capsize=3, linewidth=1)
        ax.axhline(.5, color="#555555", linewidth=.8)
        ax.set(xticks=range(len(identity)), xticklabels=[v[1] for v in identity],
               ylabel="Membership AUC", ylim=(.48, .76),
               title="Identity control among same-digit records")
        fig.savefig(args.output / "dp_ftrl_access.pdf", bbox_inches="tight")
        fig.savefig(args.output / "dp_ftrl_access.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    with (args.output / "combined_summary.csv").open("w", newline="") as stream:
        writer = None
        if args.access:
            runs["access"] = (access_rows, access_info)
        for name, (rows, info) in runs.items():
            for row in rows.values():
                merged = {"condition": name, "sigma": info["sigma_multiplier"],
                          "position": info["position"],
                          "epsilon_upper_bound": info["epsilon_conservative_zcdp_upper_bound"],
                          **row}
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(merged))
                    writer.writeheader()
                writer.writerow(merged)


if __name__ == "__main__":
    main()
