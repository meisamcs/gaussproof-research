"""Aggregate predeclared independent seeds without pooling client identities."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_pilot import write_csv


CONDITIONS = {
    "eight_clients_sigma4": [
        "eight_clients", "eight_clients_seed_20261001",
        "eight_clients_seed_20261003"],
    "two_clients_sigma025_stress": [
        "two_clients_stress", "two_clients_seed_20261001",
        "two_clients_seed_20261003"],
}
SIGMA = {"eight_clients_sigma4": 4.,
         "two_clients_sigma025_stress": .25}
METHODS = ("gaussproof_max", "gaussproof_mean", "rero_max", "rero_mean",
           "endpoint_loss", "lira_offline_fixed", "rmia_offline_a05_g1",
           "rmia_offline_a1_g2", "pierre_lira_cdf", "pierre_rmia_a05_g1",
           "informed_llr")


def run(root: Path, output: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    digests = set()
    for condition, folders in CONDITIONS.items():
        seen_seeds = set()
        for folder in folders:
            run_dir = root / folder
            metadata = json.loads((run_dir / "metadata.json").read_text())
            params = metadata["parameters"]
            seed = int(params["seed"])
            if seed in seen_seeds:
                raise ValueError(f"Duplicate seed in {condition}")
            seen_seeds.add(seed)
            digests.add(metadata["federated_emnist_sha256"])
            if params["holdout_identities"] != 80 or params["calibration_identities"] != 40:
                raise ValueError("Expected matched client counts")
            if params["rounds"] != 32 or params["position"] != 16:
                raise ValueError("Expected matched 32-round insertion")
            expected_slots = 8 if condition.startswith("eight") else 2
            if params["clients_per_round"] != expected_slots:
                raise ValueError("Client-slot condition mismatch")
            with (run_dir / "summary.csv").open() as stream:
                for row in csv.DictReader(stream):
                    if float(row["sigma"]) != SIGMA[condition]:
                        continue
                    if row["method"] not in METHODS:
                        continue
                    seed_rows.append(dict(
                        condition=condition, seed=seed,
                        distribution=row["distribution"], method=row["method"],
                        auc=float(row["auc"]),
                        tpr_at_calibrated_5pct_fpr=float(
                            row["tpr_at_calibrated_5pct_fpr"]),
                        achieved_5pct_fpr=float(row["achieved_5pct_fpr"])))
        if len(seen_seeds) != 3:
            raise ValueError(f"Expected three seeds for {condition}")
    if len(digests) != 1:
        raise ValueError("Data file digests differ")
    write_csv(output / "seed_auc.csv", seed_rows)
    groups = defaultdict(list)
    for row in seed_rows:
        groups[(row["condition"], row["distribution"], row["method"])].append(row)
    summary = []
    for (condition, distribution, method), rows in sorted(groups.items()):
        if len(rows) != 3:
            raise ValueError("Missing a seed in an AUC group")
        values = np.array([r["auc"] for r in rows])
        summary.append(dict(
            condition=condition, distribution=distribution, method=method,
            seeds=3, mean_auc=float(values.mean()),
            min_auc=float(values.min()), max_auc=float(values.max()),
            sd_auc=float(values.std(ddof=1)),
            mean_tpr_at_calibrated_5pct_fpr=float(np.mean([
                r["tpr_at_calibrated_5pct_fpr"] for r in rows])),
            mean_achieved_5pct_fpr=float(np.mean([
                r["achieved_5pct_fpr"] for r in rows]))))
    write_csv(output / "seed_summary.csv", summary)
    (output / "metadata.json").write_text(json.dumps(dict(
        status="three independent target-training seeds per condition",
        conditions=CONDITIONS, selected_sigma=SIGMA,
        federated_emnist_sha256=next(iter(digests)),
        uncertainty=("min/max across three seeds; not a confidence interval "
                     "over arbitrary deployments. Within-seed client bootstrap "
                     "intervals remain in each run's summary.csv"),
    ), indent=2))
    plot(output, summary)


def plot(output: Path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42

    methods = {
        "gaussproof_max": ("GAUSSPROOF max", "#CC6677"),
        "rero_max": ("RERO max", "#4477AA"),
        "endpoint_loss": ("Final loss", "#228833"),
        "lira_offline_fixed": ("Original LiRA", "#AA3377"),
        "rmia_offline_a05_g1": ("Original RMIA", "#66CCEE"),
        "pierre_lira_cdf": ("Pierre LiRA CDF", "#999933"),
    }
    distribution = ["natural_writer", "iid_mixed", "label_sorted"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True,
                             layout="constrained")
    for ax, condition in zip(axes, CONDITIONS):
        for method_index, (method, (label, color)) in enumerate(methods.items()):
            subset = [next(r for r in rows if r["condition"] == condition
                           and r["distribution"] == mode
                           and r["method"] == method) for mode in distribution]
            xs = np.arange(len(distribution)) + (method_index - 2.5) * .10
            y = np.array([r["mean_auc"] for r in subset])
            low = np.array([r["min_auc"] for r in subset])
            high = np.array([r["max_auc"] for r in subset])
            ax.errorbar(xs, y, yerr=[y-low, high-y], marker="o", linestyle="none",
                        color=color, capsize=2, markersize=4, label=label)
        ax.axhline(.5, linestyle="--", color=".5", linewidth=1)
        ax.set(xticks=range(3), xticklabels=["Natural writer", "IID mixed",
                                            "Label sorted"],
               xlabel="Client distribution",
               title=("8 clients/round, σ=4" if condition.startswith("eight")
                      else "2 clients/round, σ=.25 (stress)"))
    axes[0].set_ylabel("Client AUC: mean and range over 3 seeds")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(.5, -.08), frameon=False, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"replicated_auc.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
