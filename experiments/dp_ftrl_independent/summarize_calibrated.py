"""Aggregate the matched, calibration-selected DP-FTRLM endpoint audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_pilot import write_csv


SEEDS = (20261011, 20261013, 20261017)
CONDITIONS = {"low_noise": (8, .25), "high_noise": (8, 4.)}
MODES = ("natural_writer", "iid_mixed", "label_sorted")
METHODS = ("gaussproof_max", "gaussproof_mean", "rero_max", "rero_mean",
           "public_gls_max", "endpoint_loss", "informed_llr",
           "lira_offline_fixed", "rmia_offline_a1_g2",
           "rmia_offline_a05_g1", "pierre_lira_cdf",
           "pierre_rmia_a05_g1", "lira_calibrated", "rmia_calibrated",
           "endpoint_best_calibrated")
FAMILIES = ("lira_calibrated", "rmia_calibrated",
            "endpoint_best_calibrated")


def read_csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def run(root: Path, output: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    digest = None
    seed_rows = []
    selections = []
    paired = []
    for condition, (slots, sigma) in CONDITIONS.items():
        for seed in SEEDS:
            directory = root / f"seed_{seed}"
            metadata = json.loads((directory / "metadata.json").read_text())
            params = metadata["parameters"]
            expected = dict(seed=seed, clients_per_round=slots, sigma=[.25, 4.],
                            reference_models=32, population_clients=64,
                            calibration_identities=40, holdout_identities=80,
                            rounds=64, position=32, calibrate_endpoint=True,
                            modes=list(MODES), public_bank=12, public_pool=512,
                            background_pool=500, client_example_selection="uniform",
                            pixel_transform="ink", reserve_utility_clients=1124,
                            utility_seed=20260929, clip=1., client_lr=1.,
                            server_lr=.2, momentum=.9)
            if any(params.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Unmatched settings in {directory}")
            if not metadata["role_partitions_disjoint"]:
                raise ValueError("Overlapping experiment roles")
            if digest is None:
                digest = metadata["federated_emnist_sha256"]
            elif digest != metadata["federated_emnist_sha256"]:
                raise ValueError("Dataset digest differs between runs")
            rows = [r for r in read_csv(directory / "summary.csv")
                    if float(r["sigma"]) == sigma]
            observed = {(r["distribution"], r["method"]) for r in rows}
            expected_rows = {(mode, method) for mode in MODES for method in METHODS}
            if observed != expected_rows or len(rows) != len(expected_rows):
                raise ValueError(f"Missing or duplicate score row in {directory}")
            for row in rows:
                seed_rows.append(dict(condition=condition, seed=seed,
                                      distribution=row["distribution"],
                                      method=row["method"],
                                      auc=float(row["auc"]),
                                      auc_low=float(row["auc_low"]),
                                      auc_high=float(row["auc_high"]),
                                      tpr_at_calibrated_5pct_fpr=float(
                                          row["tpr_at_calibrated_5pct_fpr"]),
                                      achieved_5pct_fpr=float(
                                          row["achieved_5pct_fpr"]),
                                      mean_accuracy_present=float(
                                          row["mean_accuracy_present"])))
            selected = [r for r in read_csv(directory / "calibration_selections.csv")
                        if float(r["sigma"]) == sigma]
            observed = {(r["distribution"], r["family"]) for r in selected}
            expected_selected = {(mode, family) for mode in MODES
                                 for family in FAMILIES}
            if observed != expected_selected or len(selected) != len(expected_selected):
                raise ValueError(f"Missing or duplicate selection in {directory}")
            selections.extend(dict(condition=condition, seed=seed, **row)
                              for row in selected)
            for row in read_csv(directory / "paired_comparisons.csv"):
                if float(row["sigma"]) != sigma:
                    continue
                if row["comparison"] == "gaussproof_max_minus_lira_calibrated":
                    paired.append(dict(condition=condition, seed=seed,
                                       distribution=row["distribution"],
                                       comparison=row["comparison"],
                                       auc_difference=float(row["auc_difference"]),
                                       bootstrap_low=float(row["low"]),
                                       bootstrap_high=float(row["high"])))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "seed_auc.csv", seed_rows)
    write_csv(output / "selected_variants.csv", selections)
    write_csv(output / "paired_seed_differences.csv", paired)
    groups = defaultdict(list)
    for row in seed_rows:
        groups[(row["condition"], row["distribution"], row["method"])].append(row)
    summary = []
    for (condition, distribution, method), rows in sorted(groups.items()):
        if len(rows) != len(SEEDS):
            raise ValueError("Missing independent seed")
        auc = np.array([r["auc"] for r in rows])
        summary.append(dict(condition=condition, distribution=distribution,
                            method=method, seeds=len(SEEDS),
                            mean_auc=float(auc.mean()),
                            min_auc=float(auc.min()),
                            max_auc=float(auc.max()),
                            mean_tpr_at_calibrated_5pct_fpr=float(np.mean([
                                r["tpr_at_calibrated_5pct_fpr"] for r in rows])),
                            mean_achieved_5pct_fpr=float(np.mean([
                                r["achieved_5pct_fpr"] for r in rows])),
                            mean_accuracy_present=float(np.mean([
                                r["mean_accuracy_present"] for r in rows]))))
    write_csv(output / "summary_over_seeds.csv", summary)
    (output / "metadata.json").write_text(json.dumps(dict(
        status="calibration-selected endpoint comparison on DP-FTRLM pilot",
        seeds=SEEDS, conditions=CONDITIONS, modes=MODES,
        federated_emnist_sha256=digest,
        selection="40 calibration identities per distribution/sigma; 80 independent holdout identities",
        endpoint_resources="32 OUT DP-FTRLM references and 64 disjoint population clients",
        utility_tuning="1124 writers used for a separate utility sweep are excluded from all benchmark roles",
        limits=("Source-aligned NumPy softmax-client pilot, not official TFF "
                "CNN. Seed ranges are not deployment-wide confidence intervals. "
                "One-time candidate participation; high-noise utility is low. "
                "Calibrated 5% thresholds may exceed 5% FPR on holdout.")),
        indent=2) + "\n")
    plot(output, summary)
    return summary


def plot(output: Path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42
    selected_methods = {
        "gaussproof_max": ("GAUSSPROOF max", "#CC6677"),
        "gaussproof_mean": ("GAUSSPROOF mean", "#882255"),
        "rero_max": ("RERO max", "#4477AA"),
        "endpoint_loss": ("Final loss", "#228833"),
        "lira_calibrated": ("LiRA calibrated", "#AA3377"),
        "rmia_calibrated": ("RMIA calibrated", "#EE7733"),
    }
    for condition, title, limits in (
            ("high_noise", "8 clients/round, σ=4 (public attacks; zoomed AUC)",
             (.45, .56)),
            ("low_noise", "8 clients/round, σ=.25 (public attacks; zoomed AUC)",
             (.45, .56))):
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        for index, (method, (label, color)) in enumerate(selected_methods.items()):
            subset = [next(row for row in rows if row["condition"] == condition
                           and row["distribution"] == mode and
                           row["method"] == method) for mode in MODES]
            center = np.array([float(r["mean_auc"]) for r in subset])
            low = np.array([float(r["min_auc"]) for r in subset])
            high = np.array([float(r["max_auc"]) for r in subset])
            x = np.arange(len(MODES)) + (index - 2.5) * .11
            ax.errorbar(x, center, yerr=[center-low, high-center],
                        fmt="o", color=color, capsize=3, label=label)
        ax.axhline(.5, linestyle="--", color=".5", linewidth=1)
        ax.set(xticks=np.arange(len(MODES)),
               xticklabels=["Natural writer", "IID mixed", "Label sorted"],
               ylim=limits, xlabel="Client distribution",
               ylabel="Cross-identity client AUC",
               title=f"{title}: mean and range over three seeds")
        ax.legend(frameon=False, ncol=3, fontsize=8)
        for extension in ("pdf", "png"):
            fig.savefig(output / f"{condition}_auc.{extension}", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
