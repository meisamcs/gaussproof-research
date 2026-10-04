"""Aggregate held-out LiRA/RMIA versus +GAUSSPROOF DP-FTRLM comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_pilot import write_csv
from summarize_calibrated import METHODS as BASE_METHODS


SEEDS = (20261011, 20261013, 20261017)
CONDITIONS = {"low_noise": .25, "high_noise": 4.}
MODES = ("natural_writer", "iid_mixed", "label_sorted")
HYBRIDS = ("hybrid_equal_lira", "hybrid_calibrated_lira",
           "hybrid_equal_rmia", "hybrid_calibrated_rmia")
METHODS = (*BASE_METHODS, *HYBRIDS)
COMPARISONS = {f"{method}_minus_{family}_calibrated"
               for family in ("lira", "rmia")
               for method in (f"hybrid_equal_{family}",
                              f"hybrid_calibrated_{family}")}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def run(root: Path, output: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    digest = None
    seed_rows = []
    gain_rows = []
    selections = []
    for seed in SEEDS:
        directory = root / f"seed_{seed}"
        metadata = json.loads((directory / "metadata.json").read_text())
        parameters = metadata["parameters"]
        expected = dict(seed=seed, sigma=[.25, 4.], modes=list(MODES),
                        rounds=64, position=32, clients_per_round=8,
                        calibration_identities=40, holdout_identities=80,
                        reference_models=32, population_clients=64,
                        public_bank=12, public_pool=512,
                        background_pool=500, examples=16,
                        client_example_selection="uniform",
                        pixel_transform="ink", reserve_utility_clients=1124,
                        utility_seed=20260929, clip=1., client_lr=1.,
                        server_lr=.2, momentum=.9,
                        calibrate_endpoint=True, fuse_trajectory=True)
        if any(parameters.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Unmatched benchmark settings in {directory}")
        if not metadata["role_partitions_disjoint"] or not metadata[
                "utility_tuning_writers_disjoint"]:
            raise ValueError("Experiment roles are not disjoint")
        if digest is None:
            digest = metadata["federated_emnist_sha256"]
        elif digest != metadata["federated_emnist_sha256"]:
            raise ValueError("Dataset digest differs between seeds")
        rows = read_csv(directory / "summary.csv")
        expected_keys = {(mode, sigma, method) for mode in MODES
                         for sigma in CONDITIONS.values() for method in METHODS}
        keys = {(r["distribution"], float(r["sigma"]), r["method"])
                for r in rows}
        if keys != expected_keys or len(rows) != len(expected_keys):
            raise ValueError(f"Missing or duplicate method in {directory}")
        for row in rows:
            condition = next(name for name, sigma in CONDITIONS.items()
                             if sigma == float(row["sigma"]))
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
                                  mean_accuracy_absent=float(
                                      row["mean_accuracy_absent"])))
        for row in read_csv(directory / "fusion_selections.csv"):
            condition = next(name for name, sigma in CONDITIONS.items()
                             if sigma == float(row["sigma"]))
            selections.append(dict(condition=condition, seed=seed, **row))
        for row in read_csv(directory / "paired_comparisons.csv"):
            if row["comparison"] not in COMPARISONS:
                continue
            condition = next(name for name, sigma in CONDITIONS.items()
                             if sigma == float(row["sigma"]))
            gain_rows.append(dict(condition=condition, seed=seed,
                                  distribution=row["distribution"],
                                  comparison=row["comparison"],
                                  auc_gain=float(row["auc_difference"]),
                                  bootstrap_low=float(row["low"]),
                                  bootstrap_high=float(row["high"])))
    if len(selections) != len(SEEDS) * len(CONDITIONS) * len(MODES) * 2:
        raise ValueError("Missing fusion calibration selection")
    if len(gain_rows) != len(SEEDS) * len(CONDITIONS) * len(MODES) * 4:
        raise ValueError("Missing paired hybrid comparison")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "seed_auc.csv", seed_rows)
    write_csv(output / "fusion_selections.csv", selections)
    write_csv(output / "paired_seed_gains.csv", gain_rows)
    groups = defaultdict(list)
    for row in seed_rows:
        groups[(row["condition"], row["distribution"], row["method"])].append(row)
    summary = []
    for (condition, distribution, method), rows in sorted(groups.items()):
        values = np.asarray([r["auc"] for r in rows])
        if len(values) != len(SEEDS):
            raise ValueError("Missing AUC seed")
        summary.append(dict(condition=condition, distribution=distribution,
                            method=method, mean_auc=float(values.mean()),
                            min_auc=float(values.min()),
                            max_auc=float(values.max()),
                            mean_tpr_at_calibrated_5pct_fpr=float(np.mean([
                                r["tpr_at_calibrated_5pct_fpr"] for r in rows])),
                            mean_achieved_5pct_fpr=float(np.mean([
                                r["achieved_5pct_fpr"] for r in rows])),
                            mean_accuracy_absent=float(np.mean([
                                r["mean_accuracy_absent"] for r in rows]))))
    write_csv(output / "summary_over_seeds.csv", summary)
    groups.clear()
    for row in gain_rows:
        groups[(row["condition"], row["distribution"],
                row["comparison"])].append(row)
    gain_summary = []
    for (condition, distribution, comparison), rows in sorted(groups.items()):
        values = np.asarray([r["auc_gain"] for r in rows])
        if len(values) != len(SEEDS):
            raise ValueError("Missing gain seed")
        gain_summary.append(dict(condition=condition, distribution=distribution,
                                 comparison=comparison,
                                 mean_auc_gain=float(values.mean()),
                                 min_auc_gain=float(values.min()),
                                 max_auc_gain=float(values.max()),
                                 positive_seeds=int((values > 0).sum())))
    write_csv(output / "gain_over_seeds.csv", gain_summary)
    (output / "metadata.json").write_text(json.dumps(dict(
        seeds=SEEDS, conditions=CONDITIONS, modes=MODES,
        federated_emnist_sha256=digest,
        calibration="40 identities select endpoint variant and hybrid weight; 80 disjoint identities evaluate",
        trajectory_access="Full global checkpoints and known candidate client data; endpoint-alone attacks see final model only",
        limitations="Source-aligned NumPy softmax pilot, one candidate appearance, low utility at sigma 4; seed ranges are not confidence intervals"),
        indent=2) + "\n")
    plot(output, summary, gain_summary)
    return summary, gain_summary


def plot(output: Path, summary, gains):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42
    colors = {"lira": "#AA3377", "rmia": "#EE7733"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True,
                             layout="constrained")
    for axis, (condition, sigma) in zip(axes, CONDITIONS.items()):
        for family in ("lira", "rmia"):
            for kind, marker, offset in (("equal", "o", -.10),
                                         ("calibrated", "s", .10)):
                comparison = (f"hybrid_{kind}_{family}_minus_"
                              f"{family}_calibrated")
                rows = [next(r for r in gains if r["condition"] == condition
                             and r["distribution"] == mode and
                             r["comparison"] == comparison) for mode in MODES]
                center = np.array([r["mean_auc_gain"] for r in rows])
                low = np.array([r["min_auc_gain"] for r in rows])
                high = np.array([r["max_auc_gain"] for r in rows])
                x = np.arange(len(MODES)) + offset + (
                    -.03 if family == "lira" else .03)
                axis.errorbar(x, center, yerr=[center-low, high-center],
                              fmt=marker, color=colors[family], capsize=3,
                              label=f"{family.upper()} + GP ({kind})")
        axis.axhline(0, color=".4", linestyle="--", linewidth=1)
        axis.set(title=f"σ={sigma}", xticks=np.arange(len(MODES)),
                 xticklabels=["Natural", "IID mixed", "Label sorted"])
    axes[0].set_ylabel("Held-out AUC gain over endpoint alone")
    axes[1].legend(frameon=False, fontsize=8)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"hybrid_auc_gain.{extension}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.root, arguments.output)
