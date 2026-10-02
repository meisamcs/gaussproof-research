"""Aggregate the repeated-client, high-participation DP-FTRLM experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_pilot import write_csv
from summarize_hybrid import METHODS


SEEDS = (20261011, 20261013, 20261017)
QS = (.02, .1, .5)
SIGMAS = (2., 4.)
PLOT_METHODS = (
    ("gaussproof_mean", "GAUSSPROOF", "#4477AA", "o"),
    ("rero_mean", "RERO-style", "#228833", "^"),
    ("lira_calibrated", "LiRA", "#AA3377", "s"),
    ("hybrid_equal_lira", "LiRA + GP", "#CC6677", "D"),
    ("rmia_calibrated", "RMIA", "#EE7733", "v"),
    ("hybrid_equal_rmia", "RMIA + GP", "#CCBB44", "P"),
)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def run(root: Path, output: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    seed_auc = []
    seed_gains = []
    participation = []
    selections = []
    digest = None
    for seed in SEEDS:
        for q in QS:
            directory = root / f"seed_{seed}_q{q}"
            meta = json.loads((directory / "metadata.json").read_text())
            p = meta["parameters"]
            expected = dict(seed=seed, modes=["natural_writer"],
                            sigma=list(SIGMAS), rounds=64,
                            clients_per_round=8, participation_q=q,
                            calibration_identities=40, holdout_identities=80,
                            reference_models=32, public_bank=12,
                            public_pool=512, population_clients=64,
                            background_pool=500, examples=16,
                            client_example_selection="uniform",
                            pixel_transform="ink", clip=1.,
                            client_lr=1., server_lr=.2, momentum=.9,
                            reserve_utility_clients=1124,
                            utility_seed=20260929,
                            calibrate_endpoint=True,
                            fuse_trajectory=True)
            if any(p.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Benchmark settings differ: {directory}")
            if not meta["role_partitions_disjoint"] or not meta[
                    "utility_tuning_writers_disjoint"]:
                raise ValueError("Client role partitions are not disjoint")
            if meta["cohort_size"] != round(8 / q):
                raise ValueError("Incorrect B-of-N cohort")
            if digest is None:
                digest = meta["federated_emnist_sha256"]
            elif digest != meta["federated_emnist_sha256"]:
                raise ValueError("Dataset digest differs")
            rows = read_csv(directory / "summary.csv")
            keys = {(float(row["sigma"]), row["method"]) for row in rows}
            if keys != {(sigma, method) for sigma in SIGMAS
                        for method in METHODS} or len(rows) != len(keys):
                raise ValueError(f"Incomplete attack table: {directory}")
            for row in rows:
                seed_auc.append(dict(
                    seed=seed, q=q, sigma=float(row["sigma"]),
                    method=row["method"], auc=float(row["auc"]),
                    auc_low=float(row["auc_low"]),
                    auc_high=float(row["auc_high"]),
                    accuracy=float(row["mean_accuracy_absent"]),
                    tpr_at_calibrated_5pct_fpr=float(
                        row["tpr_at_calibrated_5pct_fpr"]),
                    achieved_5pct_fpr=float(row["achieved_5pct_fpr"])))
            for row in read_csv(directory / "paired_comparisons.csv"):
                if row["comparison"] in {
                        f"hybrid_equal_{family}_minus_{family}_calibrated"
                        for family in ("lira", "rmia")
                }:
                    seed_gains.append(dict(
                        seed=seed, q=q, sigma=float(row["sigma"]),
                        comparison=row["comparison"],
                        auc_gain=float(row["auc_difference"]),
                        bootstrap_low=float(row["low"]),
                        bootstrap_high=float(row["high"])))
            for row in read_csv(directory / "participation_counts.csv"):
                participation.append(dict(
                    seed=seed, q=q, sigma=float(row["sigma"]),
                    count=int(row["count"])))
            for row in read_csv(directory / "fusion_selections.csv"):
                selections.append(dict(seed=seed, q=q, **row))
    if len(seed_gains) != len(SEEDS) * len(QS) * len(SIGMAS) * 2:
        raise ValueError("Missing hybrid comparison")
    if len(participation) != len(SEEDS) * len(QS) * len(SIGMAS) * 120:
        raise ValueError("Missing participation counts")
    if len(selections) != len(SEEDS) * len(QS) * len(SIGMAS) * 2:
        raise ValueError("Missing fusion selections")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "seed_auc.csv", seed_auc)
    write_csv(output / "paired_seed_gains.csv", seed_gains)
    write_csv(output / "fusion_selections.csv", selections)
    groups = defaultdict(list)
    for row in seed_auc:
        groups[(row["q"], row["sigma"], row["method"])].append(row)
    summary = []
    for (q, sigma, method), rows in sorted(groups.items()):
        if len(rows) != len(SEEDS):
            raise ValueError("Incomplete seed replication")
        aucs = [r["auc"] for r in rows]
        summary.append(dict(
            q=q, sigma=sigma, method=method,
            mean_auc=float(np.mean(aucs)), min_auc=min(aucs), max_auc=max(aucs),
            mean_accuracy=float(np.mean([r["accuracy"] for r in rows])),
            mean_tpr_at_calibrated_5pct_fpr=float(np.mean([
                r["tpr_at_calibrated_5pct_fpr"] for r in rows])),
            mean_achieved_5pct_fpr=float(np.mean([
                r["achieved_5pct_fpr"] for r in rows]))))
    write_csv(output / "summary_over_seeds.csv", summary)
    groups.clear()
    for row in seed_gains:
        groups[(row["q"], row["sigma"], row["comparison"])].append(row)
    gains = []
    for (q, sigma, comparison), rows in sorted(groups.items()):
        values = [r["auc_gain"] for r in rows]
        gains.append(dict(q=q, sigma=sigma, comparison=comparison,
                          mean_auc_gain=float(np.mean(values)),
                          min_auc_gain=min(values), max_auc_gain=max(values),
                          positive_seeds=sum(value > 0 for value in values)))
    write_csv(output / "gain_over_seeds.csv", gains)
    participation_summary = []
    groups.clear()
    for row in participation:
        groups[(row["seed"], row["q"], row["sigma"])].append(row["count"])
    for (seed, q, sigma), values in sorted(groups.items()):
        if len(values) != 120:
            raise ValueError("Missing candidate participation count")
        participation_summary.append(dict(
            seed=seed, q=q, sigma=sigma, cohort_size=round(8 / q),
            mean_count=float(np.mean(values)), min_count=min(values),
            max_count=max(values),
            zero_count=sum(value == 0 for value in values)))
    write_csv(output / "participation_summary.csv", participation_summary)
    batch_rows = []
    for seed in SEEDS:
        directory = root / f"large_batch_seed_{seed}"
        meta = json.loads((directory / "metadata.json").read_text())
        p = meta["parameters"]
        expected_batch = dict(seed=seed, modes=["natural_writer"],
                              sigma=[4.], rounds=64, clients_per_round=32,
                              participation_q=.5, calibration_identities=40,
                              holdout_identities=80, reference_models=32,
                              client_example_selection="uniform",
                              pixel_transform="ink", clip=1.,
                              client_lr=1., server_lr=.2, momentum=.9,
                              calibrate_endpoint=True, fuse_trajectory=True)
        if any(p.get(key) != value for key, value in expected_batch.items()):
            raise ValueError(f"Large-batch settings differ: {directory}")
        if meta["federated_emnist_sha256"] != digest or meta[
                "cohort_size"] != 64:
            raise ValueError("Large-batch dataset or cohort differs")
        rows = read_csv(directory / "summary.csv")
        if {r["method"] for r in rows} != set(METHODS) or len(rows) != len(METHODS):
            raise ValueError("Incomplete large-batch attack table")
        for row in rows:
            batch_rows.append(dict(seed=seed, q=.5, sigma=4., rounds=64,
                                   clients_per_round=32, method=row["method"],
                                   auc=float(row["auc"]),
                                   accuracy=float(row["mean_accuracy_absent"])))
    write_csv(output / "large_batch_seed_auc.csv", batch_rows)
    groups.clear()
    for row in batch_rows:
        groups[row["method"]].append(row)
    batch_summary = []
    for method, rows in sorted(groups.items()):
        values = [r["auc"] for r in rows]
        batch_summary.append(dict(
            q=.5, sigma=4., rounds=64, clients_per_round=32,
            method=method, mean_auc=float(np.mean(values)),
            min_auc=min(values), max_auc=max(values),
            mean_accuracy=float(np.mean([r["accuracy"] for r in rows]))))
    write_csv(output / "large_batch_summary.csv", batch_summary)
    long_directory = root / "long_batch_seed_20261011"
    long_meta = json.loads((long_directory / "metadata.json").read_text())
    long_p = long_meta["parameters"]
    long_expected = dict(seed=20261011, modes=["natural_writer"],
                         sigma=[4.], rounds=128, clients_per_round=32,
                         participation_q=.5, calibration_identities=40,
                         holdout_identities=80, reference_models=32,
                         client_example_selection="uniform",
                         pixel_transform="ink", clip=1.,
                         client_lr=1., server_lr=.2, momentum=.9,
                         calibrate_endpoint=True, fuse_trajectory=True)
    if any(long_p.get(key) != value for key, value in long_expected.items()):
        raise ValueError("Long-run settings differ")
    if long_meta["federated_emnist_sha256"] != digest or long_meta[
            "cohort_size"] != 64:
        raise ValueError("Long-run dataset or cohort differs")
    long_rows = read_csv(long_directory / "summary.csv")
    if {r["method"] for r in long_rows} != set(METHODS) or len(
            long_rows) != len(METHODS):
        raise ValueError("Incomplete long-run attack table")
    write_csv(output / "long_batch_exploratory.csv", [dict(
        seed=20261011, q=.5, sigma=4., rounds=128,
        clients_per_round=32, method=row["method"],
        auc=float(row["auc"]),
        accuracy=float(row["mean_accuracy_absent"]))
        for row in long_rows])
    (output / "metadata.json").write_text(json.dumps(dict(
        seeds=SEEDS, q=QS, sigma=SIGMAS, rounds=64, clients_per_round=8,
        federated_emnist_sha256=digest,
        access="Endpoint attacks use final models; trajectory attacks additionally see all checkpoints and public clients",
        adjacency="Replace one cohort client with the candidate; fixed B=8 and N=B/q; shared schedules and tree-noise draws",
        sensitivity="Clipped replace-one shift at most 2C/B; tree-node Gaussian std sigma*C/B",
        privacy="No user-level epsilon reported for repeated participation; source one-pass bound does not apply",
        large_batch="Three-seed robustness check: B=32, N=64, q=.5, sigma=4, T=64",
        long_batch="Exploratory one-seed check: B=32, N=64, q=.5, sigma=4, T=128",
        limitations="Source-aligned NumPy softmax pilot, not official TFF CNN; seed ranges are not confidence intervals"),
        indent=2) + "\n")
    plot(output, summary)
    return summary, gains


def plot(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    matplotlib.rcParams["pdf.fonttype"] = 42
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True,
                             layout="constrained")
    for axis, sigma in zip(axes, SIGMAS):
        for method, label, color, marker in PLOT_METHODS:
            rows = [next(r for r in summary if r["q"] == q and
                         r["sigma"] == sigma and r["method"] == method)
                    for q in QS]
            center = np.asarray([r["mean_auc"] for r in rows])
            low = np.asarray([r["min_auc"] for r in rows])
            high = np.asarray([r["max_auc"] for r in rows])
            axis.errorbar(QS, center, yerr=[center - low, high - center],
                          marker=marker, color=color, linewidth=1.35,
                          markersize=5, capsize=2.5, label=label)
        axis.axhline(.5, color=".5", linestyle="--", linewidth=1)
        axis.set(title=f"Tree noise multiplier σ={sigma:g}",
                 xlabel="Per-round client participation probability q",
                 xscale="log", xticks=QS, xticklabels=[str(q) for q in QS],
                 xlim=(.015, .7))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.minorticks_off()
    axes[0].set_ylabel("Held-out client-membership AUC")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    for extension in ("pdf", "png"):
        fig.savefig(output / f"auc_by_participation.{extension}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
