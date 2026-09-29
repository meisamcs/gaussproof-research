"""Aggregate the three fixed non-DP LiRA/RMIA validation seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEEDS = (20260929, 20261001, 20261003)
METHODS = (
    ("loss", "Loss"),
    ("lira_offline_fixed", "LiRA fixed"),
    ("lira_offline_variable", "LiRA variable"),
    ("rmia_offline_a05_g1", "RMIA 0.5 / 1"),
    ("rmia_offline_a1_g2", "RMIA 1 / 2"),
    ("pierre_lira_cdf", "Pierre LiRA CDF"),
    ("pierre_rmia_a05_g1", "Pierre RMIA"),
)


def aggregate(root: Path, output: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    settings = None
    digest = None
    by_key = {}
    accuracies = []
    for seed in SEEDS:
        directory = root / f"stable_seed_{seed}"
        metadata = json.loads((directory / "metadata.json").read_text())
        parameters = dict(metadata["parameters"])
        if parameters.pop("seed") != seed:
            raise ValueError(f"Wrong seed in {directory}")
        if settings is None:
            settings = parameters
            digest = metadata["dataset_sha256"]
        elif settings != parameters or digest != metadata["dataset_sha256"]:
            raise ValueError("Seeds use different experiment settings or datasets")
        if not metadata["roles_disjoint"] or not metadata["external_pierre_parity"]:
            raise ValueError("Role or external parity check missing")
        accuracies.append((seed, metadata["target_train_accuracy"],
                           metadata["target_nonmember_accuracy"]))
        if accuracies[-1][1] < .95:
            raise ValueError("Target did not fit its training records")
        records = list(csv.DictReader((directory / "summary.csv").open()))
        expected = {(world, method) for world in ("target", "independent_control")
                    for method, _ in METHODS}
        observed = {(r["world"], r["method"]) for r in records}
        if observed != expected or len(records) != len(expected):
            raise ValueError(f"Missing or duplicate method in {directory}")
        for record in records:
            if int(record["members"]) != 120 or int(record["nonmembers"]) != 120:
                raise ValueError("Candidate count differs across seeds")
            by_key.setdefault((record["world"], record["method"]), []).append(
                float(record["auc"]))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for world in ("target", "independent_control"):
        for method, _ in METHODS:
            values = by_key[(world, method)]
            rows.append(dict(world=world, method=method,
                             auc_mean=float(np.mean(values)),
                             auc_min=float(np.min(values)),
                             auc_max=float(np.max(values)), seed_count=len(values)))
    with (output / "summary_over_seeds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (output / "target_accuracy.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("seed", "train_accuracy", "nonmember_accuracy"))
        writer.writerows(accuracies)
    (output / "metadata.json").write_text(json.dumps(dict(
        design="non-DP positive control, independent record-level targets",
        seeds=SEEDS, parameters=settings, dataset_sha256=digest,
        interval="minimum and maximum across three independent seeds",
        external_pierre_parity_on_all_seeds=True), indent=2) + "\n")
    plot(output, rows)
    return rows


def plot(output: Path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for shift, world, color, label in (
            (-.13, "target", "#4477AA", "Trained target"),
            (.13, "independent_control", "#CC6677", "Independent control")):
        selected = [next(row for row in rows if row["world"] == world and
                         row["method"] == method) for method, _ in METHODS]
        means = np.array([row["auc_mean"] for row in selected])
        lower = np.array([row["auc_min"] for row in selected])
        upper = np.array([row["auc_max"] for row in selected])
        ax.errorbar(np.arange(len(METHODS)) + shift, means,
                    yerr=[means - lower, upper - means], fmt="o", capsize=3,
                    color=color, label=label)
    ax.axhline(.5, linestyle="--", color=".5", linewidth=1)
    ax.set(xticks=np.arange(len(METHODS)),
           xticklabels=[label for _, label in METHODS],
           ylim=(.3, 1), ylabel="Record membership AUC",
           title="Non-DP EMNIST: mean and range over three seeds")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"auc_over_seeds.{extension}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.root, args.output)
