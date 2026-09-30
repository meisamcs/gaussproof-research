"""Paired comparisons across attacks, noise levels, and utility-rescue batches.

Reads private per-trial scores from ignored run directories and exports only
aggregate differences. Intervals are descriptive 95% paired-seed bootstrap
intervals; the grid contains many comparisons, so they are not adjusted
familywise or suitable for selecting a favorable cell after the fact.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from gaussproof.metrics import auc


def load_run(path):
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    if not (path / "completion.json").exists():
        raise ValueError(f"Incomplete run: {path}")
    rows = list(csv.DictReader((path / "score_rows.csv").open(newline="")))
    expected = (len(cfg["clip_values"]) * len(cfg["sigmas"]) *
                len(cfg["prefix_steps"]) * 2 *
                (cfg["calibration_pairs"] + cfg["holdout_pairs"]))
    processed_clips = json.loads((path / "completion.json").read_text()).get(
        "processed_clip_values", cfg["clip_values"])
    expected = int(expected * len(processed_clips) / len(cfg["clip_values"]))
    if len(rows) != expected:
        raise ValueError(f"Wrong number of score rows in {path}: {len(rows)} != {expected}")
    return cfg, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    data = {}
    holdout_count = None
    for directory in args.runs:
        cfg, rows = load_run(directory)
        if holdout_count is None:
            holdout_count = cfg["holdout_pairs"]
        elif cfg["holdout_pairs"] != holdout_count:
            raise ValueError("Runs have different holdout pair counts")
        for row in rows:
            if row["role"] != "holdout":
                continue
            key = (cfg["batch_size"], float(row["clip"]),
                   float(row["sigma"]), int(row["steps"]))
            observation = (int(row["pair"]), int(row["world"]))
            data.setdefault(key, {})[observation] = row
    ordered = [(pair, world) for pair in range(holdout_count) for world in (0, 1)]
    labels = np.asarray([world for _, world in ordered], dtype=int)
    sample_rng = np.random.default_rng(20260930)
    draws = sample_rng.integers(0, holdout_count,
                               size=(args.bootstrap, holdout_count))
    indices = [np.ravel(np.column_stack((2*draw, 2*draw+1))) for draw in draws]
    scores = {}
    for key, records in data.items():
        if set(records) != set(ordered):
            raise ValueError(f"Missing/duplicate held-out pair in {key}")
        scores[key] = {method: np.asarray([float(records[obs][method])
                                           for obs in ordered])
                       for method in ("trajectory_mixture", "raw_alignment",
                                      "endpoint_logprob_difference")}
    comparisons = []

    def append(kind, left_key, left_method, right_key, right_method):
        left = scores[left_key][left_method]
        right = scores[right_key][right_method]
        observed = auc(left, labels) - auc(right, labels)
        boot = np.asarray([auc(left[idx], labels[idx]) -
                           auc(right[idx], labels[idx]) for idx in indices])
        low, high = np.quantile(boot, [.025, .975])
        comparisons.append(dict(comparison=kind,
            clip=left_key[1], sigma=left_key[2], other_sigma=right_key[2],
            batch_size=left_key[0], other_batch_size=right_key[0],
            steps=left_key[3], left_method=left_method,
            right_method=right_method, delta_auc=observed,
            bootstrap_ci_low=float(low), bootstrap_ci_high=float(high),
            holdout_pairs=holdout_count, bootstrap_repetitions=args.bootstrap))

    for key in sorted(scores):
        append("GAUSSPROOF minus raw alignment", key, "trajectory_mixture",
               key, "raw_alignment")
    for batch, clip, sigma, steps in sorted(scores):
        if batch != 8:
            continue
        available = sorted({k[2] for k in scores if k[0] == batch and
                            k[1] == clip and k[3] == steps})
        pos = available.index(sigma)
        if pos:
            prior = available[pos-1]
            # A positive value means that adding noise increased attack AUC.
            append("higher noise minus lower noise", (batch,clip,sigma,steps),
                   "trajectory_mixture", (batch,clip,prior,steps),
                   "trajectory_mixture")
    for high_batch, clip, sigma, steps in sorted(scores):
        low_key = (8, clip, sigma, steps)
        if high_batch != 8 and low_key in scores:
            append("utility-rescue batch minus batch 8",
                   (high_batch,clip,sigma,steps), "trajectory_mixture",
                   low_key, "trajectory_mixture")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print(f"Wrote {len(comparisons)} aggregate comparisons to {output}")


if __name__ == "__main__":
    main()
