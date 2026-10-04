"""Held-out, calibration-selected GAUSSPROOF + final-checkpoint LiRA test.

The score rows are private intermediate data. Only aggregate results are saved.
The operating threshold and fusion weight are selected on disjoint identities.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gaussproof.metrics import auc, rates, threshold_at_fpr
from scripts.summarize_trajectory_endpoint_replication import (
    identity_roles, read_rows, write_rows,
)


def _standardize(calibration, holdout):
    center = calibration.mean(axis=0)
    scale = np.maximum(calibration.std(axis=0), 1e-9)
    return (calibration - center) / scale, (holdout - center) / scale


def analyze(raw, spec, level=.05):
    calibration_ids, test_ids = identity_roles(
        raw, int(spec["identities_per_class"]),
        int(spec["calibration_per_class"]))
    output = []
    rng = np.random.default_rng(int(spec["bootstrap_seed"]) + 31)
    for q in map(float, spec["q_values"]):
        for steps in map(int, spec["prefix_steps"]):
            group = [r for r in raw if r["q"] == q and r["steps"] == steps]
            cal = [r for r in group if r["canary"] in calibration_ids]
            test = [r for r in group if r["canary"] in test_ids]
            if len(cal) != 2 * len(calibration_ids) or len(test) != 2 * len(test_ids):
                raise ValueError("Expected exactly one H0 and H1 row per identity")
            cy = np.asarray([r["label"] for r in cal], int)
            ty = np.asarray([r["label"] for r in test], int)
            names = ("trajectory_mixture", "endpoint_lira_z")
            c, t = _standardize(
                np.asarray([[r[name] for name in names] for r in cal]),
                np.asarray([[r[name] for name in names] for r in test]))
            weights = np.linspace(0., 1., 21)
            def score(values, weight):
                return weight * values[:, 0] + (1. - weight) * values[:, 1]
            def calibrate(values):
                cutoff = threshold_at_fpr(values, cy, level)
                tpr, fpr = rates(values, cy, cutoff)
                return cutoff, tpr, fpr
            # Ties favor the simpler endpoint-only score. No test labels enter
            # either the weight search or the operating threshold.
            selected = max(weights, key=lambda w: (
                calibrate(score(c, w))[1], auc(score(c, w), cy), -w))
            indices = {identity: np.flatnonzero(
                [r["canary"] == identity for r in test]) for identity in test_ids}
            ordered = sorted(test_ids)
            draws = rng.integers(0, len(ordered), size=(
                int(spec["bootstrap_repetitions"]), len(ordered)))
            values = {"LiRA": t[:, 1], "GAUSSPROOF": t[:, 0],
                      "fusion": score(t, selected)}
            cutoffs = {"LiRA": calibrate(c[:, 1])[0],
                       "GAUSSPROOF": calibrate(c[:, 0])[0],
                       "fusion": calibrate(score(c, selected))[0]}
            baseline_pred = values["LiRA"] >= cutoffs["LiRA"]
            for method, current in values.items():
                cutoff = cutoffs[method]
                pred = current >= cutoff
                tpr, fpr = rates(current, ty, cutoff)
                tpr_gains, fpr_gains = [], []
                for draw in draws:
                    sample = np.concatenate([indices[ordered[i]] for i in draw])
                    positive = ty[sample] == 1
                    negative = ~positive
                    tpr_gains.append(float(pred[sample][positive].mean() -
                                           baseline_pred[sample][positive].mean()))
                    fpr_gains.append(float(pred[sample][negative].mean() -
                                           baseline_pred[sample][negative].mean()))
                tlo, thi = np.quantile(tpr_gains, [.025, .975])
                flo, fhi = np.quantile(fpr_gains, [.025, .975])
                output.append(dict(q=q, steps=steps, method=method,
                    trajectory_weight=float(selected) if method == "fusion" else
                    (1. if method == "GAUSSPROOF" else 0.),
                    target_calibration_fpr=level, calibration_nonmembers=int((cy == 0).sum()),
                    holdout_nonmembers=int((ty == 0).sum()),
                    holdout_members=int((ty == 1).sum()),
                    holdout_auc=auc(current, ty),
                    calibrated_holdout_tpr=tpr, calibrated_holdout_fpr=fpr,
                    tpr_gain_vs_lira=tpr - rates(values["LiRA"], ty, cutoffs["LiRA"])[0],
                    tpr_gain_ci_low=float(tlo), tpr_gain_ci_high=float(thi),
                    fpr_gain_vs_lira=fpr - rates(values["LiRA"], ty, cutoffs["LiRA"])[1],
                    fpr_gain_ci_low=float(flo), fpr_gain_ci_high=float(fhi)))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    if not (run / "completion.json").exists():
        raise ValueError("Target run is incomplete")
    spec = json.loads((run / "config.json").read_text())
    result = analyze(read_rows(run / "score_rows.csv"), spec)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(output, result)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
