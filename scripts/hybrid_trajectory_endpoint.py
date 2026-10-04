"""Calibrate trajectory plus final-model LiRA/RMIA fusion on disjoint identities."""
import argparse
from pathlib import Path

import numpy as np

from gaussproof.metrics import auc
from scripts.summarize_trajectory_endpoint_replication import (
    identity_roles, read_rows, write_rows,
)


def analyze(raw, spec):
    calibration, holdout = identity_roles(
        raw, int(spec["identities_per_class"]),
        int(spec["calibration_per_class"]))
    rng = np.random.default_rng(int(spec["bootstrap_seed"]) + 19)
    results = []
    for q in map(float, spec["q_values"]):
        for step in map(int, spec["prefix_steps"]):
            group = [r for r in raw if r["q"] == q and r["steps"] == step]
            cal = [r for r in group if r["canary"] in calibration]
            test = [r for r in group if r["canary"] in holdout]
            cal_label = np.array([r["label"] for r in cal])
            test_label = np.array([r["label"] for r in test])
            rmia_options = [name for name in group[0]
                            if name.startswith("endpoint_rmia_")]
            selected_rmia = max(enumerate(rmia_options), key=lambda item: (
                auc([r[item[1]] for r in cal], cal_label), -item[0]))[1]
            for trajectory, endpoint in (
                ("trajectory_mixture", "endpoint_lira_z"),
                ("trajectory_raw_alignment", "endpoint_lira_z"),
                ("trajectory_mixture", selected_rmia),
                ("trajectory_raw_alignment", selected_rmia),
            ):
                names = [trajectory, endpoint]
                cal_value = np.array([[r[n] for n in names] for r in cal])
                test_value = np.array([[r[n] for n in names] for r in test])
                mean = cal_value.mean(axis=0)
                scale = np.maximum(cal_value.std(axis=0), 1e-9)
                cal_value = (cal_value - mean) / scale
                test_value = (test_value - mean) / scale
                weights = np.linspace(0, 1, 21)
                # Highest calibration AUC wins; ties choose the smaller
                # trajectory weight, hence the simpler endpoint-only score.
                selected = max(weights, key=lambda w: (
                    auc(w * cal_value[:, 0] + (1 - w) * cal_value[:, 1],
                        cal_label), -w))
                fused = selected * test_value[:, 0] + (1 - selected) * test_value[:, 1]
                endpoint_scores = test_value[:, 1]
                lira_scores = np.array([r["endpoint_lira_z"] for r in test])
                identities = sorted(holdout)
                index = {identity: np.array([j for j, r in enumerate(test)
                    if r["canary"] == identity]) for identity in identities}
                draws = rng.integers(0, len(identities), size=(
                    int(spec["bootstrap_repetitions"]), len(identities)))
                gains = []
                gains_vs_lira = []
                for draw in draws:
                    choice = np.concatenate([index[identities[i]] for i in draw])
                    gains.append(auc(fused[choice], test_label[choice]) -
                                 auc(endpoint_scores[choice], test_label[choice]))
                    gains_vs_lira.append(auc(fused[choice], test_label[choice]) -
                                         auc(lira_scores[choice], test_label[choice]))
                low, high = np.quantile(gains, [.025, .975])
                lira_low, lira_high = np.quantile(gains_vs_lira, [.025, .975])
                results.append(dict(q=q, steps=step, trajectory=trajectory,
                    endpoint=endpoint,
                    calibration_selected_trajectory_weight=selected,
                    fusion_auc=auc(fused, test_label),
                    endpoint_auc=auc(endpoint_scores, test_label),
                    auc_gain=auc(fused, test_label) -
                             auc(endpoint_scores, test_label),
                    gain_ci_low=low, gain_ci_high=high,
                    lira_z_auc=auc(lira_scores, test_label),
                    gain_vs_lira=auc(fused, test_label) -
                                 auc(lira_scores, test_label),
                    gain_vs_lira_ci_low=lira_low,
                    gain_vs_lira_ci_high=lira_high,
                    calibration_identities=len(calibration),
                    holdout_identities=len(holdout)))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    if not (run / "completion.json").exists():
        raise ValueError("Target runs are incomplete")
    import json
    spec = json.loads((run / "config.json").read_text())
    rows = analyze(read_rows(run / "score_rows.csv"), spec)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(output, rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
