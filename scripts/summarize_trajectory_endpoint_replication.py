"""Aggregate paired identity holdout results without exporting row-level data."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gaussproof.metrics import auc, rates, roc, threshold_at_fpr


def read_rows(path):
    with path.open(newline="") as stream:
        return [{k: (int(v) if k in ("canary", "digit", "label", "steps")
                     else float(v)) for k, v in row.items()}
                for row in csv.DictReader(stream)]


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity_roles(rows, per_class, calibration_per_class):
    ordered = {digit: [] for digit in range(10)}
    for row in rows:
        group = ordered[row["digit"]]
        if row["canary"] not in group:
            group.append(row["canary"])
    if any(len(ids) != per_class for ids in ordered.values()):
        raise ValueError("Missing stratified identities")
    calibration = {identity for ids in ordered.values()
                   for identity in ids[:calibration_per_class]}
    holdout = {identity for ids in ordered.values()
               for identity in ids[calibration_per_class:]}
    if calibration & holdout:
        raise AssertionError("Identity roles overlap")
    return calibration, holdout


def score_vector(rows, method, selection):
    name = selection.get(method, method)
    return np.asarray([row[name] for row in rows], float)


def summarize(raw, spec):
    calibration, holdout = identity_roles(
        raw, int(spec["identities_per_class"]), int(spec["calibration_per_class"]))
    rng = np.random.default_rng(int(spec["bootstrap_seed"]))
    result, choices, differences, conditions = [], [], [], []
    for q in map(float, spec["q_values"]):
        for step in map(int, spec["prefix_steps"]):
            group = [row for row in raw if row["q"] == q and row["steps"] == step]
            cal = [row for row in group if row["canary"] in calibration]
            test = [row for row in group if row["canary"] in holdout]
            if len(cal) != 2 * len(calibration) or len(test) != 2 * len(holdout):
                raise ValueError("Missing world/identity rows")
            if any({r["label"] for r in group if r["canary"] == identity} != {0, 1}
                   for identity in calibration | holdout):
                raise ValueError("Missing paired world")
            fields = list(group[0])
            lira = [name for name in fields if name.startswith("endpoint_lira_")]
            rmia = [name for name in fields if name.startswith("endpoint_rmia_")]
            cal_labels = np.asarray([row["label"] for row in cal], int)
            def best(options):
                values = [(auc([row[name] for row in cal], cal_labels), -index, name)
                          for index, name in enumerate(options)]
                return max(values)[2]
            selected = {
                "endpoint_lira_selected": best(lira),
                "endpoint_rmia_selected": best(rmia),
            }
            selected["endpoint_best_selected"] = best([
                "endpoint_loss", selected["endpoint_lira_selected"],
                selected["endpoint_rmia_selected"]])
            choices.append(dict(q=q, steps=step, **selected,
                                calibration_identities=len(calibration),
                                holdout_identities=len(holdout)))
            methods = ["trajectory_mixture", "trajectory_sum_llr",
                       "trajectory_raw_alignment", "endpoint_loss",
                       "endpoint_lira_fixed", "endpoint_lira_z",
                       "endpoint_rmia_a050_g100", "endpoint_lira_selected",
                       "endpoint_rmia_selected", "endpoint_best_selected"]
            labels = np.asarray([row["label"] for row in test], int)
            identities = sorted(holdout)
            paired = {identity: [row for row in test if row["canary"] == identity]
                      for identity in identities}
            draws = rng.integers(0, len(identities), size=(
                int(spec["bootstrap_repetitions"]), len(identities)))
            boot = {}
            point = {}
            for method in methods:
                scores = score_vector(test, method, selected)
                point[method] = auc(scores, labels)
                sample_auc = []
                for draw in draws:
                    sample = [row for index in draw for row in paired[identities[index]]]
                    sample_auc.append(auc(score_vector(sample, method, selected),
                                          [row["label"] for row in sample]))
                boot[method] = np.asarray(sample_auc)
                low, high = np.quantile(sample_auc, [.025, .975])
                fpr, tpr, _ = roc(scores, labels)
                empirical_tpr = float(np.max(tpr[fpr <= .05]))
                cal_scores = score_vector(cal, method, selected)
                threshold = threshold_at_fpr(cal_scores, cal_labels, .05)
                achieved_tpr, achieved_fpr = rates(scores, labels, threshold)
                result.append(dict(
                    q=q, steps=step, method=method, selected_variant=selected.get(method, method),
                    auc=point[method], auc_ci_low=float(low), auc_ci_high=float(high),
                    empirical_tpr_at_fpr_5=empirical_tpr,
                    calibrated_tpr_5=achieved_tpr, calibrated_fpr_5=achieved_fpr,
                    holdout_members=len(holdout), holdout_nonmembers=len(holdout),
                    calibration_members=len(calibration),
                    calibration_nonmembers=len(calibration),
                    mean_public_accuracy=float(np.mean(
                        [row["endpoint_public_accuracy"] for row in test])),
                ))
            for method in ("trajectory_mixture", "trajectory_sum_llr",
                           "trajectory_raw_alignment"):
                for comparator in ("endpoint_best_selected", "endpoint_lira_selected",
                                   "endpoint_rmia_selected", "endpoint_loss"):
                    delta = boot[method] - boot[comparator]
                    low, high = np.quantile(delta, [.025, .975])
                    differences.append(dict(
                        q=q, steps=step, trajectory_method=method,
                        comparator_method=comparator,
                        comparator_variant=selected.get(comparator, comparator),
                        auc_gain=point[method] - point[comparator],
                        gain_ci_low=float(low), gain_ci_high=float(high)))
            delta = boot["trajectory_mixture"] - boot["trajectory_raw_alignment"]
            low, high = np.quantile(delta, [.025, .975])
            differences.append(dict(
                q=q, steps=step, trajectory_method="trajectory_mixture",
                comparator_method="trajectory_raw_alignment",
                comparator_variant="trajectory_raw_alignment",
                auc_gain=point["trajectory_mixture"] -
                         point["trajectory_raw_alignment"],
                gain_ci_low=float(low), gain_ci_high=float(high)))
            positive = [row for row in test if row["label"] == 1]
            conditions.append(dict(q=q, steps=step, expected_appearances=q * step,
                observed_mean_appearances=float(np.mean([r["appearances"] for r in positive])),
                zero_appearance_fraction=float(np.mean([r["appearances"] == 0 for r in positive])),
                expected_zero_fraction=(1-q)**step,
                mean_public_accuracy=float(np.mean(
                    [r["endpoint_public_accuracy"] for r in test]))))
    return result, choices, differences, conditions


def plot(results, out):
    methods = [("trajectory_mixture", "q-aware trajectory", "#155e75"),
               ("trajectory_raw_alignment", "raw trajectory alignment", "#c46b33"),
               ("endpoint_lira_selected", "LiRA, final model", "#4b77a9"),
               ("endpoint_rmia_selected", "RMIA, final model", "#884c86")]
    q_values = sorted({row["q"] for row in results}, reverse=True)
    fig, axes = plt.subplots(1, len(q_values), figsize=(5 * len(q_values), 3.6),
                             sharey=True, squeeze=False)
    for ax, q in zip(axes[0], q_values):
        for method, label, color in methods:
            selected = sorted((row for row in results
                               if row["q"] == q and row["method"] == method),
                              key=lambda row: row["steps"])
            ax.plot([r["steps"] for r in selected], [r["auc"] for r in selected],
                    marker="o", color=color, label=label)
            ax.fill_between([r["steps"] for r in selected],
                            [r["auc_ci_low"] for r in selected],
                            [r["auc_ci_high"] for r in selected],
                            color=color, alpha=.11)
        ax.set_title(f"Participation q={q:g}")
        ax.set_xlabel("Released updates")
        ax.set_xticks(sorted({r["steps"] for r in results if r["q"] == q}))
        ax.axhline(.5, color="#555", lw=.8)
        ax.grid(alpha=.17)
    axes[0, 0].set_ylabel("Held-out identity AUC")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"trajectory_vs_endpoint.{suffix}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if not (run / "completion.json").exists():
        raise ValueError("Target runs are incomplete")
    spec = json.loads((run / "config.json").read_text())
    raw = read_rows(run / "score_rows.csv")
    summary, selections, differences, conditions = summarize(raw, spec)
    write_rows(out / "summary.csv", summary)
    write_rows(out / "selections.csv", selections)
    write_rows(out / "paired_differences.csv", differences)
    write_rows(out / "conditions.csv", conditions)
    (out / "provenance.json").write_text((run / "completion.json").read_text())
    plot(summary, out)
    print(f"Wrote {out}; {len(raw)} private score rows stayed in ignored runs/")


if __name__ == "__main__":
    main()
