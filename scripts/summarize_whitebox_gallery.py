"""Regenerate aggregate-only equal-access results from ignored trial rows."""
import argparse
import json
from pathlib import Path

import numpy as np

from gaussproof.whitebox_gallery import (plot_summary, read_csv,
                                         select_methods, summarize, write_csv)


def paired_differences(rows, config, selected, evaluation):
    rng = np.random.default_rng(config["seed"] + 983)
    results = []
    for distribution in config["distributions"]:
        modes = ("matched",) if distribution == "balanced" else (
            "matched", "balanced_mismatch")
        for background in modes:
            for steps in config["prefix_steps"]:
                cell = [r for r in rows if r["world"] == "present"
                        and r["distribution"] == distribution
                        and r["background"] == background
                        and r["steps"] == steps
                        and r["true_position"] in evaluation]
                def identity_values(method, metric):
                    return np.asarray([np.mean([r[metric] for r in cell
                        if r["method"] == method and r["true_position"] == identity])
                        for identity in evaluation])
                comparisons = [
                    ("top1", "mixture_llr", "alignment", "top1"),
                    ("top1", selected["sparse"], "mixture_llr", "top1"),
                    ("same_class_top1", "mixture_llr", "alignment", "same_class_top1"),
                    ("same_class_top1", selected["sparse"], "mixture_llr", "same_class_top1"),
                    ("mrr", selected["sparse"], "mixture_llr", "mrr"),
                ]
                for metric, first, second, column in comparisons:
                    delta = identity_values(first, column) - identity_values(second, column)
                    indices = rng.integers(len(evaluation), size=(
                        config["bootstrap_repetitions"], len(evaluation)))
                    low, high = np.quantile(delta[indices].mean(1), [.025, .975])
                    results.append(dict(distribution=distribution, background=background,
                                        steps=steps, metric=metric, first=first, second=second,
                                        difference=float(delta.mean()), ci_low=float(low),
                                        ci_high=float(high), identities=len(evaluation)))
                reference_method = "mixture_llr"
                for method, first_column, second_column in (
                    ("posterior", "background_mse", "posterior_mse"),
                    (selected["sparse"], "background_mse", "gradient_mse"),
                    ("posterior_vs_sparse", "gradient_mse", "posterior_mse"),
                ):
                    first_method = (selected["sparse"] if first_column == "gradient_mse"
                                    else reference_method)
                    second_method = (selected["sparse"] if second_column == "gradient_mse"
                                     else reference_method)
                    gain = (identity_values(first_method, first_column)
                            - identity_values(second_method, second_column))
                    indices = rng.integers(len(evaluation), size=(
                        config["bootstrap_repetitions"], len(evaluation)))
                    low, high = np.quantile(gain[indices].mean(1), [.025, .975])
                    results.append(dict(distribution=distribution, background=background,
                                        steps=steps, metric="mse_reduction", first=method,
                                        second="background" if first_column == "background_mse"
                                        else selected["sparse"], difference=float(gain.mean()),
                                        ci_low=float(low), ci_high=float(high),
                                        identities=len(evaluation)))
    return results


def null_controls(rows, config, selected, evaluation):
    controls = []
    for distribution in config["distributions"]:
        for steps in config["prefix_steps"]:
            for method in ("mixture_llr", selected["sparse"]):
                for world in ("present", "absent"):
                    selected_rows = [r for r in rows if r["world"] == world
                        and r["distribution"] == distribution
                        and r["background"] == "matched"
                        and r["steps"] == steps and r["method"] == method
                        and r["true_position"] in evaluation[:config["null_identities"]]]
                    # Evaluation repetitions use adjacent seeds; keep the first
                    # seed per identity to pair with its absent-world control.
                    first_by_id = {}
                    for row in sorted(selected_rows, key=lambda item: item["seed"]):
                        first_by_id.setdefault(row["true_position"], row)
                    values = [row["max_score"] for row in first_by_id.values()]
                    if values:
                        controls.append(dict(distribution=distribution, steps=steps,
                            method=method, world=world, identities=len(values),
                            median_max_score=float(np.median(values)),
                            min_max_score=float(np.min(values)),
                            max_max_score=float(np.max(values))))
    return controls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    run, report = Path(args.run), Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "config.json").read_text())
    rows = read_csv(run / "trial_rows.csv")
    calibration = list(range(config["calibration_identities"]))
    evaluation = list(range(config["calibration_identities"],
                            config["calibration_identities"] + config["evaluation_identities"]))
    selected = select_methods(rows, config, calibration)
    summary = summarize(rows, config, selected, evaluation)
    write_csv(report / "summary.csv", summary)
    write_csv(report / "paired_differences.csv", paired_differences(
        rows, config, selected, evaluation))
    controls = null_controls(rows, config, selected, evaluation)
    if controls:
        write_csv(report / "null_controls.csv", controls)
    (report / "selected_methods.json").write_text(json.dumps(selected, indent=2))
    plot_summary(summary, selected, report)
    print(f"summarized {len(rows)} trial rows", flush=True)


if __name__ == "__main__":
    main()
