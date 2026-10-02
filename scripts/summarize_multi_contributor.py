"""Aggregate only: run-clustered intervals for multi-contributor DP-SGD."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gaussproof.multi_contributor import read_csv, summarize, write_csv


def analyze(run, report, repetitions):
    config = json.loads((run / "config.json").read_text())
    rng = np.random.default_rng(config["seed"] + 983)
    output, pairs, selected = [], [], {}
    for sigma in config["sigmas"]:
        rows = read_csv(run / f"sigma_{sigma:g}.csv")
        summary, sparse = summarize(rows, config)
        selected[str(sigma)] = sparse
        evaluation = [config["seed"] + 10000 + j
                      for j in range(config["evaluation_runs"])]
        by_method = {method: {r["seed"]: r for r in rows if r["method"] == method
                    and r["seed"] in evaluation}
                    for method in ("alignment", "exact", "independent", sparse)}
        index = rng.integers(len(evaluation), size=(repetitions, len(evaluation)))
        def vector(method, column):
            return np.asarray([float(by_method[method][seed][column])
                               for seed in evaluation])
        for method in ("alignment", "exact", "independent", sparse):
            row = next(r for r in summary if r["method"] == method)
            for metric in ("support_auc", "clean_gradient_mse"):
                if np.isnan(row[metric]):
                    continue
                values = vector(method, metric)
                low, high = np.quantile(values[index].mean(1), [.025, .975])
                output.append(dict(sigma=sigma, method=method, metric=metric,
                                   mean=float(values.mean()), ci_low=float(low),
                                   ci_high=float(high), runs=len(evaluation)))
        for name, method, column in (("background", "exact", "background_mse"),
                                     ("prior", "exact", "prior_mse")):
            values = vector(method, column)
            low, high = np.quantile(values[index].mean(1), [.025, .975])
            output.append(dict(sigma=sigma, method=name, metric="clean_gradient_mse",
                               mean=float(values.mean()), ci_low=float(low),
                               ci_high=float(high), runs=len(evaluation)))
        for metric, first, second, sign in (
            ("support_auc", sparse, "exact", 1),
            ("clean_gradient_mse", sparse, "exact", -1),
            ("clean_gradient_mse", sparse, "prior", -1),
            ("clean_gradient_mse", "exact", "prior", -1),
        ):
            def values(method):
                return vector("exact", "prior_mse") if method == "prior" else vector(method, metric)
            difference = sign * (values(first) - values(second))
            low, high = np.quantile(difference[index].mean(1), [.025, .975])
            pairs.append(dict(sigma=sigma, metric=metric,
                first=first, second=second, orientation=("first_minus_second" if sign == 1
                                                  else "second_minus_first_lower_error_better"),
                gain=float(difference.mean()), ci_low=float(low), ci_high=float(high),
                runs=len(evaluation)))
    write_csv(report / "summary_with_ci.csv", output)
    write_csv(report / "paired_differences.csv", pairs)
    (report / "analysis.json").write_text(json.dumps(dict(
        bootstrap_repetitions=repetitions, clustered_unit="independent target training run",
        calibration_and_evaluation_disjoint=True, selected_sparse=selected), indent=2))
    return output, selected


def plot(rows, config, selected, report):
    sigmas = np.asarray(config["sigmas"])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    panels = (("support_auc", ("alignment", "exact", "sparse"),
               "Per-round candidate support AUC"),
              ("clean_gradient_mse", ("prior", "exact", "sparse"),
               "Full-vector clean-gradient error / C²"))
    colors = dict(alignment="#777777", prior="#777777", exact="#147d91",
                  sparse="#df7838")
    labels = dict(alignment="Centered alignment", prior="Public prior",
                  exact="Exact surrogate posterior", sparse="Sparse decoder")
    width = .22
    for axis, (metric, methods, ylabel) in zip(axes, panels):
        for position, method in enumerate(methods):
            chosen = [selected[str(sigma)] if method == "sparse" else method
                      for sigma in sigmas]
            values, lower, upper = [], [], []
            for sigma, name in zip(sigmas, chosen):
                row = next(r for r in rows if r["sigma"] == sigma
                           and r["method"] == name and r["metric"] == metric)
                values.append(row["mean"])
                lower.append(row["mean"] - row["ci_low"])
                upper.append(row["ci_high"] - row["mean"])
            axis.bar(np.arange(len(sigmas)) + (position - 1) * width,
                     values, width, yerr=[lower, upper], capsize=3,
                     color=colors[method], label=labels[method])
        if metric == "support_auc":
            axis.axhline(.5, color="black", ls="--", lw=1)
            axis.set_ylim(.48, .8)
        else:
            axis.set_ylim(0, .15)
        axis.set_xticks(np.arange(len(sigmas)), [f"σ={sigma:g}" for sigma in sigmas])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(report / f"multi_contributor.{ext}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--bootstrap", type=int, default=3000)
    args = parser.parse_args()
    run, report = Path(args.run), Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    rows, selected = analyze(run, report, args.bootstrap)
    plot(rows, json.loads((run / "config.json").read_text()), selected, report)
    print(f"wrote {len(rows)} aggregate rows and paired comparisons", flush=True)


if __name__ == "__main__":
    main()
