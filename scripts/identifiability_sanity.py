"""Small exact Gaussian-mixture control for joint fingerprint identifiability.

This is a diagnostic of scoring under a fully specified synthetic model,
not an empirical DP-SGD audit or a claim about MNIST records.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from gaussproof.multi_contributor import write_csv
from gaussproof.one_run_audit import audit_guesses, joint_trajectory_score


def exact_bit_scores(observations, bank, q, sigma):
    """Enumerate all dataset-bit and active subsets for at most ten canaries."""
    m = len(bank)
    if not 1 <= m <= 10 or not 0 < q < 1 or sigma <= 0:
        raise ValueError("Invalid exact-mixture inputs")
    masks = np.arange(1 << m)
    bits = ((masks[:, None] >> np.arange(m)) & 1).astype(float)
    count = bits.sum(1)
    means = bits @ bank
    log_probability = np.zeros(1 << m)
    for observation in observations:
        log_terms = (-np.sum((means - observation) ** 2, axis=1) / (2*sigma**2)
                     + count * np.log(q/(1-q)))
        for i in range(m):
            upper = (masks & (1 << i)) != 0
            log_terms[upper] = np.logaddexp(
                log_terms[upper], log_terms[masks[upper] ^ (1 << i)])
        log_probability += log_terms + count * np.log1p(-q)
    return np.array([
        logsumexp(log_probability[bits[:, i] == 1]) -
        logsumexp(log_probability[bits[:, i] == 0]) for i in range(m)])


def simulate(draws=8, trials=200, seed=4394):
    rng = np.random.default_rng(seed)
    m, steps, q, sigma = 8, 12, .5, 1.
    rows = []
    for dimension in (2, 3, 4, 8):
        for draw in range(draws):
            bank = rng.standard_normal((m, dimension))
            bank /= np.linalg.norm(bank, axis=1, keepdims=True)
            gram = bank @ bank.T
            rank = float(np.trace(gram)**2 / np.sum(gram**2))
            grams = np.broadcast_to(gram, (steps, m, m))
            counts = np.zeros(3)
            for _ in range(trials):
                bits = (rng.random(m) < .5).astype(int)
                active = (rng.random((steps, m)) < q) * bits
                observation = active @ bank + sigma*rng.standard_normal((steps, dimension))
                projections = observation @ bank.T
                independent = np.logaddexp(np.log1p(-q), np.log(q) +
                    (projections - .5*np.diag(gram)) / sigma**2).sum(axis=0)
                joint = joint_trajectory_score(grams, projections, sigma**2, q)
                exact = exact_bit_scores(observation, bank, q, sigma)
                truth = np.where(bits, 1, -1)
                for i, score in enumerate((independent, joint, exact)):
                    counts[i] += audit_guesses(score, truth, 2, 2)[0]
            for method, count in zip(("independent_mixture", "joint_mean_field",
                                      "exact_joint"), counts):
                rows.append(dict(dimension=dimension, dictionary=draw,
                                 effective_rank=rank, method=method,
                                 mean_correct_of_4=float(count/trials),
                                 trials=trials))
    return rows


def summarize(rows, seed=561):
    rng = np.random.default_rng(seed)
    out, paired = [], []
    for dimension in sorted({r["dimension"] for r in rows}):
        group = [r for r in rows if r["dimension"] == dimension]
        draws = sorted({r["dictionary"] for r in group})
        sample = rng.integers(len(draws), size=(5000, len(draws)))
        by_method = {method: np.array([next(r["mean_correct_of_4"] for r in group
                    if r["dictionary"] == draw and r["method"] == method)
                    for draw in draws]) for method in
                    ("independent_mixture", "joint_mean_field", "exact_joint")}
        rank = np.array([next(r["effective_rank"] for r in group
                         if r["dictionary"] == draw) for draw in draws])
        for method, values in by_method.items():
            lo, hi = np.quantile(values[sample].mean(axis=1), [.025, .975])
            out.append(dict(dimension=dimension, effective_rank=float(rank.mean()),
                method=method, mean_correct_of_4=float(values.mean()),
                ci_low=float(lo), ci_high=float(hi), dictionaries=len(draws)))
        for method in ("joint_mean_field", "exact_joint"):
            difference = by_method[method] - by_method["independent_mixture"]
            lo, hi = np.quantile(difference[sample].mean(axis=1), [.025, .975])
            paired.append(dict(dimension=dimension, first=method,
                               second="independent_mixture",
                               mean_gain=float(difference.mean()),
                               ci_low=float(lo), ci_high=float(hi),
                               dictionaries=len(draws)))
    return out, paired


def plot(summary, report):
    dimensions = sorted({r["dimension"] for r in summary})
    methods = ("independent_mixture", "joint_mean_field", "exact_joint")
    colors = ("#267c91", "#a24d7c", "#324d3c")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for j, (method, color) in enumerate(zip(methods, colors)):
        cells = [next(r for r in summary if r["dimension"] == d and
                      r["method"] == method) for d in dimensions]
        x = np.arange(len(dimensions)) + (j-1)*.24
        means = np.array([r["mean_correct_of_4"] for r in cells])
        error = np.array([[r["mean_correct_of_4"]-r["ci_low"] for r in cells],
                          [r["ci_high"]-r["mean_correct_of_4"] for r in cells]])
        ax.bar(x, means, .23, yerr=error, capsize=2.5, color=color,
               label=method.replace("_", " "))
    ax.set_xticks(np.arange(len(dimensions)), [f"d={d}" for d in dimensions])
    ax.set_ylabel("Correct of four fixed guesses")
    ax.set_ylim(0, 4)
    ax.axhline(2, ls="--", lw=1, color="black", alpha=.6)
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(.5, -0.25))
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(report / f"identifiability_sanity.{extension}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()
    if args.draws < 1 or args.trials < 1:
        raise ValueError("Need positive dictionaries and trials")
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    rows = simulate(args.draws, args.trials)
    summary, paired = summarize(rows)
    write_csv(report / "per_dictionary.csv", rows)
    write_csv(report / "summary_with_ci.csv", summary)
    write_csv(report / "paired_differences.csv", paired)
    plot(summary, report)


if __name__ == "__main__":
    main()
