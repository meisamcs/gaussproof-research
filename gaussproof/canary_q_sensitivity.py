"""Participation-rate sensitivity and a genuine final-checkpoint baseline.

This experiment reuses the public initialization and disjoint splits from the
holdout-canary study.  It varies the canary's hidden per-round participation
probability q, evaluates prefixes of the same exact evolving trajectory, and
compares the q-aware full-trajectory likelihood with a candidate loss computed
from the endpoint model alone.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .canary_audit import run_sequence, scores as analytic_scores
from .data import digest, load_mnist
from .metrics import auc
from .models import initialize


def mixture_llr(round_llr, probability):
    """Log likelihood ratio for Bernoulli canary inclusion per round."""
    values = np.asarray(round_llr, dtype=np.float64)
    if not 0 < probability <= 1:
        raise ValueError("Inclusion probability must be in (0, 1]")
    if probability == 1:
        return float(values.sum())
    return float(np.logaddexp(
        np.log1p(-probability), np.log(probability) + values
    ).sum())


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stratified_identities(candidate_pool, labels, identities_per_class):
    """Choose the first fixed-split identities per class without cherry-picking."""
    selected = []
    for digit in range(10):
        matches = [int(i) for i in candidate_pool if int(labels[int(i)]) == digit]
        if len(matches) < identities_per_class:
            raise ValueError(f"Candidate pool has only {len(matches)} examples of digit {digit}")
        selected.extend(matches[:identities_per_class])
    return selected


def prefix_scores(run, sigma, clip, batch_size, probability, lengths):
    rows = []
    for length in lengths:
        truncated = {key: value[:length] for key, value in run.items()}
        score = analytic_scores(truncated, sigma * clip, batch_size)
        rows.append(dict(
            steps=length,
            sequence_llr=score["sequence_llr"],
            mixture_llr=mixture_llr(score["round_llr"], probability),
            endpoint_neg_loss=-float(run["endpoint_losses"][length - 1]),
            endpoint_confidence=float(run["endpoint_confidences"][length - 1]),
        ))
    return rows


def summarize(rows, q_values, lengths, repetitions, seed):
    """Identity/paired-sequence cluster bootstrap for AUCs and attack gaps."""
    methods = ["mixture_llr", "endpoint_neg_loss", "endpoint_confidence"]
    observed, samples, rng = {}, {}, np.random.default_rng(seed)
    for q in q_values:
        qrows = [row for row in rows if row["q"] == q]
        canaries = sorted({row["canary"] for row in qrows})
        sequences = sorted({row["sequence"] for row in qrows})
        lookup = {(row["canary"], row["sequence"], row["label"], row["steps"]): row
                  for row in qrows}
        for length in lengths:
            selected = [row for row in qrows if row["steps"] == length]
            for method in methods:
                key = (q, length, method)
                observed[key] = auc([row[method] for row in selected],
                                    [row["label"] for row in selected])
                samples[key] = []
        for _ in range(repetitions):
            chosen_canaries = rng.choice(canaries, len(canaries), replace=True)
            chosen_sequences = [rng.choice(sequences, len(sequences), replace=True)
                                for _ in chosen_canaries]
            for length in lengths:
                for method in methods:
                    values, targets = [], []
                    for slot, canary in enumerate(chosen_canaries):
                        for sequence in chosen_sequences[slot]:
                            for label in (0, 1):
                                values.append(lookup[int(canary), int(sequence), label, length][method])
                                targets.append(label)
                    samples[q, length, method].append(auc(values, targets))

    summary = []
    for q in q_values:
        for length in lengths:
            for method in methods:
                distribution = np.asarray(samples[q, length, method])
                low, high = np.quantile(distribution, [.025, .975])
                summary.append(dict(
                    q=q, steps=length, method=method,
                    auc=observed[q, length, method],
                    auc_ci_low=float(low), auc_ci_high=float(high),
                    expected_inclusions=q * length,
                    probability_at_least_one=1 - (1 - q) ** length,
                    bootstrap_repetitions=repetitions,
                    bootstrap_unit="identity and paired sequence index",
                ))

    comparisons = []
    for q in q_values:
        for length in lengths:
            trajectory = np.asarray(samples[q, length, "mixture_llr"])
            endpoint = np.asarray(samples[q, length, "endpoint_neg_loss"])
            difference = trajectory - endpoint
            low, high = np.quantile(difference, [.025, .975])
            comparisons.append(dict(
                q=q, steps=length,
                trajectory_auc=observed[q, length, "mixture_llr"],
                endpoint_auc=observed[q, length, "endpoint_neg_loss"],
                trajectory_minus_endpoint=observed[q, length, "mixture_llr"]
                    - observed[q, length, "endpoint_neg_loss"],
                difference_ci_low=float(low), difference_ci_high=float(high),
            ))
    return summary, comparisons


def inclusion_summary(rows, q_values, lengths):
    result = []
    for q in q_values:
        for length in lengths:
            positive = [row for row in rows
                        if row["q"] == q and row["steps"] == length and row["label"] == 1]
            rates = np.asarray([row["inclusion_rate"] for row in positive])
            result.append(dict(
                q=q, steps=length, positive_trajectories=len(positive),
                expected_inclusions=q * length,
                observed_mean_inclusions=float((rates * length).mean()),
                observed_zero_inclusion_fraction=float((rates == 0).mean()),
                expected_zero_inclusion_probability=(1 - q) ** length,
            ))
    return result


def plot(summary, comparisons, output):
    colors = {0.5: "#2f6f9f", 0.1: "#3b9b75", 0.02: "#d58936", 0.004: "#a34a66"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax = axes[0]
    for q in sorted(colors, reverse=True):
        selected = [row for row in summary if row["q"] == q and row["method"] == "mixture_llr"]
        x = np.asarray([row["steps"] for row in selected])
        y = np.asarray([row["auc"] for row in selected])
        low = np.asarray([row["auc_ci_low"] for row in selected])
        high = np.asarray([row["auc_ci_high"] for row in selected])
        ax.plot(x, y, marker="o", color=colors[q], label=f"q={q:g}")
        ax.fill_between(x, low, high, color=colors[q], alpha=.12)
    ax.axhline(.5, color="black", linewidth=.9, alpha=.65)
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64, 128], labels=["16", "32", "64", "128"])
    ax.set_ylim(.43, 1.02)
    ax.set_xlabel("Observed releases T")
    ax.set_ylabel("Unseen-identity AUC")
    ax.set_title("(a) q-aware trajectory detector")
    ax.grid(alpha=.18)
    ax.legend(frameon=False, ncol=2)

    ax = axes[1]
    selected = [row for row in comparisons if row["steps"] == 128]
    selected.sort(key=lambda row: row["q"])
    q = np.asarray([row["q"] for row in selected])
    ax.plot(q, [row["trajectory_auc"] for row in selected], marker="o",
            label="full trajectory", color="#2f6f9f")
    ax.plot(q, [row["endpoint_auc"] for row in selected], marker="s",
            linestyle="--", label="final checkpoint", color="#8b5e3c")
    ax.axhline(.5, color="black", linewidth=.9, alpha=.65)
    ax.set_xscale("log")
    ax.set_xticks(q, labels=[f"{value:g}" for value in q])
    ax.set_ylim(.43, 1.02)
    ax.set_xlabel("Per-round participation probability q")
    ax.set_ylabel("AUC at T=128")
    ax.set_title("(b) Trajectory versus endpoint")
    ax.grid(alpha=.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"q_sensitivity_endpoint.{extension}", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Canary participation-rate sensitivity")
    parser.add_argument("--source", default="runs/canary_holdout")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/canary_q_sensitivity.json")
    parser.add_argument("--output", default="runs/canary_q_sensitivity")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").exists():
        raise RuntimeError("Completed output exists; choose a new output directory")
    experiment = json.loads(Path(args.config).read_text())
    source_cfg = json.loads((source / "config.json").read_text())
    splits = json.loads((source / "splits.json").read_text())
    completion = json.loads((source / "completion.json").read_text())
    if completion["dataset_sha256"] != digest(args.data):
        raise ValueError("Dataset differs from source holdout run")
    lengths = sorted(experiment["prefix_steps"])
    q_values = [float(q) for q in experiment["q_values"]]
    cfg = dict(source_cfg)
    cfg["steps"] = lengths[-1]
    cfg["decouple_inclusion_rng"] = True
    torch.set_num_threads(1)
    started = time.time()
    x, labels = load_mnist(args.data)
    initial = initialize(source_cfg["seed"])
    initial.load_state_dict(torch.load(source / "public_initial_cnn.pt", weights_only=True))
    initial.eval()
    population = np.asarray(splits["population"])
    background = np.asarray(splits["background"])
    canaries = stratified_identities(
        splits["test"], labels, int(experiment["identities_per_class"]))
    sigma = float(experiment["sigma"])
    rows = []
    for q_index, q in enumerate(q_values):
        print(f"q={q:g} ({q_index + 1}/{len(q_values)})", flush=True)
        for canary_index, canary in enumerate(canaries):
            print(f"  identity {canary_index + 1}/{len(canaries)}", flush=True)
            for label, probability in ((0, 0.0), (1, q)):
                for sequence in range(int(experiment["sequences_per_identity"])):
                    # All q conditions share the deterministic seed.  Because
                    # participation uses a dedicated RNG stream, lower-q
                    # schedules are exact subsets of higher-q schedules.
                    # Positive/negative trajectories are paired as well.
                    seed = int(experiment["seed"]) + canary * 100 + sequence
                    run = run_sequence(initial, x, labels, population, background,
                                       canary, cfg, sigma, seed, probability)
                    for score in prefix_scores(run, sigma, cfg["clip"], cfg["batch_size"],
                                               q, lengths):
                        steps = score["steps"]
                        rows.append(dict(
                            q=q, sigma=sigma, canary=canary,
                            digit=int(labels[canary]), label=label, sequence=sequence,
                            inclusion_rate=float(run["included"][:steps].mean()),
                            **score,
                        ))
        write_csv(output / "scores.csv", rows)
    summary, comparisons = summarize(
        rows, q_values, lengths, int(experiment["bootstrap_repetitions"]),
        int(experiment["bootstrap_seed"]))
    inclusions = inclusion_summary(rows, q_values, lengths)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "trajectory_vs_endpoint.csv", comparisons)
    write_csv(output / "inclusion_summary.csv", inclusions)
    plot(summary, comparisons, output)
    info = dict(
        seconds=time.time() - started,
        dataset_sha256=digest(args.data), sigma=sigma, q_values=q_values,
        prefixes=lengths, holdout_identities=len(canaries),
        identities_per_class=int(experiment["identities_per_class"]),
        sequences_per_identity=int(experiment["sequences_per_identity"]),
        trajectories=len(q_values) * len(canaries) * 2 * int(experiment["sequences_per_identity"]),
        exact_evolving_trajectory=True,
        endpoint_baseline="negative candidate cross-entropy from final checkpoint only",
        q_conditions_share_nested_inclusion_draws=True,
        source_completion_sha256=hashlib.sha256((source / "completion.json").read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch=torch.__version__,
    )
    (output / "config.json").write_text(json.dumps(experiment, indent=2))
    (output / "identities.json").write_text(json.dumps(canaries, indent=2))
    (output / "completion.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
