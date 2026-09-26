"""Exact long-trajectory audit for persistent high-noise canary evidence."""
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
from .canary_learned import mixture_llr
from .data import digest, load_mnist
from .metrics import auc
from .models import initialize


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def prefix_scores(run, sigma, clip, batch_size, probability, lengths):
    rows = []
    for length in lengths:
        truncated = {key: value[:length] for key, value in run.items()}
        score = analytic_scores(truncated, sigma * clip, batch_size)
        rows.append(dict(steps=length, sequence_llr=score["sequence_llr"],
                         mixture_llr=mixture_llr(score["round_llr"], probability),
                         last_llr=score["last_llr"]))
    return rows


def cluster_bootstrap(rows, methods, lengths, repetitions, seed):
    """Resample canary identities and paired positive/negative sequence IDs."""
    rng = np.random.default_rng(seed)
    canaries = sorted(set(row["canary"] for row in rows))
    sequences = sorted(set(row["sequence"] for row in rows))
    observed, samples = {}, {(method, length): [] for method in methods for length in lengths}
    for method in methods:
        for length in lengths:
            selected = [row for row in rows if row["steps"] == length]
            observed[method, length] = auc([row[method] for row in selected],
                                            [row["label"] for row in selected])
    lookup = {(row["canary"], row["sequence"], row["label"], row["steps"]): row
              for row in rows}
    for _ in range(repetitions):
        chosen_canaries = rng.choice(canaries, len(canaries), replace=True)
        chosen_sequences = {slot: rng.choice(sequences, len(sequences), replace=True)
                            for slot in range(len(chosen_canaries))}
        for method in methods:
            for length in lengths:
                values, labels = [], []
                for slot, canary in enumerate(chosen_canaries):
                    for sequence in chosen_sequences[slot]:
                        for label in [0, 1]:
                            values.append(lookup[int(canary), int(sequence), label, length][method])
                            labels.append(label)
                samples[method, length].append(auc(values, labels))
    summary = []
    baseline = min(lengths)
    for method in methods:
        baseline_samples = np.asarray(samples[method, baseline])
        for length in lengths:
            distribution = np.asarray(samples[method, length])
            low, high = np.quantile(distribution, [.025, .975])
            difference = distribution - baseline_samples
            dlow, dhigh = np.quantile(difference, [.025, .975])
            summary.append(dict(
                steps=length, windows=length / baseline, method=method,
                auc=observed[method, length], auc_ci_low=float(low), auc_ci_high=float(high),
                auc_difference_from_16=observed[method, length] - observed[method, baseline],
                difference_ci_low=float(dlow), difference_ci_high=float(dhigh),
                bootstrap_repetitions=repetitions,
                bootstrap_unit="canary identity and paired sequence index"))
    return summary


def plot(summary, output):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    labels = {"sequence_llr": "sum LLR", "mixture_llr": "Bernoulli-mixture LLR"}
    for method, label in labels.items():
        selected = [row for row in summary if row["method"] == method]
        x = np.asarray([row["steps"] for row in selected])
        y = np.asarray([row["auc"] for row in selected])
        low = np.asarray([row["auc_ci_low"] for row in selected])
        high = np.asarray([row["auc_ci_high"] for row in selected])
        ax.plot(x, y, marker="o", label=label)
        ax.fill_between(x, low, high, alpha=.16)
    ax.axhline(.5, color="black", linewidth=1, alpha=.6)
    ax.set_xscale("log", base=2); ax.set_xticks([16, 32, 64, 128], labels=["16", "32", "64", "128"])
    ax.set_ylim(.45, 1.02); ax.set_xlabel("Observed DP-SGD releases")
    ax.set_ylabel("Unseen-canary holdout AUC")
    ax.set_title("High-noise fingerprint evidence accumulates over time (σ=4)")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"long_trajectory_auc.{extension}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Exact long high-noise canary trajectory")
    parser.add_argument("--source", default="runs/canary_holdout")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/canary_long_trajectory.json")
    parser.add_argument("--output", default="runs/canary_long_sigma4")
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
    if lengths[0] != source_cfg["steps"] or lengths[-1] < lengths[0]:
        raise ValueError("Prefixes must start at the source trajectory length")
    cfg = dict(source_cfg); cfg["steps"] = lengths[-1]
    if experiment["sequences_per_canary"] > source_cfg["sequences_per_canary"]:
        raise ValueError("Sequence count must preserve source prefix pairing")
    torch.set_num_threads(1); started = time.time()
    x, labels = load_mnist(args.data)
    initial = initialize(source_cfg["seed"])
    initial.load_state_dict(torch.load(source / "public_initial_cnn.pt", weights_only=True)); initial.eval()
    population = np.asarray(splits["population"]); background = np.asarray(splits["background"])
    canaries = [int(value) for value in splits["test_canaries"]]
    sigma = float(experiment["sigma"]); rows = []
    for canary_index, canary in enumerate(canaries):
        print(f"canary {canary_index + 1}/{len(canaries)}: {canary}", flush=True)
        for label, probability in [(0, 0.0), (1, source_cfg["canary_probability"])]:
            for sequence in range(experiment["sequences_per_canary"]):
                seed = source_cfg["seed"] + int(sigma * 10000) + canary + sequence
                run = run_sequence(initial, x, labels, population, background, canary,
                                   cfg, sigma, seed, probability)
                for score in prefix_scores(run, sigma, cfg["clip"], cfg["batch_size"],
                                           source_cfg["canary_probability"], lengths):
                    rows.append(dict(sigma=sigma, canary=canary, label=label, sequence=sequence,
                                     inclusion_rate=float(run["included"][:score["steps"]].mean()), **score))
        write_csv(output / "long_sequences.csv", rows)
    summary = cluster_bootstrap(rows, ["sequence_llr", "mixture_llr"], lengths,
                                experiment["bootstrap_repetitions"], experiment["seed"])
    write_csv(output / "summary.csv", summary)
    privacy = []
    for length in lengths:
        rho = 2 * length / sigma ** 2
        privacy.append(dict(
            sigma=sigma, steps=length, delta=source_cfg["delta"], rho=rho,
            epsilon_upper=rho + 2 * np.sqrt(rho * np.log(1 / source_cfg["delta"])),
            accounting="replace-one zCDP; no sampling amplification"))
    write_csv(output / "privacy.csv", privacy); plot(summary, output)
    info = dict(
        seconds=time.time() - started, dataset_sha256=digest(args.data), sigma=sigma,
        prefixes=lengths, holdout_canaries=canaries, trajectories=2 * len(canaries) * experiment["sequences_per_canary"],
        exact_evolving_trajectory=True,
        source_completion_sha256=hashlib.sha256((source / "completion.json").read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), torch=torch.__version__)
    (output / "config.json").write_text(json.dumps(experiment, indent=2))
    (output / "completion.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
