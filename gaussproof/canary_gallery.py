"""Recover real held-out images from a candidate gallery across data distributions."""
import argparse
import copy
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

from .canary_audit import update
from .data import digest, load_mnist
from .models import clip_gradients, initialize, per_record_gradients


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def class_probabilities(name, canary_label):
    if name == "balanced":
        probabilities = np.ones(10)
    elif name == "long_tail":
        probabilities = 1 / np.arange(1, 11, dtype=float)
    elif name == "canary_rare":
        probabilities = np.full(10, .95 / 9); probabilities[canary_label] = .05
    elif name == "canary_dominant":
        probabilities = np.full(10, .30 / 9); probabilities[canary_label] = .70
    else:
        raise ValueError(f"Unknown distribution: {name}")
    return probabilities / probabilities.sum()


def record_weights(ids, labels, class_probability):
    y = labels[ids].numpy()
    counts = np.bincount(y, minlength=10).clip(min=1)
    weights = class_probability[y] / counts[y]
    return weights / weights.sum()


def stratified_gallery(pool, labels, size):
    chosen = []
    by_label = {label: [int(i) for i in pool if int(labels[i]) == label] for label in range(10)}
    cursor = {label: 0 for label in range(10)}
    while len(chosen) < size:
        label = len(chosen) % 10
        chosen.append(by_label[label][cursor[label]])
        cursor[label] += 1
    return np.asarray(chosen, dtype=int)


def gallery_round_scores(observation, fingerprints, background, sigma, clip, batch_size,
                         probability):
    residual = fingerprints - background[None]
    delta = observation - background
    shift = residual / batch_size
    variance = (sigma * clip / batch_size) ** 2
    llr = (shift * delta[None]).sum(1) / variance - .5 * (shift.square().sum(1)) / variance
    if probability == 1:
        mixture = llr
    else:
        mixture = torch.logaddexp(
            torch.full_like(llr, float(np.log1p(-probability))),
            llr + float(np.log(probability)))
    return llr.numpy(), mixture.numpy()


def run_gallery_sequence(initial, x, labels, population, background_ids, gallery,
                         true_position, cfg, experiment, distribution, seed):
    model = copy.deepcopy(initial)
    rng = np.random.default_rng(seed)
    noise_rng = torch.Generator().manual_seed(seed + 100000)
    canary = int(gallery[true_position]); canary_label = int(labels[canary])
    probabilities = class_probabilities(distribution, canary_label)
    population_weights = record_weights(population, labels, probabilities)
    background_weights = {
        "matched": torch.tensor(record_weights(background_ids, labels, probabilities), dtype=torch.float32),
        "balanced": torch.tensor(record_weights(
            background_ids, labels, class_probabilities("balanced", canary_label)), dtype=torch.float32),
    }
    cumulative = {mode: {"sequence_llr": np.zeros(len(gallery)),
                         "mixture_llr": np.zeros(len(gallery))}
                  for mode in background_weights}
    checkpoints = {}; included = []
    probe_ids = np.r_[gallery, background_ids]
    for step in range(max(experiment["prefix_steps"])):
        probe = clip_gradients(per_record_gradients(model, x[probe_ids], labels[probe_ids]), cfg["clip"])
        fingerprints = probe[:len(gallery)]
        present = bool(rng.random() < cfg["canary_probability"])
        count = cfg["batch_size"] - int(present)
        other = rng.choice(population, count, replace=False, p=population_weights)
        ids = np.r_[[canary], other] if present else other
        clean = clip_gradients(per_record_gradients(model, x[ids], labels[ids]), cfg["clip"]).mean(0)
        noise_sd = experiment["sigma"] * cfg["clip"] / cfg["batch_size"]
        release = clean + torch.randn(clean.shape, generator=noise_rng) * noise_sd
        for mode, weights in background_weights.items():
            background = (probe[len(gallery):] * weights[:, None]).sum(0)
            llr, mixture = gallery_round_scores(
                release, fingerprints, background, experiment["sigma"], cfg["clip"],
                cfg["batch_size"], cfg["canary_probability"])
            cumulative[mode]["sequence_llr"] += llr
            cumulative[mode]["mixture_llr"] += mixture
        included.append(int(present))
        if step + 1 in experiment["prefix_steps"]:
            checkpoints[step + 1] = dict(
                scores={mode: {method: score.copy() for method, score in values.items()}
                        for mode, values in cumulative.items()},
                inclusion_rate=float(np.mean(included)))
        update(model, release, cfg["learning_rate"])
    return checkpoints


def rank_result(score, true_position, gallery, labels, images):
    order = np.argsort(-score, kind="stable")
    rank = int(np.flatnonzero(order == true_position)[0] + 1)
    closed = int(order[0])
    truth_label = int(labels[gallery[true_position]])
    same_class = np.asarray([position for position in range(len(gallery))
                             if int(labels[gallery[position]]) == truth_label])
    same_order = same_class[np.argsort(-score[same_class], kind="stable")]
    same_rank = int(np.flatnonzero(same_order == true_position)[0] + 1)
    open_order = order[order != true_position]
    opened = int(open_order[0])
    truth = images[gallery[true_position], 0]
    recovered = images[gallery[opened], 0]
    return dict(rank=rank, top1=int(rank == 1), top5=int(rank <= 5),
                reciprocal_rank=1 / rank, predicted_position=closed,
                predicted_index=int(gallery[closed]), predicted_label=int(labels[gallery[closed]]),
                predicted_label_match=int(labels[gallery[closed]] == truth_label),
                same_class_candidates=len(same_class), same_class_rank=same_rank,
                same_class_top1=int(same_rank == 1), same_class_reciprocal_rank=1 / same_rank,
                open_position=opened, open_index=int(gallery[opened]),
                open_label=int(labels[gallery[opened]]),
                open_label_match=int(labels[gallery[opened]] == labels[gallery[true_position]]),
                open_pixel_mse=float((truth - recovered).square().mean()),
                true_score=float(score[true_position]),
                top1_margin=float(score[order[0]] - score[order[1]]))


def summarize(rows, distributions, steps, repetitions, seed):
    rng = np.random.default_rng(seed); summary = []
    for distribution in distributions:
        for background_model in ["matched", "balanced"]:
            for length in steps:
                selected = [row for row in rows if row["distribution"] == distribution and
                            row["background_model"] == background_model and
                            row["steps"] == length and row["method"] == "mixture_llr"]
                canaries = sorted(set(row["true_position"] for row in selected))
                metrics = ["top1", "top5", "reciprocal_rank", "predicted_label_match",
                           "same_class_top1", "same_class_reciprocal_rank",
                           "open_label_match", "open_pixel_mse"]
                estimates = {metric: float(np.mean([row[metric] for row in selected])) for metric in metrics}
                boot = {metric: [] for metric in metrics}
                for _ in range(repetitions):
                    sampled = rng.choice(canaries, len(canaries), replace=True)
                    sample_rows = []
                    for slot, canary in enumerate(sampled):
                        available = [row for row in selected if row["true_position"] == canary]
                        chosen = rng.choice(len(available), len(available), replace=True)
                        sample_rows.extend(available[i] for i in chosen)
                    for metric in metrics:
                        boot[metric].append(np.mean([row[metric] for row in sample_rows]))
                result = dict(distribution=distribution, background_model=background_model, steps=length,
                              samples=len(selected), gallery_size=32, **estimates)
                for metric in ["top1", "top5", "reciprocal_rank", "predicted_label_match",
                               "same_class_top1", "same_class_reciprocal_rank"]:
                    result[metric + "_ci_low"], result[metric + "_ci_high"] = map(
                        float, np.quantile(boot[metric], [.025, .975]))
                summary.append(result)
    return summary


def plot_summary(summary, output):
    fig, ax = plt.subplots(figsize=(7.7, 4.5))
    for distribution in sorted(set(row["distribution"] for row in summary)):
        selected = [row for row in summary if row["distribution"] == distribution and
                    row["background_model"] == "matched"]
        line, = ax.plot([row["steps"] for row in selected],
                        [row["top1"] for row in selected],
                        marker="o", label=distribution.replace("_", " "))
        ax.fill_between([row["steps"] for row in selected],
                        [row["top1_ci_low"] for row in selected],
                        [row["top1_ci_high"] for row in selected],
                        color=line.get_color(), alpha=.08)
    ax.axhline(1 / 32, color="black", linestyle="--", linewidth=1, label="random top-1")
    ax.set_xscale("log", base=2); ax.set_xticks([16, 32, 64, 128], labels=["16", "32", "64", "128"])
    ax.set_ylim(0, .4); ax.set_xlabel("Observed DP-SGD releases")
    ax.set_ylabel("Exact-image top-1 recovery")
    ax.set_title("Real-sample recovery from a 32-image gallery (σ=4)")
    ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"gallery_recovery.{extension}", dpi=220)
    plt.close(fig)


def plot_background_robustness(summary, output):
    distributions = sorted(set(row["distribution"] for row in summary))
    x = np.arange(len(distributions)); width = .36
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for offset, mode in [(-width / 2, "matched"), (width / 2, "balanced")]:
        values = [next(row["top1"] for row in summary if row["distribution"] == distribution and
                       row["background_model"] == mode and row["steps"] == 128)
                  for distribution in distributions]
        ax.bar(x + offset, values, width, label=f"{mode} public background")
    ax.axhline(1 / 32, color="black", linestyle="--", linewidth=1, label="random top-1")
    ax.set_xticks(x, labels=[value.replace("_", " ") for value in distributions])
    ax.set_ylim(0, 1.02); ax.set_ylabel("Exact-image top-1 recovery")
    ax.set_title("Background-distribution mismatch at 128 releases (σ=4)")
    ax.grid(axis="y", alpha=.2); ax.legend(fontsize=8); fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"background_mismatch.{extension}", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for offset, mode in [(-width / 2, "matched"), (width / 2, "balanced")]:
        values = [next(row["open_label_match"] for row in summary if
                       row["distribution"] == distribution and
                       row["background_model"] == mode and row["steps"] == 128)
                  for distribution in distributions]
        ax.bar(x + offset, values, width, label=f"{mode} public background")
    ax.axhline(.071, color="black", linestyle="--", linewidth=1,
               label="random open-gallery label")
    ax.set_xticks(x, labels=[value.replace("_", " ") for value in distributions])
    ax.set_ylim(0, 1.02); ax.set_ylabel("Digit-label agreement after removing true image")
    ax.set_title("Class leakage under background-distribution mismatch")
    ax.grid(axis="y", alpha=.2); ax.legend(fontsize=8); fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"background_class_confound.{extension}", dpi=220)
    plt.close(fig)


def plot_examples(rows, gallery, x, labels, distribution, output):
    examples = [row for row in rows if row["distribution"] == distribution and
                row["background_model"] == "matched" and
                row["method"] == "mixture_llr" and row["repetition"] == 0]
    by_key = {(row["true_position"], row["steps"]): row for row in examples}
    # Keep the visualization readable when the evaluation contains more
    # identities; all identities still contribute to aggregate metrics.
    positions = sorted(set(row["true_position"] for row in examples))[:10]
    fig, axes = plt.subplots(3, len(positions), figsize=(1.45 * len(positions), 4.5))
    for column, position in enumerate(positions):
        truth_index = int(gallery[position])
        entries = [("truth", truth_index),
                   ("16 releases", by_key[position, 16]["predicted_index"]),
                   ("128 releases", by_key[position, 128]["predicted_index"])]
        for row_index, (label, image_index) in enumerate(entries):
            axes[row_index, column].imshow(x[image_index, 0], cmap="gray", vmin=0, vmax=1)
            axes[row_index, column].axis("off")
            if column == 0: axes[row_index, column].set_ylabel(label, fontsize=9)
            if row_index == 0: axes[row_index, column].set_title(str(int(labels[truth_index])), fontsize=9)
    fig.suptitle(f"Truth and recovered real images: {distribution.replace('_', ' ')}", y=.99)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"examples_{distribution}.{extension}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Canary gallery reconstruction across distributions")
    parser.add_argument("--source", default="runs/canary_holdout")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/canary_gallery.json")
    parser.add_argument("--output", default="runs/canary_gallery")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").exists():
        raise RuntimeError("Completed output exists; choose a new output directory")
    experiment = json.loads(Path(args.config).read_text())
    cfg = json.loads((source / "config.json").read_text())
    splits = json.loads((source / "splits.json").read_text())
    source_completion = json.loads((source / "completion.json").read_text())
    if source_completion["dataset_sha256"] != digest(args.data):
        raise ValueError("Dataset differs from the source holdout run")
    torch.set_num_threads(1); started = time.time()
    x, labels = load_mnist(args.data)
    initial = initialize(cfg["seed"])
    initial.load_state_dict(torch.load(source / "public_initial_cnn.pt", weights_only=True)); initial.eval()
    gallery = stratified_gallery(np.asarray(splits["test"]), labels, experiment["gallery_size"])
    true_positions = list(range(experiment["true_candidates"]))
    population = np.asarray(splits["population"]); background = np.asarray(splits["background"])
    np.savez_compressed(output / "gallery.npz", indices=gallery,
                        images=x[gallery, 0].numpy(), labels=labels[gallery].numpy())
    rows = []
    for distribution_index, distribution in enumerate(experiment["distributions"]):
        print(f"distribution {distribution_index + 1}/{len(experiment['distributions'])}: {distribution}", flush=True)
        for position in true_positions:
            for repetition in range(experiment["repetitions"]):
                seed = experiment["seed"] + int(gallery[position]) * 10 + repetition
                checkpoints = run_gallery_sequence(
                    initial, x, labels, population, background, gallery, position,
                    cfg, experiment, distribution, seed)
                for length, values in checkpoints.items():
                    for background_model, model_scores in values["scores"].items():
                        for method in ["sequence_llr", "mixture_llr"]:
                            result = rank_result(model_scores[method], position, gallery, labels, x)
                            rows.append(dict(
                                distribution=distribution, background_model=background_model,
                                sigma=experiment["sigma"], steps=length,
                                method=method, true_position=position, true_index=int(gallery[position]),
                                true_label=int(labels[gallery[position]]), repetition=repetition,
                                inclusion_rate=values["inclusion_rate"], **result))
        write_csv(output / "retrievals.csv", rows)
    summary = summarize(rows, experiment["distributions"], experiment["prefix_steps"],
                        experiment["bootstrap_repetitions"], experiment["seed"])
    write_csv(output / "summary.csv", summary); plot_summary(summary, output)
    plot_background_robustness(summary, output)
    for distribution in experiment["distributions"]:
        plot_examples(rows, gallery, x, labels, distribution, output)
    probabilities = []
    for distribution in experiment["distributions"]:
        for canary_label in range(10):
            row = dict(distribution=distribution, canary_label=canary_label)
            row.update({f"p_digit_{digit}": value for digit, value in
                        enumerate(class_probabilities(distribution, canary_label))})
            probabilities.append(row)
    write_csv(output / "class_probabilities.csv", probabilities)
    random_open_label, random_open_mse = [], []
    for position in true_positions:
        other = np.arange(len(gallery)) != position
        random_open_label.append(float((labels[gallery[other]] == labels[gallery[position]]).float().mean()))
        random_open_mse.append(float((x[gallery[other], 0] - x[gallery[position], 0]).square().mean((1, 2)).mean()))
    write_csv(output / "baselines.csv", [dict(
        gallery_size=len(gallery), random_top1=1 / len(gallery), random_top5=5 / len(gallery),
        random_reciprocal_rank=float(np.sum(1 / np.arange(1, len(gallery) + 1)) / len(gallery)),
        label_oracle_random_exact_top1=float(np.mean([
            1 / int((labels[gallery] == labels[gallery[position]]).sum()) for position in true_positions])),
        random_open_label_match=float(np.mean(random_open_label)),
        random_open_pixel_mse=float(np.mean(random_open_mse)))])
    info = dict(
        seconds=time.time() - started, dataset_sha256=digest(args.data),
        gallery_size=len(gallery), true_candidates=len(true_positions),
        trajectories=len(experiment["distributions"]) * len(true_positions) * experiment["repetitions"],
        distributions=experiment["distributions"], sigma=experiment["sigma"],
        prefixes=experiment["prefix_steps"], reconstruction="closed-gallery exact-image retrieval",
        open_control="true image removed before alternative retrieval",
        source_completion_sha256=hashlib.sha256((source / "completion.json").read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), torch=torch.__version__)
    (output / "config.json").write_text(json.dumps(experiment, indent=2))
    (output / "completion.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
