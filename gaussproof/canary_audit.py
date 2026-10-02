"""Known-canary detection from a sequence of noisy DP-SGD releases.

The canary is a known public example. The attacker knows its current gradient
fingerprint and a public background mean, but not the hidden inclusion schedule.
This measures fingerprint detectability, not reconstruction of an unknown image.
"""
import argparse
import csv
import copy
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import load_mnist, digest
from .models import clip_gradients, initialize, per_record_gradients


def auc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = labels == 1
    negatives = labels == 0
    if positives.sum() == 0 or negatives.sum() == 0:
        raise ValueError("AUC needs both positive and negative examples")
    return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2)
                 / (positives.sum() * negatives.sum()))


def update(model, release, lr):
    with torch.no_grad():
        offset = 0
        for parameter in model.parameters():
            size = parameter.numel()
            parameter.add_(release[offset:offset + size].reshape_as(parameter), alpha=-lr)
            offset += size


def run_sequence(initial, x, labels, population, background, canary, cfg, sigma,
                 seed, include_probability, query_indices=None, query_steps=None):
    """Generate one evolving noisy trajectory and hidden canary schedule."""
    model = copy.deepcopy(initial)
    rng = np.random.default_rng(seed)
    # The q-sensitivity study opts into a dedicated participation stream so
    # conditions with a shared seed have nested Bernoulli schedules.  Legacy
    # configurations retain the original single-stream behavior exactly.
    inclusion_rng = (np.random.default_rng(seed + 200000)
                     if cfg.get("decouple_inclusion_rng", False) else rng)
    noise_rng = torch.Generator().manual_seed(seed + 100000)
    observations, fingerprints, backgrounds, included = [], [], [], []
    endpoint_losses, endpoint_confidences = [], []
    if (query_indices is None) != (query_steps is None):
        raise ValueError("query_indices and query_steps must be supplied together")
    if query_indices is not None:
        query_indices = np.asarray(query_indices, dtype=np.int64)
        query_steps = set(int(step) for step in query_steps)
        if (query_indices.ndim != 1 or not len(query_indices) or not query_steps or
                min(query_steps) < 1 or max(query_steps) > cfg["steps"]):
            raise ValueError("Invalid endpoint query indices or steps")
    query_probs, query_predictions = {}, {}
    for step in range(cfg["steps"]):
        canary_gradient = clip_gradients(
            per_record_gradients(model, x[canary:canary + 1], labels[canary:canary + 1]),
            cfg["clip"])[0]
        background_gradients = clip_gradients(
            per_record_gradients(model, x[background], labels[background]), cfg["clip"])
        background_mean = background_gradients.mean(0)
        present = bool(inclusion_rng.random() < include_probability)
        if present:
            other = rng.choice(population, cfg["batch_size"] - 1, replace=False)
            ids = np.concatenate([[canary], other])
        else:
            ids = rng.choice(population, cfg["batch_size"], replace=False)
        clean = clip_gradients(per_record_gradients(model, x[ids], labels[ids]), cfg["clip"]).mean(0)
        noise_sd = sigma * cfg["clip"] / cfg["batch_size"]
        release = clean + torch.randn(clean.shape, generator=noise_rng) * noise_sd
        observations.append(release.numpy())
        fingerprints.append(canary_gradient.numpy())
        backgrounds.append(background_mean.numpy())
        included.append(int(present))
        update(model, release, cfg["learning_rate"])
        # A genuine endpoint-only baseline: after each update, query only the
        # current checkpoint on the candidate.  Saving the whole prefix lets a
        # longer run provide the endpoint score at every requested T without
        # exposing any intermediate update to this baseline.
        with torch.no_grad():
            logits = model(x[canary:canary + 1])
            endpoint_losses.append(float(torch.nn.functional.cross_entropy(
                logits, labels[canary:canary + 1])))
            endpoint_confidences.append(float(
                logits.softmax(1)[0, int(labels[canary])]))
            if query_indices is not None and step + 1 in query_steps:
                probabilities, predictions = [], []
                for indices in np.array_split(query_indices,
                                              max(1, int(np.ceil(len(query_indices) / 256)))):
                    p = model(x[indices]).softmax(1)
                    probabilities.append(p[torch.arange(len(indices)), labels[indices]].numpy())
                    predictions.append(p.argmax(1).numpy())
                query_probs[step + 1] = np.concatenate(probabilities)
                query_predictions[step + 1] = np.concatenate(predictions)
    result = dict(observations=np.asarray(observations), fingerprints=np.asarray(fingerprints),
                  backgrounds=np.asarray(backgrounds), included=np.asarray(included),
                  endpoint_losses=np.asarray(endpoint_losses),
                  endpoint_confidences=np.asarray(endpoint_confidences))
    if query_indices is not None:
        result["endpoint_query_probs"] = query_probs
        result["endpoint_query_predictions"] = query_predictions
    return result


def scores(run, sigma, batch_size):
    """Return per-round matched fingerprint scores and cumulative scores."""
    residual = run["fingerprints"] - run["backgrounds"]
    variance = (sigma / batch_size) ** 2
    delta = run["observations"] - run["backgrounds"]
    # The observed release is a batch average. Adding one canary changes its
    # mean by (h-b)/B, while the Gaussian release variance is (sigma/B)^2.
    shift = residual / batch_size
    # Gaussian matched score. The background sampling variation is deliberately
    # retained; this tests the attack under the actual DP-SGD batch distribution.
    llr = (delta * shift).sum(1) / variance - .5 * (shift ** 2).sum(1) / variance
    projection = (delta * residual).sum(1) / np.linalg.norm(residual, axis=1).clip(min=1e-12)
    return dict(round_llr=llr, round_projection=projection,
                sequence_llr=float(llr.sum()), sequence_projection=float(projection.sum()),
                max_llr=float(llr.max()), last_llr=float(llr[-1]))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Persistent known-canary fingerprint audit")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/canary_audit.json")
    parser.add_argument("--output", default="runs/canary_audit")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").exists():
        raise RuntimeError("Completed output exists; choose a new output directory")
    torch.set_num_threads(1)
    started = time.time()
    x, labels = load_mnist(args.data)
    rng = np.random.default_rng(cfg["seed"])
    permutation = rng.permutation(len(x))
    cursor = 0
    def take(size):
        nonlocal cursor
        values = permutation[cursor:cursor + size]
        cursor += size
        return values
    pretrain = take(cfg["pretrain_size"])
    background = take(cfg["background_size"])
    canary = int(take(1)[0])
    population = take(cfg["population_size"])
    test = take(cfg["test_size"])
    initial = initialize(cfg["seed"])
    optimizer = torch.optim.SGD(initial.parameters(), lr=cfg["pretrain_lr"])
    for _ in range(cfg["pretrain_steps"]):
        ids = rng.choice(pretrain, cfg["pretrain_batch"], replace=False)
        loss = torch.nn.functional.cross_entropy(initial(x[ids]), labels[ids])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    torch.save(initial.state_dict(), output / "public_initial_cnn.pt")
    (output / "splits.json").write_text(json.dumps(dict(pretrain=pretrain.tolist(), background=background.tolist(),
        canary=canary, population=population.tolist(), test=test.tolist())))
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    rows, round_rows, privacy = [], [], []
    for sigma in cfg["sigmas"]:
        print(f"sigma={sigma}: generating matched no-canary/canary trajectories", flush=True)
        for label, probability in [(0, 0.0), (1, cfg["canary_probability"])]:
            for index in range(cfg["sequences"]):
                # Reuse the same base seed for positive/negative pairs. Their
                # trajectories diverge only through the hidden canary schedule.
                run = run_sequence(initial, x, labels, population, background, canary, cfg,
                    sigma, cfg["seed"] + 100000 * int(sigma * 100) + index, probability)
                feature = scores(run, sigma, cfg["batch_size"])
                rows.append(dict(sigma=sigma, label=label, sequence=index, include_probability=probability,
                    sequence_llr=feature["sequence_llr"], sequence_projection=feature["sequence_projection"],
                    max_llr=feature["max_llr"], last_llr=feature["last_llr"],
                    inclusion_rate=float(run["included"].mean())))
                for step, present in enumerate(run["included"]):
                    round_rows.append(dict(sigma=sigma, label=label, sequence=index, round=step + 1,
                        included=int(present), llr=float(feature["round_llr"][step]),
                        projection=float(feature["round_projection"][step])))
        privacy.append(dict(sigma=sigma, clip=cfg["clip"], batch_size=cfg["batch_size"], steps=cfg["steps"],
            delta=cfg["delta"], epsilon_upper=2 * cfg["steps"] / sigma ** 2 +
            2 * np.sqrt(2 * cfg["steps"] / sigma ** 2 * np.log(1 / cfg["delta"])),
            accounting="replace-one zCDP; no sampling amplification"))
    method_names = ["sequence_llr", "sequence_projection", "max_llr", "last_llr"]
    summary = []
    for sigma in cfg["sigmas"]:
        positives = [r for r in rows if r["sigma"] == sigma and r["label"] == 1]
        negatives = [r for r in rows if r["sigma"] == sigma and r["label"] == 0]
        for method in method_names:
            summary.append(dict(sigma=sigma, method=method,
                auc=auc([r[method] for r in positives + negatives],
                        [1] * len(positives) + [0] * len(negatives)),
                positive_mean=float(np.mean([r[method] for r in positives])),
                negative_mean=float(np.mean([r[method] for r in negatives]))))
    write_csv(output / "sequences.csv", rows); write_csv(output / "rounds.csv", round_rows)
    write_csv(output / "summary.csv", summary); write_csv(output / "privacy.csv", privacy)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for method in ["sequence_llr", "last_llr", "max_llr"]:
        rr = [r for r in summary if r["method"] == method]
        ax.plot([r["sigma"] for r in rr], [r["auc"] for r in rr], marker="o", label=method.replace("_", " "))
    ax.set_xscale("log"); ax.set_ylim(.45, 1.02); ax.set_xlabel("DP-SGD noise multiplier σ")
    ax.set_ylabel("Canary detection AUC"); ax.set_title("Persistent fingerprint detection")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    for ext in ["png", "pdf"]: fig.savefig(output / f"canary_auc.{ext}", dpi=220)
    plt.close(fig)
    (output / "completion.json").write_text(json.dumps(dict(seconds=time.time() - started,
        dataset_sha256=digest(args.data), canary_index=canary, total_sequences=len(rows),
        implementation="known canary; hidden inclusion schedule; full release sequence", torch=torch.__version__), indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
