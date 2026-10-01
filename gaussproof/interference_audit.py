"""Prespecified natural-canary overlap test for the one-run audit.

The cohort is selected using clipped gradients at the public initial model,
before any audit IN/OUT bits, private minibatches, or noise are drawn. Both
cohorts use the same two MNIST classes and the same Poisson sampler.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import digest, load_mnist
from .models import clip_gradients, initialize, per_record_gradients
from .multi_contributor import write_csv
from .one_run_audit import run_audit


def select_overlap_cohorts(ids, labels, unit_gradients, classes, count_per_class,
                           pool_per_class, seed):
    """Greedy high/low within-class coherence, with matched starts and pools."""
    ids = np.asarray(ids, dtype=int)
    labels = np.asarray(labels, dtype=int)
    vectors = np.asarray(unit_gradients, dtype=float)
    if (len(ids) != len(vectors) or vectors.ndim != 2 or count_per_class < 2 or
            pool_per_class < count_per_class or len(set(classes)) != len(classes)):
        raise ValueError("Invalid overlap-selection inputs")
    rng = np.random.default_rng(seed)
    selected = {"low": [], "high": []}
    for label in classes:
        eligible = np.flatnonzero(labels[ids] == label)
        if len(eligible) < pool_per_class:
            raise ValueError(f"Too few held-out records for class {label}")
        pool = rng.choice(eligible, pool_per_class, replace=False)
        similarity = vectors[pool] @ vectors[pool].T
        start = int(rng.integers(len(pool)))
        for mode in selected:
            chosen = [start]
            while len(chosen) < count_per_class:
                remaining = np.setdiff1d(np.arange(len(pool)), chosen,
                                         assume_unique=True)
                overlap = np.abs(similarity[np.ix_(remaining, chosen)]).mean(axis=1)
                best = int(np.argmax(overlap) if mode == "high" else np.argmin(overlap))
                chosen.append(int(remaining[best]))
            selected[mode].extend(pool[chosen].tolist())
    return {mode: np.asarray(index, dtype=int) for mode, index in selected.items()}


def overlap_stats(selected, labels, unit_gradients):
    gram = unit_gradients[selected] @ unit_gradients[selected].T
    row, col = np.triu_indices(len(selected), 1)
    within = labels[selected[row]] == labels[selected[col]]
    if not within.any():
        raise ValueError("Need at least one within-class pair")
    return dict(within_abs_cosine=float(np.abs(gram[row[within], col[within]]).mean()),
                all_abs_cosine=float(np.abs(gram[row, col]).mean()),
                effective_rank=float(np.trace(gram)**2 / np.sum(gram**2)))


def main():
    parser = argparse.ArgumentParser()
    for key in ("source", "data", "config", "output", "report"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--max-repetitions", type=int, default=None,
                        help="Run a resumable prefix; the preregistered config is unchanged")
    args = parser.parse_args()
    source, output, report = Path(args.source), Path(args.output), Path(args.report)
    config = json.loads(Path(args.config).read_text())
    if digest(args.data) != json.loads((source / "completion.json").read_text())["dataset_sha256"]:
        raise ValueError("Dataset differs from public checkpoint source")
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    x, y = load_mnist(args.data)
    splits = json.loads((source / "splits.json").read_text())
    source_cfg = json.loads((source / "config.json").read_text())
    initial = initialize(source_cfg["seed"])
    initial.load_state_dict(torch.load(source / "public_initial_cnn.pt", weights_only=True))
    initial.eval()
    test = np.asarray(splits["test"], dtype=int)
    query = test[-config["query_size"]:]
    candidate_pool = test[:-config["query_size"]]
    population = np.asarray(splits["population"], dtype=int)[:config["population_size"]]
    public = np.asarray(splits["background"], dtype=int)[:config["background_size"]]
    if any(set(a) & set(b) for a, b in ((candidate_pool, population),
                                        (candidate_pool, public), (candidate_pool, query),
                                        (public, population))):
        raise ValueError("Sample roles overlap")
    if not np.isclose(config["q"] * len(population), config["batch_size"]):
        raise ValueError("Poisson q must match the expected ordinary batch size")
    if config["sampling_mode"] != "poisson":
        raise ValueError("This experiment requires matched Poisson sampling")
    gradients = clip_gradients(per_record_gradients(
        initial, x[candidate_pool], y[candidate_pool]), config["clip"]).numpy()
    unit = gradients / np.linalg.norm(gradients, axis=1, keepdims=True).clip(min=1e-12)
    labels = y.numpy()
    limit = config["repetitions"] if args.max_repetitions is None else args.max_repetitions
    if not 1 <= limit <= config["repetitions"]:
        raise ValueError("Invalid repetition prefix")
    summaries = []
    began = time.time()
    for repetition in range(limit):
        cohorts = select_overlap_cohorts(candidate_pool, labels, unit,
            config["classes"], config["canaries_per_class"],
            config["pool_per_class"], config["seed"] + repetition)
        for mode, positions in cohorts.items():
            canaries = candidate_pool[positions]
            geometry = overlap_stats(positions, labels[candidate_pool], unit)
            for sigma in config["sigmas"]:
                seed = config["seed"] + repetition
                condition = dict(config, sigma=sigma, canaries=len(canaries))
                stem = f"{mode}_sigma_{sigma:g}_rep_{repetition}"
                scores_file = output / f"{stem}_scores.npz"
                summary_file = output / f"{stem}_summary.json"
                if scores_file.exists() != summary_file.exists():
                    raise ValueError("Incomplete prior audit result")
                if summary_file.exists():
                    rows = json.loads(summary_file.read_text())
                else:
                    rows, bits, scores = run_audit(initial, x, y, population,
                        canaries, public, query, condition, seed)
                    for row in rows:
                        row.update(repetition=repetition, cohort=mode, **geometry)
                    np.savez_compressed(scores_file, bits=bits, canaries=canaries,
                                        **scores)
                    summary_file.write_text(json.dumps(rows, indent=2))
                summaries.extend(rows)
                print(f"rep={repetition+1}/{limit} {mode} sigma={sigma:g} "
                      f"coherence={geometry['within_abs_cosine']:.3f}: " +
                      ", ".join(f"{r['method']} {r['correct']}/{r['guesses']}"
                                for r in rows), flush=True)
    write_csv(report / "one_run_interference.csv", summaries)
    provenance = dict(config=config, completed_repetitions=limit,
        dataset_sha256=digest(args.data),
        source_checkpoint_sha256=hashlib.sha256(
            (source / "public_initial_cnn.pt").read_bytes()).hexdigest(),
        seconds=time.time()-began, torch=torch.__version__,
        selection="Initial public clipped gradients, fixed two classes; no audit bits or releases",
        inference="Fixed 16 IN and 16 OUT guesses; within-run Bonferroni across five scores")
    (report / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
