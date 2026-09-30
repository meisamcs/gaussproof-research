"""Paired exact DP-SGD trajectory and strong final-model attack experiment.

OUT references train on disjoint public MNIST records. Every target attack
scores the same positive/negative run; no evaluation labels fit an attack.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .attacks import rmia
from .canary_audit import run_sequence, scores as analytic_scores
from .canary_q_sensitivity import mixture_llr, stratified_identities
from .data import digest, load_mnist
from .models import initialize


def logit(probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def public_roles(size, splits, train_size, query_size, seed):
    used = {int(index) for group in splits.values() for index in group}
    available = np.asarray(sorted(set(range(size)) - used))
    if len(available) < train_size + query_size:
        raise ValueError("Insufficient disjoint public examples")
    np.random.default_rng(seed).shuffle(available)
    return available[:train_size], available[train_size:train_size + query_size]


def save_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return [{key: (int(value) if key in ("canary", "digit", "label", "steps")
                       else float(value)) for key, value in row.items()}
                for row in csv.DictReader(stream)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/trajectory_endpoint_replication.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").exists():
        raise ValueError("Completed result exists")
    spec = json.loads(Path(args.config).read_text())
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != spec:
            raise ValueError("Resume configuration changed")
    else:
        (output / "config.json").write_text(json.dumps(spec, indent=2))
    if digest(args.data) != json.loads((source / "completion.json").read_text())["dataset_sha256"]:
        raise ValueError("MNIST hash differs from source checkpoint run")
    started = time.time()
    torch.set_num_threads(1)
    x, labels = load_mnist(args.data)
    source_cfg = json.loads((source / "config.json").read_text())
    splits = json.loads((source / "splits.json").read_text())
    initial = initialize(source_cfg["seed"])
    initial.load_state_dict(torch.load(source / "public_initial_cnn.pt", weights_only=True))
    initial.eval()
    prefixes = sorted(map(int, spec["prefix_steps"]))
    q_values = list(map(float, spec["q_values"]))
    identities = np.asarray(stratified_identities(
        splits["test"], labels, int(spec["identities_per_class"])))
    public_train, public_query = public_roles(
        len(x), splits, int(source_cfg["population_size"]),
        int(spec["rmia_population"]), int(spec["seed"]) + 17)
    public_labels = labels[public_query].numpy()
    cfg = dict(source_cfg, steps=prefixes[-1], decouple_inclusion_rng=True,
               learning_rate=float(spec.get("learning_rate", source_cfg["learning_rate"])))
    sigma = float(spec["sigma"])
    cache = output / "out_references.npz"
    if cache.exists():
        with np.load(cache) as saved:
            references = {step: saved[str(step)].copy() for step in prefixes}
    else:
        query = np.concatenate([identities, public_query])
        references = {step: [] for step in prefixes}
        for index in range(int(spec["reference_models"])):
            print(f"OUT reference {index + 1}/{spec['reference_models']}", flush=True)
            run = run_sequence(initial, x, labels, public_train,
                np.asarray(splits["background"]), int(identities[0]), cfg,
                sigma, int(spec["seed"]) + 1000000 + index * 997, 0.,
                query_indices=query, query_steps=prefixes)
            for step in prefixes:
                references[step].append(run["endpoint_query_probs"][step])
        references = {step: np.asarray(values, dtype=np.float64)
                      for step, values in references.items()}
        np.savez_compressed(cache, **{str(k): v for k, v in references.items()})
    rows_path = output / "score_rows.csv"
    rows = read_rows(rows_path)
    done = {(r["q"], r["canary"], r["label"], r["steps"]) for r in rows}
    for q in q_values:
        for index, canary in enumerate(identities):
            ref = {step: references[step][:, index] for step in prefixes}
            for label in (0, 1):
                if all((q, int(canary), label, step) in done for step in prefixes):
                    continue
                print(f"q={q:g} identity={index+1}/{len(identities)} H{label}", flush=True)
                query = np.concatenate([[canary], public_query])
                run = run_sequence(initial, x, labels,
                    np.asarray(splits["population"]), np.asarray(splits["background"]),
                    int(canary), cfg, sigma, int(spec["seed"]) + int(canary) * 100,
                    q if label else 0., query_indices=query, query_steps=prefixes)
                for step in prefixes:
                    short = {k: run[k][:step] for k in
                             ("observations", "fingerprints", "backgrounds")}
                    analytic = analytic_scores(short, sigma * cfg["clip"], cfg["batch_size"])
                    target = run["endpoint_query_probs"][step]
                    out = ref[step]
                    out_population = references[step][:, len(identities):]
                    all_out_logits = logit(references[step][:, :len(identities)])
                    residual = all_out_logits - np.median(all_out_logits, axis=0)
                    std = max(float(np.sqrt(np.mean(residual ** 2))), 1e-6)
                    lira_z = float((logit(target[0]) - np.median(logit(out))) / std)
                    row = dict(
                        q=q, canary=int(canary), digit=int(labels[canary]),
                        label=label, steps=step,
                        appearances=int(run["included"][:step].sum()),
                        trajectory_mixture=mixture_llr(analytic["round_llr"], q),
                        trajectory_sum_llr=analytic["sequence_llr"],
                        trajectory_raw_alignment=float(np.einsum(
                            "td,td->", run["fingerprints"][:step],
                            run["observations"][:step]) / step),
                        endpoint_loss=float(np.log(np.clip(target[0], 1e-12, 1))),
                        endpoint_lira_fixed=.5 * lira_z ** 2,
                        endpoint_lira_z=lira_z,
                        endpoint_public_accuracy=float(np.mean(
                            run["endpoint_query_predictions"][step][1:] == public_labels)),
                    )
                    for a in (0., .5, 1.):
                        for gamma in (1., 1.05):
                            name = f"endpoint_rmia_a{int(a*100):03d}_g{int(gamma*100):03d}"
                            row[name] = float(rmia(
                                np.asarray([target[0]]), out[:, None],
                                target[1:], out_population, a=a, gamma=gamma)[0])
                    rows.append(row)
                save_rows(rows_path, rows)
    completion = dict(seconds=time.time() - started,
        data_sha256=digest(args.data),
        source_checkpoint_sha256=hashlib.sha256(
            (source / "public_initial_cnn.pt").read_bytes()).hexdigest(),
        q_values=q_values, prefixes=prefixes, identities=len(identities),
        learning_rate=float(cfg["learning_rate"]),
        reference_models=int(spec["reference_models"]),
        public_reference_pool_disjoint_from_target_roles=True,
        positive_negative_runs_paired=True, torch=torch.__version__)
    (output / "completion.json").write_text(json.dumps(completion, indent=2))
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
