"""Screen utility and trajectory signal at matched high-noise DP-SGD settings."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from gaussproof.canary_audit import run_sequence, scores
from gaussproof.canary_q_sensitivity import mixture_llr, stratified_identities
from gaussproof.data import digest, load_mnist
from gaussproof.metrics import auc
from gaussproof.models import initialize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if digest(args.data) != json.loads((source / "completion.json").read_text())["dataset_sha256"]:
        raise ValueError("Source and probe MNIST differ")
    torch.set_num_threads(1)
    x, labels = load_mnist(args.data)
    cfg = json.loads((source / "config.json").read_text())
    splits = json.loads((source / "splits.json").read_text())
    initial = initialize(cfg["seed"])
    initial.load_state_dict(torch.load(source / "public_initial_cnn.pt", weights_only=True))
    initial.eval()
    candidates = stratified_identities(splits["test"], labels, 2)
    public_query = np.asarray(splits["test"][200:328])
    if set(public_query) & set(candidates):
        raise AssertionError("Utility query overlaps canaries")
    rows = []
    for lr in (.005, .01, .02, .05):
        cfg_run = dict(cfg, steps=128, learning_rate=lr, decouple_inclusion_rng=True)
        for index, canary in enumerate(candidates):
            for label in (0, 1):
                print(f"lr={lr:g} identity={index+1}/{len(candidates)} H{label}", flush=True)
                run = run_sequence(initial, x, labels, np.asarray(splits["population"]),
                    np.asarray(splits["background"]), canary, cfg_run, 4., 20260929+canary*100,
                    .5 if label else 0., query_indices=public_query, query_steps=[128])
                score = scores(run, 4., cfg_run["batch_size"])
                rows.append(dict(lr=lr, canary=canary, label=label,
                    trajectory=mixture_llr(score["round_llr"], .5),
                    endpoint_loss=-float(run["endpoint_losses"][-1]),
                    public_accuracy=float(np.mean(
                        run["endpoint_query_predictions"][128] == labels[public_query].numpy()))))
        with (output / "screening_rows.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    summary = []
    for lr in (.005, .01, .02, .05):
        group = [row for row in rows if row["lr"] == lr]
        y = [row["label"] for row in group]
        summary.append(dict(lr=lr, sigma=4, q=.5, steps=128,
            independent_identities=len(candidates),
            trajectory_auc=auc([row["trajectory"] for row in group], y),
            endpoint_loss_auc=auc([row["endpoint_loss"] for row in group], y),
            mean_public_accuracy=float(np.mean([row["public_accuracy"] for row in group]))))
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
