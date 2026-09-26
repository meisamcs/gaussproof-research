"""Holdout-canary calibration for the known-fingerprint trajectory audit."""
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .canary_audit import auc, run_sequence, scores
from .data import digest, load_mnist
from .metrics import rates, threshold_at_fpr
from .models import initialize


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Unseen-canary fingerprint calibration")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/canary_holdout.json")
    parser.add_argument("--output", default="runs/canary_holdout")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").exists():
        raise RuntimeError("Completed output exists; choose a new output directory")
    torch.set_num_threads(1); started = time.time()
    x, labels = load_mnist(args.data); rng = np.random.default_rng(cfg["seed"])
    permutation = rng.permutation(len(x)); cursor = 0
    def take(size):
        nonlocal cursor
        result = permutation[cursor:cursor + size]; cursor += size; return result
    pretrain = take(cfg["pretrain_size"]); background = take(cfg["background_size"])
    calibration_canaries = take(cfg["calibration_canaries"])
    test_canaries = take(cfg["test_canaries"])
    population = take(cfg["population_size"]); test = take(cfg["test_size"])
    initial = initialize(cfg["seed"])
    optimizer = torch.optim.SGD(initial.parameters(), lr=cfg["pretrain_lr"])
    for _ in range(cfg["pretrain_steps"]):
        ids = rng.choice(pretrain, cfg["pretrain_batch"], replace=False)
        loss = torch.nn.functional.cross_entropy(initial(x[ids]), labels[ids])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    torch.save(initial.state_dict(), output / "public_initial_cnn.pt")
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    (output / "splits.json").write_text(json.dumps(dict(
        pretrain=pretrain.tolist(), background=background.tolist(),
        calibration_canaries=calibration_canaries.tolist(), test_canaries=test_canaries.tolist(),
        population=population.tolist(), test=test.tolist())))
    rows = []
    for sigma in cfg["sigmas"]:
        for split, canaries in [("calibration", calibration_canaries), ("holdout", test_canaries)]:
            print(f"sigma={sigma}, split={split}", flush=True)
            for canary in canaries:
                for label, probability in [(0, 0.0), (1, cfg["canary_probability"])] :
                    for index in range(cfg["sequences_per_canary"]):
                        seed = cfg["seed"] + int(sigma * 10000) + int(canary) + index
                        run = run_sequence(initial, x, labels, population, background,
                            int(canary), cfg, sigma, seed, probability)
                        feature = scores(run, sigma, cfg["batch_size"])
                        rows.append(dict(sigma=sigma, split=split, canary=int(canary), label=label,
                            sequence=index, sequence_llr=feature["sequence_llr"],
                            sequence_projection=feature["sequence_projection"],
                            max_llr=feature["max_llr"], last_llr=feature["last_llr"],
                            inclusion_rate=float(run["included"].mean())))
    methods = ["sequence_llr", "sequence_projection", "max_llr", "last_llr"]
    summary, per_canary = [], []
    for sigma in cfg["sigmas"]:
        for method in methods:
            calibration = [r for r in rows if r["sigma"] == sigma and r["split"] == "calibration"]
            holdout = [r for r in rows if r["sigma"] == sigma and r["split"] == "holdout"]
            cal_scores = np.asarray([r[method] for r in calibration]); cal_labels = np.asarray([r["label"] for r in calibration])
            test_scores = np.asarray([r[method] for r in holdout]); test_labels = np.asarray([r["label"] for r in holdout])
            threshold_1 = threshold_at_fpr(cal_scores, cal_labels, .01)
            threshold_5 = threshold_at_fpr(cal_scores, cal_labels, .05)
            tpr1, fpr1 = rates(test_scores, test_labels, threshold_1)
            tpr5, fpr5 = rates(test_scores, test_labels, threshold_5)
            summary.append(dict(sigma=sigma, method=method,
                calibration_auc=auc(cal_scores, cal_labels), holdout_auc=auc(test_scores, test_labels),
                holdout_tpr_at_calibrated_1pct=float(tpr1), holdout_fpr_at_calibrated_1pct=float(fpr1),
                holdout_tpr_at_calibrated_5pct=float(tpr5), holdout_fpr_at_calibrated_5pct=float(fpr5),
                threshold_1pct=float(threshold_1), threshold_5pct=float(threshold_5)))
            for canary in test_canaries:
                part = [r for r in holdout if r["canary"] == int(canary)]
                per_canary.append(dict(sigma=sigma, method=method, canary=int(canary),
                    auc=auc([r[method] for r in part], [r["label"] for r in part])))
    write_csv(output / "sequences.csv", rows); write_csv(output / "summary.csv", summary); write_csv(output / "per_canary.csv", per_canary)
    privacy = []
    for sigma in cfg["sigmas"]:
        rho = 2 * cfg["steps"] / sigma ** 2
        privacy.append(dict(sigma=sigma, clip=cfg["clip"], batch_size=cfg["batch_size"], steps=cfg["steps"],
            delta=cfg["delta"], epsilon_upper=rho + 2 * np.sqrt(rho * np.log(1 / cfg["delta"])),
            accounting="replace-one zCDP; no sampling amplification"))
    write_csv(output / "privacy.csv", privacy)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for method in ["sequence_llr", "last_llr", "max_llr"]:
        rr = [r for r in summary if r["method"] == method]
        ax.plot([r["sigma"] for r in rr], [r["holdout_auc"] for r in rr], marker="o", label=method.replace("_", " "))
    ax.set_xscale("log"); ax.set_ylim(.45, 1.02); ax.set_xlabel("DP-SGD noise multiplier σ")
    ax.set_ylabel("Unseen-canary holdout AUC"); ax.set_title("Fingerprint detector generalization")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    for ext in ["png", "pdf"]: fig.savefig(output / f"holdout_auc.{ext}", dpi=220)
    plt.close(fig)
    (output / "completion.json").write_text(json.dumps(dict(seconds=time.time() - started,
        dataset_sha256=digest(args.data), calibration_canaries=calibration_canaries.tolist(),
        holdout_canaries=test_canaries.tolist(), sequences=len(rows),
        calibration_is_disjoint_from_holdout=True, torch=torch.__version__), indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
