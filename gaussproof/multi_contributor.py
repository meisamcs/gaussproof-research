"""Actual CNN DP-SGD with several hidden gallery contributors per round.

This tests the *specific* structural assumption motivating a sparse
fingerprint decoder. Every estimator receives the same current public bank,
checkpoint, background, noisy release and mechanism parameters. The exact
small-gallery posterior is a tractable strong comparator, not an attack oracle.
"""
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
from scipy.special import expit, logsumexp

from .canary_audit import update
from .canary_gallery import stratified_gallery
from .data import digest, load_mnist
from .metrics import auc
from .models import clip_gradients, initialize, per_record_gradients


def bounded_sparse(gram, projection, penalty, iterations=60):
    """Minimize .5||S.T a-d||² + penalty*sum(a) over 0<=a_i<=1."""
    if penalty < 0 or gram.shape != (len(projection), len(projection)):
        raise ValueError("Invalid sparse problem")
    lipschitz = max(float(np.linalg.eigvalsh(gram)[-1]), 1e-12)
    a = np.zeros(len(projection), dtype=float)
    for _ in range(iterations):
        a = np.clip(a + (projection - gram @ a - penalty) / lipschitz, 0, 1)
    return a


def exact_subset_posterior(gram, projection, variance, q, masks):
    """Posterior inclusion marginals for the declared Bernoulli gallery prior."""
    if not 0 < q < 1 or variance <= 0:
        raise ValueError("Invalid Gaussian posterior parameters")
    counts = masks.sum(1)
    log_prior = counts * np.log(q) + (masks.shape[1] - counts) * np.log1p(-q)
    log_weight = log_prior + (masks @ projection
                             - .5 * np.einsum("si,ij,sj->s", masks, gram, masks)) / variance
    weights = np.exp(log_weight - logsumexp(log_weight))
    return weights @ masks


def score_round(release, bank, background, config, masks):
    shift = (bank - background[None, :]) / config["batch_size"]
    residual = release - background
    gram = shift @ shift.T
    dot = shift @ residual
    norms = np.diag(gram)
    variance = (config["sigma"] * config["clip"] / config["batch_size"]) ** 2
    llr = (dot - .5 * norms) / variance
    independent = expit(np.log(config["q"] / (1 - config["q"])) + llr)
    exact = exact_subset_posterior(gram, dot, variance, config["q"], masks)
    penalty_scale = np.sqrt(variance) * float(np.median(np.sqrt(norms)))
    sparse = {f"sparse_{factor:g}": bounded_sparse(
        gram, dot, factor * penalty_scale)
        for factor in config["penalty_factors"]}
    scores = dict(alignment=dot, independent=independent, exact=exact, **sparse)
    return shift, scores


def trial(initial, x, labels, gallery, background_ids, population, query,
          config, seed):
    model = copy.deepcopy(initial)
    inclusion_rng = np.random.default_rng(seed + 200000)
    background_rng = np.random.default_rng(seed)
    noise_rng = torch.Generator().manual_seed(seed + 100000)
    candidates = len(gallery)
    masks = ((np.arange(2 ** candidates)[:, None] >> np.arange(candidates)) & 1).astype(float)
    masks = masks[masks.sum(1) <= config["batch_size"]]
    names = ("alignment", "independent", "exact") + tuple(
        f"sparse_{factor:g}" for factor in config["penalty_factors"])
    event_scores = {name: [] for name in names}
    truth = []
    errors = {name: 0. for name in ("raw", "background", "prior", "oracle") +
              names if name not in ("alignment",)}
    appearances = 0
    for step in range(config["steps"]):
        public = clip_gradients(per_record_gradients(model,
            x[np.r_[gallery, background_ids]],
            labels[np.r_[gallery, background_ids]]), config["clip"]).numpy()
        bank = public[:candidates]
        background = public[candidates:].mean(0)
        active = inclusion_rng.random(candidates) < config["q"]
        # This cap affects a negligible fraction of runs for the pinned K,q,B;
        # it enforces the batch-size constraint and is recorded below.
        while active.sum() > config["batch_size"]:
            active = inclusion_rng.random(candidates) < config["q"]
        appearances += int(active.sum())
        ordinary = background_rng.choice(population,
            config["batch_size"] - int(active.sum()), replace=False)
        batch_ids = np.r_[gallery[active], ordinary]
        clean = clip_gradients(per_record_gradients(
            model, x[batch_ids], labels[batch_ids]), config["clip"]).mean(0)
        sd = config["sigma"] * config["clip"] / config["batch_size"]
        release = (clean + torch.randn(clean.shape, generator=noise_rng) * sd).numpy()
        clean = clean.numpy()
        shift, scores = score_round(release, bank, background, config, masks)
        truth.extend(active.astype(int).tolist())
        for name, values in scores.items():
            event_scores[name].extend(values.tolist())
        estimates = dict(raw=release, background=background,
                         prior=background + config["q"] * shift.sum(0),
                         oracle=background + active.astype(float) @ shift,
                         independent=background + scores["independent"] @ shift,
                         exact=background + scores["exact"] @ shift)
        for name in names:
            if name.startswith("sparse_"):
                estimates[name] = background + scores[name] @ shift
        for name, estimate in estimates.items():
            errors[name] += float(np.sum((estimate - clean) ** 2)
                                  / config["clip"] ** 2)
        update(model, torch.from_numpy(release), config["learning_rate"])
    with torch.no_grad():
        utility = float((model(x[query]).argmax(1) == labels[query]).float().mean())
    truth = np.asarray(truth)
    if truth.min() == truth.max():
        raise ValueError("Trial must contain both active and inactive candidates")
    return [dict(seed=seed, sigma=config["sigma"], steps=config["steps"],
                 method=name, support_auc=float(auc(event_scores[name], truth)),
                 clean_gradient_mse=float(errors[name] / config["steps"])
                     if name in errors else np.nan,
                 raw_mse=float(errors["raw"] / config["steps"]),
                 background_mse=float(errors["background"] / config["steps"]),
                 prior_mse=float(errors["prior"] / config["steps"]),
                 oracle_mse=float(errors["oracle"] / config["steps"]),
                 utility=utility, mean_active=float(appearances / config["steps"]))
            for name in names]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        result = list(csv.DictReader(stream))
    for row in result:
        row["seed"] = int(row["seed"])
        row["steps"] = int(row["steps"])
        for key in ("sigma", "support_auc", "clean_gradient_mse",
                    "raw_mse", "background_mse", "prior_mse", "oracle_mse",
                    "utility", "mean_active"):
            row[key] = float(row[key])
    return result


def summarize(rows, config):
    methods = ["alignment", "independent", "exact"] + [
        f"sparse_{factor:g}" for factor in config["penalty_factors"]]
    cal_seeds = [config["seed"] + index for index in range(config["calibration_runs"])]
    eval_seeds = [config["seed"] + 10000 + index
                  for index in range(config["evaluation_runs"])]
    cal = {method: [r for r in rows if r["method"] == method
                    and r["seed"] in cal_seeds] for method in methods}
    for method in methods:
        if len(cal[method]) != len(cal_seeds):
            raise ValueError("Incomplete calibration")
    penalties = [method for method in methods if method.startswith("sparse_")]
    selected = min(penalties, key=lambda name: (np.mean([
        r["clean_gradient_mse"] for r in cal[name]]), penalties.index(name)))
    output = []
    for method in methods:
        tested = [r for r in rows if r["method"] == method and r["seed"] in eval_seeds]
        if len(tested) != len(eval_seeds):
            raise ValueError("Incomplete held-out trials")
        row = dict(method=method, selected_sparse=selected,
                   evaluation_runs=len(eval_seeds))
        for key in ("support_auc", "clean_gradient_mse", "raw_mse",
                    "background_mse", "prior_mse", "oracle_mse",
                    "utility", "mean_active"):
            row[key] = float(np.mean([r[key] for r in tested]))
        output.append(row)
    return output, selected


def plot_summary(summaries, output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for sigma, rows in summaries.items():
        selected = rows[0]["selected_sparse"]
        for method in ("alignment", "independent", "exact", selected):
            row = next(r for r in rows if r["method"] == method)
            axes[0].scatter(sigma, row["support_auc"], label=f"{method}, σ={sigma:g}")
        for method in ("background", "prior", "independent", "exact", selected):
            ref = next(r for r in rows if r["method"] == "exact")
            mse = (ref[method + "_mse"] if method in ("background", "prior")
                   else next(r for r in rows if r["method"] == method)["clean_gradient_mse"])
            axes[1].scatter(sigma, mse, label=f"{method}, σ={sigma:g}")
    axes[0].axhline(.5, color="black", ls="--", lw=1)
    axes[0].set_ylabel("Per-round candidate support AUC")
    axes[1].set_ylabel("Full-vector clean-gradient error / C²")
    for ax in axes:
        ax.set_xlabel("DP-SGD noise multiplier σ")
        ax.set_xticks(sorted(summaries))
        ax.grid(alpha=.2)
        ax.legend(fontsize=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output / f"multi_contributor.{ext}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    source, output, report = Path(args.source), Path(args.output), Path(args.report)
    config = json.loads(Path(args.config).read_text())
    output.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").exists():
        raise ValueError("Completed output exists")
    saved = output / "config.json"
    if saved.exists() and json.loads(saved.read_text()) != config:
        raise ValueError("Cannot resume with changed config")
    saved.write_text(json.dumps(config, indent=2))
    if digest(args.data) != json.loads((source / "completion.json").read_text())["dataset_sha256"]:
        raise ValueError("Dataset differs from checkpoint source")
    torch.set_num_threads(1); start = time.time()
    x, labels = load_mnist(args.data)
    splits = json.loads((source / "splits.json").read_text())
    source_cfg = json.loads((source / "config.json").read_text())
    initial = initialize(source_cfg["seed"])
    initial.load_state_dict(torch.load(source / "public_initial_cnn.pt", weights_only=True))
    initial.eval()
    gallery = stratified_gallery(np.asarray(splits["test"]), labels, config["gallery_size"])
    query = np.asarray(splits["test"])[-config["query_size"]:]
    if set(gallery) & set(query):
        raise ValueError("Utility query overlaps gallery")
    population = np.asarray(splits["population"])
    background = np.asarray(splits["background"])
    summaries = {}
    for sigma in config["sigmas"]:
        condition = dict(config, sigma=sigma)
        path = output / f"sigma_{sigma:g}.csv"
        rows = read_csv(path)
        done = {r["seed"] for r in rows if r["method"] == "exact"}
        seeds = [config["seed"] + index for index in range(config["calibration_runs"])]
        seeds += [config["seed"] + 10000 + index for index in range(config["evaluation_runs"])]
        for number, seed in enumerate(seeds, 1):
            if seed in done:
                continue
            rows.extend(trial(initial, x, labels, gallery, background,
                              population, query, condition, seed))
            write_csv(path, rows)
            print(f"sigma={sigma:g}: trained/evaluated {number}/{len(seeds)} runs", flush=True)
        summary, selected = summarize(rows, config)
        write_csv(report / f"sigma_{sigma:g}_summary.csv", summary)
        summaries[sigma] = summary
    plot_summary(summaries, report)
    provenance = dict(dataset_sha256=digest(args.data), source_checkpoint_sha256=
        hashlib.sha256((source / "public_initial_cnn.pt").read_bytes()).hexdigest(),
        config=config, seconds=time.time()-start, torch=torch.__version__,
        observations="actual evolving CNN DP-SGD; public gallery/background; hidden batch and noise",
        reference="exact 2^K subset posterior under the declared Gaussian surrogate")
    (report / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (output / "completion.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
