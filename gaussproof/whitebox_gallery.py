"""Equal-access trajectory gallery and clean-gradient pilot on actual DP-SGD.

All ranking methods observe the same noisy release, current CNN checkpoint,
public background examples, and gallery. Hidden batch indices and clean
gradients are used only by the evaluator. This is closed-gallery record linkage,
not unknown-image reconstruction or an implementation of informed ReRo.
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
from scipy.special import logsumexp

from .canary_audit import update
from .canary_gallery import (class_probabilities, record_weights,
                            stratified_gallery)
from .data import digest, load_mnist
from .models import clip_gradients, initialize, per_record_gradients


BASELINES = ("alignment", "gaussian_llr", "mixture_llr")


def positive_simplex_projection(vector):
    """Euclidean projection onto {a >= 0, sum(a) <= 1}."""
    v = np.asarray(vector, dtype=np.float64)
    clipped = np.maximum(v, 0)
    if clipped.sum() <= 1:
        return clipped
    ordered = np.sort(v)[::-1]
    partial = np.cumsum(ordered)
    active = np.flatnonzero(ordered - (partial - 1) / np.arange(1, len(v) + 1) > 0)
    theta = (partial[active[-1]] - 1) / (active[-1] + 1)
    return np.maximum(v - theta, 0)


def sparse_coefficients(gram, projection, penalty, iterations=45):
    """Bounded nonnegative sparse fit using only the public candidate bank.

    Minimize .5 ||S.T a - (y-b)||² + penalty*sum(a), with a >= 0,
    sum(a) <= 1. The cap encodes at most one eligible gallery record in a
    round. It is a model assumption shared across all penalty variants.
    """
    gram = np.asarray(gram, dtype=np.float64)
    projection = np.asarray(projection, dtype=np.float64)
    if gram.shape != (len(projection), len(projection)) or penalty < 0:
        raise ValueError("Invalid sparse-decoder inputs")
    lipschitz = max(float(np.linalg.eigvalsh(gram)[-1]), 1e-12)
    a = np.zeros(len(projection), dtype=np.float64)
    for _ in range(iterations):
        a = positive_simplex_projection(
            a + (projection - gram @ a - penalty) / lipschitz)
    return a


def fixed_methods(config):
    return BASELINES + tuple(f"sparse_{factor:g}" for factor in config["penalty_factors"])


def score_round(release, clean, bank, backgrounds, config):
    """One step's same-access ranking increments and denoising errors."""
    sigma, clip, batch, q = (float(config[key]) for key in
                             ("sigma", "clip", "batch_size", "q"))
    variance = (sigma * clip / batch) ** 2
    methods = fixed_methods(config)
    output = {}
    for mode, background in backgrounds.items():
        shifts = (bank - background[None, :]) / batch
        residual = release - background
        dot = shifts @ residual
        norms = np.einsum("kd,kd->k", shifts, shifts)
        llr = (dot - 0.5 * norms) / variance
        increments = dict(alignment=dot, gaussian_llr=llr,
                          mixture_llr=np.logaddexp(np.log1p(-q), np.log(q) + llr))
        gradient_errors = {
            "raw": float(np.sum((release - clean) ** 2) / clip ** 2),
            "background": float(np.sum((background - clean) ** 2) / clip ** 2),
        }
        gram = shifts @ shifts.T
        penalty_scale = np.sqrt(variance) * float(np.median(np.sqrt(norms)))
        for factor in config["penalty_factors"]:
            name = f"sparse_{factor:g}"
            coefficients = sparse_coefficients(gram, dot, factor * penalty_scale)
            increments[name] = coefficients
            estimate = background + coefficients @ shifts
            gradient_errors[name] = float(np.sum((estimate - clean) ** 2) / clip ** 2)
        if set(increments) != set(methods):
            raise AssertionError("Unexpected score method")
        output[mode] = (increments, gradient_errors)
    return output


def temporal_posterior_mean(background, bank, cumulative_mixture, round_llr,
                            q, batch_size):
    """Same-access posterior mean under the stated one-gallery-record model.

    This online estimate uses releases through the current round only. It
    combines a uniform identity prior with the q-aware trajectory posterior
    and current-round inclusion posterior. It is an analytic comparator for
    gradient denoising, not an oracle with the hidden identity/schedule.
    """
    if not 0 < q < 1:
        raise ValueError("Posterior mean requires 0 < q < 1")
    identity_prob = np.exp(cumulative_mixture - logsumexp(cumulative_mixture))
    round_log_normalizer = np.logaddexp(np.log1p(-q), np.log(q) + round_llr)
    inclusion_prob = np.exp(np.log(q) + round_llr - round_log_normalizer)
    shifts = (bank - background[None, :]) / batch_size
    return background + (identity_prob * inclusion_prob) @ shifts


def _query_accuracy(model, x, labels, query_ids):
    with torch.no_grad():
        return float((model(x[query_ids]).argmax(1) == labels[query_ids]).float().mean())


def _rank(score, true_position, gallery, labels):
    order = np.argsort(-score, kind="stable")
    rank = int(np.flatnonzero(order == true_position)[0] + 1)
    digit = int(labels[int(gallery[true_position])])
    same = np.flatnonzero(labels[gallery].numpy() == digit)
    same_order = same[np.argsort(-score[same], kind="stable")]
    same_rank = int(np.flatnonzero(same_order == true_position)[0] + 1)
    return dict(rank=rank, top1=int(rank == 1), top5=int(rank <= 5),
                mrr=1 / rank, same_class_top1=int(same_rank == 1),
                predicted_digit=int(labels[int(gallery[int(order[0])])]))


def run_trajectory(initial, x, labels, population, background_ids, gallery,
                   query_ids, true_position, distribution, config, seed,
                   q_override=None):
    """Train one private CNN trajectory and return only evaluated scores."""
    model = copy.deepcopy(initial)
    rng = np.random.default_rng(seed)
    participation_rng = np.random.default_rng(seed + 200000)
    noise_rng = torch.Generator().manual_seed(seed + 100000)
    q = config["q"] if q_override is None else q_override
    true_id = int(gallery[true_position])
    probabilities = class_probabilities(distribution, int(labels[true_id]))
    population_weights = record_weights(population, labels, probabilities)
    background_weights = {"matched": record_weights(background_ids, labels, probabilities)}
    if distribution != "balanced":
        background_weights["balanced_mismatch"] = record_weights(
            background_ids, labels,
            class_probabilities("balanced", int(labels[true_id])))
    methods = fixed_methods(config)
    scores = {mode: {method: np.zeros(len(gallery), dtype=np.float64)
                     for method in methods} for mode in background_weights}
    errors = {mode: {name: 0.0 for name in
                     ("raw", "background") + tuple(method for method in methods
                     if method.startswith("sparse_")) + ("posterior", "oracle")}
              for mode in background_weights}
    present_count, rows = 0, []
    probe_ids = np.r_[gallery, background_ids]
    for step in range(max(config["prefix_steps"])):
        probes = clip_gradients(per_record_gradients(
            model, x[probe_ids], labels[probe_ids]), config["clip"]).numpy()
        bank, public = probes[:len(gallery)], probes[len(gallery):]
        present = bool(participation_rng.random() < q)
        present_count += int(present)
        count = config["batch_size"] - int(present)
        others = rng.choice(population, count, replace=False,
                            p=population_weights)
        ids = np.r_[[true_id], others] if present else others
        clean = clip_gradients(per_record_gradients(
            model, x[ids], labels[ids]), config["clip"]).mean(0)
        noise_sd = config["sigma"] * config["clip"] / config["batch_size"]
        release = (clean + torch.randn(clean.shape, generator=noise_rng) * noise_sd).numpy()
        clean = clean.numpy()
        backgrounds = {mode: weights @ public
                       for mode, weights in background_weights.items()}
        for mode, (increments, step_errors) in score_round(
                release, clean, bank, backgrounds, config).items():
            for method, increment in increments.items():
                scores[mode][method] += increment
            for name, error in step_errors.items():
                errors[mode][name] += error
            if q > 0:
                posterior = temporal_posterior_mean(
                    backgrounds[mode], bank, scores[mode]["mixture_llr"],
                    increments["gaussian_llr"], q, config["batch_size"])
            else:
                posterior = backgrounds[mode]
            errors[mode]["posterior"] += float(
                np.sum((posterior - clean) ** 2) / config["clip"] ** 2)
            oracle = backgrounds[mode] + (
                (bank[true_position] - backgrounds[mode]) / config["batch_size"]
                if present else 0)
            errors[mode]["oracle"] += float(
                np.sum((oracle - clean) ** 2) / config["clip"] ** 2)
        update(model, torch.from_numpy(release), config["learning_rate"])
        if step + 1 in config["prefix_steps"]:
            utility = _query_accuracy(model, x, labels, query_ids)
            for mode in scores:
                for method in methods:
                    rank = (_rank(scores[mode][method], true_position, gallery, labels)
                            if q > 0 else dict(rank="", top1="", top5="",
                                               mrr="", same_class_top1="",
                                               predicted_digit=""))
                    rows.append(dict(distribution=distribution, background=mode,
                        true_position=true_position, true_digit=int(labels[true_id]),
                        seed=seed, world="present" if q > 0 else "absent",
                        steps=step + 1, method=method, appearances=present_count,
                        utility=utility, max_score=float(scores[mode][method].max()),
                        gradient_mse=float(errors[mode].get(method, np.nan) / (step + 1)),
                        raw_mse=float(errors[mode]["raw"] / (step + 1)),
                        background_mse=float(errors[mode]["background"] / (step + 1)),
                        posterior_mse=float(errors[mode]["posterior"] / (step + 1)),
                        oracle_mse=float(errors[mode]["oracle"] / (step + 1)),
                        **rank))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    integers = ("true_position", "true_digit", "seed", "steps", "appearances")
    floats = ("utility", "max_score", "gradient_mse", "raw_mse",
              "background_mse", "posterior_mse", "oracle_mse")
    for row in rows:
        for key in integers:
            row[key] = int(row[key])
        for key in floats:
            row[key] = float(row[key])
        for key in ("rank", "top1", "top5", "mrr", "same_class_top1"):
            row[key] = float(row[key]) if row[key] else np.nan
    return rows


def select_methods(rows, config, calibration_positions):
    chosen = {}
    candidates = fixed_methods(config)
    selected = [row for row in rows if row["world"] == "present"
                and row["distribution"] == "balanced"
                and row["background"] == "matched"
                and row["steps"] == max(config["prefix_steps"])
                and row["true_position"] in calibration_positions]
    for family, methods in (("sparse", [x for x in candidates if x.startswith("sparse_")]),
                            ("same_access_baseline", BASELINES)):
        values = {method: float(np.mean([row["mrr"] for row in selected
                                       if row["method"] == method])) for method in methods}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Incomplete calibration")
        chosen[family] = sorted(methods, key=lambda method: (-values[method],
                                                                methods.index(method)))[0]
        chosen[family + "_calibration_mrr"] = values
    return chosen


def summarize(rows, config, chosen, evaluation_positions):
    rng = np.random.default_rng(config["seed"] + 789)
    output = []
    methods = BASELINES + (chosen["sparse"],)
    repetitions = int(config["bootstrap_repetitions"])
    for distribution in config["distributions"]:
        for background in (("matched",) if distribution == "balanced"
                           else ("matched", "balanced_mismatch")):
            for steps in config["prefix_steps"]:
                cell = [r for r in rows if r["world"] == "present"
                        and r["distribution"] == distribution
                        and r["background"] == background and r["steps"] == steps
                        and r["true_position"] in evaluation_positions]
                grouped = {method: {position: [r for r in cell if
                              r["method"] == method and r["true_position"] == position]
                              for position in evaluation_positions} for method in methods}
                expected_reps = config["evaluation_repetitions"]
                if any(len(grouped[m][i]) != expected_reps for m in methods
                       for i in evaluation_positions):
                    raise ValueError("Missing evaluation rows")
                identity_means = {method: {metric: np.array([
                    np.mean([r[metric] for r in grouped[method][i]])
                    for i in evaluation_positions])
                    for metric in ("top1", "top5", "mrr", "same_class_top1",
                                   "gradient_mse")}
                    for method in methods}
                boot = rng.integers(len(evaluation_positions),
                                    size=(repetitions, len(evaluation_positions)))
                baseline = chosen["same_access_baseline"]
                for method in methods:
                    data = identity_means[method]
                    row = dict(distribution=distribution, background=background,
                               steps=steps, method=method,
                               evaluation_identities=len(evaluation_positions),
                               trials=len(evaluation_positions) * expected_reps,
                               mean_utility=float(np.mean([r["utility"] for r in cell
                                   if r["method"] == method])),
                               mean_appearances=float(np.mean([r["appearances"] for r in cell
                                   if r["method"] == method])),
                               raw_mse=float(np.mean([r["raw_mse"] for r in cell
                                   if r["method"] == method])),
                               background_mse=float(np.mean([r["background_mse"] for r in cell
                                   if r["method"] == method])),
                               posterior_mse=float(np.mean([r["posterior_mse"] for r in cell
                                   if r["method"] == method])),
                               oracle_mse=float(np.mean([r["oracle_mse"] for r in cell
                                   if r["method"] == method])),
                               gradient_mse=float(np.nanmean(data["gradient_mse"]))
                                   if method.startswith("sparse_") else np.nan)
                    for metric in ("top1", "top5", "mrr", "same_class_top1"):
                        vector = data[metric]
                        row[metric] = float(vector.mean())
                        row[metric + "_ci_low"], row[metric + "_ci_high"] = map(
                            float, np.quantile(vector[boot].mean(axis=1), [.025, .975]))
                    difference = data["top1"] - identity_means[baseline]["top1"]
                    row["top1_gain_vs_selected_baseline"] = float(difference.mean())
                    row["top1_gain_ci_low"], row["top1_gain_ci_high"] = map(
                        float, np.quantile(difference[boot].mean(axis=1), [.025, .975]))
                    output.append(row)
    return output


def plot_summary(summary, selected, output):
    methods = [selected["same_access_baseline"], "mixture_llr", selected["sparse"]]
    methods = list(dict.fromkeys(methods))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    for axis, distribution in zip(axes, ("balanced", "long_tail")):
        for method in methods:
            chosen = [r for r in summary if r["distribution"] == distribution
                      and r["background"] == "matched" and r["method"] == method]
            axis.plot([r["steps"] for r in chosen], [r["top1"] for r in chosen],
                      marker="o", label=method.replace("_", " "))
            axis.fill_between([r["steps"] for r in chosen],
                              [r["top1_ci_low"] for r in chosen],
                              [r["top1_ci_high"] for r in chosen], alpha=.1)
        axis.axhline(1 / 32, color="black", ls="--", lw=1, label="random gallery")
        axis.set_title(distribution.replace("_", " "))
        axis.set_xticks([64, 128]); axis.set_xlabel("Observed DP-SGD releases")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Exact target image top-1 from known gallery")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output / f"equal_access_gallery.{ext}", dpi=300)
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
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").exists():
        raise ValueError("Completed output exists")
    saved_config = output / "config.json"
    if saved_config.exists() and json.loads(saved_config.read_text()) != config:
        raise ValueError("Cannot resume with changed configuration")
    saved_config.write_text(json.dumps(config, indent=2))
    dataset_hash = digest(args.data)
    source_hash = json.loads((source / "completion.json").read_text())["dataset_sha256"]
    if dataset_hash != source_hash:
        raise ValueError("Dataset differs from public checkpoint source")
    torch.set_num_threads(1)
    start = time.time()
    x, labels = load_mnist(args.data)
    splits = json.loads((source / "splits.json").read_text())
    source_config = json.loads((source / "config.json").read_text())
    initial = initialize(source_config["seed"])
    initial.load_state_dict(torch.load(source / "public_initial_cnn.pt", weights_only=True))
    initial.eval()
    gallery = stratified_gallery(np.asarray(splits["test"]), labels,
                                 config["gallery_size"])
    calibration = list(range(config["calibration_identities"]))
    evaluation = list(range(config["calibration_identities"],
                            config["calibration_identities"] + config["evaluation_identities"]))
    if (config["gallery_size"] < len(calibration) + len(evaluation)
            or len(set(gallery)) != len(gallery)):
        raise ValueError("Gallery does not cover unique calibration and evaluation identities")
    public_query = np.asarray(splits["test"])[-config["query_size"]:]
    if set(public_query) & set(gallery):
        raise ValueError("Utility query overlaps gallery")
    population = np.asarray(splits["population"])
    background = np.asarray(splits["background"])
    rows_path = output / "trial_rows.csv"
    rows = read_csv(rows_path)
    done = {(r["distribution"], r["true_position"], r["seed"], r["world"])
            for r in rows if r["steps"] == max(config["prefix_steps"])
            and r["background"] == "matched" and r["method"] == "alignment"}
    trials = []
    for distribution in config["distributions"]:
        for position in calibration + evaluation:
            repeats = (config["calibration_repetitions"] if position in calibration
                       else config["evaluation_repetitions"])
            for repetition in range(repeats):
                seed = config["seed"] + int(gallery[position]) * 100 + repetition
                trials.append((distribution, position, seed, None))
        for position in evaluation[:config["null_identities"]]:
            seed = config["seed"] + int(gallery[position]) * 100
            trials.append((distribution, position, seed, 0.0))
    for number, (distribution, position, seed, override) in enumerate(trials, 1):
        world = "absent" if override == 0 else "present"
        if (distribution, position, seed, world) in done:
            continue
        new = run_trajectory(initial, x, labels, population, background,
                             gallery, public_query, position, distribution,
                             config, seed, q_override=override)
        rows.extend(new)
        write_csv(rows_path, rows)
        print(f"trained/evaluated {number}/{len(trials)} trajectories", flush=True)
    selected = select_methods(rows, config, calibration)
    summary = summarize(rows, config, selected, evaluation)
    write_csv(report / "summary.csv", summary)
    (report / "selected_methods.json").write_text(json.dumps(selected, indent=2))
    plot_summary(summary, selected, report)
    provenance = dict(dataset_sha256=dataset_hash, source_checkpoint_sha256=
        hashlib.sha256((source / "public_initial_cnn.pt").read_bytes()).hexdigest(),
        source_run=str(source), config=config, calibration_positions=calibration,
        evaluation_positions=evaluation, gallery_indices_sha256=hashlib.sha256(
            gallery.tobytes()).hexdigest(), trajectories=len(trials), seconds=time.time()-start,
        role_separation="public pretraining/background/query; private population; known gallery",
        privacy_accounting="replace-one no-amplification upper bound, not tight sampling accountant",
        torch=torch.__version__)
    (report / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (output / "completion.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()
