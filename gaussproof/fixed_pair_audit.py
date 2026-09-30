"""Fixed-neighbor DP-SGD trajectory audit with a natural MNIST record pair.

Two datasets differ in one privileged record slot. Every other record, the
initial model, and the sampling rule are fixed. At each step the slot is used
with probability q; otherwise the whole batch is sampled from the common
population. This is a *specified sampler*, not uniform minibatch DP-SGD.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta

from .canary_audit import update
from .data import digest, load_mnist
from .metrics import auc, rates, threshold_at_fpr
from .models import clip_gradients, initialize, per_record_gradients


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixed_world(initial, x, labels, population, background, target, dummy,
                world, cfg, seed):
    model = initialize(cfg["source_seed"])
    model.load_state_dict(initial)
    model.eval()
    rng = np.random.default_rng(seed)
    noise_rng = torch.Generator().manual_seed(seed + 1000000)
    q, batch = cfg["q"], cfg["batch_size"]
    if not 0 <= q <= 1:
        raise ValueError("Slot inclusion probability must be in [0, 1]")
    sd = cfg["sigma"] * cfg["clip"] / batch
    aggregate, last, raw = 0., 0., 0.
    inclusions = 0
    for _ in range(cfg["steps"]):
        pair = clip_gradients(per_record_gradients(
            model, x[[dummy, target]], labels[[dummy, target]]), cfg["clip"])
        h0, h1 = pair[0], pair[1]
        b = clip_gradients(per_record_gradients(
            model, x[background], labels[background]), cfg["clip"]).mean(0)
        included = bool(rng.random() < q)
        inclusions += int(included)
        others = rng.choice(population, batch - int(included), replace=False)
        ids = np.r_[others, target if world else dummy] if included else others
        clean = clip_gradients(per_record_gradients(
            model, x[ids], labels[ids]), cfg["clip"]).mean(0)
        release = clean + torch.randn(clean.shape, generator=noise_rng) * sd
        residual = release - b
        s0, s1 = (h0 - b) / batch, (h1 - b) / batch
        l0 = (torch.dot(residual, s0) - .5 * torch.dot(s0, s0)) / sd**2
        l1 = (torch.dot(residual, s1) - .5 * torch.dot(s1, s1)) / sd**2
        if q == 0:
            last = 0.
        elif q == 1:
            last = float(l1 - l0)
        else:
            p0 = np.logaddexp(np.log1p(-q), np.log(q) + float(l0))
            p1 = np.logaddexp(np.log1p(-q), np.log(q) + float(l1))
            last = p1 - p0
        aggregate += last
        raw += float(torch.dot(residual, h1 - h0))
        update(model, release, cfg["learning_rate"])
    with torch.no_grad():
        logp = model(x[[dummy, target]]).log_softmax(1)
        endpoint = float(logp[1, labels[target]] - logp[0, labels[dummy]])
    return dict(trajectory_mixture=aggregate, last_update=last,
                raw_alignment=raw, endpoint_logprob_difference=endpoint,
                realized_inclusions=inclusions)


def one_sided_limits(successes, total, alpha):
    lower = 0. if successes == 0 else float(beta.ppf(alpha, successes, total-successes+1))
    upper = 1. if successes == total else float(beta.ppf(1-alpha, successes+1, total-successes))
    return lower, upper


def summarize(rows, cfg):
    cal = [r for r in rows if r["role"] == "calibration"]
    test = [r for r in rows if r["role"] == "holdout"]
    if len(cal) != 2 * cfg["calibration_pairs"] or len(test) != 2 * cfg["holdout_pairs"]:
        raise ValueError("Incomplete paired audit")
    result = []
    rng = np.random.default_rng(cfg["seed"] + 700000)
    for method in ("trajectory_mixture", "last_update", "endpoint_logprob_difference"):
        cs = np.asarray([r[method] for r in cal])
        cy = np.asarray([r["world"] for r in cal])
        ts = np.asarray([r[method] for r in test])
        ty = np.asarray([r["world"] for r in test])
        threshold = threshold_at_fpr(cs, cy, cfg["calibration_fpr"])
        tpr, fpr = rates(ts, ty, threshold)
        tp = int(np.sum((ts >= threshold) & (ty == 1)))
        fp = int(np.sum((ts >= threshold) & (ty == 0)))
        # Bonferroni simultaneous one-sided 95% bounds for the two rates.
        tpr_low, _ = one_sided_limits(tp, cfg["holdout_pairs"], .025)
        _, fpr_high = one_sided_limits(fp, cfg["holdout_pairs"], .025)
        empirical_epsilon_low = (max(0., math.log(
            max(tpr_low - cfg["delta"], 1e-300) / fpr_high))
            if tpr_low > cfg["delta"] else 0.)
        observed_auc = auc(ts, ty)
        pair_ids = np.asarray([r["pair"] for r in test])
        exceed = 0
        for _ in range(2000):
            flips = rng.integers(0, 2, size=cfg["holdout_pairs"])
            permuted = np.bitwise_xor(ty, flips[pair_ids])
            exceed += auc(ts, permuted) >= observed_auc
        permutation_p = (exceed + 1) / 2001
        result.append(dict(method=method, auc=observed_auc,
            calibration_fpr_target=cfg["calibration_fpr"],
            holdout_tpr=tpr, holdout_fpr=fpr, true_positives=tp,
            false_positives=fp, holdout_pairs=cfg["holdout_pairs"],
            tpr_lower_97_5=tpr_low, fpr_upper_97_5=fpr_high,
            empirical_epsilon_lower_95=empirical_epsilon_low,
            paired_label_permutation_p=permutation_p,
            delta=cfg["delta"]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    source, output, report = Path(args.source), Path(args.output), Path(args.report)
    cfg = json.loads(Path(args.config).read_text())
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").exists():
        raise ValueError("Completed output exists")
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != cfg:
            raise ValueError("Cannot resume with changed configuration")
    else:
        (output / "config.json").write_text(json.dumps(cfg, indent=2))
    expected = json.loads((source / "completion.json").read_text())["dataset_sha256"]
    observed = digest(args.data)
    if observed != expected:
        raise ValueError("MNIST digest differs from public checkpoint run")
    torch.set_num_threads(1)
    started = time.time()
    x, labels = load_mnist(args.data)
    splits = json.loads((source / "splits.json").read_text())
    source_cfg = json.loads((source / "config.json").read_text())
    runtime_cfg = dict(cfg, source_seed=source_cfg["seed"])
    initial = torch.load(source / "public_initial_cnn.pt", weights_only=True)
    dummy, target = map(int, splits["test"][:2])
    population = np.asarray(splits["population"], dtype=int)
    background = np.asarray(splits["background"], dtype=int)
    if len({dummy, target}) != 2 or {dummy, target} & set(population) or {dummy, target} & set(background):
        raise ValueError("Natural record pair overlaps the common population")
    rows_path = output / "score_rows.csv"
    rows = [] if not rows_path.exists() else [dict(r, world=int(r["world"]),
        pair=int(r["pair"]), **{k:float(r[k]) for k in (
            "trajectory_mixture", "last_update", "raw_alignment",
            "endpoint_logprob_difference", "realized_inclusions")})
        for r in csv.DictReader(rows_path.open())]
    completed = {(r["role"], r["pair"], r["world"]) for r in rows}
    for role, count, offset in (("calibration", cfg["calibration_pairs"], 0),
                                ("holdout", cfg["holdout_pairs"], 100000)):
        for pair in range(count):
            for world in (0, 1):
                if (role, pair, world) in completed:
                    continue
                seed = cfg["seed"] + offset + pair * 101
                feature = fixed_world(initial, x, labels, population, background,
                                      target, dummy, world, runtime_cfg, seed)
                rows.append(dict(role=role, pair=pair, world=world, **feature))
                write_csv(rows_path, rows)
            if (pair + 1) % 10 == 0:
                print(f"{role}: {pair+1}/{count} pairs", flush=True)
    summary = summarize(rows, cfg)
    write_csv(report / "fixed_pair_summary.csv", summary)
    upper_rho = 2 * cfg["steps"] / cfg["sigma"] ** 2
    upper_epsilon = upper_rho + 2 * math.sqrt(upper_rho * math.log(1 / cfg["delta"]))
    provenance = dict(seconds=time.time()-started, dataset_sha256=observed,
        dummy_index=dummy, target_index=target,
        dummy_digit=int(labels[dummy]), target_digit=int(labels[target]),
        n_calibration_pairs=cfg["calibration_pairs"], n_holdout_pairs=cfg["holdout_pairs"],
        sampler="Bernoulli privileged slot, fixed replace-one pair; not uniform minibatch",
        delta=cfg["delta"], conservative_zcdp_epsilon_upper_no_amplification=upper_epsilon,
        interpretation="empirical lower bound applies to this fixed neighboring pair only",
        torch=torch.__version__)
    (report / "fixed_pair_provenance.json").write_text(json.dumps(provenance, indent=2))
    (output / "completion.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
