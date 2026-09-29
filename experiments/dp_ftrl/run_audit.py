"""Checkpoint-only MNIST audit of the upstream DP-FTRL tree mechanism.

The training loop calls Google's unmodified FTRLOptimizer and tree-noise class.
The binary softmax model and its exact per-record clipped gradients are local to
this experiment. Only aggregate statistics are written to the output directory.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


UPSTREAM_COMMIT = "513500a8e31e412972a7d457e9c66756e4a48348"


def official_modules(path: Path):
    if not (path / "ftrl_noise.py").is_file() or not (path / "optimizers.py").is_file():
        raise FileNotFoundError("Clone https://github.com/google-research/DP-FTRL and pass --upstream PATH")
    commit = subprocess.check_output(
        [os.environ.get("GAUSSPROOF_GIT", "git"), "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream commit {UPSTREAM_COMMIT}, got {commit}")
    sys.path.insert(0, str(path))
    return importlib.import_module("ftrl_noise"), importlib.import_module("optimizers")


def tree_coefficients(noise_module, rounds: int, efficient: bool) -> np.ndarray:
    """Probe every independent Gaussian draw in upstream code with one basis axis.

    This yields the exact linear map from upstream node noise to its released
    prefix noise; it does not assume independent checkpoint errors.
    """
    width = 2 * rounds
    draws = 0

    def basis_draw(_mean, _std, shape):
        nonlocal draws
        assert tuple(shape) == (width,)
        if draws >= width:
            raise AssertionError("More node draws than allocated basis coordinates")
        result = torch.zeros(width, dtype=torch.float64)
        result[draws] = 1.0
        draws += 1
        return result

    cls = noise_module.CummuNoiseEffTorch if efficient else noise_module.CummuNoiseTorch
    with patch.object(torch, "normal", side_effect=basis_draw):
        generator = cls(1.0, [(width,)], "cpu")
        rows = [generator()[0].detach().clone().numpy() for _ in range(rounds)]
    expected = 2 * rounds - 1 if efficient else rounds
    if draws != expected:
        raise AssertionError(f"Upstream made {draws} node draws, expected {expected}")
    result = np.stack(rows)[:, :draws]
    covariance = result @ result.T
    np.linalg.cholesky(covariance)
    return result


def load_mnist(path: Path):
    with path.open() as stream:
        first = stream.readline().strip().split(",")
    try:
        [float(x) for x in first]
        skiprows = 0
    except ValueError:
        skiprows = 1
    a = np.loadtxt(path, delimiter=",", skiprows=skiprows, dtype=np.float32)
    if a.shape != (60000, 785) or not np.isfinite(a).all():
        raise ValueError("Expected the complete 60,000-row MNIST training CSV")
    labels = a[:, 0].astype(np.int64)
    if not np.all((a[:, 0] == labels) & (labels >= 0) & (labels < 10)):
        raise ValueError("Invalid MNIST labels")
    if a[:, 1:].min() < 0 or a[:, 1:].max() > 255:
        raise ValueError("MNIST pixels must be in [0,255]")
    x = np.empty((len(a), 785), dtype=np.float32)
    x[:, :784] = a[:, 1:] / 255.0
    x[:, 784] = 1.0
    return torch.from_numpy(x), torch.from_numpy(labels)


def clipped_gradient(weights, x, labels, clip):
    logits = x @ weights.T
    error = torch.softmax(logits, dim=1)
    error[torch.arange(len(labels)), labels] -= 1.0
    norms = torch.linalg.vector_norm(error, dim=1) * torch.linalg.vector_norm(x, dim=1)
    scales = torch.clamp(clip / norms.clamp_min(1e-12), max=1.0)
    return error.T @ (x * scales[:, None])


def pick_rows(labels, seed, calibration_per_class, holdout_per_class, rounds, batch):
    rng = np.random.default_rng(seed)
    cal, holdout = [], []
    reserved = set()
    for digit in range(10):
        ids = rng.permutation(np.flatnonzero(labels.numpy() == digit))
        cal.extend(ids[:calibration_per_class].tolist())
        holdout.extend(ids[calibration_per_class:calibration_per_class + holdout_per_class].tolist())
        reserved.update(ids[:calibration_per_class + holdout_per_class].tolist())
    pool = rng.permutation(np.array([i for i in range(len(labels)) if i not in reserved]))
    required = rounds * (batch - 1)
    if required + 1024 > len(pool):
        raise ValueError("Not enough disjoint MNIST background and validation rows")
    public = pool[:128]
    test = pool[128:1152]
    background = pool[1152:]
    return np.array(cal), np.array(holdout), background, public, test


def trajectory(
    noise_module, optimizer_module, x, y, candidate, background_schedule, public,
    validation, position, member, noise_seed, rounds, batch, clip, sigma, alpha,
    efficient, public_score, unknown_counts, decoy,
):
    torch.manual_seed(noise_seed)
    weights = torch.nn.Parameter(torch.zeros((10, 785), dtype=torch.float32))
    optimizer = optimizer_module.FTRLOptimizer([weights], momentum=0.0)
    cls = noise_module.CummuNoiseEffTorch if efficient else noise_module.CummuNoiseTorch
    tree = cls(sigma * clip, [tuple(weights.shape)], "cpu")
    background_sum = torch.zeros_like(weights)
    public_sum = torch.zeros_like(weights)
    private_residuals, public_residuals = [], []
    partial_sums = {count: torch.zeros_like(weights) for count in unknown_counts}
    partial_residuals = {count: [] for count in unknown_counts}
    fingerprint = None
    decoy_fingerprint = None
    for step in range(1, rounds + 1):
        with torch.no_grad():
            known = clipped_gradient(weights, x[background_schedule[step - 1]],
                                     y[background_schedule[step - 1]], clip)
            background_sum += known
            if public_score or unknown_counts:
                estimated = clipped_gradient(weights, x[public], y[public], clip)
                if public_score:
                    public_sum += estimated * ((batch - 1) / len(public))
                for count in unknown_counts:
                    missing = background_schedule[step - 1, -count:]
                    unknown_actual = clipped_gradient(weights, x[missing], y[missing], clip)
                    partial_sums[count] += known - unknown_actual + estimated * (count / len(public))
            if step == position:
                fingerprint = clipped_gradient(weights, x[candidate:candidate + 1],
                                               y[candidate:candidate + 1], clip)
                if decoy is not None:
                    decoy_fingerprint = clipped_gradient(weights, x[decoy:decoy + 1],
                                                         y[decoy:decoy + 1], clip)
            weights.grad = known + (fingerprint if member and step == position else 0)
            noisy_prefix = tree()[0]
            optimizer.step((alpha, [noisy_prefix]))
            # This equality is a training invariant, never an input to the attack.
            observed_prefix = -alpha * weights.detach()
            if step in (1, rounds) and not torch.allclose(
                observed_prefix, optimizer.state[weights]["grad_sum"] + noisy_prefix,
                atol=2e-3, rtol=2e-5,
            ):
                raise AssertionError("Model checkpoints do not encode the upstream noisy prefix")
            private_residuals.append((observed_prefix - background_sum).flatten().clone())
            if public_score:
                public_residuals.append((observed_prefix - public_sum).flatten().clone())
            for count in unknown_counts:
                partial_residuals[count].append(
                    (observed_prefix - partial_sums[count]).flatten().clone())
    h = fingerprint.flatten()
    projections = torch.stack(private_residuals) @ h
    decoy_projections = (torch.stack(private_residuals) @ decoy_fingerprint.flatten()
                         if decoy is not None else None)
    public_projections = (torch.stack(public_residuals) @ h) if public_score else None
    partial_projections = {count: np.asarray(torch.stack(rows) @ h, dtype=np.float64)
                           for count, rows in partial_residuals.items()}
    with torch.no_grad():
        logits = x[candidate:candidate + 1] @ weights.T
        endpoint_loss = -float(torch.nn.functional.cross_entropy(
            logits, y[candidate:candidate + 1]))
        validation_accuracy = float((x[validation] @ weights.T).argmax(1)
                                    .eq(y[validation]).float().mean())
    return np.asarray(projections, dtype=np.float64), (
        np.asarray(public_projections, dtype=np.float64) if public_score else None
    ), partial_projections, float(h @ h), (
        np.asarray(decoy_projections, dtype=np.float64) if decoy is not None else None
    ), (float(decoy_fingerprint.flatten() @ decoy_fingerprint.flatten())
        if decoy is not None else None), endpoint_loss, validation_accuracy


def score_prefixes(projections, hnorm2, position, covariance, sigma, clip, prefixes):
    result = {}
    for t in prefixes:
        shift = (np.arange(1, t + 1) >= position).astype(np.float64)
        if not shift.any():
            continue
        sub = covariance[:t, :t]
        weights = np.linalg.solve(sub, shift)
        variance = (sigma * clip) ** 2
        result[f"tree_{t}"] = float(
            (weights @ projections[:t] - 0.5 * (shift @ weights) * hnorm2) / variance
        )
    t = len(projections)
    variance = (sigma * clip) ** 2 * covariance[t - 1, t - 1]
    result["oracle_final_prefix"] = float(
        (projections[-1] - 0.5 * hnorm2) / variance
    )
    return result


def auc(pos, neg):
    # Mann-Whitney AUC with exact half credit for ties.
    p, n = np.asarray(pos), np.asarray(neg)
    return float(((p[:, None] > n).sum() + 0.5 * (p[:, None] == n).sum()) / (len(p) * len(n)))


def tpr_at_fpr(pos, neg, cap):
    p, n = np.asarray(pos), np.asarray(neg)
    # Empirical ROC, conservative at ties and without interpolation.
    thresholds = np.r_[np.inf, np.unique(np.r_[p, n])][::-1]
    feasible = [float((p >= z).mean()) for z in thresholds if (n >= z).mean() <= cap]
    return max(feasible)


def cluster_interval(rows, key, bootstrap, seed):
    rng = np.random.default_rng(seed)
    identities = sorted(set(row["identity"] for row in rows))
    by_identity = {i: [r for r in rows if r["identity"] == i] for i in identities}
    estimates = []
    for _ in range(bootstrap):
        sample = [r for i in rng.choice(identities, len(identities), replace=True)
                  for r in by_identity[i]]
        estimates.append(auc([r["positive"][key] for r in sample],
                             [r["negative"][key] for r in sample]))
    point = auc([r["positive"][key] for r in rows],
                [r["negative"][key] for r in rows])
    spread = 1.96 * float(np.std(estimates, ddof=1))
    return [max(0.0, point - spread), min(1.0, point + spread)]


def paired_gap_interval(rows, first, second, bootstrap, seed):
    rng = np.random.default_rng(seed)
    identities = sorted(set(row["identity"] for row in rows))
    by_identity = {i: [r for r in rows if r["identity"] == i] for i in identities}
    estimates = []
    for _ in range(bootstrap):
        sample = [r for i in rng.choice(identities, len(identities), replace=True)
                  for r in by_identity[i]]
        value = lambda key: auc([r["positive"][key] for r in sample],
                                [r["negative"][key] for r in sample])
        estimates.append(value(first) - value(second))
    value = lambda key: auc([r["positive"][key] for r in rows],
                            [r["negative"][key] for r in rows])
    point = value(first) - value(second)
    spread = 1.96 * float(np.std(estimates, ddof=1))
    return [point - spread, point + spread]


def run(args):
    torch.set_num_threads(1)
    upstream = Path(args.upstream).resolve()
    noise_module, optimizer_module = official_modules(upstream)
    x, y = load_mnist(Path(args.data))
    cal, holdout, pool, public, validation = pick_rows(
        y, args.seed, args.calibration_per_class, args.holdout_per_class,
        args.rounds, args.batch,
    )
    A = tree_coefficients(noise_module, args.rounds, args.efficient)
    covariance = A @ A.T
    prefixes = sorted(set(t for t in args.prefixes if 1 <= t <= args.rounds))
    if args.rounds not in prefixes:
        prefixes.append(args.rounds)
    position = args.position
    if position > args.rounds:
        raise ValueError("Canary position exceeds trajectory length")
    if any(count < 1 or count >= args.batch for count in args.unknown_counts):
        raise ValueError("Unknown background counts must be in [1, batch-1]")
    all_rows = []
    identities = [("calibration", int(i)) for i in cal] + [("holdout", int(i)) for i in holdout]
    for index, (split, candidate) in enumerate(identities):
        same_class = [row for _, row in identities
                      if row != candidate and int(y[row]) == int(y[candidate])]
        decoy = same_class[0] if args.decoy_control else None
        for rep in range(args.repetitions):
            trial_seed = int(np.random.SeedSequence([args.seed, index, rep]).generate_state(1)[0])
            schedule = np.random.default_rng(trial_seed).permutation(pool)[
                :args.rounds * (args.batch - 1)
            ].reshape(args.rounds, args.batch - 1)
            pair = {}
            for member in (False, True):
                proj, pub_proj, partial_proj, hnorm2, decoy_proj, decoy_norm2, loss, accuracy = trajectory(
                    noise_module, optimizer_module, x, y, candidate, schedule, public,
                    validation, position, member, trial_seed, args.rounds, args.batch,
                    args.clip, args.sigma, args.alpha, args.efficient, args.public_score,
                    args.unknown_counts, decoy,
                )
                scores = score_prefixes(proj, hnorm2, position, covariance,
                                        args.sigma, args.clip, prefixes)
                scores["endpoint_loss"] = loss
                scores["validation_accuracy"] = accuracy
                if args.public_score:
                    public_scores = score_prefixes(pub_proj, hnorm2, position, covariance,
                                                   args.sigma, args.clip, prefixes)
                    for k, v in public_scores.items():
                        scores[f"public_{k}"] = v
                for count, values in partial_proj.items():
                    partial_scores = score_prefixes(values, hnorm2, position, covariance,
                                                    args.sigma, args.clip, prefixes)
                    for k, v in partial_scores.items():
                        scores[f"unknown{count}_{k}"] = v
                if args.decoy_control:
                    decoy_scores = score_prefixes(decoy_proj, decoy_norm2, position,
                                                  covariance, args.sigma, args.clip, prefixes)
                    for k, v in decoy_scores.items():
                        scores[f"same_class_decoy_{k}"] = v
                        if k.startswith("tree_"):
                            scores[f"identity_contrast_{k}"] = scores[k] - v
                pair["positive" if member else "negative"] = scores
            all_rows.append({"split": split, "identity": index, **pair})
        print(f"Completed {index + 1}/{len(identities)} identities", flush=True)
    cal_rows = [r for r in all_rows if r["split"] == "calibration"]
    hold_rows = [r for r in all_rows if r["split"] == "holdout"]
    keys = [k for k in hold_rows[0]["positive"] if k != "validation_accuracy"]
    summary = []
    for key in keys:
        p = [r["positive"][key] for r in hold_rows]
        n = [r["negative"][key] for r in hold_rows]
        cal_p = [r["positive"][key] for r in cal_rows]
        cal_n = [r["negative"][key] for r in cal_rows]
        thresholds = {
            level: sorted(cal_n, reverse=True)[max(0, math.ceil(level * len(cal_n)) - 1)]
            for level in (0.01, 0.05)
        }
        # Strict > is conservative when calibration scores tie.
        at1 = float((np.asarray(p) > thresholds[0.01]).mean())
        fpr1 = float((np.asarray(n) > thresholds[0.01]).mean())
        at5 = float((np.asarray(p) > thresholds[0.05]).mean())
        fpr5 = float((np.asarray(n) > thresholds[0.05]).mean())
        ts = np.unique(np.r_[p, n])
        advantage = max(float((np.asarray(p) >= z).mean() -
                              (np.asarray(n) >= z).mean()) for z in ts)
        lo, hi = cluster_interval(hold_rows, key, args.bootstrap, args.seed + 712)
        summary.append({"attack": key, "auc": auc(p, n), "auc_ci_low": lo,
                        "auc_ci_high": hi, "max_advantage": advantage,
                        "tpr_at_1pct_fpr_empirical": tpr_at_fpr(p, n, .01),
                        "tpr_at_5pct_fpr_empirical": tpr_at_fpr(p, n, .05),
                        "calibrated_tpr_at_1pct_fpr": at1,
                        "calibrated_fpr_at_1pct_fpr": fpr1,
                        "calibrated_tpr_at_5pct_fpr": at5,
                        "calibrated_fpr_at_5pct_fpr": fpr5,
                        "n_test_identities": len(holdout), "n_test_pairs": len(hold_rows)})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    primary = f"tree_{args.rounds}"
    comparisons = []
    comparators = ([f"tree_{prefixes[0]}"] if prefixes[0] < args.rounds
                   and prefixes[0] >= position else [])
    comparators += ["oracle_final_prefix", "endpoint_loss"]
    if args.public_score:
        comparators.append(f"public_{primary}")
    if args.decoy_control:
        comparators.append(f"same_class_decoy_{primary}")
    for comparator in comparators:
        left = next(row["auc"] for row in summary if row["attack"] == primary)
        right = next(row["auc"] for row in summary if row["attack"] == comparator)
        low, high = paired_gap_interval(hold_rows, primary, comparator,
                                        args.bootstrap, args.seed + 851)
        comparisons.append({"attack": primary, "comparator": comparator,
                            "auc_difference": left - right,
                            "difference_ci_low": low, "difference_ci_high": high})
    with (out / "comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    # RDP upper bound for one participation in a standard tree; the efficient
    # variant is post-processing of the same Gaussian nodes, so it also applies.
    max_nodes = math.ceil(math.log2(args.rounds + 1))
    rho = max_nodes / (2 * args.sigma ** 2)
    delta = args.delta
    epsilon_zcdp = rho + 2 * math.sqrt(rho * math.log(1 / delta))
    metadata = {
        "upstream_url": "https://github.com/google-research/DP-FTRL",
        "upstream_commit": UPSTREAM_COMMIT,
        "mechanism": "official efficient tree" if args.efficient else "official standard tree",
        "dataset": "MNIST CSV, not redistributed",
        "model": "softmax regression, official FTRLOptimizer, momentum=0",
        "threat_model": "known canary insertion round; exact other-record and batch-schedule access for informed score",
        "public_score": "fixed disjoint 128-image background estimate" if args.public_score else "not run",
        "unknown_background_counts": args.unknown_counts,
        "same_class_decoy_control": args.decoy_control,
        "rounds": args.rounds, "position": position, "batch_size": args.batch,
        "sigma_multiplier": args.sigma, "clip": args.clip, "alpha": args.alpha,
        "calibration_identities": len(cal), "holdout_identities": len(holdout),
        "calibration_fpr_resolution": 1 / (len(cal) * args.repetitions),
        "ci_method": "95% normal interval from identity-cluster bootstrap standard deviation",
        "repetitions_per_identity": args.repetitions, "seed": args.seed,
        "node_noise_draws": int(A.shape[1]), "delta": delta,
        "rho_standard_tree_upper_bound": rho,
        "epsilon_conservative_zcdp_upper_bound": epsilon_zcdp,
        "mean_holdout_validation_accuracy": float(np.mean(
            [r["positive"]["validation_accuracy"] for r in hold_rows])),
        "mean_holdout_negative_validation_accuracy": float(np.mean(
            [r["negative"]["validation_accuracy"] for r in hold_rows])),
        "limitations": [
            "Not a run of the authors' full CNN or TensorFlow Federated pipeline.",
            "Informed gradient subtraction knows other records and the batch schedule.",
            "Public-background score uses a Gaussian likelihood that omits minibatch variance.",
            "No claim of violating the formal DP guarantee or reconstructing unknown images.",
        ],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return summary, metadata


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--upstream", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rounds", type=int, default=128)
    p.add_argument("--position", type=int, default=1)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=256.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--calibration-per-class", type=int, default=1)
    p.add_argument("--holdout-per-class", type=int, default=4)
    p.add_argument("--repetitions", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--prefixes", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--efficient", action="store_true")
    p.add_argument("--public-score", action="store_true")
    p.add_argument("--unknown-counts", type=int, nargs="*", default=[])
    p.add_argument("--decoy-control", action="store_true")
    return p


if __name__ == "__main__":
    rows, info = run(parser().parse_args())
    print(json.dumps({"summary": rows, "privacy_epsilon_upper_bound":
                      info["epsilon_conservative_zcdp_upper_bound"]}, indent=2))
