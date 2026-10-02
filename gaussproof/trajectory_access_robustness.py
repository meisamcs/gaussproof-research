"""Matched observation-access stress test for known-record trajectory attacks.

The private run is an evolving DP-SGD CNN trajectory. Access restrictions are
applied *after* generation, so every score in a row attacks the same world.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .canary_audit import run_sequence, scores as analytic_scores
from .canary_q_sensitivity import mixture_llr, stratified_identities
from .data import digest, load_mnist
from .metrics import auc, rates, threshold_at_fpr
from .models import initialize
from .trajectory_endpoint_replication import public_roles


def access_scores(run, q, sigma, clip, batch_size, cadence=1, stale=1,
                  assumed_sigma=None):
    """Score identical visible releases by mixture likelihood and projection.

    ``cadence`` releases one in every cadence noisy updates. ``stale`` retains
    the last checkpoint fingerprint for that many rounds, while all releases
    remain visible. The current release never contributes to its fingerprint.
    """
    if cadence < 1 or stale < 1:
        raise ValueError("Access intervals must be positive")
    assumed_sigma = sigma if assumed_sigma is None else assumed_sigma
    if assumed_sigma <= 0:
        raise ValueError("Assumed noise must be positive")
    indices = np.arange(cadence - 1, len(run["observations"]), cadence)
    if not len(indices):
        raise ValueError("Cadence exceeds trajectory length")
    source = (indices // stale) * stale
    view = dict(observations=run["observations"][indices],
                fingerprints=run["fingerprints"][source],
                backgrounds=run["backgrounds"][source])
    analytic = analytic_scores(view, assumed_sigma * clip, batch_size)
    residual = view["fingerprints"] - view["backgrounds"]
    observed = view["observations"] - view["backgrounds"]
    alignment = float(np.einsum("td,td->", observed, residual))
    return mixture_llr(analytic["round_llr"], q), alignment, len(indices)


def configurations(config):
    rows = [dict(name="full", cadence=1, stale=1,
                 assumed_sigma=float(config["sigma"]))]
    for cadence in config["cadences"]:
        if int(cadence) > 1:
            rows.append(dict(name=f"every_{int(cadence)}_releases",
                             cadence=int(cadence), stale=1,
                             assumed_sigma=float(config["sigma"])))
    for stale in config["stale_intervals"]:
        if int(stale) > 1:
            rows.append(dict(name=f"stale_{int(stale)}_rounds",
                             cadence=1, stale=int(stale),
                             assumed_sigma=float(config["sigma"])))
    for assumed_sigma in config["assumed_sigmas"]:
        if float(assumed_sigma) != float(config["sigma"]):
            rows.append(dict(name=f"assume_sigma_{float(assumed_sigma):g}",
                             cadence=1, stale=1,
                             assumed_sigma=float(assumed_sigma)))
    names = [row["name"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate access condition")
    return rows


def score_run(run, config, scenarios):
    rows = []
    for condition in scenarios:
        mixture, alignment, visible = access_scores(
            run, config["q"], config["sigma"], config["clip"],
            config["batch_size"], condition["cadence"], condition["stale"],
            condition["assumed_sigma"])
        rows.append(dict(scenario=condition["name"], visible_releases=visible,
                         gaussproof=mixture, alignment=alignment))
    return rows


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return [dict(row, identity=int(row["identity"]), world=int(row["world"]),
                     digit=int(row["digit"]), visible_releases=int(row["visible_releases"]),
                     appearances=int(row["appearances"]),
                     gaussproof=float(row["gaussproof"]),
                     alignment=float(row["alignment"]),
                     endpoint_loss=float(row["endpoint_loss"]),
                     public_accuracy=float(row["public_accuracy"]))
                for row in csv.DictReader(stream)]


def summarize(rows, scenarios, calibration_ids, holdout_ids, config):
    methods = ("gaussproof", "alignment", "endpoint_loss")
    rng = np.random.default_rng(config["seed"] + 900000)
    output = []
    for scenario in scenarios:
        name = scenario["name"]
        selected = [row for row in rows if row["scenario"] == name]
        cal = [row for row in selected if row["identity"] in calibration_ids]
        test = [row for row in selected if row["identity"] in holdout_ids]
        for group, identities in ((cal, calibration_ids), (test, holdout_ids)):
            if {(r["identity"], r["world"]) for r in group} != {
                    (identity, world) for identity in identities for world in (0, 1)}:
                raise ValueError(f"Missing or duplicate world in {name}")
        labels = np.asarray([r["world"] for r in test], int)
        positions = {identity: np.flatnonzero(
            np.asarray([r["identity"] for r in test]) == identity)
            for identity in holdout_ids}
        boot_indices = [np.concatenate([positions[int(identity)] for identity in
            rng.choice(holdout_ids, len(holdout_ids), replace=True)])
            for _ in range(config["bootstrap_repetitions"])]
        score_vectors = {method: np.asarray([r[method] for r in test])
                         for method in methods}
        observed = {method: auc(score_vectors[method], labels) for method in methods}
        distributions = {method: np.asarray([auc(score_vectors[method][index], labels[index])
            for index in boot_indices]) for method in methods}
        for method in methods:
            cal_scores = np.asarray([r[method] for r in cal])
            cal_labels = np.asarray([r["world"] for r in cal], int)
            threshold = threshold_at_fpr(cal_scores, cal_labels,
                                         config["calibration_fpr"])
            tpr, fpr = rates(score_vectors[method], labels, threshold)
            difference = observed[method] - observed["alignment"]
            difference_boot = distributions[method] - distributions["alignment"]
            endpoint_difference = observed[method] - observed["endpoint_loss"]
            endpoint_boot = distributions[method] - distributions["endpoint_loss"]
            ci_half = 1.96 * float(np.std(distributions[method], ddof=1))
            difference_half = 1.96 * float(np.std(difference_boot, ddof=1))
            endpoint_half = 1.96 * float(np.std(endpoint_boot, ddof=1))
            output.append(dict(scenario=name, method=method,
                visible_releases=test[0]["visible_releases"],
                auc=observed[method],
                auc_ci_low=max(0., observed[method] - ci_half),
                auc_ci_high=min(1., observed[method] + ci_half),
                gain_vs_alignment=difference,
                gain_ci_low=difference-difference_half,
                gain_ci_high=difference+difference_half,
                gain_vs_endpoint=endpoint_difference,
                endpoint_gain_ci_low=endpoint_difference-endpoint_half,
                endpoint_gain_ci_high=endpoint_difference+endpoint_half,
                calibrated_tpr=tpr, achieved_fpr=fpr,
                nominal_calibration_fpr=config["calibration_fpr"],
                mean_public_accuracy=float(np.mean([r["public_accuracy"] for r in test])),
                mean_positive_appearances=float(np.mean([r["appearances"]
                    for r in test if r["world"] == 1])),
                calibration_identities=len(calibration_ids),
                holdout_identities=len(holdout_ids)))
    return output


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
    scenarios = configurations(cfg)
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").exists():
        raise ValueError("Completed run exists")
    saved_cfg = output / "config.json"
    if saved_cfg.exists() and json.loads(saved_cfg.read_text()) != cfg:
        raise ValueError("Cannot resume with changed configuration")
    saved_cfg.write_text(json.dumps(cfg, indent=2))
    expected = json.loads((source / "completion.json").read_text())["dataset_sha256"]
    actual = digest(args.data)
    if actual != expected:
        raise ValueError("Dataset does not match source checkpoint")
    torch.set_num_threads(1)
    started = time.time()
    x, labels = load_mnist(args.data)
    splits = json.loads((source / "splits.json").read_text())
    source_cfg = json.loads((source / "config.json").read_text())
    initial = initialize(source_cfg["seed"])
    initial.load_state_dict(torch.load(source / "public_initial_cnn.pt", weights_only=True))
    initial.eval()
    identities = stratified_identities(splits["test"], labels,
                                       cfg["identities_per_class"])
    calibration_ids, holdout_ids = set(), set()
    for digit in range(10):
        members = [identity for identity in identities if int(labels[identity]) == digit]
        calibration_ids.update(members[:cfg["calibration_per_class"]])
        holdout_ids.update(members[cfg["calibration_per_class"]:])
    calibration_ids, holdout_ids = sorted(calibration_ids), sorted(holdout_ids)
    if not calibration_ids or not holdout_ids:
        raise ValueError("Both calibration and holdout identities are required")
    _, query = public_roles(len(x), splits, 0, cfg["utility_size"], cfg["seed"] + 17)
    query_labels = labels[query].numpy()
    mechanism = dict(source_cfg, steps=cfg["steps"], clip=cfg["clip"],
                     batch_size=cfg["batch_size"], learning_rate=cfg["learning_rate"],
                     decouple_inclusion_rng=True)
    rows_path = output / "score_rows.csv"
    rows = load_rows(rows_path)
    done = {(row["identity"], row["world"], row["scenario"]) for row in rows}
    for index, identity in enumerate(identities):
        for world in (0, 1):
            if all((identity, world, scenario["name"]) in done for scenario in scenarios):
                continue
            seed = cfg["seed"] + identity * 100
            run = run_sequence(initial, x, labels, np.asarray(splits["population"]),
                np.asarray(splits["background"]), identity, mechanism,
                cfg["sigma"], seed, cfg["q"] if world else 0.,
                query_indices=query, query_steps=[cfg["steps"]])
            accuracy = float(np.mean(
                run["endpoint_query_predictions"][cfg["steps"]] == query_labels))
            endpoint = -float(run["endpoint_losses"][-1])
            for feature in score_run(run, cfg, scenarios):
                row = dict(identity=identity, digit=int(labels[identity]),
                           world=world, appearances=int(run["included"].sum()),
                           endpoint_loss=endpoint, public_accuracy=accuracy,
                           **feature)
                if (identity, world, feature["scenario"]) not in done:
                    rows.append(row)
                    done.add((identity, world, feature["scenario"]))
            save_csv(rows_path, rows)
        if (index + 1) % 10 == 0:
            print(f"completed {index + 1}/{len(identities)} identities", flush=True)
    summary = summarize(rows, scenarios, calibration_ids, holdout_ids, cfg)
    save_csv(report / "summary.csv", summary)
    provenance = dict(seconds=time.time()-started, dataset_sha256=actual,
        checkpoint_sha256=hashlib.sha256(
            (source / "public_initial_cnn.pt").read_bytes()).hexdigest(),
        source_run=str(source), q=cfg["q"], sigma=cfg["sigma"],
        clip=cfg["clip"], batch_size=cfg["batch_size"], steps=cfg["steps"],
        learning_rate=cfg["learning_rate"], calibration_identities=len(calibration_ids),
        holdout_identities=len(holdout_ids), scenarios=scenarios,
        paired_world_seeds=True, model_utility_query_disjoint=True,
        torch=torch.__version__)
    (report / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (output / "completion.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
