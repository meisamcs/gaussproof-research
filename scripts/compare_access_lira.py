"""Strong endpoint LiRA control for the held-out access-stress identities."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from gaussproof.metrics import auc, rates, threshold_at_fpr
from gaussproof.trajectory_endpoint_replication import logit


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "config.json").read_text())
    with (run / "score_rows.csv").open(newline="") as stream:
        rows = [dict(row, identity=int(row["identity"]), world=int(row["world"]),
                     digit=int(row["digit"]), endpoint_loss=float(row["endpoint_loss"]),
                     gaussproof=float(row["gaussproof"]),
                     alignment=float(row["alignment"]))
                for row in csv.DictReader(stream) if row["scenario"] == "full"]
    identities = list(dict.fromkeys(row["identity"] for row in rows))
    if len(identities) != 100 or config["steps"] != 128 or config["sigma"] != 4. or config["learning_rate"] != .005:
        raise ValueError("Reference predictions require the matched 100-identity endpoint run")
    reference_run = Path(args.references).parent
    reference_config = json.loads((reference_run / "config.json").read_text())
    if (reference_config["sigma"] != config["sigma"] or
            reference_config["learning_rate"] != config["learning_rate"] or
            max(reference_config["prefix_steps"]) != config["steps"]):
        raise ValueError("OUT reference training mechanism does not match")
    with (reference_run / "score_rows.csv").open(newline="") as stream:
        reference_identities = list(dict.fromkeys(
            int(row["canary"]) for row in csv.DictReader(stream)))
    if reference_identities != identities:
        raise ValueError("OUT reference columns do not match target identity order")
    reference_completion = json.loads((reference_run / "completion.json").read_text())
    current_completion = json.loads((run / "completion.json").read_text())
    if (reference_completion["source_checkpoint_sha256"] !=
            current_completion["checkpoint_sha256"]):
        raise ValueError("OUT reference models use a different initialization")
    with np.load(args.references) as saved:
        ref = saved["128"][:, :len(identities)].copy()
    if ref.shape != (32, 100):
        raise ValueError("Expected 32 OUT references for 100 target identities")
    median = np.median(logit(ref), axis=0)
    std = max(float(np.sqrt(np.mean((logit(ref) - median[None, :])**2))), 1e-6)
    positions = {identity: j for j, identity in enumerate(identities)}
    for row in rows:
        j = positions[row["identity"]]
        target_prob = np.exp(row["endpoint_loss"])
        row["lira"] = float((logit(target_prob) - median[j]) / std)
    calibration = set()
    for digit in range(10):
        calibration.update([identity for identity in identities if
            next(row["digit"] for row in rows if row["identity"] == identity) == digit]
            [:config["calibration_per_class"]])
    holdout = [identity for identity in identities if identity not in calibration]
    if len(calibration) != 20 or len(holdout) != 80:
        raise ValueError("Calibration and holdout partitions changed")
    cal = [row for row in rows if row["identity"] in calibration]
    test = [row for row in rows if row["identity"] in holdout]
    for method in ("lira", "gaussproof", "alignment"):
        mean = float(np.mean([row[method] for row in cal]))
        sd = max(float(np.std([row[method] for row in cal])), 1e-12)
        for row in rows:
            row[f"{method}_standardized"] = (row[method]-mean)/sd
    for row in rows:
        row["fusion_half"] = .5*(row["lira_standardized"] +
                                   row["gaussproof_standardized"])
    grid = [0., .25, .5, .75, 1.]
    cal_labels = np.asarray([row["world"] for row in cal])
    calibration_aucs = []
    for weight in grid:
        candidate = [(1-weight)*row["lira_standardized"] +
                     weight*row["gaussproof_standardized"] for row in cal]
        calibration_aucs.append(auc(candidate, cal_labels))
    chosen = grid[int(np.argmax(calibration_aucs))]
    for row in rows:
        row["fusion_selected"] = ((1-chosen)*row["lira_standardized"] +
                                   chosen*row["gaussproof_standardized"])
    test_labels = np.asarray([row["world"] for row in test])
    ids = np.asarray([row["identity"] for row in test])
    positions = {identity: np.flatnonzero(ids == identity) for identity in holdout}
    rng = np.random.default_rng(config["seed"] + 991000)
    draws = [np.concatenate([positions[int(i)] for i in
        rng.choice(holdout, len(holdout), replace=True)]) for _ in range(2000)]
    methods = ("lira", "gaussproof", "alignment", "fusion_half", "fusion_selected")
    scores = {method: np.asarray([row[method] for row in test]) for method in methods}
    observed = {method: auc(scores[method], test_labels) for method in methods}
    boot = {method: np.asarray([auc(scores[method][index], test_labels[index])
        for index in draws]) for method in methods}
    summary = []
    for method in methods:
        point = observed[method]
        half = 1.96 * float(np.std(boot[method], ddof=1))
        gain = point - observed["lira"]
        gain_half = 1.96 * float(np.std(boot[method]-boot["lira"], ddof=1))
        cal_scores = np.asarray([row[method] for row in cal])
        threshold = threshold_at_fpr(cal_scores, cal_labels, .05)
        tpr, fpr = rates(scores[method], test_labels, threshold)
        summary.append(dict(method=method, auc=point,
            auc_ci_low=max(0.,point-half), auc_ci_high=min(1.,point+half),
            gain_vs_lira=gain, gain_ci_low=gain-gain_half,
            gain_ci_high=gain+gain_half, calibration_selected_weight=chosen,
            nominal_calibration_fpr=.05, calibrated_holdout_tpr=tpr,
            achieved_holdout_fpr=fpr,
            calibration_identities=20, holdout_identities=80))
    write_csv(output / "strong_endpoint_comparison.csv", summary)
    (output / "strong_endpoint_provenance.json").write_text(json.dumps(dict(
        references_sha256=hashlib.sha256(Path(args.references).read_bytes()).hexdigest(),
        reference_models=32, out_reference_predictions_reused=True,
        same_public_initialization=True, same_sigma=True,
        same_learning_rate=True, matched_candidate_order=True,
        variant="one-sided standardized OUT-logit LiRA",
        fusion_weight_grid=grid, fusion_calibration_aucs=calibration_aucs,
        fusion_selected_weight=chosen), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
