"""Compare an ordinary-rate record across two matched conservative budgets."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gaussproof.metrics import auc


def read_rows(path):
    with path.open(newline="") as stream:
        return [dict(row, identity=int(row["identity"]), world=int(row["world"]),
                     gaussproof=float(row["gaussproof"]),
                     alignment=float(row["alignment"]),
                     endpoint_loss=float(row["endpoint_loss"]),
                     public_accuracy=float(row["public_accuracy"]),
                     appearances=int(row["appearances"]))
                for row in csv.DictReader(stream) if row["scenario"] == "full"]


def output_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", required=True)
    parser.add_argument("--long", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    roots = {128: Path(args.short), 512: Path(args.long)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = {steps: json.loads((root / "config.json").read_text())
              for steps, root in roots.items()}
    if any(config[steps]["steps"] != steps for steps in roots):
        raise ValueError("T labels and run configurations disagree")
    if any(config[steps]["q"] != config[128]["q"] or
           config[steps]["batch_size"] != config[128]["batch_size"] or
           config[steps]["clip"] != config[128]["clip"] or
           config[steps]["seed"] != config[128]["seed"] for steps in roots):
        raise ValueError("Membership mechanism or seed changed across budgets")
    rho = {steps: 2 * steps / config[steps]["sigma"]**2 for steps in roots}
    if not math.isclose(rho[128], rho[512]):
        raise ValueError("Conservative no-amplification rho is not matched")
    rows = {steps: read_rows(root / "score_rows.csv") for steps, root in roots.items()}
    identities = list(dict.fromkeys(row["identity"] for row in rows[128]))
    calibration = set()
    # The main runner selects the first two identities within each digit as
    # calibration; reconstruct that role from the saved digit column.
    digits = {row["identity"]: int(row["digit"]) for row in rows[128]}
    for digit in range(10):
        calibration.update([i for i in identities if digits[i] == digit][:
                               config[128]["calibration_per_class"]])
    holdout = [i for i in identities if i not in calibration]
    if len(holdout) != 80:
        raise ValueError("Expected 80 independent holdout identities")
    lookups = {steps: {(r["identity"], r["world"]): r for r in group}
               for steps, group in rows.items()}
    expected = {(i, w) for i in identities for w in (0, 1)}
    if any(set(lookups[steps]) != expected for steps in roots):
        raise ValueError("Runs do not contain the same paired identity worlds")
    methods = ("gaussproof", "alignment", "endpoint_loss")
    labels = np.tile([0, 1], len(holdout))
    values = {steps: {method: np.asarray([
        lookups[steps][i, w][method] for i in holdout for w in (0, 1)])
        for method in methods} for steps in roots}
    rng = np.random.default_rng(config[128]["seed"] + 990000)
    draws = rng.integers(0, len(holdout), size=(2000, len(holdout)))
    indices = np.stack([2*draws, 2*draws+1], axis=-1).reshape(2000, -1)
    aucs = {(steps, method): auc(values[steps][method], labels)
            for steps in roots for method in methods}
    boot = {(steps, method): np.asarray([auc(values[steps][method][index],
                                              labels[index]) for index in indices])
            for steps in roots for method in methods}
    summary = []
    for steps in roots:
        for method in methods:
            point = aucs[steps, method]
            half = 1.96 * float(np.std(boot[steps, method], ddof=1))
            appearance = np.mean([lookups[steps][i, 1]["appearances"] for i in holdout])
            utility = np.mean([lookups[steps][i, w]["public_accuracy"]
                               for i in holdout for w in (0, 1)])
            summary.append(dict(steps=steps, sigma=config[steps]["sigma"],
                q=config[steps]["q"], method=method, auc=point,
                auc_ci_low=max(0, point-half), auc_ci_high=min(1, point+half),
                mean_public_accuracy=utility, mean_positive_appearances=appearance,
                zero_appearance_fraction=np.mean([
                    lookups[steps][i, 1]["appearances"] == 0 for i in holdout]),
                expected_appearances=steps*config[steps]["q"],
                rho_no_amplification=rho[steps],
                epsilon_upper_no_amplification=rho[steps]+2*math.sqrt(
                    rho[steps]*math.log(1e5)), holdout_identities=len(holdout)))
    output_csv(output / "summary.csv", summary)
    differences = []
    for method in methods:
        point = aucs[512, method] - aucs[128, method]
        distribution = boot[512, method] - boot[128, method]
        half = 1.96 * float(np.std(distribution, ddof=1))
        differences.append(dict(method=method, auc_gain_long_minus_short=point,
            gain_ci_low=point-half, gain_ci_high=point+half))
    output_csv(output / "paired_differences.csv", differences)
    fig, ax = plt.subplots(figsize=(6.7, 4.1))
    palette = {"gaussproof": "#155e75", "alignment": "#bb6527",
               "endpoint_loss": "#626b73"}
    for method, label in (("gaussproof", "GAUSSPROOF"),
                          ("alignment", "Linear alignment"),
                          ("endpoint_loss", "Final candidate loss")):
        chosen = [r for r in summary if r["method"] == method]
        chosen.sort(key=lambda r: r["steps"])
        y = np.asarray([r["auc"] for r in chosen])
        lo = y - np.asarray([r["auc_ci_low"] for r in chosen])
        hi = np.asarray([r["auc_ci_high"] for r in chosen]) - y
        ax.errorbar([128, 512], y, yerr=[lo, hi], marker="o", capsize=3,
                    color=palette[method], linewidth=1.8, label=label)
    ax.axhline(.5, color="#999999", linewidth=.8)
    ax.set_xticks([128, 512])
    ax.set_xlabel("Training steps T (equal conservative privacy upper bound)")
    ax.set_ylabel("Held-out membership AUC")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(output / f"ordinary_q_budget.{extension}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
