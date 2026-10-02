"""Transfer pilot-calibrated trajectory thresholds to disjoint held-out identities."""
import argparse
import csv
from pathlib import Path

from gaussproof.metrics import rates, threshold_at_fpr


def load(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def subset(rows, q, steps):
    return [r for r in rows if float(r["q"]) == q and int(r["steps"]) == steps]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    calibration, evaluation = load(args.calibration), load(args.evaluation)
    calibration_ids = {r["canary"] for r in calibration}
    evaluation_ids = {r["canary"] for r in evaluation}
    if calibration_ids & evaluation_ids:
        raise ValueError("Calibration and evaluation identities overlap")
    qs = sorted({float(r["q"]) for r in evaluation}, reverse=True)
    steps = sorted({int(r["steps"]) for r in evaluation})
    output = []
    for q in qs:
        for step in steps:
            cal, eva = subset(calibration, q, step), subset(evaluation, q, step)
            if not cal or not eva:
                raise ValueError(f"Missing scores for q={q:g}, T={step}")
            threshold = threshold_at_fpr(
                [float(r["mixture_llr"]) for r in cal],
                [int(r["label"]) for r in cal], .05)
            tpr, fpr = rates([float(r["mixture_llr"]) for r in eva],
                             [int(r["label"]) for r in eva], threshold)
            output.append(dict(q=q, steps=step, nominal_calibration_fpr=.05,
                               calibrated_threshold=threshold,
                               achieved_holdout_tpr=tpr, achieved_holdout_fpr=fpr,
                               calibration_negatives=sum(int(r["label"]) == 0 for r in cal),
                               evaluation_positives=sum(int(r["label"]) == 1 for r in eva),
                               evaluation_negatives=sum(int(r["label"]) == 0 for r in eva)))
    with Path(args.output).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)


if __name__ == "__main__":
    main()
