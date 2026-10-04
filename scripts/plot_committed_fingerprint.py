"""Rebuild the headline figure using only committed aggregate CSVs."""

import argparse
import csv
from pathlib import Path

from scripts.plot_first_principles import plot


def read_rows(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No aggregate rows in {path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    participation = read_rows(args.report / "participation_auc_matched.csv")
    noise = read_rows(args.report / "noise_auc.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(participation, noise, args.output)
    print(f"Wrote {args.output.with_suffix('.pdf')} and {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
