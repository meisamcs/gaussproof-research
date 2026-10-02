"""Paired identity bootstrap for high-participation trajectory contrasts."""
import argparse
import csv
from pathlib import Path

import numpy as np

from gaussproof.metrics import auc


def load_cube(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No score rows")
    qs = sorted({float(r["q"]) for r in rows}, reverse=True)
    steps = sorted({int(r["steps"]) for r in rows})
    identities = sorted({int(r["canary"]) for r in rows})
    sequences = sorted({int(r["sequence"]) for r in rows})
    if sequences != list(range(len(sequences))):
        raise ValueError("Sequences must be numbered consecutively from zero")
    cube = np.full((len(qs), len(steps), len(identities), len(sequences), 2), np.nan)
    q_index = {value: index for index, value in enumerate(qs)}
    step_index = {value: index for index, value in enumerate(steps)}
    identity_index = {value: index for index, value in enumerate(identities)}
    for row in rows:
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError("Membership label must be binary")
        key = (q_index[float(row["q"])], step_index[int(row["steps"])],
               identity_index[int(row["canary"])], int(row["sequence"]), label)
        if np.isfinite(cube[key]):
            raise ValueError("Duplicate score row")
        cube[key] = float(row["mixture_llr"])
    if not np.isfinite(cube).all():
        raise ValueError("Incomplete q × step × identity × sequence × label grid")
    return qs, steps, identities, cube


def auc_of_pairs(pairs):
    flat = pairs.reshape(-1, 2)
    labels = np.tile((0, 1), len(flat))
    return auc(flat.ravel(), labels)


def summarize(path, repetitions=2000, seed=20261013):
    qs, steps, identities, cube = load_cube(path)
    n_identities, n_sequences = cube.shape[2:4]
    observed = np.empty((len(qs), len(steps)))
    samples = np.empty((repetitions, len(qs), len(steps)))
    for qi in range(len(qs)):
        for ti in range(len(steps)):
            observed[qi, ti] = auc_of_pairs(cube[qi, ti])
    rng = np.random.default_rng(seed)
    for repetition in range(repetitions):
        selected_identities = rng.integers(n_identities, size=n_identities)
        selected_sequences = rng.integers(n_sequences, size=(n_identities, n_sequences))
        for qi in range(len(qs)):
            for ti in range(len(steps)):
                pairs = cube[qi, ti, selected_identities[:, None], selected_sequences]
                samples[repetition, qi, ti] = auc_of_pairs(pairs)
    contrasts = []
    def add(name, point, draws):
        low, high = np.quantile(draws, [.025, .975])
        contrasts.append(dict(contrast=name, estimate=float(point), ci_low=float(low),
                              ci_high=float(high), identities=n_identities,
                              sequences_per_identity=n_sequences,
                              bootstrap_repetitions=repetitions, bootstrap_seed=seed))
    for qi, q in enumerate(qs):
        add(f"q={q:g}: T={steps[-1]} minus T={steps[0]}",
            observed[qi, -1] - observed[qi, 0],
            samples[:, qi, -1] - samples[:, qi, 0])
    for qi in range(len(qs) - 1):
        add(f"T={steps[-1]}: q={qs[qi]:g} minus q={qs[qi + 1]:g}",
            observed[qi, -1] - observed[qi + 1, -1],
            samples[:, qi, -1] - samples[:, qi + 1, -1])
    return contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    rows = summarize(args.scores, args.bootstrap, args.seed)
    with Path(args.output).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(f"{row['contrast']}: {row['estimate']:.3f} "
              f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
