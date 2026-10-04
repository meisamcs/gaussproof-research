"""Aggregate-only check of EMNIST writer-prefix label skew."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_pilot import ClientData, write_csv


def run(sqlite: Path, output: Path, examples: int, sample_seed: int,
        utility_seed: int, reserve: int):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    prefix = ClientData(sqlite, examples, selection="first")
    uniform = ClientData(sqlite, examples, selection="uniform",
                         sample_seed=sample_seed)
    all_ids = np.asarray(prefix.ids, dtype=object)
    reserved = set(all_ids[np.random.default_rng(utility_seed)
                           .permutation(len(all_ids))[:reserve]])
    eligible = [client_id for client_id, size in prefix.connection.execute(
        "SELECT client_id,num_examples FROM federated_data ORDER BY client_id")
        if size >= examples and client_id not in reserved]
    counts = {}
    for name, data in (("first", prefix), ("uniform", uniform)):
        count = np.zeros(10, dtype=np.int64)
        for client_id in eligible:
            _, labels = data.get(client_id)
            count += np.bincount(labels, minlength=10)
        counts[name] = count
    rows = [dict(selection=name, digit=digit, count=int(count),
                 fraction=float(count / values.sum()),
                 writers=len(eligible), examples_per_writer=examples)
            for name, values in counts.items()
            for digit, count in enumerate(values)]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--sample-seed", type=int, default=20261011)
    parser.add_argument("--utility-seed", type=int, default=20260929)
    parser.add_argument("--reserve", type=int, default=1124)
    arguments = parser.parse_args()
    run(arguments.sqlite, arguments.output, arguments.examples,
        arguments.sample_seed, arguments.utility_seed, arguments.reserve)
