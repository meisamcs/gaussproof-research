"""Record-level LiRA/RMIA positive control on real EMNIST without DP.

One record is selected from each writer. Disjoint, digit-balanced writers form
target members, target nonmembers, population, independent control training,
and eight OUT reference training sets. No role IDs or record scores are saved.
The optional Pierre-Joly source check executes its attack classes on these
trained models and compares their scores to the local implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import zlib
from pathlib import Path

import msgpack
import numpy as np
import torch
from scipy.stats import norm, rankdata

from endpoint_baselines import (offline_rmia_record_scores,
                                pierre_offline_lira_cdf)
from run_pilot import array_extension


class RawClientData:
    """Read full-resolution EMNIST writers without retaining raw images."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.ids = [row[0] for row in self.connection.execute(
            "SELECT client_id FROM federated_data ORDER BY client_id")]

    def get(self, writer):
        blob, size = self.connection.execute(
            "SELECT data,num_examples FROM federated_data WHERE client_id=?",
            (writer,),
        ).fetchone()
        item = msgpack.unpackb(zlib.decompress(blob), raw=False,
                               ext_hook=array_extension)
        # EMNIST stores white background near one; foreground intensity is
        # easier to optimize with and keeps most input coordinates at zero.
        x = 1. - np.asarray(item["pixels"], np.float32).reshape(size, -1)
        y = np.asarray(item["label"], np.int64)
        if len(y) != size or x.shape[1] != 784:
            raise AssertionError("Unexpected EMNIST writer shape")
        return x, y


def balanced_roles(data: RawClientData, *, per_digit: int, population_per_digit: int,
                   references: int, seed: int):
    """Partition writers before exposing their single selected record."""
    rng = np.random.default_rng(seed)
    by_digit = {digit: [] for digit in range(10)}
    for writer in data.ids:
        x, y = data.get(writer)
        # The first record of a writer is label-skewed in this SQLite file.
        # Draw a digit and one matching record from the full writer partition.
        digit = int(rng.choice(np.unique(y)))
        index = int(rng.choice(np.flatnonzero(y == digit)))
        by_digit[digit].append((writer, x[index].copy(), digit))
    counts = {"member": per_digit, "nonmember": per_digit,
              "population": population_per_digit,
              "independent_train": per_digit}
    counts.update({f"reference_{i}": per_digit for i in range(references)})
    names = list(counts)
    roles = {name: [] for name in names}
    for digit, items in by_digit.items():
        if len(items) < sum(counts.values()):
            raise ValueError(f"Only {len(items)} independent writers for digit {digit}")
        shuffled = rng.permutation(len(items))
        cursor = 0
        for name in names:
            take = counts[name]
            roles[name].extend(items[j] for j in shuffled[cursor:cursor + take])
            cursor += take
    writers = [item[0] for role in roles.values() for item in role]
    if len(set(writers)) != len(writers):
        raise AssertionError("Writer role overlap")
    return roles


def role_arrays(role):
    x = np.stack([item[1] for item in role]).astype(np.float32)
    y = np.array([item[2] for item in role], np.int64)
    return x, y


def train_model(role, *, seed: int, epochs: int, hidden: int,
                learning_rate: float):
    torch.manual_seed(seed)
    x, y = role_arrays(role)
    features = torch.from_numpy(x)
    labels = torch.from_numpy(y)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(features), labels).backward()
        optimizer.step()
    model.eval()
    return model


def predictions(model, role):
    x, y = role_arrays(role)
    with torch.no_grad():
        logits = model(torch.from_numpy(x)).numpy().astype(np.float64)
    row = np.arange(len(y))
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    true = exp[row, y].copy()
    exp[row, y] = 0
    wrong = exp.sum(axis=1)
    probability = true / (true + wrong)
    # LiRA's score.py uses log(true-class mass) - log(other-class mass).
    # Computing this directly avoids float32 softmax rounding high confidence
    # probabilities to exactly one before the logit transformation.
    scaled = np.log(true) - np.log(wrong)
    raw = logits[row, y]
    correct = float(np.mean(logits.argmax(axis=1) == y))
    return probability, raw, correct, scaled


def tie_auc(positive, negative) -> float:
    positive = np.asarray(positive, np.float64)
    negative = np.asarray(negative, np.float64)
    if not len(positive) or not len(negative):
        raise ValueError("Both classes need examples")
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("Nonfinite scores")
    ranks = rankdata(np.r_[negative, positive], method="average")
    n = len(negative)
    m = len(positive)
    return float((ranks[n:].sum() - m * (m + 1) / 2) / (m * n))


def bootstrap_auc(positive, negative, *, seed: int, repetitions: int):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repetitions):
        p = positive[rng.integers(len(positive), size=len(positive))]
        n = negative[rng.integers(len(negative), size=len(negative))]
        scores.append(tie_auc(p, n))
    return np.quantile(scores, [.025, .975]).tolist()


def score_all(candidate_prob, candidate_raw, candidate_scaled, population_prob,
              reference_prob, reference_raw, reference_scaled,
              reference_population, reference_population_scaled):
    # All methods receive the same target, candidate, and disjoint OUT data.
    fixed_std = float(np.std(np.concatenate(
        (reference_scaled, reference_population_scaled), axis=1)))
    per_record_std = reference_scaled.std(axis=0)
    per_record_mean = np.median(reference_scaled, axis=0)
    per_record_std = np.maximum(per_record_std, 1e-12)
    # Original LiRA offline variable-variance score: negative OUT log-density.
    # Its plotting sweep negates logpdf, so higher means more membership.
    lira_variable = -norm.logpdf(candidate_scaled, per_record_mean,
                                per_record_std)
    return {
        "loss": np.log(np.maximum(candidate_prob, 1e-15)),
        "lira_offline_fixed": .5 * ((candidate_scaled - per_record_mean) /
                                      fixed_std) ** 2,
        "lira_offline_variable": lira_variable,
        "rmia_offline_a05_g1": offline_rmia_record_scores(
            candidate_prob, reference_prob, population_prob,
            reference_population, a=.5, gamma=1.),
        "rmia_offline_a1_g2": offline_rmia_record_scores(
            candidate_prob, reference_prob, population_prob,
            reference_population, a=1., gamma=2.),
        "pierre_lira_cdf": pierre_offline_lira_cdf(
            candidate_raw, reference_raw),
        "pierre_rmia_a05_g1": offline_rmia_record_scores(
            candidate_prob, reference_prob, population_prob,
            reference_population, a=.5, gamma=1.,
            population_correction=False),
    }


def verify_external(source: Path, target, references, candidate, population,
                    scores):
    """Run the pinned Pierre-Joly classes on actual trained EMNIST models."""
    from verify_pierre_joly import EXPECTED_HASHES, ToyData
    for name, expected in EXPECTED_HASHES.items():
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Unexpected pinned source: {name}")
    import sys
    sys.path.insert(0, str(source))
    import attacks.offline_lira as lira
    import attacks.offline_rmia as rmia
    from torch.utils.data import DataLoader

    def loader(data, batch_size, shuffle):
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=lambda batch: (
                              torch.stack([item[1] for item in batch]),
                              torch.tensor([item[2] for item in batch])))

    x, y = role_arrays(candidate)
    z, zy = role_arrays(population)
    candidate_data = ToyData(x, y)
    population_data = ToyData(z, zy)
    for module in (lira, rmia):
        module.get_out_dataset = lambda _: population_data
        module.get_off_shadow_models = lambda _, n: tuple(references[:n])
        module.get_data_loader = loader
        module.get_device = lambda: torch.device("cpu")
    rmia.transform_test = lambda: None
    external_rmia = rmia.OfflineRMIA(
        num_shadow_models=len(references), batch_size=32,
        reference_data="nodp_emnist", a_param=.5, gamma=1.).run_attack(
            target, candidate_data)
    external_lira = lira.OfflineLiRA(
        num_shadow_models=len(references), batch_size=32,
        reference_data="nodp_emnist").run_attack(target, candidate_data)
    np.testing.assert_allclose(external_rmia, scores["pierre_rmia_a05_g1"],
                               atol=1e-6)
    np.testing.assert_allclose(external_lira, scores["pierre_lira_cdf"],
                               atol=1e-5)


def run(args):
    torch.set_num_threads(1)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.sqlite)
    data = RawClientData(data_path)
    roles = balanced_roles(data, per_digit=args.per_digit,
                           population_per_digit=args.population_per_digit,
                           references=args.references, seed=args.seed)
    candidate = roles["member"] + roles["nonmember"]
    print("Training non-DP target, independent control, and OUT references...",
          flush=True)
    target = train_model(roles["member"], seed=args.seed + 1,
                         epochs=args.epochs, hidden=args.hidden,
                         learning_rate=args.learning_rate)
    train_accuracy = predictions(target, roles["member"])[2]
    if train_accuracy < .95:
        raise RuntimeError(f"Positive-control target did not fit: {train_accuracy:.3f}")
    control = train_model(roles["independent_train"], seed=args.seed + 2,
                          epochs=args.epochs, hidden=args.hidden,
                          learning_rate=args.learning_rate)
    references = []
    for i in range(args.references):
        references.append(train_model(
            roles[f"reference_{i}"], seed=args.seed + 100 + i,
            epochs=args.epochs, hidden=args.hidden,
            learning_rate=args.learning_rate))
        print(f"Reference {i + 1}/{args.references} trained", flush=True)

    ref_candidate = [predictions(m, candidate) for m in references]
    ref_population = [predictions(m, roles["population"]) for m in references]
    candidate_ref_p = np.stack([r[0] for r in ref_candidate])
    candidate_ref_raw = np.stack([r[1] for r in ref_candidate])
    candidate_ref_scaled = np.stack([r[3] for r in ref_candidate])
    population_ref_p = np.stack([r[0] for r in ref_population])
    population_ref_scaled = np.stack([r[3] for r in ref_population])
    labels = np.r_[np.ones(len(roles["member"]), bool),
                   np.zeros(len(roles["nonmember"]), bool)]
    target_p, target_raw, target_candidate_accuracy, target_scaled = predictions(target, candidate)
    control_p, control_raw, control_candidate_accuracy, control_scaled = predictions(control, candidate)
    target_population_p = predictions(target, roles["population"])[0]
    control_population_p = predictions(control, roles["population"])[0]
    target_scores = score_all(target_p, target_raw, target_scaled,
                              target_population_p, candidate_ref_p,
                              candidate_ref_raw, candidate_ref_scaled,
                              population_ref_p, population_ref_scaled)
    control_scores = score_all(control_p, control_raw, control_scaled,
                               control_population_p, candidate_ref_p,
                               candidate_ref_raw, candidate_ref_scaled,
                               population_ref_p, population_ref_scaled)
    if args.pierre_source:
        verify_external(Path(args.pierre_source), target, references, candidate,
                        roles["population"], target_scores)
        print("Pinned external RMIA/LiRA scores match on real EMNIST", flush=True)

    rows = []
    for name in target_scores:
        for world, scores in (("target", target_scores[name]),
                              ("independent_control", control_scores[name])):
            p = np.asarray(scores[labels])
            n = np.asarray(scores[~labels])
            low, high = bootstrap_auc(p, n, seed=args.seed + len(rows),
                                      repetitions=args.bootstrap)
            rows.append(dict(world=world, method=name, auc=tie_auc(p, n),
                             auc_low=low, auc_high=high,
                             unique_scores=len(np.unique(scores)),
                             members=len(p), nonmembers=len(n)))
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    nonmember_accuracy = predictions(target, roles["nonmember"])[2]
    metadata = dict(
        design="non-DP record-level positive control; one record per writer",
        dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        parameters={k: v for k, v in vars(args).items()
                    if k not in ("sqlite", "output", "pierre_source")},
        role_counts={k: len(v) for k, v in roles.items()},
        roles_disjoint=True, target_train_accuracy=train_accuracy,
        target_nonmember_accuracy=nonmember_accuracy,
        target_candidate_accuracy=target_candidate_accuracy,
        independent_candidate_accuracy=control_candidate_accuracy,
        external_pierre_parity=bool(args.pierre_source),
        caution=("This is a record-level, deliberately overfitted, no-DP "
                 "positive control; it is not a DP-FTRLM result"))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Target accuracy train/nonmember:", train_accuracy,
          nonmember_accuracy, flush=True)
    for row in rows:
        print(row["world"], row["method"], f'AUC={row["auc"]:.3f}',
              f'95% bootstrap [{row["auc_low"]:.3f},{row["auc_high"]:.3f}]',
              flush=True)
    plot(output, rows)


def plot(output: Path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42
    chosen = ["loss", "lira_offline_fixed", "lira_offline_variable",
              "rmia_offline_a05_g1", "rmia_offline_a1_g2",
              "pierre_lira_cdf", "pierre_rmia_a05_g1"]
    labels = ["Loss", "LiRA fixed", "LiRA variable", "RMIA .5/1",
              "RMIA 1/2", "Pierre LiRA CDF", "Pierre RMIA"]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for i, world in enumerate(("target", "independent_control")):
        selected = [next(r for r in rows if r["world"] == world and
                         r["method"] == name) for name in chosen]
        x = np.arange(len(chosen)) + (-.14 if i == 0 else .14)
        y = np.array([r["auc"] for r in selected])
        lo = np.array([r["auc_low"] for r in selected])
        hi = np.array([r["auc_high"] for r in selected])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", capsize=3,
                    color=("#4477AA" if i == 0 else "#CC6677"),
                    label=("Trained target" if i == 0 else "Independent control"))
    ax.axhline(.5, linestyle="--", color=".5", linewidth=1)
    ax.set(xticks=np.arange(len(chosen)), xticklabels=labels,
           ylim=(.3, 1.02), ylabel="Record membership AUC",
           title="Non-DP EMNIST positive control")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"nodp_mia_auc.{extension}", dpi=300)
    plt.close(fig)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pierre-source")
    p.add_argument("--per-digit", type=int, default=12)
    p.add_argument("--population-per-digit", type=int, default=24)
    p.add_argument("--references", type=int, default=8)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--bootstrap", type=int, default=400)
    p.add_argument("--seed", type=int, default=20260929)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
