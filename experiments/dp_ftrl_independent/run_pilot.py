"""Independent, source-aligned federated DP-FTRLM checkpoint pilot.

This is a NumPy reproduction of the *mechanism* in google-research/federated
at a5af8c4433c9ee2d2b1565f1bcc68c89e2000a6b, not a TFF execution.
It uses real writer-partitioned federated EMNIST, a one-step softmax client,
the official equal-weight/clip/noise scaling, and the TF Privacy 0.6 efficient
tree recurrence. Only aggregate outputs are written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
import zlib
from pathlib import Path

import msgpack
import numpy as np

UPSTREAM_SHA = "a5af8c4433c9ee2d2b1565f1bcc68c89e2000a6b"
DATA_URL = (
    "https://storage.googleapis.com/gresearch/fedjax/emnist/"
    "federated_emnist_digitsonly_train.sqlite"
)


def array_extension(code: int, data: bytes):
    if code != 1:
        raise ValueError(f"Unexpected Msgpack extension {code}")
    shape, dtype, raw = msgpack.unpackb(data, raw=False)
    return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)


class ClientData:
    def __init__(self, path: Path, examples: int, selection: str = "first",
                 sample_seed: int = 0, pixel_transform: str = "raw"):
        if (selection not in {"first", "uniform"} or examples < 1
                or pixel_transform not in {"raw", "ink"}):
            raise ValueError("Invalid per-writer example selection")
        self.connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.ids = [row[0] for row in self.connection.execute(
            "SELECT client_id FROM federated_data ORDER BY client_id")]
        self.examples = examples
        self.selection = selection
        self.sample_seed = sample_seed
        self.pixel_transform = pixel_transform
        self.cache = {}

    def get(self, client_id: bytes):
        if client_id not in self.cache:
            blob, size = self.connection.execute(
                "SELECT data,num_examples FROM federated_data WHERE client_id=?",
                (client_id,),
            ).fetchone()
            item = msgpack.unpackb(zlib.decompress(blob), raw=False,
                                   ext_hook=array_extension)
            if len(item["label"]) != size:
                raise AssertionError("Client length mismatch")
            # The archive's original order is strongly label-skewed near the
            # front. Uniform mode samples each writer without replacement,
            # using a stable writer-derived seed independent of access order.
            if self.selection == "uniform":
                writer_bytes = (client_id if isinstance(client_id, bytes)
                                else str(client_id).encode())
                digest = hashlib.sha256(
                    str(self.sample_seed).encode() + b":" + writer_bytes).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                indices = rng.choice(size, min(size, self.examples),
                                     replace=False)
            else:
                indices = np.arange(min(size, self.examples))
            image = np.asarray(item["pixels"][indices], np.float32)
            labels = np.asarray(item["label"][indices], np.int64)
            image = image.reshape(-1, 7, 4, 7, 4).mean(axis=(2, 4))
            if self.pixel_transform == "ink":
                image = 1.0 - image
            x = np.concatenate((image.reshape(len(image), -1),
                                np.ones((len(image), 1), np.float32)), axis=1)
            if len(x) == 0 or not np.isfinite(x).all():
                raise AssertionError("Empty or nonfinite client")
            self.cache[client_id] = (x, labels)
        return self.cache[client_id]


class EfficientTree:
    """TF Privacy 0.6 EfficientTreeAggregator, scalar or vector node values."""

    def __init__(self, draw):
        self.draw = draw
        self.levels = [0]
        self.values = [draw()]

    def next(self):
        result = sum(v / (2.0 - 2.0 ** (-level))
                     for level, v in zip(self.levels, self.values))
        new_value = self.draw()
        idx = 0
        while idx < len(self.levels) and self.levels[idx] == idx:
            new_value = .5 * (self.values[idx] + new_value) + self.draw()
            idx += 1
        self.levels = [idx] + self.levels[idx:]
        self.values = [new_value] + self.values[idx:]
        return result


def tree_matrix(rounds: int) -> np.ndarray:
    # The TF Privacy API prepares the next leaf after emitting each prefix,
    # so its final (unused) update also consumes Gaussian draws.
    width = 3 * rounds + 2
    counter = 0

    def basis():
        nonlocal counter
        if counter >= width:
            raise AssertionError("Too many tree nodes")
        x = np.zeros(width, np.float64)
        x[counter] = 1
        counter += 1
        return x

    tree = EfficientTree(basis)
    result = np.stack([tree.next() for _ in range(rounds)])[:, :counter]
    if not rounds <= counter <= width:
        raise AssertionError(f"Unexpected efficient-tree nodes: {counter}")
    np.linalg.cholesky(result @ result.T)
    return result


def clipped_client_delta(weights, client, client_lr: float, clip: float):
    x, labels = client
    logits = x @ weights.T
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(labels)), labels] -= 1
    delta = -client_lr * (probs.T @ x) / len(labels)
    norm = np.linalg.norm(delta)
    return delta * min(1.0, clip / max(norm, 1e-12))


def candidate_loss(weights, client):
    x, labels = client
    logits = x @ weights.T
    scores = logits[np.arange(len(labels)), labels]
    maximum = logits.max(axis=1)
    logsum = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
    return float(np.mean(scores - logsum))


def server_step(initial, previous_velocity, cumulative_grad, tree_noise,
                learning_rate, momentum):
    """Official DPFTRLMServerOptimizer recurrence, with its noise sign."""
    noised_sum = cumulative_grad - tree_noise
    velocity = momentum * previous_velocity + noised_sum
    return initial - learning_rate * velocity, velocity


def recover_noised_sum(initial, current, previous, learning_rate, momentum):
    velocity = (initial - current) / learning_rate
    previous_velocity = (initial - previous) / learning_rate
    return velocity - momentum * previous_velocity


def accuracy(weights, clients):
    correct = total = 0
    for x, y in clients:
        correct += int(((x @ weights.T).argmax(axis=1) == y).sum())
        total += len(y)
    return correct / total


def sparse_coordinate(bank: np.ndarray, observation: np.ndarray,
                      penalty: float, iterations: int) -> float:
    """The repository's bounded nonnegative sparse decoder at one round."""
    gram = bank @ bank.T
    lipschitz = max(float(np.linalg.eigvalsh(gram)[-1]), 1e-12)
    b = bank @ observation
    x = np.zeros(len(bank))
    v = x.copy()
    momentum = 1.0
    for _ in range(iterations):
        nxt = np.clip(v - (gram @ v - b + penalty) / lipschitz, 0, 1)
        newer = (1 + math.sqrt(1 + 4 * momentum * momentum)) / 2
        v = nxt + (momentum - 1) / newer * (nxt - x)
        x, momentum = nxt, newer
    return float(x[0])


def simulate(data, candidate_id, background_ids, public_ids, validation,
             *, member, rounds, position, clients_per_round, sigma, clip,
             client_lr, server_lr, momentum, seed, matrix, penalty,
             sparse_iterations, capture_final=False):
    dim = 50 * 10
    weights = np.zeros((10, 50), np.float64)
    initial = weights.copy()
    velocity = np.zeros_like(weights)
    cumulative_grad = np.zeros_like(weights)
    cumulative_background = np.zeros_like(weights)
    cumulative_public = np.zeros_like(weights)
    previous_observed = np.zeros_like(weights)
    previous_noise = np.zeros_like(weights)
    noise_rng = np.random.default_rng(seed)
    node_std = sigma * clip / clients_per_round
    tree = EfficientTree(lambda: noise_rng.normal(0, node_std, dim)
                         .reshape(weights.shape))
    candidate = data.get(candidate_id)
    public = [data.get(i) for i in public_ids]
    baseline = [data.get(i) for i in validation]
    projection = []
    oracle_prefix = []
    rero = []
    sparse = []
    late = []
    noise_prefix = []
    candidate_norms = []
    candidate_fingerprints = []
    public_prefixes = []
    insertion_count = 0
    for t in range(rounds):
        candidate_grad = -clipped_client_delta(
            weights, candidate, client_lr, clip) / clients_per_round
        candidate_fingerprints.append(candidate_grad.reshape(-1).copy())
        public_grads = np.stack([
            -clipped_client_delta(weights, item, client_lr, clip)
            / clients_per_round for item in public
        ])
        known = np.zeros_like(weights)
        for client_id in background_ids[t]:
            known -= clipped_client_delta(weights, data.get(client_id),
                                          client_lr, clip) / clients_per_round
        if member and t == position:
            insertion_count += 1
            grad = known + candidate_grad
        else:
            grad = known
        cumulative_grad += grad
        cumulative_background += known
        sampled_noise = tree.next()
        old_weights = weights
        weights, velocity = server_step(initial, velocity, cumulative_grad,
                                        sampled_noise, server_lr, momentum)
        # The attacker derives this from checkpoints and public hyperparameters.
        recovered_prefix = recover_noised_sum(
            initial, weights, old_weights, server_lr, momentum)
        if not np.allclose(recovered_prefix, cumulative_grad - sampled_noise,
                           atol=1e-10):
            raise AssertionError("Momentum inversion failed")
        late.append(weights.copy())
        release = recovered_prefix - previous_observed
        previous_observed = recovered_prefix
        noise_prefix.append(sampled_noise.copy())  # validation only
        if not np.allclose(release, grad - (sampled_noise - previous_noise)):
            raise AssertionError("Tree release mismatch")
        previous_noise = sampled_noise
        # Public-only background estimate: expected seven ordinary clients.
        expected = (clients_per_round - 1) * public_grads.mean(axis=0)
        cumulative_public += expected
        residual = release - expected
        public_prefixes.append((recovered_prefix - cumulative_public)
                               .reshape(-1).copy())
        norm2 = float(np.sum(candidate_grad ** 2))
        candidate_norms.append(norm2)
        score = float(np.sum(candidate_grad * residual) /
                      math.sqrt(max(norm2, 1e-15)))
        rero.append(score)
        centered_bank = np.concatenate((candidate_grad[None, ...],
                                        public_grads), axis=0).reshape(-1, dim)
        sparse.append(sparse_coordinate(centered_bank, residual.reshape(-1),
                                        penalty, sparse_iterations))
        projection.append(float(np.sum(candidate_grad * residual)))
        oracle_prefix.append((recovered_prefix - cumulative_background).copy())
    if insertion_count != int(member):
        raise AssertionError("Candidate inserted wrong number of times")
    shifted = np.zeros(rounds)
    shifted[position:] = 1
    covariance = matrix @ matrix.T
    shifts = np.tri(rounds, dtype=np.float64)
    gls_weights = np.linalg.solve(covariance, shifts)
    public_rows = np.stack(public_prefixes)
    fingerprints = np.stack(candidate_fingerprints)
    projected_public = np.einsum("pd,dp->p", fingerprints,
                                 public_rows.T @ gls_weights)
    shift_energy = np.einsum("tp,tp->p", shifts, gls_weights)
    public_gls = ((projected_public - .5 * shift_energy
                   * np.sum(fingerprints * fingerprints, axis=1))
                  / (node_std * node_std))
    # Informed, known-round Gaussian reference; its background is private.
    h = -clipped_client_delta(
        late[position - 1] if position else initial, candidate,
        client_lr, clip) / clients_per_round
    hflat = h.reshape(-1)
    projected = np.asarray([np.sum(hflat * r.reshape(-1)) for r in oracle_prefix])
    solve = np.linalg.solve(covariance, shifted)
    informed = float((solve @ projected - .5 * (shifted @ solve)
                      * (hflat @ hflat)) / (node_std * node_std))
    result = {
        "rero_max": float(max(rero)),
        "rero_mean": float(np.mean(rero)),
        "gaussproof_max": float(max(sparse)),
        "gaussproof_mean": float(np.mean(sparse)),
        "public_gls_max": float(np.max(public_gls)),
        "endpoint_loss": candidate_loss(weights, candidate),
        "informed_llr": informed,
        "accuracy": accuracy(weights, baseline),
        "sparse_saturation": float(np.mean(np.asarray(sparse) >= .999)),
        "noise_check": float(np.linalg.norm(noise_prefix[-1])),
    }
    if capture_final:
        # In-memory access for matched endpoint attacks. Never written by
        # run_pilot, which records only aggregate metrics.
        result["final_weights"] = weights.copy()
    return result


def auc(pos, neg):
    p, n = np.asarray(pos), np.asarray(neg)
    return float(((p[:, None] > n).sum() + .5 * (p[:, None] == n).sum())
                 / (len(p) * len(n)))


def cross_identity_auc(records, key, sampled_ids=None):
    """AUC on distinct client identities; avoids paired self-match inflation."""
    if sampled_ids is None:
        sampled_ids = np.arange(len(records))
    sampled_ids = np.asarray(sampled_ids)
    p = np.asarray([records[i]["positive"][key] for i in sampled_ids])
    n = np.asarray([records[i]["negative"][key] for i in sampled_ids])
    distinct = sampled_ids[:, None] != sampled_ids[None, :]
    if not distinct.any():
        return .5
    wins = (p[:, None] > n) + .5 * (p[:, None] == n)
    return float(wins[distinct].mean())


def identity_interval(records, key, repeats, seed):
    rng = np.random.default_rng(seed)
    ids = np.arange(len(records))
    values = []
    for _ in range(repeats):
        sample = rng.choice(ids, len(ids), replace=True)
        values.append(cross_identity_auc(records, key, sample))
    return np.quantile(values, [.025, .975])


def paired_difference_interval(records, left, right, repeats, seed):
    rng = np.random.default_rng(seed)
    ids = np.arange(len(records))
    values = []
    for _ in range(repeats):
        sample = rng.choice(ids, len(ids), replace=True)
        values.append(cross_identity_auc(records, left, sample)
                      - cross_identity_auc(records, right, sample))
    return np.quantile(values, [.025, .975])


def advantage(pos, neg):
    p, n = np.asarray(pos), np.asarray(neg)
    thresholds = np.unique(np.r_[p, n])
    return float(max(((p >= z).mean() - (n >= z).mean())
                     for z in thresholds))


def calibrated_rates(calibration, heldout, method, cap):
    negatives = sorted((r["negative"][method] for r in calibration),
                       reverse=True)
    if len(negatives) * cap < 1:
        # The calibration set cannot resolve a nonzero false-positive count.
        return math.nan, math.nan
    allowed = math.floor(cap * len(negatives) + 1e-12)
    if allowed == len(negatives):
        threshold = -math.inf
    else:
        threshold = np.nextafter(negatives[allowed], math.inf)
    pos = np.asarray([r["positive"][method] for r in heldout])
    neg = np.asarray([r["negative"][method] for r in heldout])
    return float((pos >= threshold).mean()), float((neg >= threshold).mean())


def epsilon_upper(rounds, sigma, delta):
    rho = math.ceil(math.log2(rounds + 1)) / (2 * sigma * sigma)
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.sqlite)
    if not source.is_file():
        raise FileNotFoundError(f"Download {DATA_URL} to {source}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = ClientData(source, args.examples)
    rng = np.random.default_rng(args.seed)
    ids = np.asarray(data.ids, dtype=object)[rng.permutation(len(data.ids))]
    count = args.calibration_identities + args.holdout_identities
    if args.public < 1 or args.public > args.public_pool:
        raise ValueError("Require 1 <= public <= public-pool")
    candidates = ids[:count]
    public_ids = ids[count:count + args.public]
    validation = ids[count + args.public_pool:count + args.public_pool + 40]
    backgrounds = ids[count + args.public_pool + 40:]
    if len(backgrounds) < args.rounds * (args.clients_per_round - 1):
        raise ValueError("Insufficient distinct background clients")
    if args.position < 0 or args.position >= args.rounds:
        raise ValueError("Invalid insertion round")
    matrix = tree_matrix(args.rounds)
    methods = ("rero_max", "rero_mean", "gaussproof_max",
               "gaussproof_mean", "public_gls_max", "endpoint_loss",
               "informed_llr")
    table = []
    comparisons = []
    for sigma in args.sigma:
        records = []
        for idx, candidate in enumerate(candidates):
            run_rng = np.random.default_rng(args.seed + 7919 * (idx + 1))
            bg = run_rng.choice(backgrounds,
                                args.rounds * (args.clients_per_round - 1),
                                replace=False).reshape(args.rounds, -1)
            paired_seed = args.seed + 100003 * (idx + 1)
            kw = dict(rounds=args.rounds, position=args.position,
                      clients_per_round=args.clients_per_round, sigma=sigma,
                      clip=args.clip, client_lr=args.client_lr,
                      server_lr=args.server_lr, momentum=args.momentum,
                      seed=paired_seed, matrix=matrix,
                      penalty=args.penalty, sparse_iterations=args.iterations)
            negative = simulate(data, candidate, bg, public_ids, validation,
                                member=False, **kw)
            positive = simulate(data, candidate, bg, public_ids, validation,
                                member=True, **kw)
            if abs(positive["noise_check"] - negative["noise_check"]) > 1e-10:
                raise AssertionError("Paired worlds used different tree noise")
            records.append(dict(positive=positive, negative=negative))
            print(f"sigma={sigma} client={idx + 1}/{count}", flush=True)
        calibration = records[:args.calibration_identities]
        heldout = records[args.calibration_identities:]
        for method in methods:
            p = [r["positive"][method] for r in heldout]
            n = [r["negative"][method] for r in heldout]
            ci = identity_interval(heldout, method, args.bootstrap,
                                   args.seed + 17)
            tpr1, fpr1 = calibrated_rates(calibration, heldout, method, .01)
            tpr5, fpr5 = calibrated_rates(calibration, heldout, method, .05)
            table.append(dict(
                sigma=sigma, method=method,
                auc=cross_identity_auc(heldout, method),
                pooled_auc=auc(p, n),
                auc_low=float(ci[0]), auc_high=float(ci[1]),
                advantage=advantage(p, n),
                calibrated_tpr_1pct=tpr1, calibrated_fpr_1pct=fpr1,
                calibrated_tpr_5pct=tpr5, calibrated_fpr_5pct=fpr5,
                mean_accuracy_present=float(np.mean([
                    r["positive"]["accuracy"] for r in heldout])),
                mean_accuracy_absent=float(np.mean([
                    r["negative"]["accuracy"] for r in heldout])),
                mean_sparse_saturation=float(np.mean([
                    r["negative"]["sparse_saturation"] for r in heldout])),
                epsilon_upper=epsilon_upper(args.rounds, sigma, args.delta),
                delta=args.delta,
                calibration_clients=len(calibration),
                heldout_clients=len(heldout),
                participation_round=args.position + 1,
            ))
        for left, baseline in (("gaussproof_max", "rero_max"),
                               ("gaussproof_max", "endpoint_loss"),
                               ("gaussproof_mean", "rero_mean"),
                               ("gaussproof_mean", "endpoint_loss")):
            pos_left = [r["positive"][left] for r in heldout]
            neg_left = [r["negative"][left] for r in heldout]
            pos_right = [r["positive"][baseline] for r in heldout]
            neg_right = [r["negative"][baseline] for r in heldout]
            interval = paired_difference_interval(
                heldout, left, baseline, args.bootstrap, args.seed + 29)
            comparisons.append(dict(
                sigma=sigma, comparison=f"{left}_minus_{baseline}",
                auc_difference=(cross_identity_auc(heldout, left)
                                - cross_identity_auc(heldout, baseline)),
                low=float(interval[0]), high=float(interval[1]),
                heldout_clients=len(heldout),
            ))
    write_csv(output / "summary.csv", table)
    write_csv(output / "paired_comparisons.csv", comparisons)
    metadata = dict(
        status="source-aligned NumPy/PyTorch-free pilot; official TFF did not run",
        upstream_sha=UPSTREAM_SHA,
        tf_privacy_version="0.6.0 algorithm transcribed",
        federated_emnist_sha256=digest,
        dataset_url=DATA_URL,
        parameters=vars(args),
        note=("One-step 7x7-pooled softmax clients, not the upstream EMNIST CNN. "
              "Checkpoint attack has server-log access. Background clients "
              "are hidden from the public attacks. The informed LLR uses their "
              "updates and the insertion round. Results are an exploratory pilot."),
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    make_plot(output, table)


def make_plot(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["rero_max", "gaussproof_max", "gaussproof_mean",
               "public_gls_max", "endpoint_loss", "informed_llr"]
    names = ["RERO max", "Sparse max", "Sparse mean", "Public GLS",
             "Final model", "Informed LLR"]
    colors = ["#4477AA", "#EE6677", "#CCBB44", "#66CCEE",
              "#228833", "#AA3377"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
    for method, name, color in zip(methods, names, colors):
        subset = sorted((r for r in rows if r["method"] == method),
                        key=lambda r: r["sigma"])
        ax.plot([r["sigma"] for r in subset], [r["auc"] for r in subset],
                marker="o", color=color, label=name)
    ax.axhline(.5, color="0.5", linestyle="--", linewidth=1)
    ax.set(xlabel="Tree noise multiplier σ", ylabel="Cross-identity AUC",
           ylim=(0, 1), title="One-time client membership (exploratory pilot)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"auc_vs_noise.{ext}", dpi=300)
    plt.close(fig)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rounds", type=int, default=128)
    p.add_argument("--position", type=int, default=64,
                   help="Zero-based one-time insertion round")
    p.add_argument("--clients-per-round", type=int, default=8)
    p.add_argument("--calibration-identities", type=int, default=10)
    p.add_argument("--holdout-identities", type=int, default=10)
    p.add_argument("--public", type=int, default=12)
    p.add_argument("--public-pool", type=int, default=64,
                   help="Reserve this many disjoint public writers for paired bank-size ablations")
    p.add_argument("--examples", type=int, default=16)
    p.add_argument("--sigma", nargs="+", type=float, default=[1.0, 4.0, 8.0])
    p.add_argument("--clip", type=float, default=.25)
    p.add_argument("--client-lr", type=float, default=.2)
    p.add_argument("--server-lr", type=float, default=.005)
    p.add_argument("--momentum", type=float, default=.9)
    p.add_argument("--penalty", type=float, default=.0001)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--bootstrap", type=int, default=400)
    p.add_argument("--seed", type=int, default=20260929)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
