"""One-run white-box DP-SGD canary audit with identical access for all scores.

The canary inclusion bits are independent fair coins chosen before training.
Conditional on an included bit, a canary joins each step with probability q.
The ordinary batch is sampled independently of those bits, either at fixed
size or by uniform Poisson sampling, and canary gradients are added to its
sum. Dividing by the public expected ordinary batch size gives add/remove
sensitivity C/B even with several active canaries.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import binom

from .canary_audit import update
from .canary_gallery import stratified_gallery
from .data import digest, load_mnist
from .metrics import auc
from .models import clip_gradients, initialize, per_record_gradients
from .multi_contributor import bounded_sparse, write_csv


SCORES = ("nasr_dot", "centered_dot", "gaussian_mixture", "sparse",
          "joint_trajectory")


def audit_guesses(scores, bits, positive_guesses, negative_guesses):
    """Fixed extreme-score guesses; no outcome-dependent choice of k."""
    if positive_guesses + negative_guesses > len(bits):
        raise ValueError("Too many audit guesses")
    order = np.argsort(np.asarray(scores), kind="stable")
    guesses = np.zeros(len(bits), dtype=int)
    guesses[order[:negative_guesses]] = -1
    guesses[order[-positive_guesses:]] = 1
    correct = int(np.count_nonzero(guesses * np.asarray(bits) == 1))
    return correct, guesses


def conservative_epsilon_lower(correct, guesses, canaries, delta, alpha):
    """Invert Binomial domination with the conservative 2mδ slack.

    This is deliberately weaker than the exact Theorem 5.2 implementation in
    Steinke--Nasr--Jagielski. A zero result means this pilot lacks evidence;
    it is not evidence that the mechanism has zero privacy loss.
    """
    r = int(guesses)
    slack = 2 * canaries * delta
    if not 0 <= correct <= r or not 0 < alpha < 1 or slack >= alpha:
        raise ValueError("Invalid or uninformative audit parameters")
    def pvalue(epsilon):
        return float(binom.sf(correct - 1, r, expit(epsilon)) + slack)
    if pvalue(0.0) >= alpha:
        return 0.0
    lower, upper = 0.0, 1.0
    while pvalue(upper) < alpha and upper < 1024:
        upper *= 2
    for _ in range(60):
        middle = (lower + upper) / 2
        if pvalue(middle) < alpha:
            lower = middle
        else:
            upper = middle
    return lower


def paper_audit_pvalue(correct, guesses, canaries, epsilon, delta):
    """Corollary 5.4 / Appendix D of Steinke--Nasr--Jagielski (2023)."""
    v, r, m = int(correct), int(guesses), int(canaries)
    if not 0 <= v <= r <= m or epsilon < 0 or not 0 <= delta <= 1:
        raise ValueError("Invalid one-run audit parameters")
    q = expit(epsilon)
    beta = float(binom.sf(v - 1, r, q))
    correction = 0.0
    mass = 0.0
    for i in range(1, v + 1):
        mass += float(binom.pmf(v - i, r, q))
        correction = max(correction, mass / i)
    return min(1.0, beta + correction * 2 * m * delta)


def paper_epsilon_lower(correct, guesses, canaries, delta, alpha):
    """Invert the paper's exact ternary-guess audit p-value at fixed delta."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    pvalue = lambda eps: paper_audit_pvalue(
        correct, guesses, canaries, eps, delta)
    if pvalue(0.) >= alpha:
        return 0.
    lower, upper = 0., 1.
    while pvalue(upper) < alpha and upper < 1024:
        upper *= 2
    for _ in range(60):
        middle = (lower + upper) / 2
        if pvalue(middle) < alpha:
            lower = middle
        else:
            upper = middle
    return lower


def score_step(release, fingerprints, background, config):
    """Nasr alignment and three fingerprint patches on exactly one release."""
    batch, clip, sigma, q = (float(config[key]) for key in
                             ("batch_size", "clip", "sigma", "q"))
    bank = fingerprints / batch
    residual = release - background
    dot = bank @ residual
    gram = bank @ bank.T
    variance = (sigma * clip / batch) ** 2
    llr = (dot - .5 * np.diag(gram)) / variance
    penalty_scale = np.sqrt(variance) * float(np.median(np.sqrt(np.diag(gram))))
    sparse = bounded_sparse(gram, dot, config["sparse_penalty_factor"] * penalty_scale)
    return dict(nasr_dot=fingerprints @ release,
                centered_dot=fingerprints @ residual,
                gaussian_mixture=np.logaddexp(np.log1p(-q), np.log(q) + llr),
                sparse=sparse)


def joint_trajectory_score(grams, projections, variance, q, iterations=8,
                           damping=.5):
    """Mean-field scores for persistent IN/OUT bits over the whole trajectory.

    The public surrogate is d_t = sum_i A_it h_it/B + Gaussian noise, with
    A_it~Bernoulli(q) only when the dataset-level bit S_i is IN. Other
    canaries are replaced by their current posterior mean in each candidate's
    likelihood. This is an approximate *score*, not an exact privacy bound or
    a claim that the Gaussian surrogate models random ordinary batches.
    """
    grams = np.asarray(grams, dtype=np.float64)
    projections = np.asarray(projections, dtype=np.float64)
    if (grams.ndim != 3 or projections.shape != grams.shape[:2] or
            grams.shape[1] != grams.shape[2] or variance <= 0 or not 0 < q < 1 or
            iterations < 1 or not 0 < damping <= 1):
        raise ValueError("Invalid trajectory posterior inputs")
    diagonal = np.diagonal(grams, axis1=1, axis2=2)
    belief = np.full(grams.shape[1], .5)
    for _ in range(iterations):
        expected = q * belief
        other = np.einsum("tij,j->ti", grams, expected) - diagonal * expected
        llr = (projections - other - .5 * diagonal) / variance
        score = np.logaddexp(np.log1p(-q), np.log(q) + llr).sum(axis=0)
        belief = (1 - damping) * belief + damping * expit(score)
    return score


def run_audit(initial, x, labels, population, canaries, public, query, config, seed):
    model = copy.deepcopy(initial)
    mechanism_rng = np.random.default_rng(seed)
    inclusion_rng = np.random.default_rng(seed + 200000)
    noise_rng = torch.Generator().manual_seed(seed + 100000)
    bits = np.where(mechanism_rng.random(len(canaries)) < .5, 1, -1)
    methods = SCORES if config.get("joint_iterations") else SCORES[:-1]
    total = {name: np.zeros(len(canaries), dtype=np.float64) for name in methods}
    trajectory_grams, trajectory_projections = [], []
    active_total = 0
    probe_ids = np.r_[canaries, public]
    for step in range(config["steps"]):
        probes = clip_gradients(per_record_gradients(
            model, x[probe_ids], labels[probe_ids]), config["clip"]).numpy()
        fingerprints = probes[:len(canaries)]
        background = probes[len(canaries):].mean(axis=0)
        if config.get("sampling_mode", "fixed_ordinary") == "poisson":
            ordinary = population[mechanism_rng.random(len(population)) < config["q"]]
        elif config.get("sampling_mode", "fixed_ordinary") == "fixed_ordinary":
            ordinary = mechanism_rng.choice(
                population, config["batch_size"], replace=False)
        else:
            raise ValueError("Unknown sampling mode")
        active = (bits == 1) & (inclusion_rng.random(len(canaries)) < config["q"])
        active_total += int(active.sum())
        # Included canaries add to the independently sampled ordinary records;
        # no hidden batch-size cap or replacement changes the audit-bit channel.
        ordinary_g = (clip_gradients(per_record_gradients(
            model, x[ordinary], labels[ordinary]), config["clip"]).sum(0)
            if len(ordinary) else torch.zeros(fingerprints.shape[1]))
        canary_sum = torch.from_numpy(fingerprints[active].sum(axis=0))
        sd = config["sigma"] * config["clip"] / config["batch_size"]
        release = ((ordinary_g + canary_sum) / config["batch_size"]
                   + torch.randn(ordinary_g.shape, generator=noise_rng) * sd)
        for name, increment in score_step(
                release.numpy(), fingerprints, background, config).items():
            total[name] += increment
        bank = fingerprints / config["batch_size"]
        trajectory_grams.append(bank @ bank.T)
        trajectory_projections.append(bank @ (release.numpy() - background))
        update(model, release, config["learning_rate"])
    variance = (config["sigma"] * config["clip"] / config["batch_size"]) ** 2
    if "joint_trajectory" in total:
        total["joint_trajectory"] = joint_trajectory_score(
            trajectory_grams, trajectory_projections, variance, config["q"],
            iterations=config["joint_iterations"], damping=config["joint_damping"])
    with torch.no_grad():
        utility = float((model(x[query]).argmax(1) == labels[query]).float().mean())
    results = []
    positives = (bits == 1).astype(int)
    r = config["positive_guesses"] + config["negative_guesses"]
    # Each independent training run is reported, never selected for its best
    # lower bound. Simultaneity is across the scores *within* that run.
    per_method_alpha = config["familywise_alpha"] / len(methods)
    for name, scores in total.items():
        correct, guesses = audit_guesses(scores, bits, config["positive_guesses"],
                                         config["negative_guesses"])
        results.append(dict(sigma=config["sigma"], seed=seed, method=name,
            canaries=len(canaries), included=int(positives.sum()), guesses=r,
            correct=correct, guess_accuracy=correct/r,
            auc=(float(auc(scores, positives)) if 0 < positives.sum() < len(positives)
                 else float("nan")),
            epsilon_lower_paper=paper_epsilon_lower(
                correct, r, len(canaries), config["delta"], per_method_alpha),
            epsilon_lower_conservative=conservative_epsilon_lower(
                correct, r, len(canaries), config["delta"], per_method_alpha),
            familywise_alpha=config["familywise_alpha"],
            per_test_alpha=per_method_alpha, delta=config["delta"],
            mean_active_per_round=active_total/config["steps"], utility=utility))
    return results, bits, total


def main():
    parser = argparse.ArgumentParser()
    for key in ("source", "data", "config", "output", "report"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    source, output, report = Path(args.source), Path(args.output), Path(args.report)
    config = json.loads(Path(args.config).read_text())
    if digest(args.data) != json.loads((source / "completion.json").read_text())["dataset_sha256"]:
        raise ValueError("Dataset differs from public checkpoint source")
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    x, labels = load_mnist(args.data)
    splits = json.loads((source / "splits.json").read_text())
    source_cfg = json.loads((source / "config.json").read_text())
    initial = initialize(source_cfg["seed"])
    initial.load_state_dict(torch.load(source / "public_initial_cnn.pt", weights_only=True))
    initial.eval()
    test_ids = np.asarray(splits["test"])
    query = test_ids[-config["query_size"]:]
    gallery_pool = test_ids[:-config["query_size"]]
    public = np.asarray(splits["background"])[:config["background_size"]]
    population = np.asarray(splits["population"])[
        :config.get("population_size", len(splits["population"]))]
    if config.get("sampling_mode") == "poisson" and not np.isclose(
            config["q"] * len(population), config["batch_size"]):
        raise ValueError("Poisson q must give the declared expected ordinary batch size")
    if (set(gallery_pool) & set(query) or set(gallery_pool) & set(public) or
            set(gallery_pool) & set(population) or set(public) & set(population)):
        raise ValueError("Candidate, public, query, and private roles must be disjoint")
    summaries = []
    began = time.time()
    for repetition in range(config["repetitions"]):
        seed = config["seed"] + repetition
        shuffled_pool = np.random.default_rng(seed + 300000).permutation(gallery_pool)
        canaries = stratified_gallery(shuffled_pool, labels, config["canaries"])
        for sigma in config["sigmas"]:
            condition = dict(config, sigma=sigma)
            file = output / f"sigma_{sigma:g}_rep_{repetition}_scores.npz"
            summary_file = output / f"sigma_{sigma:g}_rep_{repetition}_summary.json"
            if file.exists() != summary_file.exists():
                raise ValueError("Incomplete prior audit result")
            if file.exists():
                rows = json.loads(summary_file.read_text())
                for row in rows:
                    if "epsilon_lower_paper" not in row:
                        row["epsilon_lower_paper"] = paper_epsilon_lower(
                            row["correct"], row["guesses"], row["canaries"],
                            row["delta"], row["per_test_alpha"])
            else:
                rows, bits, scores = run_audit(initial, x, labels, population,
                    canaries, public, query, condition, seed)
                for row in rows:
                    row["repetition"] = repetition
                np.savez_compressed(file, bits=bits, canaries=canaries, **scores)
                summary_file.write_text(json.dumps(rows, indent=2))
            summaries.extend(rows)
            print(f"rep={repetition+1}/{config['repetitions']} sigma={sigma:g}: " + ", ".join(
                f"{r['method']} {r['correct']}/{r['guesses']}" for r in rows), flush=True)
    write_csv(report / "one_run_audit.csv", summaries)
    provenance = dict(dataset_sha256=digest(args.data),
        source_checkpoint_sha256=hashlib.sha256(
            (source / "public_initial_cnn.pt").read_bytes()).hexdigest(),
        config=config, seconds=time.time()-began, torch=torch.__version__,
        audit="independent fair dataset-level canary bits, one evolving CNN run per condition and repetition",
        inference=f"Within-run Bonferroni across {len(SCORES) if config.get('joint_iterations') else len(SCORES)-1} methods; no best-run selection; conservative Binomial tail plus 2m*delta")
    (report / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
