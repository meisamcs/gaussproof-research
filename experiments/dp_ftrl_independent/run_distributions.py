"""Matched DP-FTRLM client-membership audit across EMNIST client distributions.

Endpoint RMIA and LiRA are faithful *offline record-level* scores, averaged
within each candidate client's bag. The resulting client score is an explicit
adaptation. All public and reference data are disjoint from target candidates.
Only aggregate CSVs and plots are saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from calibrated_endpoint import apply_calibration, score_variants
from hybrid_fusion import apply_hybrids
from endpoint_baselines import (offline_lira_fixed_record_scores,
                                offline_rmia_record_scores,
                                pierre_offline_lira_cdf,
                                reference_global_std, train_out_reference,
                                true_label_logits, true_label_probabilities)
from run_pilot import (ClientData, DATA_URL, UPSTREAM_SHA, advantage,
                       calibrated_rates, cross_identity_auc, epsilon_upper,
                       identity_interval, paired_difference_interval, simulate,
                       tree_matrix, write_csv)


class MaterializedData:
    def __init__(self, items):
        self.items = items

    def get(self, client_id):
        return self.items[client_id]


def replacement_schedule(cohort, rounds, clients_per_round, rng):
    """Sample a B-of-N cohort every round, marking the replace-one slot.

    The last member of the cohort is the H0-only client. Under H1, the known
    candidate replaces that client whenever selected. Both worlds therefore
    have exactly B clients every round, with participation q=B/N.
    """
    cohort = np.asarray(cohort, dtype=object)
    if len(cohort) < clients_per_round:
        raise ValueError("The cohort must have at least B clients")
    special = cohort[-1]
    schedule = []
    selected = []
    for t in range(rounds):
        row = cohort[rng.choice(len(cohort), clients_per_round,
                                replace=False)]
        if special in row:
            row = np.concatenate((row[row != special], [special]))
            selected.append(t)
        schedule.append(row)
    return np.stack(schedule), tuple(selected)


def materialize_distribution(source, partitions, mode, examples, seed):
    """Make record-disjoint natural, IID-mixed, or label-sorted client bags.

    Mixing is restricted to each role partition. Thus an example used in a
    candidate client can never enter a reference/background/population client.
    IID and label-sorted conditions are *virtual* clients, not EMNIST writers.
    """
    if mode not in {"natural_writer", "iid_mixed", "label_sorted"}:
        raise ValueError(f"Unknown distribution: {mode}")
    rng = np.random.default_rng(seed)
    items = {}
    purity = {}
    for role, ids in partitions.items():
        ids = list(ids)
        xs = []
        ys = []
        for client_id in ids:
            x, y = source.get(client_id)
            if len(x) < examples:
                raise ValueError(f"Client in {role} has fewer than {examples} records")
            xs.append(np.asarray(x[:examples], np.float64))
            ys.append(np.asarray(y[:examples], np.int64))
        x = np.stack(xs)
        y = np.stack(ys)
        if mode != "natural_writer":
            flat_x = x.reshape(-1, x.shape[-1])
            flat_y = y.reshape(-1)
            if mode == "iid_mixed":
                order = rng.permutation(len(flat_y))
            else:
                order = np.lexsort((rng.random(len(flat_y)), flat_y))
            x = flat_x[order].reshape(len(ids), examples, -1)
            y = flat_y[order].reshape(len(ids), examples)
        for client_id, images, labels in zip(ids, x, y):
            items[client_id] = (images, labels)
        purity[role] = float(np.mean([
            np.bincount(row, minlength=10).max() / examples for row in y
        ]))
    if len(items) != sum(len(v) for v in partitions.values()):
        raise AssertionError("Client identities overlap between partitions")
    return MaterializedData(items), purity


def score_endpoint(weights, candidate, population, ref_candidate,
                   ref_candidate_logits, ref_population, lira_std,
                   rmia_variants, calibrate_endpoint=False):
    x = true_label_probabilities(weights, candidate)
    raw = true_label_logits(weights, candidate)
    z = np.concatenate([true_label_probabilities(weights, item)
                        for item in population])
    results = {"lira_offline_fixed": float(np.mean(
        offline_lira_fixed_record_scores(x, ref_candidate, lira_std))),
        "pierre_lira_cdf": float(np.mean(pierre_offline_lira_cdf(
            raw, ref_candidate_logits)))}
    for name, a, gamma, population_correction in rmia_variants:
        results[name] = float(np.mean(offline_rmia_record_scores(
            x, ref_candidate, z, ref_population, a=a, gamma=gamma,
            population_correction=population_correction)))
    if calibrate_endpoint:
        results.update(score_variants(x, raw, ref_candidate,
                                      ref_candidate_logits, z,
                                      ref_population, lira_std))
    return results


def run(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    if args.rounds < 2 or not 0 <= args.position < args.rounds:
        raise ValueError("Invalid round count or insertion position")
    if args.clients_per_round < 2 or args.public_bank < 1:
        raise ValueError("Invalid client slots or public bank")
    if args.participation_q is None:
        cohort_size = None
        needed_public = args.rounds * (args.clients_per_round - 1)
    else:
        if not 0 < args.participation_q <= 1:
            raise ValueError("Require 0 < participation q <= 1")
        cohort_size = round(args.clients_per_round / args.participation_q)
        if not np.isclose(args.clients_per_round / cohort_size,
                          args.participation_q):
            raise ValueError("Choose q=B/N for an integer cohort size N")
        needed_public = cohort_size
        if args.background_pool < cohort_size:
            raise ValueError("Background pool is smaller than the cohort")
    if args.public_pool < needed_public or args.public_bank > args.public_pool:
        raise ValueError(f"Public pool must have at least {needed_public} clients")
    if min(args.calibration_identities, args.holdout_identities,
           args.population_clients, args.background_pool) < 2:
        raise ValueError("Insufficient clients in a role")
    if not args.sigma or any(s <= 0 for s in args.sigma):
        raise ValueError("Noise multipliers must be positive")
    if not 0 < args.delta < 1:
        raise ValueError("Require 0 < delta < 1")
    if args.clip <= 0 or args.client_lr <= 0 or args.server_lr <= 0:
        raise ValueError("Clipping and learning rates must be positive")
    if not 0 <= args.momentum < 1 or args.penalty < 0 or args.iterations < 1:
        raise ValueError("Invalid momentum or sparse-decoder parameters")
    if args.reference_models < 2:
        raise ValueError("Offline LiRA requires at least two OUT references")
    if args.fuse_trajectory and not args.calibrate_endpoint:
        raise ValueError("Trajectory fusion requires endpoint calibration")
    if args.reserve_utility_clients < 0:
        raise ValueError("Utility reservation must be nonnegative")
    source_path = Path(args.sqlite)
    if not source_path.is_file():
        raise FileNotFoundError(f"Download {DATA_URL} to {source_path}")
    source = ClientData(source_path, args.examples,
                        selection=args.client_example_selection,
                        sample_seed=args.seed,
                        pixel_transform=args.pixel_transform)
    count = args.calibration_identities + args.holdout_identities
    required = (count + args.public_pool + args.population_clients
                + args.background_pool)
    all_ids = np.asarray(source.ids, dtype=object)
    ids = np.asarray([client_id for client_id, size in
                      source.connection.execute(
                          "SELECT client_id,num_examples FROM federated_data "
                          "ORDER BY client_id") if size >= args.examples],
                     dtype=object)
    utility_ids = set()
    if args.reserve_utility_clients:
        utility_ids = set(all_ids[np.random.default_rng(args.utility_seed)
                                  .permutation(len(all_ids))
                                  [:args.reserve_utility_clients]])
        ids = np.asarray([client_id for client_id in ids
                          if client_id not in utility_ids], dtype=object)
    if required > len(ids):
        raise ValueError(f"Need {required} eligible clients after utility "
                         f"reservation; have {len(ids)}")
    eligible_count = len(ids)
    ids = ids[np.random.default_rng(args.seed).permutation(len(ids))][:required]
    if utility_ids.intersection(ids):
        raise AssertionError("Utility-tuning writer entered the audit")
    cuts = np.cumsum([count, args.public_pool,
                      args.population_clients, args.background_pool])
    candidate_ids, public_ids, population_ids, background_ids = np.split(ids, cuts[:-1])
    partitions = dict(candidate=candidate_ids, public=public_ids,
                      population=population_ids, background=background_ids)
    matrix = tree_matrix(args.rounds)
    methods = ("rero_max", "rero_mean", "gaussproof_max",
               "gaussproof_mean", "public_gls_max", "endpoint_loss",
               "informed_llr", "lira_offline_fixed",
               "rmia_offline_a1_g2", "rmia_offline_a05_g1",
               "pierre_lira_cdf", "pierre_rmia_a05_g1")
    if args.calibrate_endpoint:
        methods += ("lira_calibrated", "rmia_calibrated",
                    "endpoint_best_calibrated")
    if args.fuse_trajectory:
        methods += ("hybrid_equal_lira", "hybrid_calibrated_lira",
                    "hybrid_equal_rmia", "hybrid_calibrated_rmia")
    # The first is the official repository's default offline a=1,gamma=2.
    # The second is the pre-existing GAUSSPROOF benchmark's fixed setting.
    rmia_variants = (("rmia_offline_a1_g2", 1., 2., True),
                     ("rmia_offline_a05_g1", .5, 1., True),
                     ("pierre_rmia_a05_g1", .5, 1., False))
    table = []
    comparisons = []
    selections = []
    fusion_selections = []
    participation_counts = []
    distribution_purity = {}
    output.mkdir(parents=True, exist_ok=True)
    for mode_index, mode in enumerate(args.modes):
        data, purity = materialize_distribution(
            source, partitions, mode, args.examples, args.seed + mode_index * 101)
        distribution_purity[mode] = purity
        population = [data.get(i) for i in population_ids]
        for sigma in args.sigma:
            references = []
            for index in range(args.reference_models):
                rng = np.random.default_rng(args.seed + 1009 * (index + 1))
                if cohort_size is None:
                    schedule = rng.choice(public_ids, needed_public,
                                          replace=False).reshape(args.rounds, -1)
                else:
                    cohort = rng.choice(public_ids, cohort_size, replace=False)
                    schedule, _ = replacement_schedule(
                        cohort, args.rounds, args.clients_per_round, rng)
                references.append(train_out_reference(
                    data, schedule, sigma=sigma, clip=args.clip,
                    client_lr=args.client_lr, server_lr=args.server_lr,
                    momentum=args.momentum,
                    clients_per_round=args.clients_per_round,
                    seed=args.seed + 7907 * (index + 1)))
            reference_population = np.stack([
                np.concatenate([true_label_probabilities(w, item)
                                for item in population]) for w in references
            ])
            reference_candidates = {
                client_id: np.stack([
                    true_label_probabilities(w, data.get(client_id))
                    for w in references]) for client_id in candidate_ids
            }
            reference_candidate_logits = {
                client_id: np.stack([
                    true_label_logits(w, data.get(client_id))
                    for w in references]) for client_id in candidate_ids
            }
            all_out = np.concatenate((reference_population,
                np.concatenate(list(reference_candidates.values()), axis=1)), axis=1)
            lira_std = reference_global_std(all_out)
            records = []
            for index, candidate_id in enumerate(candidate_ids):
                bg_rng = np.random.default_rng(args.seed + 7919 * (index + 1))
                if cohort_size is None:
                    background = bg_rng.choice(background_ids, needed_public,
                                               replace=False).reshape(args.rounds, -1)
                    selected = None
                else:
                    cohort = bg_rng.choice(background_ids, cohort_size,
                                           replace=False)
                    background, selected = replacement_schedule(
                        cohort, args.rounds, args.clients_per_round, bg_rng)
                    participation_counts.append(dict(
                        distribution=mode, sigma=sigma, client_index=index,
                        count=len(selected)))
                settings = dict(rounds=args.rounds, position=args.position,
                                clients_per_round=args.clients_per_round,
                                sigma=sigma, clip=args.clip,
                                client_lr=args.client_lr,
                                server_lr=args.server_lr,
                                momentum=args.momentum,
                                seed=args.seed + 100003 * (index + 1),
                                matrix=matrix, penalty=args.penalty,
                                sparse_iterations=args.iterations,
                                capture_final=True,
                                participation_rounds=selected,
                                replace_background=cohort_size is not None)
                world = {}
                for member, name in ((False, "negative"), (True, "positive")):
                    result = simulate(data, candidate_id, background,
                                      public_ids[:args.public_bank],
                                      population_ids, member=member, **settings)
                    final = result.pop("final_weights")
                    result.update(score_endpoint(
                        final, data.get(candidate_id), population,
                        reference_candidates[candidate_id],
                        reference_candidate_logits[candidate_id],
                        reference_population, lira_std, rmia_variants,
                        calibrate_endpoint=args.calibrate_endpoint))
                    world[name] = result
                if abs(world["positive"]["noise_check"]
                       - world["negative"]["noise_check"]) > 1e-10:
                    raise AssertionError("Paired tree noise differs")
                records.append(world)
                print(f"{mode} sigma={sigma:g} client={index+1}/{count}",
                      flush=True)
            calibration = records[:args.calibration_identities]
            holdout = records[args.calibration_identities:]
            if args.calibrate_endpoint:
                selections.extend(dict(distribution=mode, sigma=sigma, **row)
                                  for row in apply_calibration(
                                      records, args.calibration_identities))
            if args.fuse_trajectory:
                fusion_selections.extend(dict(distribution=mode, sigma=sigma,
                                              **row) for row in apply_hybrids(
                                                  records, args.calibration_identities))
            for method in methods:
                interval = identity_interval(holdout, method, args.bootstrap,
                                             args.seed + 17)
                tpr1, fpr1 = calibrated_rates(calibration, holdout, method, .01)
                tpr5, fpr5 = calibrated_rates(calibration, holdout, method, .05)
                positive = [r["positive"][method] for r in holdout]
                negative = [r["negative"][method] for r in holdout]
                table.append(dict(
                    distribution=mode, sigma=sigma, method=method,
                    auc=cross_identity_auc(holdout, method),
                    auc_low=float(interval[0]), auc_high=float(interval[1]),
                    advantage=advantage(positive, negative),
                    tpr_at_calibrated_1pct_fpr=tpr1,
                    achieved_1pct_fpr=fpr1,
                    tpr_at_calibrated_5pct_fpr=tpr5,
                    achieved_5pct_fpr=fpr5,
                    mean_accuracy_present=float(np.mean([
                        r["positive"]["accuracy"] for r in holdout])),
                    mean_accuracy_absent=float(np.mean([
                        r["negative"]["accuracy"] for r in holdout])),
                    calibration_clients=len(calibration),
                    holdout_clients=len(holdout),
                    population_records=len(reference_population[0]),
                    reference_models=len(references),
                    mean_client_class_purity=purity["candidate"],
                    rounds=args.rounds,
                    participation_round=(args.position + 1
                                         if cohort_size is None else None),
                    participation_probability=(args.participation_q
                                               if cohort_size else None),
                    expected_participations=(args.rounds * args.participation_q
                                             if cohort_size else 1),
                    conservative_one_pass_epsilon_upper=(epsilon_upper(
                        args.rounds, sigma, args.delta)
                        if cohort_size is None else float("nan")),
                    delta=args.delta,
                ))
            comparison_pairs = (
                ("gaussproof_mean", "rero_mean"),
                ("gaussproof_mean", "endpoint_loss"),
                ("gaussproof_mean", "lira_offline_fixed"),
                ("gaussproof_mean", "rmia_offline_a1_g2"),
                ("gaussproof_mean", "rmia_offline_a05_g1"),
                ("gaussproof_mean", "pierre_lira_cdf"),
                ("gaussproof_mean", "pierre_rmia_a05_g1"),
                ("gaussproof_max", "rero_max"),
                ("gaussproof_max", "endpoint_loss"),
                ("gaussproof_max", "lira_offline_fixed"),
                ("gaussproof_max", "rmia_offline_a1_g2"),
                ("gaussproof_max", "rmia_offline_a05_g1"),
                ("gaussproof_max", "pierre_lira_cdf"),
                ("gaussproof_max", "pierre_rmia_a05_g1"))
            if args.calibrate_endpoint:
                comparison_pairs += tuple(
                    (left, baseline)
                    for left in ("gaussproof_max", "gaussproof_mean")
                    for baseline in ("lira_calibrated", "rmia_calibrated",
                                     "endpoint_best_calibrated"))
            if args.fuse_trajectory:
                comparison_pairs += tuple(
                    (f"hybrid_{kind}_{family}", f"{family}_calibrated")
                    for family in ("lira", "rmia")
                    for kind in ("equal", "calibrated"))
            for left, baseline in comparison_pairs:
                diff = paired_difference_interval(
                    holdout, left, baseline, args.bootstrap,
                    args.seed + 29)
                comparisons.append(dict(
                    distribution=mode, sigma=sigma,
                    comparison=f"{left}_minus_{baseline}",
                    auc_difference=(cross_identity_auc(holdout, left)
                                    - cross_identity_auc(holdout, baseline)),
                    low=float(diff[0]), high=float(diff[1]),
                    holdout_clients=len(holdout)))
    write_csv(output / "summary.csv", table)
    write_csv(output / "paired_comparisons.csv", comparisons)
    if args.calibrate_endpoint:
        write_csv(output / "calibration_selections.csv", selections)
    if args.fuse_trajectory:
        write_csv(output / "fusion_selections.csv", fusion_selections)
    if cohort_size is not None:
        write_csv(output / "participation_counts.csv", participation_counts)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    metadata = dict(
        status="source-aligned NumPy exploratory pilot; not official TFF driver",
        upstream_sha=UPSTREAM_SHA, federated_emnist_sha256=digest,
        dataset_url=DATA_URL, parameters=vars(args),
        role_counts={name: len(ids) for name, ids in partitions.items()},
        role_partitions_disjoint=True,
        utility_tuning_writers_reserved=args.reserve_utility_clients,
        utility_tuning_writers_disjoint=True,
        eligible_writers_after_reservation=eligible_count,
        participation_design=(
            "One fixed insertion with one empty slot under H0"
            if cohort_size is None else
            "Each round samples B clients without replacement from N=B/q. "
            "H1 replaces one H0 cohort client with the candidate. Both worlds "
            "have B clients every round; the same schedule and tree-noise draws "
            "are paired. q is the candidate's per-round sampling probability. "
            "The one-pass epsilon bound does not apply to repeated participation. "
            "Replace-one sensitivity is at most 2C/B while node noise "
            "standard deviation is sigma*C/B."),
        cohort_size=cohort_size,
        endpoint_calibration=(
            "Calibration identities alone select LiRA/RMIA score variant, "
            "mean/max record aggregation, and orientation separately for each "
            "distribution and sigma; holdout identities select nothing. "
            "The endpoint_best family may also select final-model loss. "
            "This is a deliberately generous client-level adaptation, not "
            "an original-paper attack implementation."
            if args.calibrate_endpoint else None),
        trajectory_fusion=(
            "Calibrated endpoint and GAUSSPROOF mean scores are standardized "
            "using calibration identities only. A 50:50 fusion is fixed. "
            "A second fusion selects trajectory weight from "
            "{0,.25,.5,.75,1} using calibration AUC only. Both are evaluated "
            "on separate heldout identities."
            if args.fuse_trajectory else None),
        candidate_class_purity=distribution_purity,
        reference_training=("Same fixed-slot DP-FTRLM recurrence; only public "
                            "clients; all candidate and population records OUT"),
        rmia=("Original record-level offline ratio and population dominance; "
              "record scores averaged within each client"),
        lira=("Original offline fixed-variance OUT Gaussian score on "
              "true-label logit; record scores averaged within each client"),
        pierre_joly_reference=(
            "https://github.com/Pierre-Joly/Membership-Inference-Attacks/"
            "tree/9182ed809d9fa3d5141d50816b7e83a06590371b"),
        pierre_joly_variants=("OfflineRMIA applies a=.5 correction to x but "
                              "uses mean OUT for population z. OfflineLiRA "
                              "uses raw correct-class logits, per-record OUT "
                              "mean/std, and Gaussian CDF. Both are explicitly "
                              "separate from original-paper variants."),
        caution=("IID and label-sorted clients are synthetic regroupings of "
                 "genuine EMNIST digit records. This is not an official "
                 "TensorFlow Federated application or privacy break."),
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    plot(output, table, args.modes)


def plot(output, rows, modes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42

    endpoint = {"endpoint_loss": ("Final loss", "#228833"),
                "lira_offline_fixed": ("LiRA (offline)", "#AA3377"),
                "rmia_offline_a1_g2": ("RMIA (a=1, γ=2)", "#EE7733"),
                "rmia_offline_a05_g1": ("RMIA (a=.5, γ=1)", "#66CCEE"),
                "pierre_lira_cdf": ("Pierre LiRA CDF", "#999933"),
                "pierre_rmia_a05_g1": ("Pierre RMIA", "#882255")}
    variants = {
        "auc_by_distribution": {
            "gaussproof_max": ("GAUSSPROOF max", "#CC6677"),
            "rero_max": ("RERO max", "#4477AA"), **endpoint},
        "auc_mean_by_distribution": {
            "gaussproof_mean": ("GAUSSPROOF mean", "#CC6677"),
            "rero_mean": ("RERO mean", "#4477AA"), **endpoint},
    }
    for basename, methods in variants.items():
        fig, axes = plt.subplots(1, len(modes),
                                 figsize=(4.1 * len(modes), 3.7),
                                 sharey=True, layout="constrained")
        axes = np.atleast_1d(axes)
        visible = [r for r in rows if r["method"] in methods]
        lower = min(.5, *(float(r["auc_low"]) for r in visible))
        upper = max(.5, *(float(r["auc_high"]) for r in visible))
        limits = (max(0, lower - .03), min(1, upper + .03))
        for axis, mode in zip(axes, modes):
            for method, (label, color) in methods.items():
                subset = sorted((r for r in rows if r["distribution"] == mode
                                 and r["method"] == method),
                                key=lambda x: float(x["sigma"]))
                axis.errorbar([float(r["sigma"]) for r in subset],
                              [float(r["auc"]) for r in subset],
                              yerr=[[max(0., float(r["auc"]) - float(r["auc_low"]))
                                     for r in subset],
                                    [max(0., float(r["auc_high"]) - float(r["auc"]))
                                     for r in subset]],
                              marker="o", markersize=4, linewidth=1.4,
                              capsize=2, color=color, label=label)
            axis.axhline(.5, color=".5", linewidth=1, linestyle="--")
            axis.set(title=mode.replace("_", " "),
                     xlabel="Tree noise multiplier σ", ylim=limits)
        axes[0].set_ylabel("Cross-identity client AUC")
        handles, legend_labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", ncol=4,
                   bbox_to_anchor=(.5, -.08), frameon=False, fontsize=8)
        for ext in ("pdf", "png"):
            fig.savefig(output / f"{basename}.{ext}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--modes", nargs="+", default=[
        "natural_writer", "iid_mixed", "label_sorted"])
    p.add_argument("--rounds", type=int, default=32)
    p.add_argument("--position", type=int, default=16)
    p.add_argument("--participation-q", type=float, default=None,
                   help="B/N replacement-client sampling rate per round; "
                        "omission runs the legacy one-insertion control")
    p.add_argument("--clients-per-round", type=int, default=8)
    p.add_argument("--calibration-identities", type=int, default=40)
    p.add_argument("--holdout-identities", type=int, default=80)
    p.add_argument("--public-bank", type=int, default=12)
    p.add_argument("--public-pool", type=int, default=256)
    p.add_argument("--population-clients", type=int, default=20)
    p.add_argument("--background-pool", type=int, default=500)
    p.add_argument("--reference-models", type=int, default=8)
    p.add_argument("--examples", type=int, default=16)
    p.add_argument("--client-example-selection", choices=("first", "uniform"),
                   default="first")
    p.add_argument("--pixel-transform", choices=("raw", "ink"), default="raw")
    p.add_argument("--reserve-utility-clients", type=int, default=0)
    p.add_argument("--utility-seed", type=int, default=20260929)
    p.add_argument("--sigma", nargs="+", type=float, default=[1., 4.])
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=.25)
    p.add_argument("--client-lr", type=float, default=.2)
    p.add_argument("--server-lr", type=float, default=.005)
    p.add_argument("--momentum", type=float, default=.9)
    p.add_argument("--penalty", type=float, default=.0001)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--bootstrap", type=int, default=400)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--calibrate-endpoint", action="store_true")
    p.add_argument("--fuse-trajectory", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
