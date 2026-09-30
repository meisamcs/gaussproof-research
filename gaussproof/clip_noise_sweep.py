"""Resumable matched clipping/noise sweep for a fixed natural record pair.

The report contains only aggregate measurements. Individual MNIST scores and
world labels remain in ignored runs/. Each grid cell has independent
calibration/holdout trial seeds; those seeds are paired across C and sigma.
"""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import digest, load_mnist
from .fixed_pair_audit import fixed_world, one_sided_limits, write_csv
from .metrics import auc, rates, threshold_at_fpr

METHODS = ("trajectory_mixture", "raw_alignment", "last_update",
           "endpoint_logprob_difference")


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return [{k: (v if k == "role" else int(v) if k in
                 ("pair", "world", "steps") else float(v))
                 for k, v in r.items()} for r in csv.DictReader(stream)]


def epsilon_upper(steps, sigma, delta):
    """Replace-one zCDP bound without any participation amplification."""
    rho = 2 * steps / sigma**2
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def summarize_cell(rows, cfg, clip, sigma, index):
    result = []
    n_cal, n_test = cfg["calibration_pairs"], cfg["holdout_pairs"]
    conditions = len(cfg["clip_values"]) * len(cfg["sigmas"]) * len(cfg["prefix_steps"])
    simultaneous_tests = conditions * len(METHODS)
    for steps in cfg["prefix_steps"]:
        selected = [r for r in rows if r["clip"] == clip and r["sigma"] == sigma
                    and r["steps"] == steps]
        cal = [r for r in selected if r["role"] == "calibration"]
        test = [r for r in selected if r["role"] == "holdout"]
        if len(cal) != 2*n_cal or len(test) != 2*n_test:
            raise ValueError(f"Incomplete cell C={clip}, sigma={sigma}, T={steps}")
        for role, group, count in (("calibration", cal, n_cal), ("holdout", test, n_test)):
            observed = {(r["pair"], r["world"]) for r in group}
            expected = {(p, w) for p in range(count) for w in (0, 1)}
            if observed != expected:
                raise ValueError(f"Missing or duplicate {role} world/seed rows")
        cy, ty = (np.asarray([r["world"] for r in group], int)
                  for group in (cal, test))
        pairs = np.asarray([r["pair"] for r in test])
        pair_positions = [np.flatnonzero(pairs == i) for i in range(n_test)]
        rng = np.random.default_rng(cfg["seed"] + 500000 + index*1000 + steps)
        draws = rng.integers(0, n_test, size=(cfg["bootstrap_repetitions"], n_test))
        samples = [np.concatenate([pair_positions[i] for i in draw]) for draw in draws]
        scores = {method: np.asarray([r[method] for r in test]) for method in METHODS}
        boot = {method: np.asarray([auc(values[s], ty[s]) for s in samples])
                for method, values in scores.items()}
        endpoint = "endpoint_logprob_difference"
        endpoint_auc = auc(scores[endpoint], ty)
        for method in METHODS:
            cs = np.asarray([r[method] for r in cal])
            ts = scores[method]
            threshold = threshold_at_fpr(cs, cy, cfg["calibration_fpr"])
            tpr, fpr = rates(ts, ty, threshold)
            tp = int(np.sum((ts >= threshold) & (ty == 1)))
            fp = int(np.sum((ts >= threshold) & (ty == 0)))
            tpr_low, _ = one_sided_limits(tp, n_test, .025)
            _, fpr_high = one_sided_limits(fp, n_test, .025)
            family_alpha = .05 / (2 * simultaneous_tests)
            family_tpr_low, _ = one_sided_limits(tp, n_test, family_alpha)
            _, family_fpr_high = one_sided_limits(fp, n_test, family_alpha)
            def eps_lower(lower, upper):
                return max(0., math.log((lower-cfg["delta"])/upper)) if lower > cfg["delta"] else 0.
            observed_auc = auc(ts, ty)
            gain_distribution = boot[method] - boot[endpoint]
            result.append(dict(clip=clip, sigma=sigma, q=cfg["q"], steps=steps,
                epsilon_upper_no_amplification=epsilon_upper(steps, sigma, cfg["delta"]),
                method=method, auc=observed_auc,
                auc_ci_low=float(np.quantile(boot[method], .025)),
                auc_ci_high=float(np.quantile(boot[method], .975)),
                endpoint_auc=endpoint_auc, auc_gain_vs_endpoint=observed_auc-endpoint_auc,
                gain_ci_low=float(np.quantile(gain_distribution, .025)),
                gain_ci_high=float(np.quantile(gain_distribution, .975)),
                calibrated_holdout_tpr=tpr, calibrated_holdout_fpr=fpr,
                true_positives=tp, false_positives=fp,
                empirical_epsilon_lower_95_marginal=eps_lower(tpr_low,fpr_high),
                empirical_epsilon_lower_95_familywise=eps_lower(
                    family_tpr_low,family_fpr_high),
                mean_public_accuracy=float(np.mean([r["utility_accuracy"] for r in test])),
                mean_clipped_fraction=float(np.mean([
                    r["mean_batch_clipped_fraction"] for r in test])),
                mean_positive_appearances=float(np.mean([
                    r["realized_inclusions"] for r in test if r["world"] == 1])),
                calibration_pairs=n_cal, holdout_pairs=n_test,
                delta=cfg["delta"]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--only-clip", type=float,
                        help="Run one clip group from the unchanged full-grid configuration")
    args = parser.parse_args()
    source, output, report = Path(args.source), Path(args.output), Path(args.report)
    cfg = json.loads(Path(args.config).read_text())
    if cfg["calibration_pairs"] < 20 or cfg["holdout_pairs"] < 40:
        print("Warning: low-FPR calibration and inference will be coarse", flush=True)
    if not (0 < cfg["q"] <= 1 and cfg["delta"] > 0 and
            min(cfg["clip_values"]) > 0 and min(cfg["sigmas"]) > 0):
        raise ValueError("Invalid mechanism parameters")
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").exists():
        raise ValueError("Completed sweep exists")
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != cfg:
            raise ValueError("Cannot resume a run under different conditions")
    else:
        (output / "config.json").write_text(json.dumps(cfg, indent=2))
    if (report / "config.json").exists():
        if json.loads((report / "config.json").read_text()) != cfg:
            raise ValueError("Report directory belongs to a different sweep")
    else:
        (report / "config.json").write_text(json.dumps(cfg, indent=2))
    expected = json.loads((source / "completion.json").read_text())["dataset_sha256"]
    observed = digest(args.data)
    if observed != expected:
        raise ValueError("MNIST checksum does not match the public checkpoint")
    torch.set_num_threads(1)
    x, labels = load_mnist(args.data)
    splits = json.loads((source / "splits.json").read_text())
    source_cfg = json.loads((source / "config.json").read_text())
    initial = torch.load(source / "public_initial_cnn.pt", weights_only=True)
    dummy, target = map(int, splits["test"][:2])
    utility = np.asarray(splits["test"][2:2+cfg["utility_size"]], int)
    population = np.asarray(splits["population"], int)
    background = np.asarray(splits["background"], int)
    if (len(utility) != cfg["utility_size"] or
        len(set(utility) & (set(population)|set(background)|{dummy,target}))):
        raise ValueError("Utility and audit roles overlap")
    runtime = dict(q=cfg["q"], batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"], source_seed=source_cfg["seed"],
        steps=max(cfg["prefix_steps"]), prefix_steps=cfg["prefix_steps"])
    rows_path = output / "score_rows.csv"
    rows = read_rows(rows_path)
    done = {(r["clip"],r["sigma"],r["role"],r["pair"],r["world"],r["steps"])
            for r in rows}
    summary_path = report / "summary.csv"
    if summary_path.exists():
        with summary_path.open(newline="") as stream:
            summary = list(csv.DictReader(stream))
    else:
        summary = []
    summarized = {(float(r["clip"]),float(r["sigma"])) for r in summary}
    started = time.time()
    all_conditions = [(float(c),float(s)) for c in cfg["clip_values"] for s in cfg["sigmas"]]
    conditions = [(index,c,s) for index,(c,s) in enumerate(all_conditions)
                  if args.only_clip is None or c == args.only_clip]
    if not conditions:
        raise ValueError("Requested clip value is absent from the fixed grid")
    for ordinal,(index,clip,sigma) in enumerate(conditions):
        condition_cfg = dict(runtime, clip=clip, sigma=sigma)
        print(f"Condition {ordinal+1}/{len(conditions)} (grid {index+1}/{len(all_conditions)}): "
              f"C={clip:g} sigma={sigma:g}", flush=True)
        for role,count,offset in (("calibration",cfg["calibration_pairs"],0),
                                  ("holdout",cfg["holdout_pairs"],100000)):
            for pair in range(count):
                for world in (0,1):
                    if all((clip,sigma,role,pair,world,step) in done
                           for step in cfg["prefix_steps"]):
                        continue
                    seed = cfg["seed"] + offset + pair*101
                    snapshots = fixed_world(initial,x,labels,population,background,
                        target,dummy,world,condition_cfg,seed,utility_indices=utility)
                    for step in cfg["prefix_steps"]:
                        key = (clip,sigma,role,pair,world,step)
                        if key in done:
                            continue
                        row = dict(clip=clip,sigma=sigma,role=role,pair=pair,
                                   world=world,steps=step,**snapshots[step])
                        rows.append(row)
                        done.add(key)
                    write_csv(rows_path,rows)
                if (pair+1)%10==0:
                    print(f"  {role} {pair+1}/{count} pairs",flush=True)
        if (clip,sigma) not in summarized:
            summary.extend(summarize_cell(rows,cfg,clip,sigma,index))
            write_csv(summary_path,summary)
            summarized.add((clip,sigma))
            (report / "progress.json").write_text(json.dumps(dict(
                completed_conditions=len(summarized),total_conditions=len(conditions),
                last_clip=clip,last_sigma=sigma,elapsed_seconds=time.time()-started),indent=2))
    checkpoint_hash = hashlib.sha256((source / "public_initial_cnn.pt").read_bytes()).hexdigest()
    provenance = dict(dataset_sha256=observed,source_checkpoint_sha256=checkpoint_hash,
        dummy_index=dummy,target_index=target,dummy_digit=int(labels[dummy]),
        target_digit=int(labels[target]),calibration_pairs=cfg["calibration_pairs"],
        holdout_pairs=cfg["holdout_pairs"],epsilon_accounting="replace-one zCDP; no sampling amplification",
        sampler="Bernoulli privileged slot; not ordinary uniform minibatch DP-SGD",
        paired_randomness_across_conditions=True,
        processed_clip_values=sorted({c for _,c,_ in conditions}),
        processed_conditions=len(conditions),
        configured_conditions=len(all_conditions),delta=cfg["delta"],
        elapsed_seconds=time.time()-started,torch=torch.__version__)
    (report / "provenance.json").write_text(json.dumps(provenance,indent=2))
    (output / "completion.json").write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance,indent=2),flush=True)


if __name__ == "__main__":
    main()
