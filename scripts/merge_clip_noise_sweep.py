"""Validate and merge independently executed clip groups and low-epsilon tail."""
import argparse
import csv
import json
import math
from pathlib import Path

from gaussproof.fixed_pair_audit import one_sided_limits


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return (float(row["clip"]), float(row["sigma"]), int(float(row["steps"])),
            row["method"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs=3, required=True)
    parser.add_argument("--tail", required=True)
    parser.add_argument("--weak-clip", required=True)
    parser.add_argument("--utility-rescue", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    groups = [Path(v) for v in args.groups]
    tail_dir, weak_dir, output = Path(args.tail), Path(args.weak_clip), Path(args.output)
    configs = [json.loads((p/"config.json").read_text()) for p in groups]
    if any(c != configs[0] for c in configs[1:]):
        raise ValueError("Clip groups used different scientific configurations")
    cfg = configs[0]
    provenances = [json.loads((p/"provenance.json").read_text()) for p in groups]
    reference = provenances[0]
    for record in provenances[1:]:
        for field in ("dataset_sha256", "source_checkpoint_sha256",
                      "dummy_index", "target_index", "calibration_pairs", "holdout_pairs"):
            if record[field] != reference[field]:
                raise ValueError(f"Mismatch in {field} across clip groups")
    group_rows = []
    for index, directory in enumerate(groups):
        # The first group is also the final output directory. Preserve its
        # original five-cell table so merging can be safely repeated.
        source = (directory / "core_C1_summary.csv"
                  if index == 0 and (directory / "core_C1_summary.csv").exists()
                  else directory / "summary.csv")
        group_rows.append(read_csv(source))
    rows = [r for part in group_rows for r in part]
    expected = {(float(c),float(s),int(t),m)
                for c in cfg["clip_values"] for s in cfg["sigmas"]
                for t in cfg["prefix_steps"]
                for m in ("trajectory_mixture", "raw_alignment", "last_update",
                          "endpoint_logprob_difference")}
    observed = [key(r) for r in rows]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError("Core grid has missing or duplicate aggregate rows")
    rows.sort(key=lambda r:(float(r["clip"]),float(r["sigma"]),
                            int(float(r["steps"])),r["method"]))
    tail_cfg = json.loads((tail_dir/"config.json").read_text())
    for field in ("seed", "q", "batch_size", "learning_rate", "prefix_steps",
                  "calibration_pairs", "holdout_pairs", "utility_size", "delta"):
        if tail_cfg[field] != cfg[field]:
            raise ValueError(f"Low-epsilon tail mismatches {field}")
    tail_provenance = json.loads((tail_dir/"provenance.json").read_text())
    for field in ("dataset_sha256", "source_checkpoint_sha256",
                  "dummy_index", "target_index"):
        if tail_provenance[field] != reference[field]:
            raise ValueError(f"Tail mismatches {field}")
    tail = read_csv(tail_dir/"summary.csv")
    expected_tail = {(float(c),float(s),int(t),m)
                     for c in tail_cfg["clip_values"] for s in tail_cfg["sigmas"]
                     for t in tail_cfg["prefix_steps"] for m in
                     ("trajectory_mixture", "raw_alignment", "last_update",
                      "endpoint_logprob_difference")}
    if set(map(key,tail)) != expected_tail or len(tail) != len(expected_tail):
        raise ValueError("Low-epsilon tail is incomplete")
    tail.sort(key=lambda r:(float(r["clip"]),float(r["sigma"]),
                            int(float(r["steps"])),r["method"]))
    weak_cfg = json.loads((weak_dir/"config.json").read_text())
    for field in ("seed", "q", "batch_size", "learning_rate", "prefix_steps",
                  "calibration_pairs", "holdout_pairs", "utility_size", "delta", "sigmas"):
        if weak_cfg[field] != cfg[field]:
            raise ValueError(f"Weak-clipping control mismatches {field}")
    weak_provenance = json.loads((weak_dir/"provenance.json").read_text())
    for field in ("dataset_sha256", "source_checkpoint_sha256",
                  "dummy_index", "target_index"):
        if weak_provenance[field] != reference[field]:
            raise ValueError(f"Weak-clipping control mismatches {field}")
    weak = read_csv(weak_dir/"summary.csv")
    expected_weak = {(float(c),float(s),int(t),m)
                     for c in weak_cfg["clip_values"] for s in weak_cfg["sigmas"]
                     for t in weak_cfg["prefix_steps"] for m in
                     ("trajectory_mixture", "raw_alignment", "last_update",
                      "endpoint_logprob_difference")}
    if set(map(key,weak)) != expected_weak or len(weak) != len(expected_weak):
        raise ValueError("Weak-clipping control is incomplete")
    weak.sort(key=lambda r:(float(r["clip"]),float(r["sigma"]),
                            int(float(r["steps"])),r["method"]))
    utility_rows = []
    utility_conditions = []
    for path_string in args.utility_rescue:
        utility_dir = Path(path_string)
        utility_cfg = json.loads((utility_dir/"config.json").read_text())
        for field in ("seed", "q", "learning_rate", "prefix_steps",
                      "calibration_pairs", "holdout_pairs", "utility_size", "delta"):
            if utility_cfg[field] != cfg[field]:
                raise ValueError(f"Utility-rescue run mismatches {field}")
        utility_provenance = json.loads((utility_dir/"provenance.json").read_text())
        for field in ("dataset_sha256", "source_checkpoint_sha256",
                      "dummy_index", "target_index"):
            if utility_provenance[field] != reference[field]:
                raise ValueError(f"Utility-rescue run mismatches {field}")
        part = read_csv(utility_dir/"summary.csv")
        expected_part = {(float(c),float(s),int(t),m)
                         for c in utility_cfg["clip_values"]
                         for s in utility_cfg["sigmas"]
                         for t in utility_cfg["prefix_steps"]
                         for m in ("trajectory_mixture", "raw_alignment",
                                   "last_update", "endpoint_logprob_difference")}
        if set(map(key,part)) != expected_part or len(part) != len(expected_part):
            raise ValueError(f"Incomplete utility-rescue report: {utility_dir}")
        utility_rows.extend(dict(r, batch_size=utility_cfg["batch_size"]) for r in part)
        utility_conditions.extend((utility_cfg["batch_size"], c, s)
                                  for c in utility_cfg["clip_values"]
                                  for s in utility_cfg["sigmas"])
    if len(set(utility_conditions)) != len(utility_conditions):
        raise ValueError("Duplicate utility-rescue conditions")
    utility_rows.sort(key=lambda r:(float(r["batch_size"]),float(r["clip"]),
                                    float(r["sigma"]),int(float(r["steps"])),r["method"]))
    output.mkdir(parents=True, exist_ok=True)
    if output == groups[0] and not (output/"core_C1_summary.csv").exists():
        write_csv(output/"core_C1_summary.csv",group_rows[0])
    write_csv(output/"summary.csv",rows)
    write_csv(output/"low_epsilon_tail.csv",tail)
    write_csv(output/"weak_clipping_control.csv",weak)
    write_csv(output/"utility_rescue.csv",utility_rows)
    combined = [dict(r,batch_size=8) for r in rows+tail+weak] + utility_rows
    comparisons = len(combined)
    alpha = .05/(2*comparisons)
    for row in combined:
        n = int(row["holdout_pairs"])
        tp,fp = int(row["true_positives"]),int(row["false_positives"])
        tpr_lower,_ = one_sided_limits(tp,n,alpha)
        _,fpr_upper = one_sided_limits(fp,n,alpha)
        delta = float(row["delta"])
        row["empirical_epsilon_lower_95_across_all_cells"] = (
            max(0.,math.log((tpr_lower-delta)/fpr_upper))
            if tpr_lower > delta else 0.)
    combined.sort(key=lambda r:(float(r["batch_size"]),float(r["clip"]),float(r["sigma"]),
                                int(float(r["steps"])),r["method"]))
    write_csv(output/"all_cells_summary.csv",combined)
    provenance = dict(reference,
        grid_clip_values=cfg["clip_values"],grid_sigmas=cfg["sigmas"],
        low_epsilon_tail_sigmas=tail_cfg["sigmas"],
        weak_clipping_control_clip_values=weak_cfg["clip_values"],
        grid_conditions=len(cfg["clip_values"])*len(cfg["sigmas"]),
        tail_conditions=len(tail_cfg["clip_values"])*len(tail_cfg["sigmas"]),
        weak_clipping_conditions=len(weak_cfg["clip_values"])*len(weak_cfg["sigmas"]),
        utility_rescue_conditions=len(utility_conditions),
        utility_rescue_batch_sizes=sorted({b for b,_,_ in utility_conditions}),
        familywise_confidence_tests=comparisons,
        clip_group_reports=[str(p) for p in groups],tail_report=str(tail_dir),
        weak_clipping_report=str(weak_dir),
        utility_rescue_reports=args.utility_rescue)
    provenance["auc_interval"] = "observed AUC +/- 1.96 paired-bootstrap standard errors"
    provenance["gain_interval"] = "observed AUC gap +/- 1.96 paired-bootstrap standard errors"
    (output/"provenance.json").write_text(json.dumps(provenance,indent=2))
    (output/"progress.json").write_text(json.dumps(dict(
        completed_conditions=provenance["grid_conditions"]+provenance["tail_conditions"]+
                             provenance["weak_clipping_conditions"]+
                             provenance["utility_rescue_conditions"],
        total_conditions=provenance["grid_conditions"]+provenance["tail_conditions"]+
                         provenance["weak_clipping_conditions"]+
                         provenance["utility_rescue_conditions"],
        complete=True),indent=2))
    print(f"Merged {len(rows)} core, {len(tail)} low-epsilon, "
          f"{len(weak)} weak-clipping, and {len(utility_rows)} utility-rescue rows")


if __name__ == "__main__":
    main()
