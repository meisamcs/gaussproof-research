"""Check the public evidence trail without MNIST or optional dependencies."""

import csv
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "canary_first_principles_holdout"
PREFIXES = (16, 32, 64, 128, 256)


def read_indexed(path, fields):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise AssertionError(f"Empty report: {path}")
    indexed = {}
    for row in rows:
        key = tuple(float(row[field]) for field in fields)
        if key in indexed:
            raise AssertionError(f"Duplicate report key {key} in {path}")
        value = float(row["auc"])
        low, high = float(row["auc_ci_low"]), float(row["auc_ci_high"])
        if not (math.isfinite(value) and 0 <= low <= value <= high <= 1):
            raise AssertionError(f"Invalid AUC or interval for {key} in {path}")
        indexed[key] = row
    return indexed


def verify_headline_data():
    # The full summary has a string method, so index it separately.
    with (REPORT / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    primary = {}
    for row in rows:
        key = (float(row["q"]), int(row["steps"]), row["method"])
        if key in primary:
            raise AssertionError(f"Duplicate primary score {key}")
        primary[key] = row
    if len(primary) != 3 * len(PREFIXES) * 3:
        raise AssertionError("Incomplete primary q × prefix × method grid")
    expected = {(0.5, 16): .633, (0.5, 256): .843,
                (0.1, 16): .530, (0.1, 256): .589}
    for (q, step), shown in expected.items():
        row = primary[q, step, "mixture_llr"]
        if abs(float(row["auc"]) - shown) >= .0005:
            raise AssertionError(f"Headline AUC drift: q={q}, T={step}")
    matched = read_indexed(REPORT / "participation_auc_matched.csv", ("q", "steps"))
    noise = read_indexed(REPORT / "noise_auc.csv", ("sigma", "steps"))
    if len(matched) != 3 * len(PREFIXES) or len(noise) != 3 * len(PREFIXES):
        raise AssertionError("Incomplete matched participation or noise grid")
    for rows in (matched, noise):
        for row in rows.values():
            if int(row["identities"]) != 80 or int(row["sequences_per_identity"]) != 1:
                raise AssertionError("Matched comparison must use 80 identities and one sequence")
    for sigma, shown in ((2, .910), (4, .843), (8, .737)):
        if abs(float(noise[float(sigma), 256.0]["auc"]) - shown) >= .0005:
            raise AssertionError(f"Noise-check AUC drift: sigma={sigma}")
    with (ROOT / "reports" / "canary_q_sensitivity" / "summary.csv").open(newline="") as stream:
        q_rows = list(csv.DictReader(stream))
    low_q = [row for row in q_rows if float(row["q"]) == .004
             and int(row["steps"]) == 128 and row["method"] == "mixture_llr"]
    if len(low_q) != 1 or abs(float(low_q[0]["auc"]) - .503) >= .0005:
        raise AssertionError("Ordinary-q boundary is missing or changed")


def verify_links():
    for relative in ("README.md", "REPRODUCIBILITY.md", "STUDENT_HANDOFF.md",
                     "paper/satml2027/README.md"):
        path = ROOT / relative
        contents = path.read_text()
        for target in re.findall(r"!?\[[^]]*\]\(([^)]+)\)", contents):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            local = (path.parent / target.split("#", 1)[0]).resolve()
            if not local.is_file() and not local.is_dir():
                raise AssertionError(f"Broken link in {relative}: {target}")


def verify_paper_inputs():
    paper = ROOT / "paper" / "satml2027"
    source = paper / "main.tex"
    if source.read_bytes() != (paper / "gaussproof_satml2027_revised.tex").read_bytes():
        raise AssertionError("Manuscript source copies differ")
    for target in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source.read_text()):
        if not (paper / target).is_file():
            raise AssertionError(f"Missing manuscript figure: {target}")


def main():
    verify_headline_data()
    verify_links()
    verify_paper_inputs()
    print("Public artifact checks passed: result grids, links, and manuscript inputs")


if __name__ == "__main__":
    main()
