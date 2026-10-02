"""Plot held-out participation and noise checks for the fingerprint hypothesis."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gaussproof.metrics import auc
from scripts.summarize_first_principles import load_cube


def one_sequence_auc(pairs):
    flat = pairs.reshape(-1, 2)
    return auc(flat.ravel(), np.tile((0, 1), len(flat)))


def participation_summary(path, repetitions, seed):
    qs, steps, identities, cube = load_cube(path)
    rng = np.random.default_rng(seed)
    rows = []
    draws = np.empty((repetitions, len(qs), len(steps)))
    for repetition in range(repetitions):
        selected = rng.integers(len(identities), size=len(identities))
        for qi in range(len(qs)):
            for ti in range(len(steps)):
                draws[repetition, qi, ti] = one_sequence_auc(cube[qi, ti, selected, 0, :])
    for qi, q in enumerate(qs):
        for ti, step in enumerate(steps):
            low, high = np.quantile(draws[:, qi, ti], [.025, .975])
            rows.append(dict(q=q, sigma=4.0, steps=step,
                             auc=one_sequence_auc(cube[qi, ti, :, 0, :]),
                             auc_ci_low=float(low), auc_ci_high=float(high),
                             identities=len(identities), sequences_per_identity=1,
                             bootstrap_repetitions=repetitions, bootstrap_seed=seed))
    return rows


def noise_summary(paths, repetitions, seed):
    arrays, steps, identities = {}, None, None
    for sigma, path in paths.items():
        qs, current_steps, current_ids, cube = load_cube(path)
        if 0.5 not in qs:
            raise ValueError(f"q=0.5 missing from sigma={sigma:g}")
        if steps is not None and (steps != current_steps or identities != current_ids):
            raise ValueError("Noise arms do not use identical identities and prefixes")
        steps, identities = current_steps, current_ids
        arrays[sigma] = cube[qs.index(0.5), :, :, 0, :]
    rng = np.random.default_rng(seed)
    ordered = sorted(arrays)
    observed = np.asarray([[one_sequence_auc(arrays[sigma][ti])
                            for ti in range(len(steps))] for sigma in ordered])
    samples = np.empty((repetitions, len(ordered), len(steps)))
    for repetition in range(repetitions):
        selected = rng.integers(len(identities), size=len(identities))
        for si, sigma in enumerate(ordered):
            for ti in range(len(steps)):
                samples[repetition, si, ti] = one_sequence_auc(arrays[sigma][ti, selected])
    rows = []
    for si, sigma in enumerate(ordered):
        for ti, step in enumerate(steps):
            low, high = np.quantile(samples[:, si, ti], [.025, .975])
            rows.append(dict(sigma=sigma, q=0.5, steps=step, auc=observed[si, ti],
                             auc_ci_low=float(low), auc_ci_high=float(high),
                             identities=len(identities), sequences_per_identity=1,
                             bootstrap_repetitions=repetitions, bootstrap_seed=seed))
    contrasts = []
    for left, right in ((2.0, 4.0), (4.0, 8.0)):
        li, ri = ordered.index(left), ordered.index(right)
        values = samples[:, li, -1] - samples[:, ri, -1]
        low, high = np.quantile(values, [.025, .975])
        contrasts.append(dict(contrast=f"sigma={left:g} minus sigma={right:g} at T={steps[-1]}",
                              auc_difference=float(observed[li, -1] - observed[ri, -1]),
                              ci_low=float(low), ci_high=float(high),
                              identities=len(identities), bootstrap_repetitions=repetitions,
                              bootstrap_seed=seed))
    return rows, contrasts


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot(participation, noise, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True)
    colors_q = {0.5: "#276b9c", 0.25: "#d68127", 0.1: "#3f946a"}
    colors_sigma = {2.0: "#7952a3", 4.0: "#276b9c", 8.0: "#bd5658"}
    for q in (0.5, 0.25, 0.1):
        rows = sorted((r for r in participation if float(r["q"]) == q),
                      key=lambda r: int(r["steps"]))
        x = np.asarray([int(r["steps"]) for r in rows])
        y = np.asarray([float(r["auc"]) for r in rows])
        lo = np.asarray([float(r["auc_ci_low"]) for r in rows])
        hi = np.asarray([float(r["auc_ci_high"]) for r in rows])
        axes[0].plot(x, y, marker="o", linewidth=2, color=colors_q[q], label=f"q={q:g}")
        axes[0].fill_between(x, lo, hi, color=colors_q[q], alpha=.13)
    for sigma in (2.0, 4.0, 8.0):
        rows = sorted((r for r in noise if float(r["sigma"]) == sigma),
                      key=lambda r: int(r["steps"]))
        x = np.asarray([int(r["steps"]) for r in rows])
        y = np.asarray([float(r["auc"]) for r in rows])
        lo = np.asarray([float(r["auc_ci_low"]) for r in rows])
        hi = np.asarray([float(r["auc_ci_high"]) for r in rows])
        axes[1].plot(x, y, marker="o", linewidth=2, color=colors_sigma[sigma],
                     label=f"σ={sigma:g}")
        axes[1].fill_between(x, lo, hi, color=colors_sigma[sigma], alpha=.13)
    for ax in axes:
        ax.axhline(.5, color="#555555", linewidth=.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 32, 64, 128, 256], labels=["16", "32", "64", "128", "256"])
        ax.set_xlim(14, 285)
        ax.set_ylim(.48, 1.01)
        ax.set_xlabel("Observed releases T")
        ax.grid(alpha=.15)
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("Held-out candidate detection AUC")
    axes[0].set_title("(a) More participation strengthens signal")
    axes[1].set_title("(b) More noise weakens signal at q=0.5")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(Path(output).with_suffix("." + suffix), dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma2", required=True)
    parser.add_argument("--sigma4", required=True)
    parser.add_argument("--sigma8", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    paths = {2.0: args.sigma2, 4.0: args.sigma4, 8.0: args.sigma8}
    participation = participation_summary(args.sigma4, args.bootstrap, args.seed)
    noise, contrasts = noise_summary(paths, args.bootstrap, args.seed)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_csv(destination.parent / "noise_auc.csv", noise)
    write_csv(destination.parent / "noise_contrasts.csv", contrasts)
    write_csv(destination.parent / "participation_auc_matched.csv", participation)
    plot(participation, noise, destination)
    for row in contrasts:
        print(f"{row['contrast']}: {row['auc_difference']:.3f} "
              f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
