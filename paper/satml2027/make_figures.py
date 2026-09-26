from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, xy, wh, text, face, edge="#243044", fontsize=8.5):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#172033", linespacing=1.2)
    return patch


def arrow(ax, start, end, label=None, bend=0.0):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.0,
        color="#46566f", connectionstyle=f"arc3,rad={bend}",
    )
    ax.add_patch(patch)
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .025,
                label, fontsize=7.3, ha="center", va="bottom", color="#35445e")


def framework():
    fig, ax = plt.subplots(figsize=(7.15, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (.02, .55), (.17, .27),
        "Private batch\nrecord clipped to C", "#e8f1fb")
    box(ax, (.25, .55), (.17, .27),
        "DP-SGD release\nyₜ = mean + Gaussian noise", "#f7eadf")
    box(ax, (.48, .55), (.17, .27),
        "White-box observer\nmodel state θₜ and yₜ", "#e8f1fb")
    box(ax, (.71, .55), (.26, .27),
        "Trajectory decision\nmembership or gallery rank", "#e7f3ea")

    box(ax, (.25, .10), (.17, .22),
        "Public background\nestimate bₜ", "#f2eff8")
    box(ax, (.48, .10), (.17, .22),
        "Candidate fingerprint\nhₜ = clip(∇ℓ(θₜ; z))", "#f2eff8")
    box(ax, (.71, .10), (.26, .22),
        "Accumulate evidence\nΣₜ log[(1-q)+q exp(ℓₜ)]", "#fff4cf")

    arrow(ax, (.19, .685), (.25, .685))
    arrow(ax, (.42, .685), (.48, .685))
    arrow(ax, (.65, .685), (.71, .685))
    arrow(ax, (.42, .21), (.48, .21))
    arrow(ax, (.65, .21), (.71, .21))
    arrow(ax, (.335, .55), (.335, .32))
    arrow(ax, (.565, .55), (.565, .32))
    arrow(ax, (.84, .32), (.84, .55))

    ax.text(.02, .94, "GAUSSPROOF audit", fontsize=11, weight="bold", color="#172033")
    ax.text(.995, .01,
            "The observer never receives the private batch, hidden inclusion schedule, or sampled noise.",
            fontsize=7.2, ha="right", va="bottom", color="#536079")
    fig.tight_layout(pad=.2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"framework.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    framework()
