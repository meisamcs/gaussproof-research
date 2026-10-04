"""Render a compact, paper-width GAUSSPROOF system and adversary diagram.

The manuscript gives the likelihood equations.  The figure shows the release
boundary, auxiliary knowledge, hidden quantities, and attack workflow.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "satml2027" / "figures" / "system_adversary_model"


def box(ax, x, y, w, h, label, *, fill, edge, fontsize=8.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.025",
        facecolor=fill, edgecolor=edge, linewidth=1.05,
    ))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color="#192b3e", fontsize=fontsize, linespacing=1.12)


def arrow(ax, start, end, *, color="#52677f", width=1.1):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9,
        linewidth=width, color=color,
    ))


def make():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.15, 2.88))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, .018, .035, .430, .920, "", fill="#f5f8fc", edge="#9bb1c8")
    box(ax, .552, .035, .430, .920, "", fill="#f6faf8", edge="#95b5a3")
    ax.text(.044, .914, "SYSTEM  /  DP-SGD", fontsize=9.4, weight="bold",
            color="#265472", va="center")
    ax.text(.578, .914, "ADVERSARY  /  GAUSSPROOF", fontsize=9.4,
            weight="bold", color="#286a4d", va="center")

    box(ax, .047, .692, .372, .144,
        "Secret: candidate absent or eligible\n"
        "for sampling at rate $q$ each round",
        fill="#faeded", edge="#c58790")
    box(ax, .047, .390, .372, .220,
        "Clip private batch gradients at $\\theta_t$\n"
        "Release noisy average $y_t$",
        fill="#e8f1fa", edge="#7899bb", fontsize=8.3)
    box(ax, .047, .212, .372, .107,
        "Update $\\theta_{t+1}=\\theta_t-\\eta y_t$; repeat",
        fill="#e8f1fa", edge="#7899bb", fontsize=8.3)
    ax.text(.233, .111, "Hidden: batch, inclusion bits, noise draws",
            ha="center", va="center", fontsize=7.7, color="#905960")

    box(ax, .581, .692, .372, .144,
        "Known: candidate $z$, public $D_{\\mathrm{pub}}$,\n"
        "model and DP-SGD settings",
        fill="#e9f2fb", edge="#7899bb")
    box(ax, .581, .499, .372, .117,
        "At $\\theta_t$, recompute $h_t$ and $b_t$\n"
        "from candidate and public data",
        fill="#eeeafa", edge="#9887ba", fontsize=8.0)
    box(ax, .581, .323, .372, .108,
        "Compare $y_t-b_t$ with shift $(h_t-b_t)/B$",
        fill="#fff1d5", edge="#c5a466", fontsize=8.0)
    box(ax, .581, .117, .372, .137,
        "Accumulate $q$-aware evidence $M_T(z)$\n"
        "Test membership or rank a gallery",
        fill="#e3f1e8", edge="#70a081", fontsize=8.1)

    arrow(ax, (.233, .692), (.233, .610))
    arrow(ax, (.233, .390), (.233, .319))
    arrow(ax, (.767, .692), (.767, .616))
    arrow(ax, (.767, .499), (.767, .431))
    arrow(ax, (.767, .323), (.767, .254))
    arrow(ax, (.419, .515), (.581, .558), color="#286a4d", width=1.8)
    ax.text(.500, .670, "visible", ha="center", va="center",
            color="#286a4d", fontsize=7.8, weight="bold")
    ax.text(.500, .622, "$(\\theta_t,y_t)$", ha="center", va="center",
            color="#286a4d", fontsize=8.4)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT.with_suffix("." + extension), dpi=300,
                    bbox_inches="tight", pad_inches=.035, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make()
