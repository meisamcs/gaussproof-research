"""Render the GAUSSPROOF system and passive white-box adversary model."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "satml2027" / "figures" / "system_adversary_model"


def rounded(ax, x, y, width, height, text, *, fill, edge, fontsize=9.2,
            text_color="#17263d", linewidth=1.15, style="round,pad=0.010,rounding_size=0.015"):
    patch = FancyBboxPatch((x, y), width, height, boxstyle=style,
                           facecolor=fill, edgecolor=edge, linewidth=linewidth)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, linespacing=1.25)
    return patch


def arrow(ax, start, end, *, color="#53677d", linewidth=1.55, style="arc3,rad=0"):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                            linewidth=linewidth, color=color, connectionstyle=style)
    ax.add_patch(patch)


def make():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(15.2, 7.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(.035, .967, "GAUSSPROOF  |  system and adversary model",
            ha="left", va="center", fontsize=17, weight="bold", color="#17263d")
    ax.text(.965, .967, "Known candidate · passive white-box trajectory audit",
            ha="right", va="center", fontsize=10, color="#59697e")

    # Three regions make the information boundary explicit.
    rounded(ax, .025, .055, .305, .85, "", fill="#f5f8fc", edge="#9ab0c7")
    rounded(ax, .353, .055, .257, .85, "", fill="#f8fbfd", edge="#9ab0c7")
    rounded(ax, .633, .055, .342, .85, "", fill="#f7fbf8", edge="#96b6a1")
    ax.text(.045, .865, "SYSTEM  ·  DP-SGD", fontsize=11.5, weight="bold", color="#245071")
    ax.text(.373, .865, "EXPOSURE INTERFACE", fontsize=11.5, weight="bold", color="#245071")
    ax.text(.653, .865, "ADVERSARY  ·  GAUSSPROOF", fontsize=11.5, weight="bold", color="#286a4c")

    # System side: both the dataset-level secret and per-round inclusion are hidden.
    rounded(ax, .047, .727, .261, .098,
            "Hidden membership bit $S$\n$H_0:S=0\\Rightarrow a_t=0$     $H_1:S=1\\Rightarrow a_t\\sim\\mathrm{Bern}(q)$",
            fill="#fbeeee", edge="#c78a90", fontsize=9.1)
    rounded(ax, .047, .601, .261, .092,
            "Sample private minibatch $\\mathcal{B}_t$\nif $a_t=1$: replace one ordinary record by $z$",
            fill="#eaf2fb", edge="#7e9dbd", fontsize=8.9)
    rounded(ax, .047, .475, .261, .092,
            "Clip each per-record gradient\n$g_t(x)=\\operatorname{clip}_C(\\nabla_\\theta\\ell(\\theta_t,x))$",
            fill="#eaf2fb", edge="#7e9dbd", fontsize=9.1)
    rounded(ax, .047, .349, .261, .092,
            "Average and add Gaussian noise\n$y_t=B^{-1}\\!\\sum_{x\\in\\mathcal{B}_t} g_t(x)+\\xi_t$",
            fill="#fff1de", edge="#c99b62", fontsize=9.2)
    rounded(ax, .047, .223, .261, .092,
            "Update model, then repeat\n$\\theta_{t+1}=\\theta_t-\\eta y_t$",
            fill="#eaf2fb", edge="#7e9dbd", fontsize=9.4)
    ax.text(.1775, .138, "$\\xi_t\\sim\\mathcal{N}(0,(\\sigma C/B)^2I)$",
            ha="center", va="center", fontsize=10, color="#6d5030")
    for top, bottom in ((.727, .693), (.601, .567), (.475, .441), (.349, .315)):
        arrow(ax, (.1775, top), (.1775, bottom), linewidth=1.2)

    # Interface: visible transcript, known auxiliary inputs, and hidden variables.
    rounded(ax, .374, .626, .215, .133,
            "Known auxiliary inputs\nrecord $z$, public $D_{\\mathrm{pub}}$\n$B,C,\\sigma,q,\\eta$ and model/loss",
            fill="#e9f3fb", edge="#7e9dbd", fontsize=9.0)
    rounded(ax, .374, .349, .215, .127,
            "Visible transcript\n$\\mathcal{O}_T=\\{(\\theta_t,y_t)\\}_{t=1}^{T}$\nall intermediate states and noisy updates",
            fill="#e5f2e9", edge="#6d9f7d", fontsize=9.0)
    rounded(ax, .374, .165, .215, .118,
            "Never observed\n$\\mathcal{B}_t$, $a_t$, $\\xi_t$\nor other private records",
            fill="#fbeeee", edge="#c78a90", fontsize=9.0)
    arrow(ax, (.308, .395), (.374, .395), color="#286a4c", linewidth=2.1)
    ax.text(.341, .414, "release", ha="center", va="bottom", fontsize=8.4,
            color="#286a4c")

    # Adversary: recompute only candidate-conditioned quantities from visible states.
    rounded(ax, .655, .628, .298, .124,
            "At each checkpoint $\\theta_t$, recompute\n"
            "$h_t=\\operatorname{clip}_C(\\nabla_\\theta\\ell(\\theta_t,z))$\n"
            "$b_t\\approx\\mathbb{E}_{x\\sim D_{\\mathrm{pub}}}g_t(x)$",
            fill="#eeeafa", edge="#9c8cbd", fontsize=9.0)
    rounded(ax, .655, .450, .298, .108,
            "Candidate shift and released residual\n"
            "$s_t=(h_t-b_t)/B$     $r_t=y_t-b_t$",
            fill="#eeeafa", edge="#9c8cbd", fontsize=9.4)
    rounded(ax, .655, .279, .298, .101,
            "Gaussian matched evidence $\\lambda_t$\n"
            "Hidden-inclusion mixture $m_t=\\log(1-q+qe^{\\lambda_t})$",
            fill="#fff3cf", edge="#b9a15c", fontsize=9.2)
    rounded(ax, .655, .109, .298, .101,
            "Accumulate $M_T=m_1+\\cdots+m_T$\n"
            "test $H_1$ versus $H_0$ or rank a known gallery",
            fill="#e5f2e9", edge="#6d9f7d", fontsize=9.3)
    arrow(ax, (.589, .690), (.655, .690), color="#6570a0", linewidth=1.7)
    arrow(ax, (.589, .413), (.655, .504), color="#286a4c", linewidth=1.8)
    arrow(ax, (.804, .628), (.804, .558), linewidth=1.3)
    arrow(ax, (.804, .450), (.804, .380), linewidth=1.3)
    arrow(ax, (.804, .279), (.804, .210), linewidth=1.3)

    ax.text(.5, .012,
            "Output is candidate-conditioned membership evidence; the attack does not reveal the hidden schedule or synthesize an unknown image.",
            ha="center", va="center", fontsize=9.2, color="#536479")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(OUTPUT.with_suffix("." + extension), dpi=260,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make()
