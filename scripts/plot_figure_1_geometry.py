from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw_panel(ax: plt.Axes, weights: tuple[float, float], title: str, scale_word: str) -> None:
    x1 = np.linspace(-2.6, 2.6, 240)
    x2 = np.linspace(-2.6, 2.6, 240)
    xx, yy = np.meshgrid(x1, x2)
    zz = weights[0] * xx + weights[1] * yy

    levels = np.linspace(-4.0, 4.0, 11)
    ax.contour(xx, yy, zz, levels=levels, cmap="Blues", linewidths=1.1)
    ax.plot([0.2, 0.2], [-2.2, 2.2], color="#4b5563", linestyle="--", linewidth=1.4)
    ax.arrow(0.2, -1.8, 0.0, 2.8, width=0.03, head_width=0.18, head_length=0.22, color="#374151")
    ax.scatter([0.2], [-1.8], s=28, color="#111827")
    ax.text(
        0.48,
        1.15,
        "drift path",
        color="#374151",
        fontsize=11,
        zorder=5,
        bbox={
            "facecolor": "white",
            "edgecolor": "#9ca3af",
            "linewidth": 1.1,
            "alpha": 1.0,
            "boxstyle": "round,pad=0.22",
        },
    )
    ax.text(
        -2.33,
        2.17,
        scale_word,
        color="#1f2937",
        fontsize=11,
        zorder=5,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9, "pad": 2.2},
    )
    ax.text(
        -2.02,
        1.62,
        r"$\left| J_f(x)\,v \right|$",
        color="#111827",
        fontsize=17,
        zorder=6,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.94, "pad": 1.8},
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    out_path = base_dir / "figures" / "figure_1_geometry.png"
    pdf_path = out_path.with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    draw_panel(
        axes[0],
        weights=(0.35, 1.55),
        title="Steep Along Drift",
        scale_word="large",
    )
    draw_panel(
        axes[1],
        weights=(1.55, 0.18),
        title="Flat Along Drift",
        scale_word="small",
    )
    fig.suptitle("Dynamic shift matters through the directional derivative along the drift path", fontsize=12)
    fig.savefig(out_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
