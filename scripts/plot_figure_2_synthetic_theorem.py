from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_synthetic_theorem_experiment import ensure_experiment_outputs


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    out_path = base_dir / "figures" / "figure_2_synthetic_theorem.png"
    results = ensure_experiment_outputs(base_dir)
    summary = results["summary"]
    standard_color = "#a8b0bb"
    dtr_color = "#334155"
    label_color = "#000000"
    mean_edge = "#111827"

    standard = summary[summary["lambda"] == 0.0]
    dtr = summary[summary["lambda"] > 0.0]

    upper = float(max(summary["bound_fd"].max(), summary["volatility"].max()) * 1.08)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    ax.plot([0.0, upper], [0.0, upper], color="#9ca3af", linestyle="--", linewidth=1.2, label=r"$y=x$")
    ax.scatter(
        standard["bound_fd"],
        standard["volatility"],
        s=78,
        c=standard_color,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.62,
        label="standard",
        zorder=2,
    )
    ax.scatter(
        dtr["bound_fd"],
        dtr["volatility"],
        s=86,
        c=dtr_color,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.68,
        marker="^",
        label="DTR",
        zorder=2,
    )

    # Use a monotone display transform so the near-zero cluster is legible in print.
    ax.set_xscale("function", functions=(np.sqrt, np.square))
    ax.set_yscale("function", functions=(np.sqrt, np.square))
    ax.set_xlim(0.0, upper)
    ax.set_ylim(0.0, upper)
    ax.set_xlabel(
        r"estimated derivative-energy bound $\frac{T}{\pi^2}\int_0^T (r'(t))^2 dt$",
        fontsize=16,
    )
    ax.set_ylabel(r"empirical risk volatility $\mathrm{Var}_U(r(U))$")
    ax.grid(color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Synthetic theorem sanity check", fontsize=16)
    ax.legend(frameon=False, loc="upper left")
    for label in ax.get_xticklabels():
        label.set_rotation(28)
        label.set_ha("right")

    ax.text(
        0.97,
        0.08,
        "sqrt display scale on both axes\nfor readability near zero",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color=label_color,
        bbox={
            "facecolor": "white",
            "edgecolor": "#d1d5db",
            "alpha": 1.0,
            "boxstyle": "round,pad=0.35",
        },
    )

    standard_center = (
        float(standard["bound_fd"].mean()),
        float(standard["volatility"].mean()),
    )
    dtr_center = (
        float(dtr["bound_fd"].mean()),
        float(dtr["volatility"].mean()),
    )
    ax.annotate(
        "standard mean",
        xy=standard_center,
        xytext=(-70, 34),
        textcoords="offset points",
        fontsize=12,
        color=label_color,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.94, "pad": 2.0},
        arrowprops={"arrowstyle": "->", "color": mean_edge, "linewidth": 1.4},
        zorder=6,
    )
    ax.annotate(
        "DTR mean",
        xy=dtr_center,
        xytext=(24, 46),
        textcoords="offset points",
        fontsize=12,
        color=label_color,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.94, "pad": 2.0},
        arrowprops={"arrowstyle": "->", "color": mean_edge, "linewidth": 1.4},
        zorder=6,
    )
    ax.scatter(
        *standard_center,
        s=390,
        c="white",
        edgecolor=mean_edge,
        linewidth=2.1,
        marker="o",
        zorder=4,
    )
    ax.scatter(
        *standard_center,
        s=230,
        c=standard_color,
        edgecolor=mean_edge,
        linewidth=1.4,
        marker="o",
        zorder=5,
    )
    ax.scatter(
        *dtr_center,
        s=470,
        c="white",
        edgecolor=dtr_color,
        linewidth=3.0,
        marker="^",
        zorder=5,
    )

    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
