from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from run_synthetic_theorem_experiment import ensure_experiment_outputs


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    out_path = base_dir / "figures" / "figure_2_synthetic_theorem.png"
    pdf_path = out_path.with_suffix(".pdf")
    results = ensure_experiment_outputs(base_dir)
    summary = results["summary"]
    standard_color = "#a8b0bb"
    dtr_color = "#334155"
    mean_edge = "#111827"

    standard = summary[summary["lambda"] == 0.0]
    dtr = summary[summary["lambda"] > 0.0]

    x_lower = float(summary["bound_fd"].min() * 0.75)
    x_upper = float(summary["bound_fd"].max() * 1.15)
    y_lower = float(summary["volatility"].min() * 0.70)
    y_upper = float(summary["volatility"].max() * 1.25)

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 15,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    diagonal_lower = max(x_lower, y_lower)
    diagonal_upper = min(x_upper, y_upper)
    ax.plot(
        [diagonal_lower, diagonal_upper],
        [diagonal_lower, diagonal_upper],
        color="#9ca3af",
        linestyle="--",
        linewidth=1.2,
        label=r"$y=x$",
    )
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

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)
    ax.set_xlabel("derivative-energy bound", fontsize=17)
    ax.set_ylabel("risk volatility")
    ax.grid(which="both", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Temporal volatility vs. derivative energy", fontsize=17)
    ax.legend(frameon=False, loc="upper left")

    standard_center = (
        float(standard["bound_fd"].mean()),
        float(standard["volatility"].mean()),
    )
    dtr_center = (
        float(dtr["bound_fd"].mean()),
        float(dtr["volatility"].mean()),
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
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
