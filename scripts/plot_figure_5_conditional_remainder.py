from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    figures_dir = base_dir / "figures"
    summary = pd.read_csv(figures_dir / "conditional_remainder_summary.csv")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    colors = {"standard": "#8d99ae", "dtr": "#1f4e79"}
    labels = {"standard": "standard", "dtr": "DTR"}

    for method in ["standard", "dtr"]:
        subset = summary[summary["method"] == method].sort_values("conditional_slope")
        x = subset["conditional_slope"].to_numpy()
        mean = subset["volatility_mean"].to_numpy()
        sem = subset["volatility_std"].to_numpy() / np.sqrt(100.0)
        axes[0].plot(x, mean, marker="o", color=colors[method], label=labels[method])
        axes[0].fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, color=colors[method], alpha=0.16)

    axes[0].set_title("Conditional variation limits tangent control")
    axes[0].set_xlabel(r"conditional slope $c$")
    axes[0].set_ylabel(r"risk volatility")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    dtr = summary[summary["method"] == "dtr"].sort_values("conditional_slope")
    x = dtr["conditional_slope"].to_numpy()
    axes[1].plot(x, dtr["volatility_mean"], marker="o", color="#222222", label="observed volatility")
    axes[1].plot(x, dtr["jacobian_bound_mean"], marker="s", linestyle="--", color="#6b7280", label="Jacobian-only RHS")
    axes[1].plot(x, dtr["residual_bound_mean"], marker="^", linestyle="-.", color="#1f4e79", label="Jacobian + remainder RHS")
    axes[1].set_title("The remainder restores valid control")
    axes[1].set_xlabel(r"conditional slope $c$")
    axes[1].set_ylabel("time-domain quantity")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout(w_pad=1.8)
    fig.savefig(figures_dir / "figure_5_conditional_remainder.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "figure_5_conditional_remainder.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
