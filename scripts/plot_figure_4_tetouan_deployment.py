from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BENCHMARK_SCRIPTS = Path(__file__).resolve().parents[1] / "benchmark_package" / "scripts"
if str(BENCHMARK_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SCRIPTS))

from run_tetouan_power_benchmark import run_suite as run_tetouan_benchmark


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    out_path = base_dir / "figures" / "figure_4_tetouan_deployment.png"
    pdf_path = out_path.with_suffix(".pdf")
    results = run_tetouan_benchmark(base_dir, force=False)
    trajectory = results["selected_trajectories"].copy()
    trajectory["block_start"] = pd.to_datetime(trajectory["block"] + "-01")
    trajectory["risk_e8"] = trajectory["risk"] / 1e8

    standard_color = "#a8b0bb"
    isotropic_color = "#6b7280"
    dtr_color = "#334155"

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.35), constrained_layout=True)

    for method, color, marker, label in [
        ("standard", standard_color, "o", "standard"),
        ("isotropic", isotropic_color, "s", "isotropic"),
        ("dtr", dtr_color, "^", "DTR"),
    ]:
        subset = trajectory[trajectory["method"] == method]
        ax.plot(
            subset["block_start"],
            subset["risk_e8"],
            color=color,
            linewidth=2.0,
            linestyle="--" if method == "dtr" else "-",
            marker=marker,
            markersize=4.1,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=label,
        )

    ax.grid(color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(r"deployment MSE ($\times 10^8$)")
    ax.set_xlabel("deployment month")
    ax.set_title("Tetouan deployment risk")
    ax.legend(frameon=False, loc="upper right", ncol=1, handlelength=2.2)

    tick_positions = pd.to_datetime(
        ["2017-07-01", "2017-08-01", "2017-09-01", "2017-10-01", "2017-11-01", "2017-12-01"]
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.tick_params(length=4.2, width=0.9)

    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
