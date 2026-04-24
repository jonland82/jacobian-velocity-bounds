from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from run_air_quality_experiment import ensure_air_quality_outputs


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    out_path = base_dir / "figures" / "figure_3_air_quality_monitoring.png"
    results = ensure_air_quality_outputs(base_dir)
    trajectory = results["selected_trajectories_all_seeds"].copy()
    trajectory["block_start"] = pd.to_datetime(trajectory["block_start"])
    trajectory = (
        trajectory.groupby(["method", "model", "block"], as_index=False)
        .agg(
            block_start=("block_start", "first"),
            risk=("risk", "mean"),
            hazard=("hazard", "mean"),
            speed=("speed", "mean"),
        )
        .sort_values(["method", "block"])
        .reset_index(drop=True)
    )
    standard_color = "#a8b0bb"
    dtr_color = "#334155"

    standard = trajectory[trajectory["model"] == "standard"]
    dtr = trajectory[trajectory["model"] == "dtr"]

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)

    axes[0].plot(
        standard["block_start"],
        standard["risk"],
        color=standard_color,
        linewidth=2.5,
        linestyle="-",
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="standard",
    )
    axes[0].plot(
        dtr["block_start"],
        dtr["risk"],
        color=dtr_color,
        linewidth=2.5,
        linestyle="--",
        marker="^",
        markersize=4.6,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="DTR",
    )
    axes[0].grid(color="#e5e7eb", linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].set_ylabel("deployment MSE")
    axes[0].set_xlabel("deployment block start")
    axes[0].set_title("Air Quality mean deployment risk")
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(
        standard["block_start"],
        standard["hazard"],
        color=standard_color,
        linewidth=2.3,
        linestyle="-",
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="standard hazard",
    )
    axes[1].plot(
        dtr["block_start"],
        dtr["hazard"],
        color=dtr_color,
        linewidth=2.3,
        linestyle="--",
        marker="^",
        markersize=4.4,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="DTR hazard",
    )
    speed_axis = axes[1].twinx()
    speed_band = speed_axis.fill_between(
        standard["block_start"],
        0.0,
        standard["speed"],
        color="#d1d5db",
        alpha=0.22,
        label="drift magnitude",
    )
    speed_band.set_hatch("//")
    speed_band.set_edgecolor("#9ca3af")
    speed_band.set_linewidth(0.0)

    axes[1].grid(color="#e5e7eb", linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].set_ylabel(r"$h_t = s_t^2 g_t$")
    speed_axis.set_ylabel(r"block drift $s_t$")
    axes[1].set_xlabel("deployment block start")
    axes[1].set_title("Mean hazard score under block drift")
    axes[1].legend(frameon=False, loc="upper right")

    locator = mdates.MonthLocator(interval=2)
    formatter = mdates.DateFormatter("%b %Y")
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        for label in ax.get_xticklabels():
            label.set_rotation(18)
            label.set_ha("right")
    for axis in [*axes, speed_axis]:
        axis.tick_params(length=4.5, width=0.9)

    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
