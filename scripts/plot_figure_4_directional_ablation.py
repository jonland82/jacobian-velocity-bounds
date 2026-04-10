from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_synthetic_directional_ablation import ensure_directional_ablation_outputs


METRICS = [
    ("bound_chain", "derivative\nenergy"),
    ("volatility", "risk\nvolatility"),
    ("mean_gain", "directional\ngain"),
    ("terminal_risk", "terminal\nrisk"),
]


def load_summary(base_dir: Path) -> dict[str, object]:
    summary_path = base_dir / "figures" / "synthetic_directional_summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ratio_values(summary_block: dict[str, dict[str, float]], baseline_name: str) -> dict[str, list[float]]:
    baseline = summary_block[baseline_name]
    ratios: dict[str, list[float]] = {}
    for name, metrics in summary_block.items():
        ratios[name] = [metrics[key] / baseline[key] for key, _ in METRICS]
    return ratios


def plot_ratio_panel(
    ax: plt.Axes,
    ratios: dict[str, list[float]],
    series: list[tuple[str, str, dict[str, object]]],
    title: str,
    xlabel: str,
    show_ylabels: bool,
) -> None:
    y_base = np.arange(len(METRICS))
    offsets = np.linspace(-0.22, 0.22, len(series))
    for offset, (key, display, style) in zip(offsets, series):
        ax.scatter(
            ratios[key],
            y_base + offset,
            s=88,
            edgecolors="#111827",
            linewidths=0.7,
            label=display,
            **style,
        )

    ax.axvline(1.0, color="#9ca3af", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_yticks(y_base)
    if show_ylabels:
        ax.set_yticklabels([label for _, label in METRICS])
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    out_path = base_dir / "figures" / "figure_4_directional_ablation.png"
    ensure_directional_ablation_outputs(base_dir)
    summary = load_summary(base_dir)

    comparison = summary["comparison_selected_mean"]
    misspec = summary["misspecification_mean"]
    comparison_ratios = ratio_values(comparison, "standard")
    misspec_ratios = ratio_values(misspec, "aligned")

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

    comparison_series = [
        ("standard", "standard", {"color": "#a8b0bb", "marker": "o"}),
        ("isotropic", "isotropic", {"color": "#6b7280", "marker": "s"}),
        ("dtr", "DTR", {"color": "#334155", "marker": "^"}),
    ]
    plot_ratio_panel(
        axes[0],
        comparison_ratios,
        comparison_series,
        r"Matched $\lambda=0.03$ comparison",
        "metric / standard",
        show_ylabels=True,
    )
    axes[0].set_xlim(0.03, 1.4)
    axes[0].set_xticks([0.05, 0.1, 0.2, 0.5, 1.0])
    axes[0].set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1.0"])
    axes[0].legend(frameon=True, framealpha=0.92, edgecolor="#d1d5db", loc="center", bbox_to_anchor=(0.58, 0.5))
    axes[0].text(
        0.03,
        0.92,
        "lower is better",
        transform=axes[0].transAxes,
        fontsize=9.5,
        color="#4b5563",
    )

    misspec_series = [
        ("aligned", "correct $V$", {"color": "#334155", "marker": "^"}),
        ("rotated20", "20$^\\circ$ rot.", {"color": "#6b7280", "marker": "s"}),
        ("wrong", "wrong $V$", {"color": "#111827", "marker": "X"}),
    ]
    plot_ratio_panel(
        axes[1],
        misspec_ratios,
        misspec_series,
        "Misspecified drift subspace",
        "metric / aligned DTR",
        show_ylabels=False,
    )
    axes[1].set_xlim(0.8, 80.0)
    axes[1].set_xticks([1, 2, 5, 10, 20, 50])
    axes[1].set_xticklabels(["1", "2", "5", "10", "20", "50"])
    axes[1].legend(frameon=True, framealpha=0.92, edgecolor="#d1d5db", loc="center", bbox_to_anchor=(0.38, 0.5))

    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
