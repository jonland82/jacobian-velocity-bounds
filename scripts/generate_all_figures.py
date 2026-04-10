from __future__ import annotations

from pathlib import Path

from run_air_quality_experiment import run_suite as run_air_quality_experiment
from run_synthetic_directional_ablation import run_suite as run_directional_ablation
from run_synthetic_theorem_experiment import run_suite as run_synthetic_experiment
from plot_figure_1_geometry import main as run_figure_1
from plot_figure_2_synthetic_theorem import main as run_figure_2
from plot_figure_3_air_quality_monitoring import main as run_figure_3
from plot_figure_4_directional_ablation import main as run_figure_4


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    run_synthetic_experiment(base_dir / "figures")
    run_directional_ablation(base_dir / "figures")
    run_air_quality_experiment(base_dir)
    run_figure_1()
    run_figure_2()
    run_figure_3()
    run_figure_4()


if __name__ == "__main__":
    main()
