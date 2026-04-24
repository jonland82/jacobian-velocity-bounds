from __future__ import annotations

import argparse
from pathlib import Path

from run_air_quality_subspace_ablation import run_suite as run_air_quality_subspace_ablation
from run_real_deployment_reporting import run_suite as run_real_deployment_reporting
from run_synthetic_directional_ablation import run_suite as run_directional_ablation
from run_synthetic_theorem_experiment import run_suite as run_synthetic_experiment
from plot_figure_1_geometry import main as run_figure_1
from plot_figure_2_synthetic_theorem import main as run_figure_2
from plot_figure_3_air_quality_monitoring import main as run_figure_3
from plot_figure_4_directional_ablation import main as run_figure_4
from plot_figure_5_tetouan_deployment import main as run_figure_5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    run_synthetic_experiment(base_dir / "figures", force=args.force)
    run_directional_ablation(base_dir / "figures", force=args.force)
    run_real_deployment_reporting(base_dir, force=args.force)
    run_air_quality_subspace_ablation(base_dir, force=args.force)
    run_figure_1()
    run_figure_2()
    run_figure_3()
    run_figure_4()
    run_figure_5()


if __name__ == "__main__":
    main()
