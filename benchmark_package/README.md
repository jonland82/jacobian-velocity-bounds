# Benchmark Package

This package holds the retained follow-on benchmark used by the paper, plus benchmark-search notes kept for transparency. The retained executable benchmark is the UCI Tetouan City power-consumption study that appears as the second real deployment result in the manuscript.

Contents:

- `tetouan_city_power_consumption/`: cached Tetouan outputs used to summarize the paper result.
- `scripts/run_tetouan_power_benchmark.py`: isolated Tetouan benchmark entry point.
- `scripts/common_temporal_regression.py`: shared temporal-regression utilities for the Tetouan run.
- `data/power_consumption_of_tetouan_city/`: extracted raw dataset used by the Tetouan benchmark.
- `dtr_benchmark_suitability_notes.md`: benchmark-selection notes explaining why Air Quality and Tetouan were retained and why several exploratory alternatives were not used in the manuscript.
- `uci_gas_sensor_benchmark_report.md`: historical diagnostic report for an exploratory gas-sensor benchmark that was not retained as a main paper result.

Reading order:

- Open `tetouan_city_power_consumption/README.md` for the protocol and headline result.
- Use `selected_summary.csv`, `selected_trajectories.csv`, `selected_trajectories_all_seeds.csv`, and `summary.json` for the detailed outputs behind the manuscript numbers.
- Read `dtr_benchmark_suitability_notes.md` only as benchmark-search context; it is not an additional manuscript experiment.
- The cross-dataset real-deployment reports live under the repository-level `figures/` directory: `real_deployment_summary_stats.csv`, `real_deployment_paired_comparisons.csv`, `real_deployment_conservative_gain_summary.csv`, `real_deployment_conservative_gain_paired.csv`, `air_quality_dtr_lambda_path.csv`, the Air Quality subspace-ablation CSVs, `monitoring_blockwise_selected_dtr.csv`, `monitoring_volatility_ablation.csv`, and `monitoring_volatility_bootstrap.csv`.

Reproduction:

- `python benchmark_package/scripts/run_tetouan_power_benchmark.py --force`
