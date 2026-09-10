# Benchmark Package

This package contains the paper's central UCI gas-sensor drift benchmark and the complementary UCI Tetouan City power-consumption benchmark.

Contents:

- `gas_sensor_array_drift/`: cached gas-sensor dataset archive and matched-seed outputs for the predeployment rank sweep, rank-4 random-subspace controls, anisotropic hybrid sweep, and classwise analysis.
- `tetouan_city_power_consumption/`: cached Tetouan outputs used to summarize the paper result.
- `scripts/run_gas_sensor_dtr_benchmark.py`: gas-sensor benchmark utilities and endpoint sweep.
- `scripts/run_gas_sensor_rank_sweep.py`: strict predeployment rank-selection experiment.
- `scripts/run_gas_sensor_rank4_controls.py`: ten ambient-random rank-4 controls.
- `scripts/run_gas_sensor_hybrid_sweep.py`: two-parameter anisotropic DTR sweep.
- `scripts/analyze_gas_sensor_*.py`: paired, aggregate, and classwise gas-sensor reports.
- `scripts/run_tetouan_power_benchmark.py`: isolated Tetouan benchmark entry point.
- `scripts/common_temporal_regression.py`: shared temporal-regression utilities for the Tetouan run.
- `data/power_consumption_of_tetouan_city/`: extracted raw dataset used by the Tetouan benchmark.
- `dtr_benchmark_suitability_notes.md`: benchmark-selection notes retained for transparency.

Reading order:

- Open `gas_sensor_array_drift/results/` for the gas-sensor summaries underlying the main field study.
- Open `tetouan_city_power_consumption/README.md` for the complementary regression protocol and headline result.
- Use `selected_summary.csv`, `selected_trajectories.csv`, `selected_trajectories_all_seeds.csv`, and `summary.json` for the detailed outputs behind the manuscript numbers.
- Read `dtr_benchmark_suitability_notes.md` only as benchmark-search context; it is not an additional manuscript experiment.
- The cross-dataset real-deployment reports live under the repository-level `figures/` directory: `real_deployment_summary_stats.csv`, `real_deployment_paired_comparisons.csv`, `real_deployment_conservative_gain_summary.csv`, `real_deployment_conservative_gain_paired.csv`, `air_quality_dtr_lambda_path.csv`, the Air Quality subspace-ablation CSVs, `monitoring_blockwise_selected_dtr.csv`, `monitoring_volatility_ablation.csv`, and `monitoring_volatility_bootstrap.csv`.

Reproduction:

- `python benchmark_package/scripts/run_gas_sensor_rank_sweep.py`
- `python benchmark_package/scripts/run_gas_sensor_rank4_controls.py`
- `python benchmark_package/scripts/run_gas_sensor_hybrid_sweep.py --force`
- `python benchmark_package/scripts/run_tetouan_power_benchmark.py --force`
