# Benchmark Package

This package now holds only the retained follow-on benchmark used by the paper: the UCI Tetouan City power-consumption study that appears as the second real deployment result in the manuscript.

Contents:

- `tetouan_city_power_consumption/`: cached Tetouan outputs used to summarize the paper result.
- `scripts/run_tetouan_power_benchmark.py`: isolated Tetouan benchmark entry point.
- `scripts/common_temporal_regression.py`: shared temporal-regression utilities for the Tetouan run.
- `data/power_consumption_of_tetouan_city/`: extracted raw dataset used by the Tetouan benchmark.

Reading order:

- Open `tetouan_city_power_consumption/README.md` for the protocol and headline result.
- Use `selected_summary.csv`, `selected_trajectories.csv`, and `summary.json` for the detailed outputs behind the manuscript numbers.

Reproduction:

- `python benchmark_package/scripts/run_tetouan_power_benchmark.py --force`
