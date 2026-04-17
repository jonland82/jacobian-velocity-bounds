from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common_temporal_regression import BenchmarkConfig, run_temporal_regression_benchmark


def load_tetouan_power(base_dir: Path) -> pd.DataFrame:
    csv_path = (
        base_dir
        / "benchmark_package"
        / "data"
        / "power_consumption_of_tetouan_city"
        / "Tetuan City power consumption.csv"
    )
    frame = pd.read_csv(csv_path)
    frame["timestamp"] = pd.to_datetime(frame["DateTime"], errors="coerce")
    return frame


def run_suite(base_dir: Path, force: bool = False) -> dict[str, pd.DataFrame]:
    frame = load_tetouan_power(base_dir)
    config = BenchmarkConfig(
        name="tetouan_zone1_power",
        target_col="Zone 1 Power Consumption",
        timestamp_col="timestamp",
        numeric_cols=[
            "Temperature",
            "Humidity",
            "Wind Speed",
            "general diffuse flows",
            "diffuse flows",
        ],
        categorical_cols=[],
        train_end="2017-05-01",
        val_end="2017-07-01",
        block_freq="M",
        description=(
            "Power Consumption of Tetouan City. Predict Zone 1 power consumption from weather, "
            "solar diffuse-flow channels, and calendar covariates under a chronological frozen-model split."
        ),
        epochs=30,
        batch_size=512,
        hidden_width=64,
        subspace_dim=2,
    )
    output_dir = base_dir / "benchmark_package" / "tetouan_city_power_consumption"
    return run_temporal_regression_benchmark(frame, config, output_dir, force=force)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    results = run_suite(args.base_dir, force=args.force)
    print(
        "Wrote Tetouan power benchmark outputs to "
        f"{args.base_dir / 'benchmark_package' / 'tetouan_city_power_consumption'} "
        f"({len(results['sweep_summary'])} sweep rows)."
    )


if __name__ == "__main__":
    main()
