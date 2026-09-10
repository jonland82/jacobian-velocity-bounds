from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch


BASE_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPTS = BASE_DIR / "benchmark_package" / "scripts"
if str(BENCHMARK_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SCRIPTS))

import run_air_quality_experiment as air
from common_temporal_regression import (
    LAMBDA_GRID,
    SEEDS,
    PreparedRegressionSplit,
    evaluate_model as evaluate_tetouan_model,
    prepare_regression_split,
    train_model as train_tetouan_model,
)
from run_tetouan_power_benchmark import benchmark_config, load_tetouan_power


torch.set_num_threads(1)
RANDOM_SEED = 2026


def _basis_from_blocks(values: np.ndarray, block_ids: np.ndarray, k: int) -> np.ndarray:
    labels = list(dict.fromkeys(block_ids.tolist()))
    means = np.asarray([values[block_ids == label].mean(axis=0) for label in labels])
    diffs = np.diff(means, axis=0)
    if len(diffs) < k:
        raise ValueError(f"Need at least {k} block differences, found {len(diffs)}")
    _, _, vh = np.linalg.svd(diffs, full_matrices=False)
    return vh[:k].T.astype(np.float32)


def _random_basis(dim: int, k: int, projection: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    draw = rng.normal(size=(dim, k))
    if projection is not None:
        draw = projection @ draw
    q, _ = np.linalg.qr(draw)
    return q[:, :k].astype(np.float32)


def _principal_angle_degrees(basis: np.ndarray, oracle: np.ndarray) -> float:
    singular_values = np.linalg.svd(basis.T @ oracle, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return float(np.degrees(np.arccos(singular_values.min())))


def _explained_drift_ratio(diffs: np.ndarray, basis: np.ndarray) -> float:
    denominator = float(np.square(diffs).sum())
    if denominator == 0.0:
        return 0.0
    return float(np.square(diffs @ basis).sum() / denominator)


def _air_bases(
    frame: pd.DataFrame, split: air.SplitData
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    start = frame["timestamp"].min()
    val_end = start + pd.Timedelta(weeks=air.TRAIN_WEEKS + air.VAL_WEEKS)
    pre_frame = frame[frame["timestamp"] < val_end]
    pre_x = np.concatenate([split.train_x, split.val_x], axis=0)
    if len(pre_frame) != len(pre_x):
        raise RuntimeError("Air Quality predeployment timestamps and features are misaligned")
    pre_block_ids = (
        (pre_frame["timestamp"] - start) / pd.Timedelta(days=air.BLOCK_DAYS)
    ).astype(int).to_numpy()

    sensor_idx = air.sensor_indices()
    train_sensor = split.train_x[:, sensor_idx]
    pre_sensor = pre_x[:, sensor_idx]
    deploy_sensor = split.deploy_x[:, sensor_idx]
    pre_orthogonal = air.target_orthogonalize(pre_sensor, train_sensor, split.train_y)
    deploy_orthogonal = air.target_orthogonalize(deploy_sensor, train_sensor, split.train_y)

    prospective_local = _basis_from_blocks(pre_orthogonal, pre_block_ids, air.SUBSPACE_DIM)
    deploy_means = np.asarray(
        [
            deploy_orthogonal[split.block_index == block].mean(axis=0)
            for block in range(len(split.block_dates))
        ]
    )
    deploy_diffs_local = np.diff(deploy_means, axis=0)

    target_direction, *_ = np.linalg.lstsq(train_sensor, split.train_y, rcond=None)
    target_direction = target_direction / np.linalg.norm(target_direction)
    projection = np.eye(len(sensor_idx)) - np.outer(target_direction, target_direction)
    random_local = _random_basis(len(sensor_idx), air.SUBSPACE_DIM, projection=projection)

    def embed(local_basis: np.ndarray) -> np.ndarray:
        full = np.zeros((len(air.FEATURE_COLS), air.SUBSPACE_DIM), dtype=np.float32)
        full[np.asarray(sensor_idx), :] = local_basis
        return full

    bases = {
        "oracle_deployment": split.drift_basis.astype(np.float32),
        "prospective_predeployment": embed(prospective_local),
        "random_control": embed(random_local),
    }
    deploy_diffs = np.zeros((len(deploy_diffs_local), len(air.FEATURE_COLS)), dtype=np.float32)
    deploy_diffs[:, np.asarray(sensor_idx)] = deploy_diffs_local
    return bases, deploy_diffs


def _tetouan_bases(
    frame: pd.DataFrame, split: PreparedRegressionSplit
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    config = split.config
    data = frame.copy()
    data[config.timestamp_col] = pd.to_datetime(data[config.timestamp_col])
    data = data.sort_values(config.timestamp_col).reset_index(drop=True)
    data = data.dropna(subset=[config.target_col]).reset_index(drop=True)
    timestamps = data[config.timestamp_col]
    pre_mask = timestamps < pd.Timestamp(config.val_end)
    pre_x = np.concatenate([split.train_x, split.val_x], axis=0)
    pre_timestamps = timestamps.loc[pre_mask]
    if len(pre_timestamps) != len(pre_x):
        raise RuntimeError("Tetouan predeployment timestamps and features are misaligned")
    pre_block_ids = pre_timestamps.dt.to_period(config.block_freq).astype(str).to_numpy()
    prospective = _basis_from_blocks(pre_x, pre_block_ids, config.subspace_dim)
    random = _random_basis(pre_x.shape[1], config.subspace_dim)

    deploy_means = np.asarray(
        [
            split.deploy_x[split.deploy_block_ids == label].mean(axis=0)
            for label in split.deploy_block_labels
        ]
    )
    deploy_diffs = np.diff(deploy_means, axis=0)
    bases = {
        "oracle_deployment": split.drift_basis.astype(np.float32),
        "prospective_predeployment": prospective,
        "random_control": random,
    }
    return bases, deploy_diffs


def _run_air_variant(
    split: air.SplitData, basis: np.ndarray, variant: str
) -> pd.DataFrame:
    variant_split = replace(split, drift_basis=basis)
    rows: list[dict[str, float | int | str]] = []
    for penalty_lambda in air.REGULARIZATION_LAMBDAS:
        for seed in air.SWEEP_SEEDS:
            model = air.train_model(
                variant_split, seed=seed, method="dtr", penalty_lambda=penalty_lambda
            )
            _, summary = air.evaluate_model(
                model,
                split=variant_split,
                lambda_value=penalty_lambda,
                seed=seed,
                method="dtr",
            )
            rows.append({"dataset": "Air Quality", "basis": variant, **summary})
    return pd.DataFrame(rows)


def _run_tetouan_variant(
    split: PreparedRegressionSplit, basis: np.ndarray, variant: str
) -> pd.DataFrame:
    variant_split = replace(split, drift_basis=basis)
    rows: list[dict[str, float | int | str]] = []
    for penalty_lambda in LAMBDA_GRID:
        for seed in SEEDS:
            model = train_tetouan_model(
                variant_split, seed=seed, method="dtr", penalty_lambda=penalty_lambda
            )
            _, summary = evaluate_tetouan_model(
                model,
                split=variant_split,
                seed=seed,
                method="dtr",
                penalty_lambda=penalty_lambda,
            )
            rows.append({"basis": variant, **summary, "dataset": "Tetouan"})
    return pd.DataFrame(rows)


def _select_lambdas(sweep: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[pd.DataFrame] = []
    lambda_rows: list[dict[str, float | str]] = []
    for (dataset, basis), subset in sweep.groupby(["dataset", "basis"], sort=False):
        if basis == "standard":
            chosen_lambda = 0.0
        else:
            grouped = (
                subset.groupby("lambda", as_index=False)[["val_mse", "val_directional_gain"]]
                .mean()
                .sort_values(["val_mse", "val_directional_gain", "lambda"])
            )
            chosen_lambda = float(grouped.iloc[0]["lambda"])
        chosen = subset[np.isclose(subset["lambda"], chosen_lambda)].copy()
        selected_rows.append(chosen)
        lambda_rows.append(
            {"dataset": dataset, "basis": basis, "selected_lambda": chosen_lambda}
        )
    return pd.concat(selected_rows, ignore_index=True), pd.DataFrame(lambda_rows)


def _summarize_selected(
    selected: pd.DataFrame,
    selected_lambdas: pd.DataFrame,
    geometry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        selected.groupby(["dataset", "basis"], as_index=False)
        .agg(
            deploy_mse_mean=("deploy_mse", "mean"),
            deploy_mse_std=("deploy_mse", "std"),
            volatility_mean=("volatility", "mean"),
            volatility_std=("volatility", "std"),
            terminal_risk_mean=("terminal_risk", "mean"),
            mean_gain_mean=("mean_gain", "mean"),
        )
        .merge(selected_lambdas, on=["dataset", "basis"], how="left")
        .merge(geometry, on=["dataset", "basis"], how="left")
    )

    paired_rows: list[dict[str, float | int | str]] = []
    for dataset, dataset_rows in selected.groupby("dataset"):
        standard = dataset_rows[dataset_rows["basis"] == "standard"]
        for basis in ["oracle_deployment", "prospective_predeployment", "random_control"]:
            candidate = dataset_rows[dataset_rows["basis"] == basis]
            paired = candidate.merge(
                standard[["seed", "deploy_mse", "volatility"]],
                on="seed",
                suffixes=("_candidate", "_standard"),
            )
            for row in paired.itertuples(index=False):
                paired_rows.append(
                    {
                        "dataset": dataset,
                        "basis": basis,
                        "seed": int(row.seed),
                        "deploy_mse_difference": float(
                            row.deploy_mse_candidate - row.deploy_mse_standard
                        ),
                        "volatility_difference": float(
                            row.volatility_candidate - row.volatility_standard
                        ),
                    }
                )
            mask = (summary["dataset"] == dataset) & (summary["basis"] == basis)
            summary.loc[mask, "mse_wins_vs_standard"] = int(
                (paired["deploy_mse_candidate"] < paired["deploy_mse_standard"]).sum()
            )
            summary.loc[mask, "volatility_wins_vs_standard"] = int(
                (paired["volatility_candidate"] < paired["volatility_standard"]).sum()
            )
    return summary, pd.DataFrame(paired_rows)


def run_suite(
    base_dir: Path, force: bool = False, resummarize: bool = False
) -> dict[str, pd.DataFrame]:
    figures_dir = base_dir / "figures"
    sweep_path = figures_dir / "prospective_subspace_sweep.csv"
    selected_path = figures_dir / "prospective_subspace_selected.csv"
    paired_path = figures_dir / "prospective_subspace_paired.csv"
    lambdas_path = figures_dir / "prospective_subspace_selected_lambdas.csv"
    json_path = figures_dir / "prospective_subspace_summary.json"
    required = [sweep_path, selected_path, paired_path, lambdas_path, json_path]
    if not force and not resummarize and all(path.exists() for path in required):
        return {
            "sweep": pd.read_csv(sweep_path),
            "selected": pd.read_csv(selected_path),
            "paired": pd.read_csv(paired_path),
            "selected_lambdas": pd.read_csv(lambdas_path),
        }

    air_frame = air.load_air_quality(base_dir / "data" / "air_quality.csv")
    air_split = air.build_splits(air_frame)
    air_bases, air_deploy_diffs = _air_bases(air_frame, air_split)

    tetouan_frame = load_tetouan_power(base_dir)
    tetouan_split = prepare_regression_split(tetouan_frame, benchmark_config())
    tetouan_bases, tetouan_deploy_diffs = _tetouan_bases(tetouan_frame, tetouan_split)

    geometry_rows: list[dict[str, float | str]] = []
    for dataset, bases, diffs in [
        ("Air Quality", air_bases, air_deploy_diffs),
        ("Tetouan", tetouan_bases, tetouan_deploy_diffs),
    ]:
        oracle = bases["oracle_deployment"]
        geometry_rows.append(
            {
                "dataset": dataset,
                "basis": "standard",
                "max_principal_angle_deg": np.nan,
                "deploy_drift_energy_ratio": np.nan,
            }
        )
        for name, basis in bases.items():
            geometry_rows.append(
                {
                    "dataset": dataset,
                    "basis": name,
                    "max_principal_angle_deg": _principal_angle_degrees(basis, oracle),
                    "deploy_drift_energy_ratio": _explained_drift_ratio(diffs, basis),
                }
            )
    geometry = pd.DataFrame(geometry_rows)

    if resummarize:
        sweep = pd.read_csv(sweep_path)
        sweep["dataset"] = sweep["dataset"].replace({"tetouan_zone1_power": "Tetouan"})
    else:
        air_existing = air.run_suite(base_dir, force=False)["summary"]
        air_standard = air_existing[air_existing["method"] == "standard"].copy()
        air_standard["dataset"] = "Air Quality"
        air_standard["basis"] = "standard"
        air_oracle = air_existing[air_existing["method"] == "dtr"].copy()
        air_oracle["dataset"] = "Air Quality"
        air_oracle["basis"] = "oracle_deployment"

        tetouan_dir = base_dir / "benchmark_package" / "tetouan_city_power_consumption"
        tetouan_existing = pd.read_csv(tetouan_dir / "sweep_summary.csv")
        tetouan_standard = tetouan_existing[tetouan_existing["method"] == "standard"].copy()
        tetouan_standard["dataset"] = "Tetouan"
        tetouan_standard["basis"] = "standard"
        tetouan_oracle = tetouan_existing[tetouan_existing["method"] == "dtr"].copy()
        tetouan_oracle["dataset"] = "Tetouan"
        tetouan_oracle["basis"] = "oracle_deployment"

        frames = [air_standard, air_oracle, tetouan_standard, tetouan_oracle]
        for name in ["prospective_predeployment", "random_control"]:
            frames.append(_run_air_variant(air_split, air_bases[name], name))
            frames.append(_run_tetouan_variant(tetouan_split, tetouan_bases[name], name))

        common_columns = sorted(set.intersection(*(set(frame.columns) for frame in frames)))
        sweep = pd.concat([frame[common_columns] for frame in frames], ignore_index=True)
    sweep = sweep.sort_values(["dataset", "basis", "lambda", "seed"]).reset_index(drop=True)
    selected_seed_rows, selected_lambdas = _select_lambdas(sweep)
    selected, paired = _summarize_selected(selected_seed_rows, selected_lambdas, geometry)
    selected = selected.sort_values(["dataset", "basis"]).reset_index(drop=True)

    sweep.to_csv(sweep_path, index=False)
    selected.to_csv(selected_path, index=False)
    paired.to_csv(paired_path, index=False)
    selected_lambdas.to_csv(lambdas_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selection_rule": "Mean validation MSE across matched seeds; directional energy breaks ties.",
                "random_seed": RANDOM_SEED,
                "selected": selected.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )
    return {
        "sweep": sweep,
        "selected": selected,
        "paired": paired,
        "selected_lambdas": selected_lambdas,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resummarize", action="store_true")
    args = parser.parse_args()
    results = run_suite(args.base_dir, force=args.force, resummarize=args.resummarize)
    print(results["selected"].to_string(index=False))


if __name__ == "__main__":
    main()
