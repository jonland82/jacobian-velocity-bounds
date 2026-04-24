from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from run_air_quality_experiment import (
    FEATURE_COLS,
    REGULARIZATION_LAMBDAS,
    SELECTED_SEED,
    SWEEP_SEEDS,
    SplitData,
    build_splits,
    evaluate_model,
    load_air_quality,
    train_model,
)


SENSOR_PREFIX = "PT08."
GAIN_TARGET = 0.05
METRICS = ["deploy_mse", "volatility", "mean_gain", "terminal_risk"]


def _sensor_indices() -> list[int]:
    return [idx for idx, name in enumerate(FEATURE_COLS) if name.startswith(SENSOR_PREFIX)]


def _weather_indices() -> list[int]:
    sensor_idx = set(_sensor_indices())
    return [idx for idx in range(len(FEATURE_COLS)) if idx not in sensor_idx]


def _basis_from_block_values(
    values: np.ndarray,
    block_index: np.ndarray,
    feature_indices: list[int],
    k: int,
) -> np.ndarray:
    num_blocks = int(block_index.max()) + 1
    block_means = np.asarray(
        [values[block_index == block_id].mean(axis=0) for block_id in range(num_blocks)],
        dtype=np.float32,
    )
    block_diffs = np.diff(block_means, axis=0)
    _, _, right_vectors = np.linalg.svd(block_diffs, full_matrices=False)
    local_basis = right_vectors[:k].T.astype(np.float32)
    full_basis = np.zeros((len(FEATURE_COLS), k), dtype=np.float32)
    full_basis[np.asarray(feature_indices), :] = local_basis
    return full_basis


def _sensor_residuals(split: SplitData, values: np.ndarray) -> np.ndarray:
    sensor_idx = _sensor_indices()
    weather_idx = _weather_indices()
    predeploy_x = np.vstack([split.train_x, split.val_x])
    design = np.column_stack(
        [
            predeploy_x[:, weather_idx],
            np.ones(len(predeploy_x), dtype=np.float32),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, predeploy_x[:, sensor_idx], rcond=None)
    values_design = np.column_stack(
        [
            values[:, weather_idx],
            np.ones(len(values), dtype=np.float32),
        ]
    )
    return (values[:, sensor_idx] - values_design @ coefficients).astype(np.float32)


def _target_orthogonalized(values: np.ndarray, train_values: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    direction, *_ = np.linalg.lstsq(train_values, train_y, rcond=None)
    direction = direction.astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return values.astype(np.float32)
    unit_direction = direction / norm
    projection = np.outer(values @ unit_direction, unit_direction)
    return (values - projection).astype(np.float32)


def _subspace_variants(split: SplitData) -> dict[str, np.ndarray]:
    sensor_idx = _sensor_indices()
    all_idx = list(range(len(FEATURE_COLS)))
    sensor_deploy = split.deploy_x[:, sensor_idx]
    sensor_train = split.train_x[:, sensor_idx]
    sensor_residual_deploy = _sensor_residuals(split, split.deploy_x)
    sensor_residual_train = _sensor_residuals(split, split.train_x)
    sensor_target_orth_deploy = _target_orthogonalized(sensor_deploy, sensor_train, split.train_y)
    sensor_residual_target_orth_deploy = _target_orthogonalized(
        sensor_residual_deploy,
        sensor_residual_train,
        split.train_y,
    )
    return {
        "all_features_k2": _basis_from_block_values(split.deploy_x, split.block_index, all_idx, k=2),
        "sensor_only_k1": _basis_from_block_values(sensor_deploy, split.block_index, sensor_idx, k=1),
        "sensor_only_k2": _basis_from_block_values(sensor_deploy, split.block_index, sensor_idx, k=2),
        "sensor_target_orth_k1": _basis_from_block_values(
            sensor_target_orth_deploy,
            split.block_index,
            sensor_idx,
            k=1,
        ),
        "sensor_target_orth_k2": _basis_from_block_values(
            sensor_target_orth_deploy,
            split.block_index,
            sensor_idx,
            k=2,
        ),
        "sensor_residual_k1": _basis_from_block_values(
            sensor_residual_deploy,
            split.block_index,
            sensor_idx,
            k=1,
        ),
        "sensor_residual_k2": _basis_from_block_values(
            sensor_residual_deploy,
            split.block_index,
            sensor_idx,
            k=2,
        ),
        "sensor_residual_k3": _basis_from_block_values(
            sensor_residual_deploy,
            split.block_index,
            sensor_idx,
            k=3,
        ),
        "sensor_residual_target_orth_k2": _basis_from_block_values(
            sensor_residual_target_orth_deploy,
            split.block_index,
            sensor_idx,
            k=2,
        ),
        "sensor_residual_target_orth_k3": _basis_from_block_values(
            sensor_residual_target_orth_deploy,
            split.block_index,
            sensor_idx,
            k=3,
        ),
    }


def _with_basis(split: SplitData, basis: np.ndarray) -> SplitData:
    return replace(split, drift_basis=basis.astype(np.float32))


def _summarize_selected(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[pd.DataFrame] = []
    selection_records: list[dict[str, object]] = []
    for subspace, subset in summary.groupby("subspace"):
        standard = subset[subset["method"] == "standard"].copy()
        dtr = subset[subset["method"] == "dtr"].copy()
        standard_val_mse = float(standard["val_mse"].mean())
        standard_val_gain = float(standard["val_directional_gain"].mean())
        grouped = (
            dtr.groupby("lambda", as_index=False)[["val_mse", "val_directional_gain"]]
            .mean()
            .sort_values("lambda")
            .reset_index(drop=True)
        )

        selectors = []
        mse_best = grouped.sort_values(["val_mse", "val_directional_gain", "lambda"]).iloc[0]
        selectors.append(("validation_mse", float(mse_best["lambda"])))

        grouped["val_gain_reduction"] = 1.0 - grouped["val_directional_gain"] / standard_val_gain
        eligible = grouped[
            (grouped["val_mse"] <= standard_val_mse)
            & (grouped["val_gain_reduction"] >= GAIN_TARGET)
        ]
        if eligible.empty:
            selectors.append(("conservative_gain_fallback_mse", float(mse_best["lambda"])))
        else:
            selectors.append(("conservative_gain", float(eligible.iloc[0]["lambda"])))

        for rule, lambda_value in selectors:
            chosen = dtr[np.isclose(dtr["lambda"], lambda_value)].copy()
            chosen["selection_rule"] = rule
            selected_rows.append(chosen)

            record: dict[str, object] = {
                "subspace": subspace,
                "selection_rule": rule,
                "selected_lambda": lambda_value,
                "standard_val_mse": standard_val_mse,
                "standard_val_directional_gain": standard_val_gain,
                "selected_val_mse": float(chosen["val_mse"].mean()),
                "selected_val_directional_gain": float(chosen["val_directional_gain"].mean()),
                "n_seeds": int(chosen["seed"].nunique()),
            }
            for metric in METRICS:
                record[f"{metric}_mean"] = float(chosen[metric].mean())
                record[f"{metric}_std"] = float(chosen[metric].std(ddof=1))
            selection_records.append(record)

    selected = pd.concat(selected_rows, ignore_index=True)
    selected_summary = pd.DataFrame(selection_records).sort_values(
        ["selection_rule", "deploy_mse_mean", "volatility_mean", "subspace"]
    )
    return selected, selected_summary.reset_index(drop=True)


def _paired_against_standard(summary: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (subspace, rule), chosen in selected.groupby(["subspace", "selection_rule"]):
        standard = summary[
            (summary["subspace"] == subspace) & (summary["method"] == "standard")
        ].set_index("seed")
        dtr = chosen.set_index("seed")
        matched_seeds = sorted(set(standard.index).intersection(set(dtr.index)))
        for metric in METRICS:
            diff = (
                dtr.loc[matched_seeds, metric].to_numpy(dtype=np.float64)
                - standard.loc[matched_seeds, metric].to_numpy(dtype=np.float64)
            )
            rows.append(
                {
                    "subspace": subspace,
                    "selection_rule": rule,
                    "metric": metric,
                    "n_seeds": int(len(diff)),
                    "wins": int(np.sum(diff < 0.0)),
                    "losses": int(np.sum(diff > 0.0)),
                    "mean_difference": float(diff.mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["selection_rule", "subspace", "metric"])


def run_suite(base_dir: Path, force: bool = False) -> dict[str, pd.DataFrame]:
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = figures_dir / "air_quality_subspace_ablation_summary.csv"
    selected_path = figures_dir / "air_quality_subspace_ablation_selected.csv"
    paired_path = figures_dir / "air_quality_subspace_ablation_paired.csv"
    trajectory_path = figures_dir / "air_quality_subspace_ablation_selected_seed_trajectories.csv"

    if (
        not force
        and summary_path.exists()
        and selected_path.exists()
        and paired_path.exists()
        and trajectory_path.exists()
    ):
        return {
            "summary": pd.read_csv(summary_path),
            "selected": pd.read_csv(selected_path),
            "paired": pd.read_csv(paired_path),
            "trajectories": pd.read_csv(trajectory_path),
        }

    cache_path = base_dir / "data" / "air_quality.csv"
    split = build_splits(load_air_quality(cache_path))
    variants = _subspace_variants(split)

    summary_frames: list[pd.DataFrame] = []
    trajectory_frames: list[pd.DataFrame] = []

    standard_models = {
        seed: train_model(split, seed=seed, method="standard", penalty_lambda=0.0)
        for seed in SWEEP_SEEDS
    }

    for subspace_name, basis in variants.items():
        variant_split = _with_basis(split, basis)
        standard_summaries: list[dict[str, object]] = []
        for seed, model in standard_models.items():
            trajectory, summary = evaluate_model(
                model,
                split=variant_split,
                lambda_value=0.0,
                seed=seed,
                method="standard",
            )
            summary["subspace"] = subspace_name
            summary["source"] = "subspace_ablation"
            standard_summaries.append(summary)
            if seed == SELECTED_SEED:
                trajectory = trajectory.copy()
                trajectory["subspace"] = subspace_name
                trajectory["source"] = "subspace_ablation"
                trajectory_frames.append(trajectory)
        summary_frames.append(pd.DataFrame(standard_summaries))

        dtr_summaries: list[dict[str, object]] = []
        for lambda_value in REGULARIZATION_LAMBDAS:
            for seed in SWEEP_SEEDS:
                model = train_model(
                    variant_split,
                    seed=seed,
                    method="dtr",
                    penalty_lambda=lambda_value,
                )
                trajectory, summary = evaluate_model(
                    model,
                    split=variant_split,
                    lambda_value=lambda_value,
                    seed=seed,
                    method="dtr",
                )
                summary["subspace"] = subspace_name
                summary["source"] = "subspace_ablation"
                dtr_summaries.append(summary)
                if seed == SELECTED_SEED:
                    trajectory = trajectory.copy()
                    trajectory["subspace"] = subspace_name
                    trajectory["source"] = "subspace_ablation"
                    trajectory_frames.append(trajectory)
        summary_frames.append(pd.DataFrame(dtr_summaries))

    summary = (
        pd.concat(summary_frames, ignore_index=True)
        .sort_values(["subspace", "method", "lambda", "seed"])
        .reset_index(drop=True)
    )
    selected, selected_summary = _summarize_selected(summary)
    paired = _paired_against_standard(summary, selected)
    trajectories = pd.concat(trajectory_frames, ignore_index=True)

    summary.to_csv(summary_path, index=False)
    selected_summary.to_csv(selected_path, index=False)
    paired.to_csv(paired_path, index=False)
    trajectories.to_csv(trajectory_path, index=False)
    return {
        "summary": summary,
        "selected": selected_summary,
        "paired": paired,
        "trajectories": trajectories,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    results = run_suite(args.base_dir, force=args.force)
    print(
        "Wrote Air Quality subspace ablation "
        f"({len(results['summary'])} sweep rows, {len(results['selected'])} selected rows) "
        f"to {args.base_dir / 'figures'}"
    )


if __name__ == "__main__":
    main()
