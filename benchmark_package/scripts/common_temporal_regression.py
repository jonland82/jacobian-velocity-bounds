from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


torch.set_num_threads(1)

DEVICE = torch.device("cpu")
SEEDS = list(range(10))
LAMBDA_GRID = [3e-4, 1e-3, 3e-3, 1e-2]


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    target_col: str
    timestamp_col: str
    numeric_cols: list[str]
    categorical_cols: list[str]
    train_end: str
    val_end: str
    block_freq: str
    description: str
    hidden_width: int = 64
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    subspace_dim: int = 2


@dataclass(frozen=True)
class PreparedRegressionSplit:
    config: BenchmarkConfig
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    deploy_x: np.ndarray
    deploy_y: np.ndarray
    deploy_block_ids: np.ndarray
    deploy_block_labels: list[str]
    drift_basis: np.ndarray
    feature_columns: list[str]
    target_mean: float
    target_std: float
    split_summary: dict[str, object]


class RegressionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def add_calendar_features(frame: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = frame.copy()
    ts = out[timestamp_col]
    hour = ts.dt.hour.astype(np.float32)
    dow = ts.dt.dayofweek.astype(np.float32)
    month = ts.dt.month.astype(np.float32)
    dayofyear = ts.dt.dayofyear.astype(np.float32)

    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    out["doy_sin"] = np.sin(2.0 * np.pi * dayofyear / 366.0)
    out["doy_cos"] = np.cos(2.0 * np.pi * dayofyear / 366.0)
    return out


def prepare_regression_split(frame: pd.DataFrame, config: BenchmarkConfig) -> PreparedRegressionSplit:
    data = frame.copy()
    data[config.timestamp_col] = pd.to_datetime(data[config.timestamp_col])
    data = data.sort_values(config.timestamp_col).reset_index(drop=True)
    data = data.dropna(subset=[config.target_col]).reset_index(drop=True)
    data = add_calendar_features(data, config.timestamp_col)

    calendar_cols = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
    ]
    numeric_cols = [*config.numeric_cols, *calendar_cols]
    categorical_cols = list(config.categorical_cols)

    train_end = pd.Timestamp(config.train_end)
    val_end = pd.Timestamp(config.val_end)
    timestamp = data[config.timestamp_col]
    train_mask = timestamp < train_end
    val_mask = (timestamp >= train_end) & (timestamp < val_end)
    deploy_mask = timestamp >= val_end

    for column in numeric_cols:
        train_median = float(data.loc[train_mask, column].median())
        data[column] = data[column].fillna(train_median)

    for column in categorical_cols:
        train_mode = data.loc[train_mask, column].astype(str).replace("nan", np.nan).mode()
        fill_value = str(train_mode.iloc[0]) if not train_mode.empty else "missing"
        data[column] = data[column].astype(str).replace("nan", fill_value).fillna(fill_value)

    feature_frame = data[numeric_cols].copy()
    if categorical_cols:
        categorical_frame = pd.get_dummies(data[categorical_cols], prefix=categorical_cols, dtype=np.float32)
        feature_frame = pd.concat([feature_frame, categorical_frame], axis=1)

    feature_frame = feature_frame.astype(np.float32)
    feature_columns = feature_frame.columns.tolist()

    train_x_raw = feature_frame.loc[train_mask].to_numpy(dtype=np.float32)
    val_x_raw = feature_frame.loc[val_mask].to_numpy(dtype=np.float32)
    deploy_x_raw = feature_frame.loc[deploy_mask].to_numpy(dtype=np.float32)

    feature_mean = train_x_raw.mean(axis=0).astype(np.float32)
    feature_std = train_x_raw.std(axis=0).astype(np.float32)
    feature_std[feature_std == 0.0] = 1.0

    train_x = ((train_x_raw - feature_mean) / feature_std).astype(np.float32)
    val_x = ((val_x_raw - feature_mean) / feature_std).astype(np.float32)
    deploy_x = ((deploy_x_raw - feature_mean) / feature_std).astype(np.float32)

    train_y_raw = data.loc[train_mask, config.target_col].to_numpy(dtype=np.float32)
    val_y_raw = data.loc[val_mask, config.target_col].to_numpy(dtype=np.float32)
    deploy_y_raw = data.loc[deploy_mask, config.target_col].to_numpy(dtype=np.float32)
    target_mean = float(train_y_raw.mean())
    target_std = float(train_y_raw.std())
    if target_std == 0.0:
        target_std = 1.0

    train_y = ((train_y_raw - target_mean) / target_std).astype(np.float32)
    val_y = ((val_y_raw - target_mean) / target_std).astype(np.float32)
    deploy_y = ((deploy_y_raw - target_mean) / target_std).astype(np.float32)

    deploy_periods = data.loc[deploy_mask, config.timestamp_col].dt.to_period(config.block_freq)
    deploy_block_ids = deploy_periods.astype(str).to_numpy()
    deploy_block_labels = list(dict.fromkeys(deploy_block_ids))
    block_means = np.asarray(
        [deploy_x[deploy_block_ids == block_id].mean(axis=0) for block_id in deploy_block_labels],
        dtype=np.float32,
    )
    diffs = np.diff(block_means, axis=0)
    if len(diffs) == 0:
        drift_basis = np.eye(train_x.shape[1], config.subspace_dim, dtype=np.float32)
    else:
        _, _, vh = np.linalg.svd(diffs, full_matrices=False)
        drift_basis = vh[: config.subspace_dim].T.astype(np.float32)

    split_summary = {
        "dataset": config.name,
        "target": config.target_col,
        "train_end": config.train_end,
        "val_end": config.val_end,
        "block_freq": config.block_freq,
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "deploy_rows": int(deploy_mask.sum()),
        "deploy_blocks": int(len(deploy_block_labels)),
        "num_features": int(train_x.shape[1]),
    }

    return PreparedRegressionSplit(
        config=config,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        deploy_x=deploy_x,
        deploy_y=deploy_y,
        deploy_block_ids=deploy_block_ids,
        deploy_block_labels=deploy_block_labels,
        drift_basis=drift_basis,
        feature_columns=feature_columns,
        target_mean=target_mean,
        target_std=target_std,
        split_summary=split_summary,
    )


def jacobian_penalty(
    predictions: torch.Tensor,
    inputs: torch.Tensor,
    method: str,
    drift_basis: torch.Tensor | None,
) -> torch.Tensor:
    grads = torch.autograd.grad(predictions.sum(), inputs, create_graph=True, retain_graph=True)[0]
    if method == "standard":
        return torch.zeros((), device=inputs.device)
    if method == "isotropic":
        return grads.square().sum(dim=1).mean()
    if method == "dtr" and drift_basis is not None:
        projected = grads @ drift_basis
        return projected.square().sum(dim=1).mean()
    raise ValueError(f"Unsupported method: {method}")


def train_model(
    split: PreparedRegressionSplit,
    seed: int,
    method: str,
    penalty_lambda: float,
) -> RegressionMLP:
    torch.manual_seed(seed)
    model = RegressionMLP(split.train_x.shape[1], split.config.hidden_width).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=split.config.learning_rate)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.from_numpy(split.train_x), torch.from_numpy(split.train_y))
    loader = DataLoader(
        dataset,
        batch_size=split.config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 41),
    )

    drift_basis = None
    if method == "dtr":
        drift_basis = torch.from_numpy(split.drift_basis).to(DEVICE)

    for _ in range(split.config.epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE).clone().detach().requires_grad_(True)
            batch_y = batch_y.to(DEVICE)

            predictions = model(batch_x)
            fit_loss = criterion(predictions, batch_y)
            penalty = jacobian_penalty(predictions, batch_x, method, drift_basis)
            weight_decay = split.config.weight_decay * sum(p.square().sum() for p in model.parameters())
            objective = fit_loss + penalty_lambda * penalty + weight_decay

            optimizer.zero_grad()
            objective.backward()
            optimizer.step()

    return model


def evaluate_regression_predictions(
    pred_std: np.ndarray,
    target_std: np.ndarray,
    target_mean: float,
    target_scale: float,
) -> dict[str, float]:
    pred_raw = pred_std * target_scale + target_mean
    target_raw = target_std * target_scale + target_mean
    mse = float(np.mean((pred_raw - target_raw) ** 2))
    mae = float(np.mean(np.abs(pred_raw - target_raw)))
    return {"mse": mse, "mae": mae}


def mean_projected_jacobian_energy(
    model: RegressionMLP,
    x_np: np.ndarray,
    drift_basis_np: np.ndarray,
    batch_size: int,
) -> float:
    model.eval()
    drift_basis = torch.from_numpy(drift_basis_np).to(DEVICE)
    energies: list[float] = []
    counts: list[int] = []
    for start in range(0, len(x_np), batch_size):
        batch = torch.from_numpy(x_np[start : start + batch_size]).to(DEVICE).requires_grad_(True)
        predictions = model(batch)
        grads = torch.autograd.grad(predictions.sum(), batch)[0]
        projected = grads @ drift_basis
        energy = projected.square().sum(dim=1)
        energies.append(float(energy.sum().detach().cpu().item()))
        counts.append(int(len(batch)))
    return float(np.sum(energies) / np.sum(counts))


def evaluate_model(
    model: RegressionMLP,
    split: PreparedRegressionSplit,
    seed: int,
    method: str,
    penalty_lambda: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        val_predictions = model(torch.from_numpy(split.val_x).to(DEVICE)).cpu().numpy()
        deploy_predictions = model(torch.from_numpy(split.deploy_x).to(DEVICE)).cpu().numpy()

    val_metrics = evaluate_regression_predictions(
        val_predictions,
        split.val_y,
        split.target_mean,
        split.target_std,
    )
    deploy_metrics = evaluate_regression_predictions(
        deploy_predictions,
        split.deploy_y,
        split.target_mean,
        split.target_std,
    )
    val_directional_gain = mean_projected_jacobian_energy(
        model,
        split.val_x,
        split.drift_basis,
        split.config.batch_size,
    )

    rows: list[dict[str, float | int | str]] = []
    previous_mean: np.ndarray | None = None
    for block_label in split.deploy_block_labels:
        mask = split.deploy_block_ids == block_label
        block_x = split.deploy_x[mask]
        block_y = split.deploy_y[mask]

        inputs = torch.from_numpy(block_x).to(DEVICE).requires_grad_(True)
        predictions = model(inputs)
        block_metrics = evaluate_regression_predictions(
            predictions.detach().cpu().numpy(),
            block_y,
            split.target_mean,
            split.target_std,
        )

        current_mean = block_x.mean(axis=0)
        if previous_mean is None:
            speed = 0.0
            gain = 0.0
        else:
            delta = current_mean - previous_mean
            speed = float(np.linalg.norm(delta))
            if speed > 0.0:
                direction = torch.from_numpy((delta / speed).astype(np.float32)).to(DEVICE)
                grads = torch.autograd.grad(predictions.sum(), inputs)[0]
                gain = float((grads @ direction).square().mean().detach().cpu().item())
            else:
                gain = 0.0
        previous_mean = current_mean

        rows.append(
            {
                "dataset": split.config.name,
                "method": method,
                "lambda": float(penalty_lambda),
                "seed": int(seed),
                "block": block_label,
                "risk": block_metrics["mse"],
                "mae": block_metrics["mae"],
                "speed": speed,
                "gain": gain,
                "hazard": float((speed**2) * gain),
                "n_obs": int(mask.sum()),
            }
        )

    trajectory = pd.DataFrame(rows)
    risk = trajectory["risk"].to_numpy(dtype=np.float64)
    mae = trajectory["mae"].to_numpy(dtype=np.float64)
    summary = {
        "dataset": split.config.name,
        "method": method,
        "lambda": float(penalty_lambda),
        "seed": int(seed),
        "val_mse": val_metrics["mse"],
        "val_mae": val_metrics["mae"],
        "val_directional_gain": val_directional_gain,
        "deploy_mse": deploy_metrics["mse"],
        "deploy_mae": deploy_metrics["mae"],
        "volatility": float(np.mean((risk - risk.mean()) ** 2)),
        "initial_risk": float(risk[0]),
        "terminal_risk": float(risk[-1]),
        "max_risk": float(risk.max()),
        "mean_block_mae": float(mae.mean()),
        "terminal_mae": float(mae[-1]),
        "mean_gain": float(trajectory["gain"].mean()),
        "mean_hazard": float(trajectory["hazard"].mean()),
        "max_hazard": float(trajectory["hazard"].max()),
    }
    return trajectory, summary


def run_temporal_regression_benchmark(
    frame: pd.DataFrame,
    config: BenchmarkConfig,
    output_dir: Path,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    trajectory_path = output_dir / "selected_trajectories.csv"
    all_selected_trajectory_path = output_dir / "selected_trajectories_all_seeds.csv"
    selected_lambda_path = output_dir / "selected_lambdas.csv"
    selected_summary_path = output_dir / "selected_summary.csv"
    summary_json_path = output_dir / "summary.json"

    if (
        not force
        and sweep_path.exists()
        and trajectory_path.exists()
        and all_selected_trajectory_path.exists()
        and selected_lambda_path.exists()
        and selected_summary_path.exists()
        and summary_json_path.exists()
    ):
        return {
            "sweep_summary": pd.read_csv(sweep_path),
            "selected_trajectories": pd.read_csv(trajectory_path),
            "selected_trajectories_all_seeds": pd.read_csv(all_selected_trajectory_path),
            "selected_lambdas": pd.read_csv(selected_lambda_path),
            "selected_summary": pd.read_csv(selected_summary_path),
        }

    split = prepare_regression_split(frame, config)
    summary_rows: list[dict[str, float | int | str]] = []
    trajectories: list[pd.DataFrame] = []

    experiment_specs: list[tuple[str, float]] = [("standard", 0.0)]
    experiment_specs.extend(("isotropic", lam) for lam in LAMBDA_GRID)
    experiment_specs.extend(("dtr", lam) for lam in LAMBDA_GRID)

    for method, penalty_lambda in experiment_specs:
        for seed in SEEDS:
            model = train_model(split, seed=seed, method=method, penalty_lambda=penalty_lambda)
            trajectory_df, summary = evaluate_model(
                model,
                split=split,
                seed=seed,
                method=method,
                penalty_lambda=penalty_lambda,
            )
            summary_rows.append(summary)
            trajectories.append(trajectory_df)

    sweep_summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["method", "lambda", "seed"])
        .reset_index(drop=True)
    )
    all_trajectories = (
        pd.concat(trajectories, ignore_index=True)
        .sort_values(["method", "lambda", "seed", "block"])
        .reset_index(drop=True)
    )

    selected_lambda_rows: list[dict[str, float | str]] = []
    for method in ["standard", "isotropic", "dtr"]:
        subset = sweep_summary[sweep_summary["method"] == method]
        if method == "standard":
            best_lambda = 0.0
            selection_val_mse = float(subset["val_mse"].mean())
        else:
            grouped = (
                subset.groupby("lambda", as_index=False)[["val_mse", "val_directional_gain"]]
                .mean()
                .sort_values(["val_mse", "val_directional_gain", "lambda"])
                .reset_index(drop=True)
            )
            best = grouped.iloc[0]
            best_lambda = float(best["lambda"])
            selection_val_mse = float(best["val_mse"])
            selection_val_directional_gain = float(best["val_directional_gain"])
        if method == "standard":
            selection_val_directional_gain = float(subset["val_directional_gain"].mean())
        selected_lambda_rows.append(
            {
                "dataset": config.name,
                "method": method,
                "selected_lambda": best_lambda,
                "selection_val_mse": selection_val_mse,
                "selection_val_directional_gain": selection_val_directional_gain,
            }
        )

    selected_lambdas = pd.DataFrame(selected_lambda_rows)
    selected_summary_frames: list[pd.DataFrame] = []
    selected_trajectory_frames: list[pd.DataFrame] = []
    all_selected_trajectory_frames: list[pd.DataFrame] = []
    display_seed = SEEDS[1]

    for row in selected_lambdas.itertuples(index=False):
        summary_subset = sweep_summary[
            (sweep_summary["method"] == row.method)
            & np.isclose(sweep_summary["lambda"], row.selected_lambda)
        ]
        selected_summary_frames.append(summary_subset)

        all_trajectory_subset = all_trajectories[
            (all_trajectories["method"] == row.method)
            & np.isclose(all_trajectories["lambda"], row.selected_lambda)
        ]
        all_selected_trajectory_frames.append(all_trajectory_subset)

        trajectory_subset = all_trajectories[
            (all_trajectories["method"] == row.method)
            & np.isclose(all_trajectories["lambda"], row.selected_lambda)
            & (all_trajectories["seed"] == display_seed)
        ]
        selected_trajectory_frames.append(trajectory_subset)

    selected_summary = (
        pd.concat(selected_summary_frames, ignore_index=True)
        .groupby(["dataset", "method", "lambda"], as_index=False)
        .mean(numeric_only=True)
        .merge(
            selected_lambdas.rename(columns={"selected_lambda": "lambda"}),
            on=["dataset", "method", "lambda"],
            how="left",
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    selected_trajectories = (
        pd.concat(selected_trajectory_frames, ignore_index=True)
        .sort_values(["method", "block"])
        .reset_index(drop=True)
    )
    selected_trajectories_all_seeds = (
        pd.concat(all_selected_trajectory_frames, ignore_index=True)
        .sort_values(["method", "seed", "block"])
        .reset_index(drop=True)
    )

    summary_json = {
        "dataset": config.name,
        "description": config.description,
        "split_summary": split.split_summary,
        "selection_rule": (
            "Per-method lambda chosen by mean validation MSE across matched seeds, "
            "with validation directional gain used only as a secondary tie-breaker."
        ),
        "selected_summary": selected_summary.to_dict(orient="records"),
    }

    sweep_summary.to_csv(sweep_path, index=False)
    selected_trajectories.to_csv(trajectory_path, index=False)
    selected_trajectories_all_seeds.to_csv(all_selected_trajectory_path, index=False)
    selected_lambdas.to_csv(selected_lambda_path, index=False)
    selected_summary.to_csv(selected_summary_path, index=False)
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, indent=2)

    return {
        "sweep_summary": sweep_summary,
        "selected_trajectories": selected_trajectories,
        "selected_trajectories_all_seeds": selected_trajectories_all_seeds,
        "selected_lambdas": selected_lambdas,
        "selected_summary": selected_summary,
    }
