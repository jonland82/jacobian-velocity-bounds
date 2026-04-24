from __future__ import annotations

import argparse
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


DATA_URL = "https://archive.ics.uci.edu/static/public/360/data.csv"
FEATURE_COLS = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]
TARGET_COL = "CO(GT)"
TRAIN_WEEKS = 12
VAL_WEEKS = 4
BLOCK_DAYS = 14
HIDDEN_WIDTH = 32
EPOCHS = 35
LEARNING_RATE = 0.01
SWEEP_LAMBDAS = [0.0, 0.003, 0.01, 0.03, 0.08]
REGULARIZATION_LAMBDAS = [0.003, 0.01, 0.03, 0.08]
SWEEP_SEEDS = list(range(10))
SUBSPACE_DIM = 2
SUBSPACE_METHOD = "sensor_target_orthogonal"
SELECTED_SEED = 1
DEVICE = torch.device("cpu")


@dataclass(frozen=True)
class SplitData:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    deploy_x: np.ndarray
    deploy_y: np.ndarray
    deploy_timestamps: pd.Series
    block_index: np.ndarray
    block_dates: list[pd.Timestamp]
    drift_basis: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_std: float
    split_summary: dict[str, str | int]


class AirQualityRegressor(nn.Module):
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


def load_air_quality(cache_path: Path) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        raw = pd.read_csv(cache_path)
    else:
        raw = pd.read_csv(DATA_URL)
        raw.to_csv(cache_path, index=False)

    clean = raw.replace(-200, np.nan).copy()
    clean["timestamp"] = pd.to_datetime(
        clean["Date"] + " " + clean["Time"],
        format="%m/%d/%Y %H:%M:%S",
        errors="coerce",
    )
    keep_cols = ["timestamp", *FEATURE_COLS, TARGET_COL]
    clean = clean[keep_cols].dropna().sort_values("timestamp").reset_index(drop=True)
    return clean


def sensor_indices() -> list[int]:
    return [idx for idx, name in enumerate(FEATURE_COLS) if name.startswith("PT08.")]


def target_orthogonalize(
    values: np.ndarray,
    train_values: np.ndarray,
    train_y: np.ndarray,
) -> np.ndarray:
    direction, *_ = np.linalg.lstsq(train_values, train_y, rcond=None)
    direction = direction.astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return values.astype(np.float32)
    unit_direction = direction / norm
    return (values - np.outer(values @ unit_direction, unit_direction)).astype(np.float32)


def estimate_sensor_target_orthogonal_subspace(
    train_x: np.ndarray,
    train_y: np.ndarray,
    deploy_x: np.ndarray,
    block_index: np.ndarray,
    subspace_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    sensor_idx = sensor_indices()
    train_sensor = train_x[:, sensor_idx]
    deploy_sensor = deploy_x[:, sensor_idx]
    deploy_orthogonal = target_orthogonalize(deploy_sensor, train_sensor, train_y)
    num_blocks = int(block_index.max()) + 1
    block_means = np.asarray(
        [deploy_orthogonal[block_index == block_id].mean(axis=0) for block_id in range(num_blocks)],
        dtype=np.float32,
    )
    block_diffs = np.diff(block_means, axis=0)
    _, singular_values, right_vectors = np.linalg.svd(block_diffs, full_matrices=False)
    local_basis = right_vectors[:subspace_dim].T.astype(np.float32)
    full_basis = np.zeros((len(FEATURE_COLS), subspace_dim), dtype=np.float32)
    full_basis[np.asarray(sensor_idx), :] = local_basis
    return full_basis, singular_values


def build_splits(df: pd.DataFrame) -> SplitData:
    start_time = df["timestamp"].min()
    train_end = start_time + pd.Timedelta(weeks=TRAIN_WEEKS)
    val_end = train_end + pd.Timedelta(weeks=VAL_WEEKS)
    block_width = pd.Timedelta(days=BLOCK_DAYS)

    train_df = df[df["timestamp"] < train_end].copy()
    val_df = df[(df["timestamp"] >= train_end) & (df["timestamp"] < val_end)].copy()
    deploy_df = df[df["timestamp"] >= val_end].copy()

    feature_mean = train_df[FEATURE_COLS].mean().to_numpy(dtype=np.float32)
    feature_std = train_df[FEATURE_COLS].std().replace(0.0, 1.0).to_numpy(dtype=np.float32)

    def scale_features(frame: pd.DataFrame) -> np.ndarray:
        return ((frame[FEATURE_COLS].to_numpy(dtype=np.float32) - feature_mean) / feature_std).astype(
            np.float32
        )

    train_x = scale_features(train_df)
    val_x = scale_features(val_df)
    deploy_x = scale_features(deploy_df)

    target_mean = float(train_df[TARGET_COL].mean())
    target_std = float(train_df[TARGET_COL].std())
    if target_std == 0.0:
        target_std = 1.0

    def scale_target(frame: pd.DataFrame) -> np.ndarray:
        raw = frame[TARGET_COL].to_numpy(dtype=np.float32)
        return ((raw - target_mean) / target_std).astype(np.float32)

    train_y = scale_target(train_df)
    val_y = scale_target(val_df)
    deploy_y = scale_target(deploy_df)

    raw_block_index = (
        (deploy_df["timestamp"] - deploy_df["timestamp"].min()) / block_width
    ).astype(int)
    block_index = raw_block_index.to_numpy(dtype=int)
    num_blocks = int(block_index.max()) + 1
    drift_basis, singular_values = estimate_sensor_target_orthogonal_subspace(
        train_x,
        train_y,
        deploy_x,
        block_index,
        SUBSPACE_DIM,
    )

    block_dates = [
        pd.Timestamp(deploy_df.loc[block_index == block_id, "timestamp"].iloc[0])
        for block_id in range(num_blocks)
    ]

    split_summary = {
        "train_start": str(train_df["timestamp"].min().date()),
        "train_end": str(train_df["timestamp"].max().date()),
        "val_start": str(val_df["timestamp"].min().date()),
        "val_end": str(val_df["timestamp"].max().date()),
        "deploy_start": str(deploy_df["timestamp"].min().date()),
        "deploy_end": str(deploy_df["timestamp"].max().date()),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "deploy_rows": int(len(deploy_df)),
        "deploy_blocks": int(num_blocks),
        "block_days": int(BLOCK_DAYS),
        "target": TARGET_COL,
        "subspace_method": SUBSPACE_METHOD,
        "subspace_dim": int(SUBSPACE_DIM),
    }
    for idx, value in enumerate(singular_values[:SUBSPACE_DIM], start=1):
        split_summary[f"drift_sv_{idx}"] = float(value)

    return SplitData(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        deploy_x=deploy_x,
        deploy_y=deploy_y,
        deploy_timestamps=deploy_df["timestamp"].reset_index(drop=True),
        block_index=block_index,
        block_dates=block_dates,
        drift_basis=drift_basis,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_std=target_std,
        split_summary=split_summary,
    )


def jacobian_penalty(
    predictions: torch.Tensor,
    inputs: torch.Tensor,
    method: str,
    drift_basis: torch.Tensor | None,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        predictions.sum(),
        inputs,
        create_graph=True,
        retain_graph=True,
    )[0]
    if method == "standard":
        return torch.zeros((), device=inputs.device)
    if method == "isotropic":
        return grads.square().sum(dim=1).mean()
    if method == "dtr" and drift_basis is not None:
        projected = grads @ drift_basis
        return projected.square().sum(dim=1).mean()
    raise ValueError(f"Unsupported method: {method}")


def mean_projected_jacobian_energy(
    model: AirQualityRegressor,
    x_np: np.ndarray,
    drift_basis_np: np.ndarray,
    batch_size: int = 512,
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


def train_model(
    split: SplitData,
    seed: int,
    method: str,
    penalty_lambda: float,
) -> AirQualityRegressor:
    torch.manual_seed(seed)
    model = AirQualityRegressor(len(FEATURE_COLS), HIDDEN_WIDTH).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    drift_basis = None
    if method == "dtr":
        drift_basis = torch.from_numpy(split.drift_basis).to(DEVICE)
    dataset = TensorDataset(torch.from_numpy(split.train_x), torch.from_numpy(split.train_y))
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 17),
    )

    for _ in range(EPOCHS):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE).clone().detach().requires_grad_(True)
            batch_y = batch_y.to(DEVICE)

            predictions = model(batch_x)
            fit_loss = criterion(predictions, batch_y)
            penalty = jacobian_penalty(predictions, batch_x, method, drift_basis)
            objective = fit_loss + penalty_lambda * penalty

            optimizer.zero_grad()
            objective.backward()
            optimizer.step()

    return model


def evaluate_model(
    model: AirQualityRegressor,
    split: SplitData,
    lambda_value: float,
    seed: int,
    method: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    with torch.no_grad():
        val_predictions = model(torch.from_numpy(split.val_x).to(DEVICE)).cpu().numpy()
        deploy_predictions = model(torch.from_numpy(split.deploy_x).to(DEVICE)).cpu().numpy()
    val_mse = float(np.mean(((val_predictions - split.val_y) * split.target_std) ** 2))
    deploy_mse = float(np.mean(((deploy_predictions - split.deploy_y) * split.target_std) ** 2))
    val_directional_gain = mean_projected_jacobian_energy(
        model,
        split.val_x,
        split.drift_basis,
    )

    rows: list[dict[str, float | int | str]] = []
    previous_mean: np.ndarray | None = None
    num_blocks = len(split.block_dates)

    for block_id in range(num_blocks):
        mask = split.block_index == block_id
        block_x = split.deploy_x[mask]
        block_y = split.deploy_y[mask]
        block_dates = split.deploy_timestamps[mask]

        inputs = torch.from_numpy(block_x).to(DEVICE).requires_grad_(True)
        predictions = model(inputs)
        block_risk = float(
            np.mean(((predictions.detach().cpu().numpy() - block_y) * split.target_std) ** 2)
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
                gain = float((grads @ direction).square().mean().detach().cpu().numpy())
            else:
                gain = 0.0
        previous_mean = current_mean

        rows.append(
            {
                "block": int(block_id),
                "block_start": str(block_dates.iloc[0].date()),
                "lambda": float(lambda_value),
                "seed": int(seed),
                "method": method,
                "model": method,
                "risk": block_risk,
                "speed": speed,
                "gain": gain,
                "hazard": float((speed**2) * gain),
                "n_obs": int(mask.sum()),
            }
        )

    trajectory = pd.DataFrame(rows)
    risk = trajectory["risk"].to_numpy()
    summary = {
        "lambda": float(lambda_value),
        "seed": int(seed),
        "method": method,
        "model": method,
        "val_mse": val_mse,
        "val_directional_gain": val_directional_gain,
        "deploy_mse": deploy_mse,
        "volatility": float(np.mean((risk - risk.mean()) ** 2)),
        "initial_risk": float(risk[0]),
        "terminal_risk": float(risk[-1]),
        "mean_gain": float(trajectory["gain"].mean()),
        "mean_hazard": float(trajectory["hazard"].mean()),
        "max_hazard": float(trajectory["hazard"].max()),
    }
    return trajectory, summary


def build_paper_summary(
    selected_lambdas: pd.DataFrame,
    selected_summary: pd.DataFrame,
    split: SplitData,
) -> dict[str, object]:
    return {
        "split": split.split_summary,
        "selection_rule": (
            "Per-method lambda chosen by mean validation MSE across matched seeds, "
            "with validation directional gain used only as a secondary tie-breaker."
        ),
        "selected_lambdas": selected_lambdas.to_dict(orient="records"),
        "selected_summary": selected_summary.to_dict(orient="records"),
    }


def run_suite(base_dir: Path, force: bool = False) -> dict[str, pd.DataFrame]:
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = figures_dir / "air_quality_summary.csv"
    trajectory_path = figures_dir / "air_quality_selected_trajectories.csv"
    all_selected_trajectory_path = figures_dir / "air_quality_selected_trajectories_all_seeds.csv"
    selected_lambda_path = figures_dir / "air_quality_selected_lambdas.csv"
    selected_summary_path = figures_dir / "air_quality_selected_summary.csv"
    paper_summary_path = figures_dir / "air_quality_summary.json"
    cache_path = base_dir / "data" / "air_quality.csv"

    if (
        not force
        and summary_path.exists()
        and trajectory_path.exists()
        and all_selected_trajectory_path.exists()
        and selected_lambda_path.exists()
        and selected_summary_path.exists()
        and paper_summary_path.exists()
    ):
        return {
            "summary": pd.read_csv(summary_path),
            "trajectory": pd.read_csv(trajectory_path),
            "selected_trajectories_all_seeds": pd.read_csv(all_selected_trajectory_path),
            "selected_lambdas": pd.read_csv(selected_lambda_path),
            "selected_summary": pd.read_csv(selected_summary_path),
        }

    split = build_splits(load_air_quality(cache_path))
    summary_rows: list[dict[str, float | int | str]] = []
    trajectories: list[pd.DataFrame] = []

    experiment_specs: list[tuple[str, float]] = [("standard", 0.0)]
    experiment_specs.extend(("isotropic", lambda_value) for lambda_value in REGULARIZATION_LAMBDAS)
    experiment_specs.extend(("dtr", lambda_value) for lambda_value in REGULARIZATION_LAMBDAS)

    for method, lambda_value in experiment_specs:
        for seed in SWEEP_SEEDS:
            model = train_model(
                split,
                seed=seed,
                method=method,
                penalty_lambda=lambda_value,
            )
            trajectory_df, summary = evaluate_model(
                model,
                split=split,
                lambda_value=lambda_value,
                seed=seed,
                method=method,
            )
            summary_rows.append(summary)
            trajectories.append(trajectory_df)

    summary_df = (
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
        subset = summary_df[summary_df["method"] == method]
        if method == "standard":
            best_lambda = 0.0
            selection_val_mse = float(subset["val_mse"].mean())
            selection_val_directional_gain = float(subset["val_directional_gain"].mean())
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
        selected_lambda_rows.append(
            {
                "dataset": "air_quality",
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

    for row in selected_lambdas.itertuples(index=False):
        summary_subset = summary_df[
            (summary_df["method"] == row.method)
            & np.isclose(summary_df["lambda"], row.selected_lambda)
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
            & (all_trajectories["seed"] == SELECTED_SEED)
        ]
        selected_trajectory_frames.append(trajectory_subset)

    selected_summary = (
        pd.concat(selected_summary_frames, ignore_index=True)
        .groupby(["method", "lambda"], as_index=False)
        .mean(numeric_only=True)
        .merge(
            selected_lambdas.rename(columns={"selected_lambda": "lambda"}),
            on=["method", "lambda"],
            how="left",
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    trajectory_df = (
        pd.concat(selected_trajectory_frames, ignore_index=True)
        .sort_values(["method", "block"])
        .reset_index(drop=True)
    )
    selected_trajectories_all_seeds = (
        pd.concat(all_selected_trajectory_frames, ignore_index=True)
        .sort_values(["method", "seed", "block"])
        .reset_index(drop=True)
    )
    paper_summary = build_paper_summary(selected_lambdas, selected_summary, split)

    summary_df.to_csv(summary_path, index=False)
    trajectory_df.to_csv(trajectory_path, index=False)
    selected_trajectories_all_seeds.to_csv(all_selected_trajectory_path, index=False)
    selected_lambdas.to_csv(selected_lambda_path, index=False)
    selected_summary.to_csv(selected_summary_path, index=False)
    with paper_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(paper_summary, handle, indent=2)

    return {
        "summary": summary_df,
        "trajectory": trajectory_df,
        "selected_trajectories_all_seeds": selected_trajectories_all_seeds,
        "selected_lambdas": selected_lambdas,
        "selected_summary": selected_summary,
    }


def ensure_air_quality_outputs(base_dir: Path, force: bool = False) -> dict[str, pd.DataFrame]:
    return run_suite(base_dir, force=force)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    results = run_suite(args.base_dir, force=args.force)
    print(f"Wrote {len(results['summary'])} Air Quality sweep rows to {args.base_dir / 'figures'}")


if __name__ == "__main__":
    main()
