from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TIME_HORIZON = 1.0
TIME_GRID = np.linspace(0.0, TIME_HORIZON, 41)
SWEEP_LAMBDAS = [0.0, 0.01, 0.03, 0.08]
SWEEP_SEEDS = list(range(20))
SELECTED_DTR_LAMBDA = 0.03
SELECTED_SEED = 0
TRAIN_SIZE = 1024
EVAL_SIZE = 2048
DRIFT_DIRECTION = np.array([0.0, 1.0], dtype=np.float32)
DEVICE = torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def speed_schedule(t: np.ndarray | float) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    return (
        0.55
        + 1.05 * np.exp(-((t_arr - 0.33) / 0.10) ** 2)
        + 0.75 * np.exp(-((t_arr - 0.76) / 0.08) ** 2)
    )


_DENSE_T = np.linspace(0.0, TIME_HORIZON, 4001)
_DENSE_SPEED = speed_schedule(_DENSE_T)
_DENSE_SHIFT = np.concatenate(
    (
        [0.0],
        np.cumsum(0.5 * (_DENSE_SPEED[1:] + _DENSE_SPEED[:-1]) * np.diff(_DENSE_T)),
    )
)


def shift_schedule(t: np.ndarray | float) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    return np.interp(t_arr, _DENSE_T, _DENSE_SHIFT)


def sample_features(n: int, t: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    sign = 2.0 * y - 1.0
    x1 = rng.normal(loc=1.05 * sign, scale=0.90, size=n)
    x2 = rng.normal(loc=1.25 * sign, scale=0.55, size=n)
    x2 = x2 + shift_schedule(t)
    x = np.stack([x1, x2], axis=1).astype(np.float32)
    return x, y


class SmallReLUNet(nn.Module):
    def __init__(self, hidden: int = 12) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def directional_penalty(
    logits: torch.Tensor, inputs: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    grads = torch.autograd.grad(
        logits.sum(), inputs, create_graph=True, retain_graph=True
    )[0]
    directional = grads @ direction
    return directional.square().mean()


def train_model(seed: int, dtr_lambda: float) -> SmallReLUNet:
    set_seed(seed)
    x_np, y_np = sample_features(TRAIN_SIZE, t=0.0, seed=seed + 17)
    dataset = TensorDataset(torch.from_numpy(x_np), torch.from_numpy(y_np))
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 31),
    )

    model = SmallReLUNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    direction = torch.tensor(DRIFT_DIRECTION, device=DEVICE)

    for _ in range(140):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE).clone().detach().requires_grad_(True)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            risk = criterion(logits, batch_y)
            penalty = directional_penalty(logits, batch_x, direction)
            weight_decay = 1e-4 * sum(param.square().sum() for param in model.parameters())
            objective = risk + dtr_lambda * penalty + weight_decay
            objective.backward()
            optimizer.step()

    return model


def evaluate_trajectory(
    model: SmallReLUNet, lambda_value: float, seed: int, label: str
) -> pd.DataFrame:
    criterion = nn.BCEWithLogitsLoss()
    direction = torch.tensor(DRIFT_DIRECTION, device=DEVICE)
    rows: list[dict[str, float | int | str]] = []

    for idx, t in enumerate(TIME_GRID):
        eval_seed = 1000 + 97 * seed + idx
        x_np, y_np = sample_features(EVAL_SIZE, t=float(t), seed=eval_seed)
        inputs = torch.from_numpy(x_np).to(DEVICE).requires_grad_(True)
        targets = torch.from_numpy(y_np).to(DEVICE)

        logits = model(inputs)
        risk = criterion(logits, targets).item()
        error = ((logits > 0).float() != targets).float().mean().item()
        grads = torch.autograd.grad(logits.sum(), inputs)[0]
        directional = grads @ direction
        loss_slope = torch.sigmoid(logits) - targets
        gain = directional.square().mean().item()
        speed = float(speed_schedule(t))
        rprime = float((loss_slope * directional).mean().item() * speed)
        rows.append(
            {
                "t": float(t),
                "lambda": float(lambda_value),
                "seed": int(seed),
                "model": label,
                "risk": float(risk),
                "error": float(error),
                "gain": float(gain),
                "speed": speed,
                "shift": float(shift_schedule(t)),
                "hazard": float((speed**2) * gain),
                "rprime": rprime,
            }
        )

    return pd.DataFrame(rows)


def summarize_trajectory(df: pd.DataFrame) -> dict[str, float]:
    risk = df["risk"].to_numpy()
    hazard = df["hazard"].to_numpy()
    rprime = df["rprime"].to_numpy()
    times = df["t"].to_numpy()
    volatility = float(np.mean((risk - risk.mean()) ** 2))
    fd_rprime = np.gradient(risk, times)
    bound_fd = float((TIME_HORIZON / math.pi**2) * np.trapezoid(fd_rprime**2, times))
    bound_jv = float((TIME_HORIZON / math.pi**2) * np.trapezoid(hazard, times))
    bound_chain = float((TIME_HORIZON / math.pi**2) * np.trapezoid(rprime**2, times))
    return {
        "volatility": volatility,
        "bound_fd": bound_fd,
        "bound_chain": bound_chain,
        "bound_jv": bound_jv,
        "terminal_risk": float(risk[-1]),
        "initial_risk": float(risk[0]),
        "max_risk": float(risk.max()),
        "mean_gain": float(df["gain"].mean()),
        "mean_hazard": float(df["hazard"].mean()),
    }


def build_summary(
    summary_df: pd.DataFrame, trajectory_df: pd.DataFrame
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    standard = summary_df[summary_df["lambda"] == 0.0]
    dtr = summary_df[summary_df["lambda"] > 0.0]
    out["standard_mean"] = {
        "volatility": float(standard["volatility"].mean()),
        "bound_fd": float(standard["bound_fd"].mean()),
        "bound_chain": float(standard["bound_chain"].mean()),
        "bound_jv": float(standard["bound_jv"].mean()),
        "terminal_risk": float(standard["terminal_risk"].mean()),
        "mean_gain": float(standard["mean_gain"].mean()),
    }
    out["dtr_mean"] = {
        "volatility": float(dtr["volatility"].mean()),
        "bound_fd": float(dtr["bound_fd"].mean()),
        "bound_chain": float(dtr["bound_chain"].mean()),
        "bound_jv": float(dtr["bound_jv"].mean()),
        "terminal_risk": float(dtr["terminal_risk"].mean()),
        "mean_gain": float(dtr["mean_gain"].mean()),
    }

    selected = trajectory_df[trajectory_df["model"].isin(["standard", "dtr"])]
    for model_name in ["standard", "dtr"]:
        model_df = selected[selected["model"] == model_name]
        risk = model_df["risk"].to_numpy()
        out[f"selected_{model_name}"] = {
            "initial_risk": float(risk[0]),
            "terminal_risk": float(risk[-1]),
            "max_risk": float(risk.max()),
            "volatility": float(np.mean((risk - risk.mean()) ** 2)),
            "max_hazard": float(model_df["hazard"].max()),
            "mean_gain": float(model_df["gain"].mean()),
        }

    if out["standard_mean"]["volatility"] > 0.0:
        out["relative_reduction"] = {
            "volatility_pct": float(
                100.0
                * (
                    1.0
                    - out["dtr_mean"]["volatility"] / out["standard_mean"]["volatility"]
                )
            ),
            "mean_gain_pct": float(
                100.0
                * (1.0 - out["dtr_mean"]["mean_gain"] / out["standard_mean"]["mean_gain"])
            ),
        }
    return out


def run_suite(output_dir: Path, force: bool = False) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "synthetic_theorem_summary.csv"
    trajectory_path = output_dir / "synthetic_theorem_selected_trajectories.csv"
    paper_summary_path = output_dir / "synthetic_theorem_summary.json"

    if (
        not force
        and summary_path.exists()
        and trajectory_path.exists()
        and paper_summary_path.exists()
    ):
        return {
            "summary": pd.read_csv(summary_path),
            "trajectory": pd.read_csv(trajectory_path),
        }

    summary_rows: list[dict[str, float | int | str]] = []
    selected_frames: list[pd.DataFrame] = []

    for lambda_value in SWEEP_LAMBDAS:
        for seed in SWEEP_SEEDS:
            model = train_model(seed=seed, dtr_lambda=lambda_value)
            label = "standard" if lambda_value == 0.0 else f"dtr_lambda_{lambda_value:.2f}"
            trajectory = evaluate_trajectory(model, lambda_value=lambda_value, seed=seed, label=label)
            summary = summarize_trajectory(trajectory)
            summary_rows.append(
                {
                    "lambda": float(lambda_value),
                    "seed": int(seed),
                    **summary,
                }
            )
            if seed == SELECTED_SEED and lambda_value in {0.0, SELECTED_DTR_LAMBDA}:
                selected_label = "standard" if lambda_value == 0.0 else "dtr"
                selected_frames.append(
                    trajectory.assign(model=selected_label, seed=seed, **{"lambda": lambda_value})
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(["lambda", "seed"]).reset_index(drop=True)
    trajectory_df = (
        pd.concat(selected_frames, ignore_index=True)
        .sort_values(["model", "t"])
        .reset_index(drop=True)
    )
    paper_summary = build_summary(summary_df, trajectory_df)

    summary_df.to_csv(summary_path, index=False)
    trajectory_df.to_csv(trajectory_path, index=False)
    with paper_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(paper_summary, handle, indent=2)

    return {"summary": summary_df, "trajectory": trajectory_df}


def ensure_experiment_outputs(base_dir: Path, force: bool = False) -> dict[str, pd.DataFrame]:
    return run_suite(base_dir / "figures", force=force)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    results = run_suite(args.output_dir, force=args.force)
    print(f"Wrote {len(results['summary'])} sweep rows to {args.output_dir}")


if __name__ == "__main__":
    main()
