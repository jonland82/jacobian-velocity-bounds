from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TIME_GRID = np.linspace(0.0, 1.0, 401)
CONDITIONAL_SLOPES = [0.0, 0.5, 1.0, 1.5, 2.0]
METHODS = {"standard": 0.0, "dtr": 0.5}
SEEDS = list(range(100))
SOURCE_SIZE = 128
SOURCE_SCALE = 0.30
SOURCE_LOG_ODDS = 1.0
DEPLOY_DISTANCE = 2.0


def sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    out = np.empty_like(value, dtype=np.float64)
    out[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


def fit_logistic_source(seed: int, conditional_slope: float, penalty_lambda: float) -> np.ndarray:
    rng = np.random.default_rng(seed + 1907)
    x = rng.normal(0.0, SOURCE_SCALE, size=SOURCE_SIZE)
    eta = sigmoid(SOURCE_LOG_ODDS + conditional_slope * x)
    y = rng.binomial(1, eta).astype(np.float64)
    design = np.column_stack([np.ones_like(x), x])
    mean_y = np.clip(y.mean(), 1e-4, 1.0 - 1e-4)
    parameters = np.array([math.log(mean_y / (1.0 - mean_y)), 0.0])

    for _ in range(50):
        probability = sigmoid(design @ parameters)
        weights = np.maximum(probability * (1.0 - probability), 1e-8)
        gradient = design.T @ (probability - y) / SOURCE_SIZE
        gradient[1] += 2.0 * penalty_lambda * parameters[1]
        hessian = design.T @ (weights[:, None] * design) / SOURCE_SIZE
        hessian[1, 1] += 2.0 * penalty_lambda
        step = np.linalg.solve(hessian + 1e-10 * np.eye(2), gradient)
        parameters -= step
        if np.linalg.norm(step) < 1e-10:
            break
    return parameters


def evaluate_path(
    parameters: np.ndarray,
    conditional_slope: float,
) -> dict[str, float]:
    intercept, score_slope = parameters
    x = DEPLOY_DISTANCE * TIME_GRID
    velocity = DEPLOY_DISTANCE
    eta = sigmoid(SOURCE_LOG_ODDS + conditional_slope * x)
    eta_prime = conditional_slope * eta * (1.0 - eta)
    score = intercept + score_slope * x
    probability = sigmoid(score)
    risk = np.logaddexp(0.0, score) - eta * score

    model_derivative = (probability - eta) * score_slope * velocity
    residual_derivative = -eta_prime * score * velocity
    risk_derivative = model_derivative + residual_derivative
    gamma = np.full_like(TIME_GRID, abs(score_slope * velocity))
    remainder = np.abs(residual_derivative)

    volatility = float(np.mean(np.square(risk - risk.mean())))
    derivative_bound = float(np.trapezoid(np.square(risk_derivative), TIME_GRID) / math.pi**2)
    jacobian_bound = float(np.trapezoid(np.square(gamma), TIME_GRID) / math.pi**2)
    residual_bound = float(
        np.trapezoid(np.square(gamma + remainder), TIME_GRID) / math.pi**2
    )
    separated_bound = float(
        2.0
        * (
            np.trapezoid(np.square(gamma), TIME_GRID)
            + np.trapezoid(np.square(remainder), TIME_GRID)
        )
        / math.pi**2
    )
    return {
        "intercept": float(intercept),
        "score_slope": float(score_slope),
        "directional_gain": float(score_slope**2),
        "volatility": volatility,
        "derivative_bound": derivative_bound,
        "jacobian_bound": jacobian_bound,
        "residual_bound": residual_bound,
        "separated_bound": separated_bound,
        "mean_abs_model_derivative": float(np.mean(np.abs(model_derivative))),
        "mean_abs_remainder": float(np.mean(remainder)),
        "jacobian_bound_holds": bool(volatility <= jacobian_bound + 1e-12),
        "residual_bound_holds": bool(volatility <= residual_bound + 1e-12),
        "derivative_bound_holds": bool(volatility <= derivative_bound + 1e-12),
    }


def run_suite(output_dir: Path, force: bool = False) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "conditional_remainder_raw.csv"
    summary_path = output_dir / "conditional_remainder_summary.csv"
    json_path = output_dir / "conditional_remainder_summary.json"
    if not force and raw_path.exists() and summary_path.exists() and json_path.exists():
        return {"raw": pd.read_csv(raw_path), "summary": pd.read_csv(summary_path)}

    rows: list[dict[str, float | int | str | bool]] = []
    for conditional_slope in CONDITIONAL_SLOPES:
        for method, penalty_lambda in METHODS.items():
            for seed in SEEDS:
                parameters = fit_logistic_source(seed, conditional_slope, penalty_lambda)
                metrics = evaluate_path(parameters, conditional_slope)
                rows.append(
                    {
                        "conditional_slope": conditional_slope,
                        "method": method,
                        "penalty_lambda": penalty_lambda,
                        "seed": seed,
                        **metrics,
                    }
                )
    raw = pd.DataFrame(rows).sort_values(["conditional_slope", "method", "seed"])
    numeric_metrics = [
        "score_slope",
        "directional_gain",
        "volatility",
        "derivative_bound",
        "jacobian_bound",
        "residual_bound",
        "separated_bound",
        "mean_abs_model_derivative",
        "mean_abs_remainder",
    ]
    summary = raw.groupby(["conditional_slope", "method"], as_index=False)[numeric_metrics].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    hold_rates = (
        raw.groupby(["conditional_slope", "method"], as_index=False)[
            ["jacobian_bound_holds", "residual_bound_holds", "derivative_bound_holds"]
        ]
        .mean()
        .rename(
            columns={
                "jacobian_bound_holds": "jacobian_bound_hold_rate",
                "residual_bound_holds": "residual_bound_hold_rate",
                "derivative_bound_holds": "derivative_bound_hold_rate",
            }
        )
    )
    summary = summary.merge(hold_rates, on=["conditional_slope", "method"])

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "setup": {
                    "source_size": SOURCE_SIZE,
                    "source_scale": SOURCE_SCALE,
                    "source_log_odds": SOURCE_LOG_ODDS,
                    "deploy_distance": DEPLOY_DISTANCE,
                    "seeds": len(SEEDS),
                    "dtr_lambda": METHODS["dtr"],
                },
                "summary": summary.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )
    return {"raw": raw, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    results = run_suite(args.output_dir, force=args.force)
    print(results["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
