from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_synthetic_theorem_experiment import (
    DEVICE,
    SELECTED_DTR_LAMBDA,
    SWEEP_SEEDS,
    TRAIN_SIZE,
    DRIFT_DIRECTION,
    SmallReLUNet,
    evaluate_trajectory,
    sample_features,
    set_seed,
    summarize_trajectory,
)


COMPARISON_LAMBDAS = [0.01, 0.03, 0.08]
MISSPEC_LAMBDA = SELECTED_DTR_LAMBDA
MISSPEC_CASES = [
    ("aligned", 0.0),
    ("rotated20", 20.0),
    ("wrong", 90.0),
]


def jacobian_penalty(
    logits: torch.Tensor,
    inputs: torch.Tensor,
    method: str,
    direction: torch.Tensor | None,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        logits.sum(), inputs, create_graph=True, retain_graph=True
    )[0]
    if method == "standard":
        return torch.zeros((), device=inputs.device)
    if method == "isotropic":
        return grads.square().sum(dim=1).mean()
    if method == "dtr" and direction is not None:
        return (grads @ direction).square().mean()
    raise ValueError(f"Unsupported method: {method}")


def train_model(
    seed: int,
    method: str,
    penalty_lambda: float,
    direction: np.ndarray | None = None,
) -> SmallReLUNet:
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
    direction_tensor = None
    if direction is not None:
        direction_tensor = torch.tensor(direction, device=DEVICE, dtype=torch.float32)

    for _ in range(140):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE).clone().detach().requires_grad_(True)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            risk = criterion(logits, batch_y)
            penalty = jacobian_penalty(logits, batch_x, method, direction_tensor)
            weight_decay = 1e-4 * sum(param.square().sum() for param in model.parameters())
            (risk + penalty_lambda * penalty + weight_decay).backward()
            optimizer.step()

    return model


def rotation_direction(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    direction = np.array([math.sin(angle), math.cos(angle)], dtype=np.float32)
    direction /= np.linalg.norm(direction)
    return direction


def mean_metrics(df: pd.DataFrame) -> dict[str, float]:
    metric_columns = [
        "volatility",
        "bound_fd",
        "bound_chain",
        "bound_jv",
        "terminal_risk",
        "initial_risk",
        "max_risk",
        "mean_gain",
        "mean_hazard",
    ]
    return {column: float(df[column].mean()) for column in metric_columns}


def build_paper_summary(
    comparison_df: pd.DataFrame,
    misspec_df: pd.DataFrame,
) -> dict[str, object]:
    family_means = {}
    for method in ["standard", "isotropic", "dtr"]:
        subset = comparison_df[comparison_df["method"] == method]
        family_means[method] = mean_metrics(subset)

    selected_means = {}
    selected_df = comparison_df[
        (comparison_df["method"] == "standard")
        | (
            comparison_df["method"].isin(["isotropic", "dtr"])
            & np.isclose(comparison_df["lambda"], MISSPEC_LAMBDA)
        )
    ]
    for method in ["standard", "isotropic", "dtr"]:
        selected_means[method] = mean_metrics(selected_df[selected_df["method"] == method])

    misspec_means = {
        case: mean_metrics(misspec_df[misspec_df["case"] == case])
        for case, _ in MISSPEC_CASES
    }

    ratio_metrics = ["bound_chain", "volatility", "mean_gain", "terminal_risk"]
    comparison_ratios = {}
    standard_selected = selected_means["standard"]
    for method in ["isotropic", "dtr"]:
        comparison_ratios[method] = {
            metric: float(selected_means[method][metric] / standard_selected[metric])
            for metric in ratio_metrics
        }

    aligned = misspec_means["aligned"]
    misspec_ratios = {}
    for case in ["rotated20", "wrong"]:
        misspec_ratios[case] = {
            metric: float(misspec_means[case][metric] / aligned[metric])
            for metric in ratio_metrics
        }

    return {
        "comparison_family_mean": family_means,
        "comparison_selected_lambda": float(MISSPEC_LAMBDA),
        "comparison_selected_mean": selected_means,
        "comparison_selected_ratios": comparison_ratios,
        "misspecification_lambda": float(MISSPEC_LAMBDA),
        "misspecification_mean": misspec_means,
        "misspecification_ratios": misspec_ratios,
    }


def run_suite(output_dir: Path, force: bool = False) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "synthetic_directional_comparison_summary.csv"
    misspec_path = output_dir / "synthetic_directional_misspecification_summary.csv"
    summary_path = output_dir / "synthetic_directional_summary.json"

    if (
        not force
        and comparison_path.exists()
        and misspec_path.exists()
        and summary_path.exists()
    ):
        return {
            "comparison": pd.read_csv(comparison_path),
            "misspecification": pd.read_csv(misspec_path),
        }

    comparison_rows: list[dict[str, float | int | str]] = []
    comparison_specs = [("standard", 0.0, None)]
    comparison_specs.extend(("isotropic", lam, None) for lam in COMPARISON_LAMBDAS)
    comparison_specs.extend(("dtr", lam, DRIFT_DIRECTION) for lam in COMPARISON_LAMBDAS)

    for method, penalty_lambda, direction in comparison_specs:
        for seed in SWEEP_SEEDS:
            model = train_model(
                seed=seed,
                method=method,
                penalty_lambda=penalty_lambda,
                direction=direction,
            )
            trajectory = evaluate_trajectory(
                model,
                lambda_value=penalty_lambda,
                seed=seed,
                label=method,
            )
            comparison_rows.append(
                {
                    "method": method,
                    "lambda": float(penalty_lambda),
                    "seed": int(seed),
                    **summarize_trajectory(trajectory),
                }
            )

    misspec_rows: list[dict[str, float | int | str]] = []
    for case, angle_deg in MISSPEC_CASES:
        direction = rotation_direction(angle_deg)
        alignment = float(abs(direction @ DRIFT_DIRECTION))
        for seed in SWEEP_SEEDS:
            model = train_model(
                seed=seed,
                method="dtr",
                penalty_lambda=MISSPEC_LAMBDA,
                direction=direction,
            )
            trajectory = evaluate_trajectory(
                model,
                lambda_value=MISSPEC_LAMBDA,
                seed=seed,
                label=case,
            )
            misspec_rows.append(
                {
                    "case": case,
                    "angle_deg": float(angle_deg),
                    "alignment": alignment,
                    "seed": int(seed),
                    **summarize_trajectory(trajectory),
                }
            )

    comparison_df = (
        pd.DataFrame(comparison_rows)
        .sort_values(["method", "lambda", "seed"])
        .reset_index(drop=True)
    )
    misspec_df = (
        pd.DataFrame(misspec_rows)
        .sort_values(["angle_deg", "seed"])
        .reset_index(drop=True)
    )
    paper_summary = build_paper_summary(comparison_df, misspec_df)

    comparison_df.to_csv(comparison_path, index=False)
    misspec_df.to_csv(misspec_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(paper_summary, handle, indent=2)

    return {"comparison": comparison_df, "misspecification": misspec_df}


def ensure_directional_ablation_outputs(
    base_dir: Path, force: bool = False
) -> dict[str, pd.DataFrame]:
    return run_suite(base_dir / "figures", force=force)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    results = run_suite(args.output_dir, force=args.force)
    print(
        "Wrote "
        f"{len(results['comparison'])} comparison rows and "
        f"{len(results['misspecification'])} misspecification rows "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
