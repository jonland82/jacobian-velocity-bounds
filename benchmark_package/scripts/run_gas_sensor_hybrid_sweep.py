from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_gas_sensor_dtr_benchmark import (  # noqa: E402
    DEVICE,
    NUM_CLASSES,
    OUTPUT_DIR,
    Classifier,
    PreparedData,
    balanced_metrics,
    class_weights,
    load_dataset,
    prepare_data,
)


LEVELS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
HYBRID_PERP = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
HYBRID_PARALLEL = [1e-1, 3e-1, 1.0, 3.0, 10.0]


def configuration_grid() -> list[tuple[float, float, str]]:
    configs: set[tuple[float, float, str]] = {(0.0, value, "dtr") for value in LEVELS}
    configs.update((value, value, "isotropic") for value in LEVELS[1:])
    configs.add((0.0, 0.0, "standard"))
    for lambda_perp in HYBRID_PERP:
        for lambda_parallel in HYBRID_PARALLEL:
            if lambda_parallel > lambda_perp:
                configs.add((lambda_perp, lambda_parallel, "hybrid"))
    return sorted(configs, key=lambda item: (item[2], item[0], item[1]))


def weighted_jacobian_penalty(
    logits: torch.Tensor,
    inputs: torch.Tensor,
    basis: torch.Tensor,
    lambda_perp: float,
    lambda_parallel: float,
) -> torch.Tensor:
    if lambda_perp == 0.0 and lambda_parallel == 0.0:
        return torch.zeros((), device=inputs.device)
    centered = logits - logits.mean(dim=1, keepdim=True)
    isotropic_energy = torch.zeros((), device=inputs.device)
    directional_energy = torch.zeros((), device=inputs.device)
    for class_index in range(NUM_CLASSES):
        gradient = torch.autograd.grad(
            centered[:, class_index].sum(),
            inputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        isotropic_energy += gradient.square().sum(dim=1).mean() / NUM_CLASSES
        directional_energy += (gradient @ basis).square().sum(dim=1).mean() / NUM_CLASSES
    return (
        lambda_perp * isotropic_energy
        + (lambda_parallel - lambda_perp) * directional_energy
    )


def train_hybrid(
    data: PreparedData,
    seed: int,
    lambda_perp: float,
    lambda_parallel: float,
    epochs: int,
) -> Classifier:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Classifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights(data.train_y))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data.train_x), torch.from_numpy(data.train_y)),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 311),
    )
    basis = torch.from_numpy(data.predeploy_span[:, :4]).to(DEVICE)
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE).clone().detach().requires_grad_(True)
            batch_y = batch_y.to(DEVICE)
            logits = model(batch_x)
            fit_loss = criterion(logits, batch_y)
            penalty = weighted_jacobian_penalty(
                logits, batch_x, basis, lambda_perp, lambda_parallel
            )
            optimizer.zero_grad()
            (fit_loss + penalty).backward()
            optimizer.step()
    return model


def block_metrics(
    logits: np.ndarray, labels: np.ndarray, batches: np.ndarray
) -> tuple[list[float], list[float]]:
    losses: list[float] = []
    accuracies: list[float] = []
    for batch in sorted(np.unique(batches)):
        mask = batches == batch
        loss, accuracy = balanced_metrics(logits[mask], labels[mask])
        losses.append(loss)
        accuracies.append(accuracy)
    return losses, accuracies


def evaluate_hybrid(
    model: Classifier,
    data: PreparedData,
    seed: int,
    lambda_perp: float,
    lambda_parallel: float,
    family: str,
) -> dict[str, float | int | str]:
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(data.val_x).to(DEVICE)).cpu().numpy()
        deploy_logits = model(torch.from_numpy(data.deploy_x).to(DEVICE)).cpu().numpy()
    val_ce, val_accuracy = balanced_metrics(val_logits, data.val_y)
    deploy_ce, deploy_accuracy = balanced_metrics(deploy_logits, data.deploy_y)
    val_block_ce, val_block_accuracy = block_metrics(
        val_logits, data.val_y, data.val_batches
    )
    deploy_block_ce, deploy_block_accuracy = block_metrics(
        deploy_logits, data.deploy_y, data.deploy_batches
    )
    return {
        "seed": seed,
        "family": family,
        "lambda_perp": lambda_perp,
        "lambda_parallel": lambda_parallel,
        "val_ce": val_ce,
        "val_macro_accuracy": val_accuracy,
        "val_volatility": float(np.var(val_block_ce)),
        "val_terminal_ce": float(val_block_ce[-1]),
        "deploy_ce": deploy_ce,
        "deploy_macro_accuracy": deploy_accuracy,
        "deploy_volatility": float(np.var(deploy_block_ce)),
        "deploy_terminal_ce": float(deploy_block_ce[-1]),
        "val_block_ce": json.dumps(val_block_ce),
        "val_block_macro_accuracy": json.dumps(val_block_accuracy),
        "deploy_block_ce": json.dumps(deploy_block_ce),
        "deploy_block_macro_accuracy": json.dumps(deploy_block_accuracy),
    }


def pareto_mask(frame: pd.DataFrame) -> np.ndarray:
    objectives = frame[["val_ce_mean", "val_volatility_mean"]].to_numpy(dtype=float)
    negative_accuracy = -frame["val_macro_accuracy_mean"].to_numpy(dtype=float)
    objectives = np.column_stack([objectives, negative_accuracy])
    keep = np.ones(len(frame), dtype=bool)
    for index, point in enumerate(objectives):
        dominated = np.all(objectives <= point, axis=1) & np.any(objectives < point, axis=1)
        dominated[index] = False
        if dominated.any():
            keep[index] = False
    return keep


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(["family", "lambda_perp", "lambda_parallel"], as_index=False)
        .agg(
            val_ce_mean=("val_ce", "mean"),
            val_ce_std=("val_ce", "std"),
            val_macro_accuracy_mean=("val_macro_accuracy", "mean"),
            val_volatility_mean=("val_volatility", "mean"),
            deploy_ce_mean=("deploy_ce", "mean"),
            deploy_ce_std=("deploy_ce", "std"),
            deploy_macro_accuracy_mean=("deploy_macro_accuracy", "mean"),
            deploy_macro_accuracy_std=("deploy_macro_accuracy", "std"),
            deploy_volatility_mean=("deploy_volatility", "mean"),
            deploy_volatility_std=("deploy_volatility", "std"),
            deploy_terminal_ce_mean=("deploy_terminal_ce", "mean"),
            deploy_terminal_ce_std=("deploy_terminal_ce", "std"),
        )
        .sort_values(["family", "lambda_perp", "lambda_parallel"])
        .reset_index(drop=True)
    )
    summary["validation_pareto"] = pareto_mask(summary)
    summary["accuracy_constraint"] = summary["val_macro_accuracy_mean"] >= 0.90
    return summary


def selected_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    candidates: list[pd.Series] = []
    for family in ["standard", "dtr", "isotropic", "hybrid"]:
        family_rows = summary[summary["family"] == family]
        if family_rows.empty:
            continue
        best_ce = family_rows.sort_values(
            ["val_ce_mean", "val_volatility_mean", "lambda_parallel"]
        ).iloc[0].copy()
        best_ce["selection"] = "minimum_validation_ce"
        candidates.append(best_ce)
        constrained = family_rows[family_rows["accuracy_constraint"]]
        if not constrained.empty:
            best_stability = constrained.sort_values(
                ["val_volatility_mean", "val_ce_mean", "lambda_parallel"]
            ).iloc[0].copy()
            best_stability["selection"] = "minimum_validation_volatility_at_90pct_accuracy"
            candidates.append(best_stability)
    return pd.DataFrame(candidates)


def run(force: bool) -> None:
    torch.set_num_threads(1)
    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / "hybrid_sweep_raw.csv"
    summary_path = results_dir / "hybrid_sweep_summary.csv"
    selected_path = results_dir / "hybrid_selected_candidates.csv"
    configs = configuration_grid()
    data = prepare_data(load_dataset())

    if force or not raw_path.exists():
        raw = pd.DataFrame()
    else:
        raw = pd.read_csv(raw_path)
    total = len(configs) * 10
    completed = 0
    for lambda_perp, lambda_parallel, family in configs:
        for seed in range(10):
            already_done = False
            if not raw.empty:
                already_done = bool(
                    (
                        (raw["family"] == family)
                        & np.isclose(raw["lambda_perp"], lambda_perp)
                        & np.isclose(raw["lambda_parallel"], lambda_parallel)
                        & (raw["seed"] == seed)
                    ).any()
                )
            if not already_done:
                started = time.perf_counter()
                model = train_hybrid(
                    data, seed, lambda_perp, lambda_parallel, epochs=35
                )
                row = evaluate_hybrid(
                    model, data, seed, lambda_perp, lambda_parallel, family
                )
                row["seconds"] = time.perf_counter() - started
                raw = pd.concat([raw, pd.DataFrame([row])], ignore_index=True)
            completed += 1
            print(
                f"hybrid sweep {completed}/{total}: {family} "
                f"perp={lambda_perp:g} parallel={lambda_parallel:g} seed={seed}",
                flush=True,
            )
        raw.to_csv(raw_path, index=False)

    summary = summarize(raw)
    selected = selected_candidates(summary)
    summary.to_csv(summary_path, index=False)
    selected.to_csv(selected_path, index=False)
    print("\nValidation-selected candidates", flush=True)
    print(selected.to_string(index=False), flush=True)
    print("\nValidation Pareto frontier", flush=True)
    print(summary[summary["validation_pareto"]].to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.force)


if __name__ == "__main__":
    main()
