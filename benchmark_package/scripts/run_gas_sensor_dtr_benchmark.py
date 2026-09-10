from __future__ import annotations

import argparse
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "benchmark_package" / "gas_sensor_array_drift"
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/270/"
    "gas+sensor+array+drift+dataset+at+different+concentrations.zip"
)
ZIP_PATH = OUTPUT_DIR / "gas_sensor_array_drift.zip"
NUM_FEATURES = 128
NUM_CLASSES = 5
SUBSPACE_DIM = 2
FULL_LAMBDAS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
PILOT_LAMBDAS = [0.0, 1e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
DEVICE = torch.device("cpu")


@dataclass(frozen=True)
class PreparedData:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    val_batches: np.ndarray
    deploy_x: np.ndarray
    deploy_y: np.ndarray
    deploy_batches: np.ndarray
    prospective_basis: np.ndarray
    oracle_basis: np.ndarray
    predeploy_span: np.ndarray
    deploy_diffs: np.ndarray
    summary: dict[str, object]


class Classifier(nn.Module):
    def __init__(self, hidden_width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_dataset() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"Downloading {DATA_URL}", flush=True)
        urlretrieve(DATA_URL, ZIP_PATH)

    rows: list[dict[str, float | int]] = []
    with zipfile.ZipFile(ZIP_PATH) as archive:
        for batch in range(1, 11):
            with archive.open(f"batch{batch}.dat") as handle:
                for raw_line in handle:
                    parts = raw_line.decode("utf-8").strip().split()
                    gas_text, concentration_text = parts[0].split(";")
                    gas = int(gas_text)
                    if gas == 6:
                        continue
                    row: dict[str, float | int] = {
                        "batch": batch,
                        "gas": gas - 1,
                        "concentration": float(concentration_text),
                    }
                    for item in parts[1:]:
                        index_text, value_text = item.split(":")
                        row[f"x{int(index_text):03d}"] = float(value_text)
                    rows.append(row)
    frame = pd.DataFrame(rows)
    feature_columns = [f"x{index:03d}" for index in range(1, NUM_FEATURES + 1)]
    return frame[["batch", "gas", "concentration", *feature_columns]]


def exposure_design(gas: np.ndarray, concentration: np.ndarray) -> np.ndarray:
    log_concentration = np.log(np.maximum(concentration, 1e-6))
    one_hot = np.eye(NUM_CLASSES, dtype=np.float64)[gas]
    return np.concatenate(
        [
            one_hot,
            one_hot * log_concentration[:, None],
            one_hot * np.square(log_concentration[:, None]),
        ],
        axis=1,
    )


def residual_block_differences(
    x: np.ndarray,
    gas: np.ndarray,
    concentration: np.ndarray,
    batches: np.ndarray,
    coefficients: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = exposure_design(gas, concentration)
    if coefficients is None:
        coefficients, *_ = np.linalg.lstsq(design, x, rcond=None)
    residual = x - design @ coefficients
    labels = sorted(np.unique(batches).tolist())
    means = np.asarray([residual[batches == label].mean(axis=0) for label in labels])
    return np.diff(means, axis=0), coefficients, residual


def svd_basis(diffs: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    _, singular_values, vh = np.linalg.svd(diffs, full_matrices=False)
    return vh[:rank].T.astype(np.float32), singular_values


def prepare_data(frame: pd.DataFrame) -> PreparedData:
    feature_columns = [f"x{index:03d}" for index in range(1, NUM_FEATURES + 1)]
    train_mask = frame["batch"] <= 2
    val_mask = frame["batch"].between(3, 5)
    deploy_mask = frame["batch"] >= 6

    train_raw = frame.loc[train_mask, feature_columns].to_numpy(dtype=np.float64)
    mean = train_raw.mean(axis=0)
    scale = train_raw.std(axis=0)
    scale[scale == 0.0] = 1.0

    all_x = ((frame[feature_columns].to_numpy(dtype=np.float64) - mean) / scale).astype(np.float32)
    gas = frame["gas"].to_numpy(dtype=np.int64)
    concentration = frame["concentration"].to_numpy(dtype=np.float64)
    batches = frame["batch"].to_numpy(dtype=np.int64)

    pre_mask = batches <= 5
    pre_diffs, exposure_coefficients, _ = residual_block_differences(
        all_x[pre_mask], gas[pre_mask], concentration[pre_mask], batches[pre_mask]
    )
    prospective_basis, pre_singular_values = svd_basis(pre_diffs, SUBSPACE_DIM)
    predeploy_span, _ = svd_basis(pre_diffs, min(len(pre_diffs), 4))

    deploy_diffs, _, _ = residual_block_differences(
        all_x[deploy_mask],
        gas[deploy_mask],
        concentration[deploy_mask],
        batches[deploy_mask],
        coefficients=exposure_coefficients,
    )
    oracle_basis, deploy_singular_values = svd_basis(deploy_diffs, SUBSPACE_DIM)

    summary = {
        "dataset": "UCI Gas Sensor Array Drift at Different Concentrations",
        "excluded_gas": "Toluene (class 6; absent from several middle batches)",
        "task": "five-gas classification",
        "train_batches": [1, 2],
        "validation_batches": [3, 4, 5],
        "deployment_batches": [6, 7, 8, 9, 10],
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(val_mask.sum()),
        "deployment_rows": int(deploy_mask.sum()),
        "features": NUM_FEATURES,
        "subspace_rank": SUBSPACE_DIM,
        "predeploy_singular_values": pre_singular_values.tolist(),
        "deployment_singular_values": deploy_singular_values.tolist(),
    }
    return PreparedData(
        train_x=all_x[train_mask],
        train_y=gas[train_mask],
        val_x=all_x[val_mask],
        val_y=gas[val_mask],
        val_batches=batches[val_mask],
        deploy_x=all_x[deploy_mask],
        deploy_y=gas[deploy_mask],
        deploy_batches=batches[deploy_mask],
        prospective_basis=prospective_basis,
        oracle_basis=oracle_basis,
        predeploy_span=predeploy_span,
        deploy_diffs=deploy_diffs.astype(np.float32),
        summary=summary,
    )


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    weights = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def centered_logit_jacobian_penalty(
    logits: torch.Tensor,
    inputs: torch.Tensor,
    method: str,
    basis: torch.Tensor | None,
) -> torch.Tensor:
    if method == "standard":
        return torch.zeros((), device=inputs.device)
    centered = logits - logits.mean(dim=1, keepdim=True)
    total = torch.zeros((), device=inputs.device)
    for class_index in range(NUM_CLASSES):
        gradient = torch.autograd.grad(
            centered[:, class_index].sum(),
            inputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        if method == "isotropic":
            total = total + gradient.square().sum(dim=1).mean()
        elif method == "dtr" and basis is not None:
            total = total + (gradient @ basis).square().sum(dim=1).mean()
        else:
            raise ValueError(f"Unsupported method: {method}")
    return total / NUM_CLASSES


def train_model(
    data: PreparedData,
    seed: int,
    method: str,
    penalty_lambda: float,
    basis: np.ndarray | None,
    epochs: int,
) -> Classifier:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Classifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights(data.train_y))
    dataset = TensorDataset(torch.from_numpy(data.train_x), torch.from_numpy(data.train_y))
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 311),
    )
    basis_tensor = None if basis is None else torch.from_numpy(basis).to(DEVICE)

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE).clone().detach().requires_grad_(True)
            batch_y = batch_y.to(DEVICE)
            logits = model(batch_x)
            fit_loss = criterion(logits, batch_y)
            penalty = centered_logit_jacobian_penalty(logits, batch_x, method, basis_tensor)
            objective = fit_loss + penalty_lambda * penalty
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
    return model


def balanced_metrics(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    losses = -log_probabilities[np.arange(len(labels)), labels]
    predictions = logits.argmax(axis=1)
    class_losses = [losses[labels == class_index].mean() for class_index in range(NUM_CLASSES)]
    class_accuracies = [
        (predictions[labels == class_index] == class_index).mean()
        for class_index in range(NUM_CLASSES)
    ]
    return float(np.mean(class_losses)), float(np.mean(class_accuracies))


def projected_gain(
    model: Classifier,
    values: np.ndarray,
    basis: np.ndarray,
    batch_size: int = 512,
) -> float:
    model.eval()
    basis_tensor = torch.from_numpy(basis).to(DEVICE)
    total = 0.0
    count = 0
    for start in range(0, len(values), batch_size):
        inputs = torch.from_numpy(values[start : start + batch_size]).to(DEVICE).requires_grad_(True)
        logits = model(inputs)
        centered = logits - logits.mean(dim=1, keepdim=True)
        batch_energy = torch.zeros(len(inputs), device=DEVICE)
        for class_index in range(NUM_CLASSES):
            gradient = torch.autograd.grad(
                centered[:, class_index].sum(), inputs, retain_graph=True
            )[0]
            batch_energy += (gradient @ basis_tensor).square().sum(dim=1) / NUM_CLASSES
        total += float(batch_energy.sum().detach().cpu())
        count += len(inputs)
    return total / count


def evaluate_model(
    model: Classifier,
    data: PreparedData,
    seed: int,
    method: str,
    basis_name: str,
    penalty_lambda: float,
    basis: np.ndarray | None,
) -> dict[str, float | int | str]:
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(data.val_x).to(DEVICE)).cpu().numpy()
        deploy_logits = model(torch.from_numpy(data.deploy_x).to(DEVICE)).cpu().numpy()
    val_ce, val_macro_accuracy = balanced_metrics(val_logits, data.val_y)
    deploy_ce, deploy_macro_accuracy = balanced_metrics(deploy_logits, data.deploy_y)

    block_ce: list[float] = []
    block_accuracy: list[float] = []
    for batch in sorted(np.unique(data.deploy_batches)):
        mask = data.deploy_batches == batch
        ce, accuracy = balanced_metrics(deploy_logits[mask], data.deploy_y[mask])
        block_ce.append(ce)
        block_accuracy.append(accuracy)

    gain_basis = data.prospective_basis if basis is None else basis
    val_directional_gain = projected_gain(model, data.val_x, gain_basis)
    return {
        "seed": seed,
        "method": method,
        "basis": basis_name,
        "lambda": penalty_lambda,
        "val_ce": val_ce,
        "val_macro_accuracy": val_macro_accuracy,
        "val_directional_gain": val_directional_gain,
        "deploy_ce": deploy_ce,
        "deploy_macro_accuracy": deploy_macro_accuracy,
        "volatility": float(np.var(block_ce)),
        "terminal_ce": float(block_ce[-1]),
        "worst_block_ce": float(np.max(block_ce)),
        "block_ce": json.dumps(block_ce),
        "block_macro_accuracy": json.dumps(block_accuracy),
    }


def fit_and_evaluate(
    data: PreparedData,
    seed: int,
    method: str,
    basis_name: str,
    penalty_lambda: float,
    basis: np.ndarray | None,
    epochs: int,
) -> dict[str, float | int | str]:
    started = time.perf_counter()
    model = train_model(data, seed, method, penalty_lambda, basis, epochs)
    row = evaluate_model(model, data, seed, method, basis_name, penalty_lambda, basis)
    row["seconds"] = time.perf_counter() - started
    return row


def random_basis(rng: np.random.Generator, ambient_basis: np.ndarray | None = None) -> np.ndarray:
    if ambient_basis is None:
        draw = rng.normal(size=(NUM_FEATURES, SUBSPACE_DIM))
    else:
        draw = ambient_basis @ rng.normal(size=(ambient_basis.shape[1], SUBSPACE_DIM))
    q, _ = np.linalg.qr(draw)
    return q[:, :SUBSPACE_DIM].astype(np.float32)


def principal_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    singular_values = np.linalg.svd(first.T @ second, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(singular_values.min(), 0.0, 1.0))))


def captured_energy(diffs: np.ndarray, basis: np.ndarray) -> float:
    denominator = float(np.square(diffs).sum())
    return float(np.square(diffs @ basis).sum() / denominator) if denominator > 0.0 else 0.0


def select_lambda(rows: pd.DataFrame, basis_name: str) -> float:
    subset = rows[rows["basis"] == basis_name]
    means = (
        subset.groupby("lambda", as_index=False)[["val_ce", "val_directional_gain"]]
        .mean()
        .sort_values(["val_ce", "val_directional_gain", "lambda"])
    )
    return float(means.iloc[0]["lambda"])


def summarize(selected: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    result = (
        selected.groupby(["method", "basis"], as_index=False)
        .agg(
            lambda_value=("lambda", "first"),
            deploy_ce_mean=("deploy_ce", "mean"),
            deploy_ce_std=("deploy_ce", "std"),
            volatility_mean=("volatility", "mean"),
            volatility_std=("volatility", "std"),
            terminal_ce_mean=("terminal_ce", "mean"),
            terminal_ce_std=("terminal_ce", "std"),
            worst_block_ce_mean=("worst_block_ce", "mean"),
            deploy_macro_accuracy_mean=("deploy_macro_accuracy", "mean"),
        )
        .merge(geometry, on="basis", how="left")
    )
    standard = selected[selected["basis"] == "standard"]
    for index, summary_row in result.iterrows():
        if summary_row["basis"] == "standard":
            continue
        candidate = selected[selected["basis"] == summary_row["basis"]]
        paired = candidate.merge(
            standard[["seed", "deploy_ce", "volatility", "terminal_ce"]],
            on="seed",
            suffixes=("_candidate", "_standard"),
        )
        result.loc[index, "deploy_ce_wins_vs_standard"] = int(
            (paired["deploy_ce_candidate"] < paired["deploy_ce_standard"]).sum()
        )
        result.loc[index, "volatility_wins_vs_standard"] = int(
            (paired["volatility_candidate"] < paired["volatility_standard"]).sum()
        )
        result.loc[index, "terminal_wins_vs_standard"] = int(
            (paired["terminal_ce_candidate"] < paired["terminal_ce_standard"]).sum()
        )
    return result


def run(mode: str, force: bool) -> None:
    torch.set_num_threads(1)
    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / f"{mode}_raw.csv"
    selected_path = results_dir / f"{mode}_selected.csv"
    summary_path = results_dir / f"{mode}_summary.csv"
    json_path = results_dir / f"{mode}_summary.json"
    if not force and all(path.exists() for path in [raw_path, selected_path, summary_path, json_path]):
        print(pd.read_csv(summary_path).to_string(index=False))
        return

    frame = load_dataset()
    data = prepare_data(frame)
    seeds = list(range(3 if mode == "pilot" else 10))
    lambdas = PILOT_LAMBDAS if mode == "pilot" else FULL_LAMBDAS
    random_count = 3 if mode == "pilot" else 10
    epochs = 20 if mode == "pilot" else 35
    print(json.dumps(data.summary, indent=2), flush=True)

    rows: list[dict[str, float | int | str]] = []
    main_specs = [
        ("standard", "standard", None),
        ("isotropic", "isotropic", None),
        ("dtr", "prospective", data.prospective_basis),
        ("dtr", "oracle_deployment", data.oracle_basis),
    ]
    total_main = len(seeds) + (len(main_specs) - 1) * len(lambdas) * len(seeds)
    completed = 0
    for method, basis_name, basis in main_specs:
        method_lambdas = [0.0] if method == "standard" else lambdas
        for penalty_lambda in method_lambdas:
            for seed in seeds:
                rows.append(
                    fit_and_evaluate(
                        data, seed, method, basis_name, penalty_lambda, basis, epochs
                    )
                )
                completed += 1
                print(
                    f"main {completed}/{total_main}: {basis_name} lambda={penalty_lambda:g} seed={seed}",
                    flush=True,
                )

    main = pd.DataFrame(rows)
    selected_prospective_lambda = select_lambda(main, "prospective")
    rng = np.random.default_rng(2026)
    random_specs: list[tuple[str, np.ndarray]] = []
    for index in range(random_count):
        random_specs.append((f"ambient_random_{index:02d}", random_basis(rng)))
    for index in range(random_count):
        random_specs.append(
            (f"drift_span_random_{index:02d}", random_basis(rng, data.predeploy_span))
        )

    total_random = len(random_specs) * len(seeds)
    completed = 0
    for basis_name, basis in random_specs:
        for seed in seeds:
            rows.append(
                fit_and_evaluate(
                    data,
                    seed,
                    "dtr",
                    basis_name,
                    selected_prospective_lambda,
                    basis,
                    epochs,
                )
            )
            completed += 1
            print(f"controls {completed}/{total_random}: {basis_name} seed={seed}", flush=True)

    raw = pd.DataFrame(rows)
    selected_frames = [raw[raw["basis"] == "standard"]]
    for basis_name in ["isotropic", "prospective", "oracle_deployment"]:
        chosen_lambda = select_lambda(raw, basis_name)
        selected_frames.append(
            raw[(raw["basis"] == basis_name) & np.isclose(raw["lambda"], chosen_lambda)]
        )
    selected_frames.append(raw[raw["basis"].str.contains("random")])
    selected = pd.concat(selected_frames, ignore_index=True)

    basis_lookup = {
        "prospective": data.prospective_basis,
        "oracle_deployment": data.oracle_basis,
        **dict(random_specs),
    }
    geometry = pd.DataFrame(
        [
            {
                "basis": name,
                "angle_to_oracle_deg": np.nan,
                "captured_deploy_drift_energy": np.nan,
            }
            for name in ["standard", "isotropic"]
        ]
        + [
            {
                "basis": name,
                "angle_to_oracle_deg": principal_angle_degrees(basis, data.oracle_basis),
                "captured_deploy_drift_energy": captured_energy(data.deploy_diffs, basis),
            }
            for name, basis in basis_lookup.items()
        ]
    )
    summary = summarize(selected, geometry)

    random_rows = summary[summary["basis"].str.contains("random")].copy()
    aggregate_random = (
        random_rows.assign(
            control_type=np.where(
                random_rows["basis"].str.startswith("ambient"),
                "ambient_random_distribution",
                "drift_span_random_distribution",
            )
        )
        .groupby("control_type", as_index=False)
        .agg(
            bases=("basis", "count"),
            deploy_ce_mean=("deploy_ce_mean", "mean"),
            deploy_ce_basis_std=("deploy_ce_mean", "std"),
            volatility_mean=("volatility_mean", "mean"),
            volatility_basis_std=("volatility_mean", "std"),
            terminal_ce_mean=("terminal_ce_mean", "mean"),
            captured_energy_mean=("captured_deploy_drift_energy", "mean"),
            angle_to_oracle_mean=("angle_to_oracle_deg", "mean"),
        )
    )

    raw.to_csv(raw_path, index=False)
    selected.to_csv(selected_path, index=False)
    summary.to_csv(summary_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "setup": {**data.summary, "mode": mode, "seeds": len(seeds), "epochs": epochs},
                "selection": {
                    "rule": "Mean class-balanced validation CE; directional gain breaks ties.",
                    "prospective_lambda_used_for_random_controls": selected_prospective_lambda,
                },
                "main_results": summary[
                    summary["basis"].isin(
                        ["standard", "isotropic", "prospective", "oracle_deployment"]
                    )
                ].to_dict(orient="records"),
                "random_control_distributions": aggregate_random.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )
    print("\nSelected main results", flush=True)
    print(
        summary[
            summary["basis"].isin(
                ["standard", "isotropic", "prospective", "oracle_deployment"]
            )
        ].to_string(index=False),
        flush=True,
    )
    print("\nRandom control distributions", flush=True)
    print(aggregate_random.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.mode, args.force)


if __name__ == "__main__":
    main()
