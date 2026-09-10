from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_gas_sensor_dtr_benchmark import DEVICE, OUTPUT_DIR, load_dataset, prepare_data  # noqa: E402
from run_gas_sensor_hybrid_sweep import train_hybrid  # noqa: E402


GAS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone"]
METRICS = [
    "deploy_ce",
    "deploy_volatility",
    "deploy_terminal_ce",
    "deploy_macro_accuracy",
]
SPECS = {
    "standard": ("standard", 0.0, 0.0),
    "dtr_ce": ("dtr", 0.0, 1.0),
    "isotropic_ce": ("isotropic", 1.0, 1.0),
    "hybrid_ce": ("hybrid", 0.003, 3.0),
    "hybrid_stable": ("hybrid", 0.3, 10.0),
}


def bootstrap_interval(
    differences: np.ndarray, rng: np.random.Generator, draws: int = 50_000
) -> list[float]:
    sampled = rng.choice(
        differences, size=(draws, len(differences)), replace=True
    ).mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def rows_for(raw: pd.DataFrame, name: str) -> pd.DataFrame:
    family, lambda_perp, lambda_parallel = SPECS[name]
    return raw[
        (raw["family"] == family)
        & np.isclose(raw["lambda_perp"], lambda_perp)
        & np.isclose(raw["lambda_parallel"], lambda_parallel)
    ].copy()


def paired_comparison(
    raw: pd.DataFrame,
    candidate_name: str,
    reference_name: str,
    rng: np.random.Generator,
) -> dict[str, object]:
    paired = rows_for(raw, candidate_name).merge(
        rows_for(raw, reference_name), on="seed", suffixes=("_candidate", "_reference")
    )
    result: dict[str, object] = {
        "candidate": candidate_name,
        "reference": reference_name,
        "seeds": int(len(paired)),
    }
    for metric in METRICS:
        differences = (
            paired[f"{metric}_candidate"] - paired[f"{metric}_reference"]
        ).to_numpy(dtype=float)
        lower_is_better = metric != "deploy_macro_accuracy"
        wins = differences < 0 if lower_is_better else differences > 0
        result[metric] = {
            "mean_difference": float(differences.mean()),
            "bootstrap_95_ci": bootstrap_interval(differences, rng),
            "wins": int(wins.sum()),
        }
    return result


def classwise_results(results_dir: Path) -> pd.DataFrame:
    data = prepare_data(load_dataset())
    raw_path = results_dir / "hybrid_classwise_raw.csv"
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    for name in ["hybrid_ce", "hybrid_stable"]:
        family, lambda_perp, lambda_parallel = SPECS[name]
        for seed in range(10):
            if not existing.empty and bool(
                ((existing["configuration"] == name) & (existing["seed"] == seed)).any()
            ):
                continue
            model = train_hybrid(data, seed, lambda_perp, lambda_parallel, epochs=35)
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(data.deploy_x).to(DEVICE)).cpu().numpy()
            shifted = logits - logits.max(axis=1, keepdims=True)
            log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
            predictions = logits.argmax(axis=1)
            for class_index, gas_name in enumerate(GAS_NAMES):
                mask = data.deploy_y == class_index
                rows.append(
                    {
                        "configuration": name,
                        "family": family,
                        "lambda_perp": lambda_perp,
                        "lambda_parallel": lambda_parallel,
                        "seed": seed,
                        "gas": gas_name,
                        "cross_entropy": float(-log_probabilities[mask, class_index].mean()),
                        "accuracy": float((predictions[mask] == class_index).mean()),
                        "rows": int(mask.sum()),
                    }
                )
            print(f"classwise {name}: seed {seed}", flush=True)
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(raw_path, index=False)
    summary = (
        combined.groupby(["configuration", "gas"], as_index=False)
        .agg(
            cross_entropy_mean=("cross_entropy", "mean"),
            cross_entropy_std=("cross_entropy", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            rows=("rows", "first"),
        )
        .sort_values(["configuration", "gas"])
    )
    summary.to_csv(results_dir / "hybrid_classwise_summary.csv", index=False)
    return summary


def main() -> None:
    torch.set_num_threads(1)
    results_dir = OUTPUT_DIR / "results"
    raw = pd.read_csv(results_dir / "hybrid_sweep_raw.csv")
    rng = np.random.default_rng(84_721)
    comparisons = [
        paired_comparison(raw, candidate, reference, rng)
        for candidate in ["hybrid_ce", "hybrid_stable"]
        for reference in ["standard", "dtr_ce", "isotropic_ce"]
    ]
    classwise = classwise_results(results_dir)
    report = {"paired_comparisons": comparisons}
    with (results_dir / "hybrid_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print("\nClasswise deployment results")
    print(classwise.to_string(index=False))


if __name__ == "__main__":
    main()
