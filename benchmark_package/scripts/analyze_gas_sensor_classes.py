from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_gas_sensor_dtr_benchmark import (  # noqa: E402
    DEVICE,
    OUTPUT_DIR,
    load_dataset,
    prepare_data,
    train_model,
)


GAS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone"]


def main() -> None:
    torch.set_num_threads(1)
    data = prepare_data(load_dataset())
    specs = [
        ("standard", "standard", 0.0, None),
        ("isotropic", "isotropic", 1.0, None),
        ("dtr", "prospective", 1.0, data.prospective_basis),
        ("dtr", "prospective_k4", 1.0, data.predeploy_span[:, :4]),
        ("dtr", "oracle_deployment", 1.0, data.oracle_basis),
    ]
    results_dir = OUTPUT_DIR / "results"
    raw_path = results_dir / "full_classwise_raw.csv"
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    for method, basis_name, penalty_lambda, basis in specs:
        if not existing.empty and basis_name in set(existing["basis"]):
            continue
        for seed in range(10):
            model = train_model(data, seed, method, penalty_lambda, basis, epochs=35)
            with torch.no_grad():
                logits = model(torch.from_numpy(data.deploy_x).to(DEVICE)).cpu().numpy()
            shifted = logits - logits.max(axis=1, keepdims=True)
            log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
            predictions = logits.argmax(axis=1)
            for class_index, gas_name in enumerate(GAS_NAMES):
                mask = data.deploy_y == class_index
                rows.append(
                    {
                        "basis": basis_name,
                        "seed": seed,
                        "gas": gas_name,
                        "cross_entropy": float(
                            -log_probabilities[mask, class_index].mean()
                        ),
                        "accuracy": float((predictions[mask] == class_index).mean()),
                        "rows": int(mask.sum()),
                    }
                )
            print(f"{basis_name}: seed {seed}", flush=True)
    raw = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    summary = (
        raw.groupby(["basis", "gas"], as_index=False)
        .agg(
            cross_entropy_mean=("cross_entropy", "mean"),
            cross_entropy_std=("cross_entropy", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            rows=("rows", "first"),
        )
        .sort_values(["basis", "gas"])
    )
    raw.to_csv(raw_path, index=False)
    summary.to_csv(results_dir / "full_classwise_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
