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
    NUM_FEATURES,
    OUTPUT_DIR,
    fit_and_evaluate,
    load_dataset,
    prepare_data,
)


def main() -> None:
    torch.set_num_threads(1)
    data = prepare_data(load_dataset())
    rng = np.random.default_rng(7319)
    rows: list[dict[str, float | int | str]] = []
    geometry: list[dict[str, float | str]] = []
    for basis_index in range(10):
        q, _ = np.linalg.qr(rng.normal(size=(NUM_FEATURES, 4)))
        basis = q[:, :4].astype(np.float32)
        energy = float(
            np.square(data.deploy_diffs @ basis).sum()
            / np.square(data.deploy_diffs).sum()
        )
        geometry.append(
            {
                "basis": f"ambient_random_k4_{basis_index:02d}",
                "captured_deploy_drift_energy": energy,
            }
        )
        for seed in range(10):
            rows.append(
                fit_and_evaluate(
                    data,
                    seed,
                    "dtr",
                    f"ambient_random_k4_{basis_index:02d}",
                    1.0,
                    basis,
                    epochs=35,
                )
            )
            print(f"rank-4 controls {basis_index * 10 + seed + 1}/100", flush=True)

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("basis", as_index=False)
        .agg(
            deploy_ce_mean=("deploy_ce", "mean"),
            volatility_mean=("volatility", "mean"),
            terminal_ce_mean=("terminal_ce", "mean"),
            deploy_macro_accuracy_mean=("deploy_macro_accuracy", "mean"),
        )
        .merge(pd.DataFrame(geometry), on="basis")
    )
    results_dir = OUTPUT_DIR / "results"
    raw.to_csv(results_dir / "full_rank4_random_raw.csv", index=False)
    summary.to_csv(results_dir / "full_rank4_random_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
