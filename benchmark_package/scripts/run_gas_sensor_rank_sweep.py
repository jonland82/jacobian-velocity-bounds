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

from run_gas_sensor_dtr_benchmark import (  # noqa: E402
    FULL_LAMBDAS,
    OUTPUT_DIR,
    fit_and_evaluate,
    load_dataset,
    prepare_data,
)


def main() -> None:
    torch.set_num_threads(1)
    results_dir = OUTPUT_DIR / "results"
    output_path = results_dir / "full_prospective_rank_sweep.csv"
    summary_path = results_dir / "full_prospective_rank_summary.csv"
    json_path = results_dir / "full_prospective_rank_selection.json"

    data = prepare_data(load_dataset())
    existing = pd.read_csv(results_dir / "full_raw.csv")
    rank_two = existing[existing["basis"] == "prospective"].copy()
    rank_two["rank"] = 2
    rank_two["basis"] = "prospective_k2"
    rows = [rank_two]

    total = 3 * len(FULL_LAMBDAS) * 10
    completed = 0
    for rank in [1, 3, 4]:
        basis = data.predeploy_span[:, :rank].copy()
        rank_rows: list[dict[str, float | int | str]] = []
        for penalty_lambda in FULL_LAMBDAS:
            for seed in range(10):
                rank_rows.append(
                    {
                        **fit_and_evaluate(
                            data,
                            seed,
                            "dtr",
                            f"prospective_k{rank}",
                            penalty_lambda,
                            basis,
                            epochs=35,
                        ),
                        "rank": rank,
                    }
                )
                completed += 1
                print(
                    f"rank sweep {completed}/{total}: k={rank} lambda={penalty_lambda:g} seed={seed}",
                    flush=True,
                )
        rows.append(pd.DataFrame(rank_rows))

    raw = pd.concat(rows, ignore_index=True)
    summary = (
        raw.groupby(["rank", "lambda"], as_index=False)
        .agg(
            val_ce_mean=("val_ce", "mean"),
            val_macro_accuracy_mean=("val_macro_accuracy", "mean"),
            deploy_ce_mean=("deploy_ce", "mean"),
            deploy_ce_std=("deploy_ce", "std"),
            volatility_mean=("volatility", "mean"),
            volatility_std=("volatility", "std"),
            terminal_ce_mean=("terminal_ce", "mean"),
            deploy_macro_accuracy_mean=("deploy_macro_accuracy", "mean"),
        )
        .sort_values(["val_ce_mean", "rank", "lambda"])
    )
    selected = summary.iloc[0].to_dict()
    raw.to_csv(output_path, index=False)
    summary.to_csv(summary_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selection_rule": "Minimum mean class-balanced validation cross-entropy over rank and lambda.",
                "selected": selected,
                "best_by_rank": (
                    summary.sort_values("val_ce_mean").groupby("rank", as_index=False).first()
                ).to_dict(orient="records"),
            },
            handle,
            indent=2,
        )
    print("\nBest by rank", flush=True)
    print(
        summary.sort_values("val_ce_mean").groupby("rank", as_index=False).first().to_string(
            index=False
        ),
        flush=True,
    )
    print("\nSelected", flush=True)
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()
