from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from run_air_quality_experiment import run_suite as run_air_quality_experiment


BENCHMARK_SCRIPTS = Path(__file__).resolve().parents[1] / "benchmark_package" / "scripts"
if str(BENCHMARK_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SCRIPTS))

from run_tetouan_power_benchmark import run_suite as run_tetouan_benchmark


METRICS = ["deploy_mse", "volatility", "mean_gain", "terminal_risk"]
METHOD_ORDER = ["standard", "isotropic", "dtr"]
DISPLAY_DATASETS = {
    "air_quality": "Air Quality",
    "tetouan_zone1_power": "Tetouan",
}
BOOTSTRAP_RESAMPLES = 10000
GAIN_TARGET = 0.05


def _selected_rows(
    sweep_summary: pd.DataFrame,
    selected_lambdas: pd.DataFrame,
    dataset_key: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in selected_lambdas.itertuples(index=False):
        subset = sweep_summary[
            (sweep_summary["method"] == row.method)
            & np.isclose(sweep_summary["lambda"], row.selected_lambda)
        ].copy()
        subset["dataset_key"] = dataset_key
        subset["selected_lambda"] = float(row.selected_lambda)
        frames.append(subset)
    return pd.concat(frames, ignore_index=True)


def _summary_stats(selected_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset_key, method), subset in selected_rows.groupby(["dataset_key", "method"]):
        out: dict[str, object] = {
            "dataset_key": dataset_key,
            "dataset": DISPLAY_DATASETS[dataset_key],
            "method": method,
            "selected_lambda": float(subset["selected_lambda"].iloc[0]),
            "n_seeds": int(subset["seed"].nunique()),
        }
        for metric in METRICS:
            out[f"{metric}_mean"] = float(subset[metric].mean())
            out[f"{metric}_std"] = float(subset[metric].std(ddof=1))
        rows.append(out)

    order = {(dataset, method): idx for idx, (dataset, method) in enumerate(
        (dataset, method)
        for dataset in DISPLAY_DATASETS
        for method in METHOD_ORDER
    )}
    return (
        pd.DataFrame(rows)
        .assign(_order=lambda df: [order[(row.dataset_key, row.method)] for row in df.itertuples()])
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def _bootstrap_ci(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(diff) == 0:
        return float("nan"), float("nan")
    samples = rng.choice(diff, size=(BOOTSTRAP_RESAMPLES, len(diff)), replace=True)
    means = samples.mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _paired_comparisons(selected_rows: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260423)
    rows: list[dict[str, object]] = []
    for dataset_key, dataset_df in selected_rows.groupby("dataset_key"):
        dtr = dataset_df[dataset_df["method"] == "dtr"].set_index("seed")
        for baseline_name in ["standard", "isotropic"]:
            baseline = dataset_df[dataset_df["method"] == baseline_name].set_index("seed")
            matched_seeds = sorted(set(dtr.index).intersection(set(baseline.index)))
            for metric in METRICS:
                diff = (
                    dtr.loc[matched_seeds, metric].to_numpy(dtype=np.float64)
                    - baseline.loc[matched_seeds, metric].to_numpy(dtype=np.float64)
                )
                ci_lo, ci_hi = _bootstrap_ci(diff, rng)
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": DISPLAY_DATASETS[dataset_key],
                        "comparison": f"dtr_minus_{baseline_name}",
                        "metric": metric,
                        "n_seeds": int(len(diff)),
                        "wins": int(np.sum(diff < 0.0)),
                        "losses": int(np.sum(diff > 0.0)),
                        "ties": int(np.sum(diff == 0.0)),
                        "mean_difference": float(diff.mean()),
                        "bootstrap_ci_low": ci_lo,
                        "bootstrap_ci_high": ci_hi,
                    }
                )
    return pd.DataFrame(rows)


def _conservative_gain_selected_rows(
    sweep_summary: pd.DataFrame,
    dataset_key: str,
    gain_target: float = GAIN_TARGET,
) -> pd.DataFrame:
    standard = sweep_summary[
        (sweep_summary["method"] == "standard") & np.isclose(sweep_summary["lambda"], 0.0)
    ].copy()
    standard_val_mse = float(standard["val_mse"].mean())
    standard_val_gain = float(standard["val_directional_gain"].mean())
    standard["dataset_key"] = dataset_key
    standard["selected_lambda"] = 0.0
    standard["selection_rule"] = "standard"

    frames = [standard]
    for method in ["isotropic", "dtr"]:
        method_rows = sweep_summary[sweep_summary["method"] == method]
        grouped = (
            method_rows.groupby("lambda", as_index=False)[["val_mse", "val_directional_gain"]]
            .mean()
            .sort_values("lambda")
            .reset_index(drop=True)
        )
        grouped["val_gain_reduction"] = 1.0 - grouped["val_directional_gain"] / standard_val_gain
        eligible = grouped[
            (grouped["val_mse"] <= standard_val_mse)
            & (grouped["val_gain_reduction"] >= gain_target)
        ]
        if eligible.empty:
            chosen_lambda = float(
                grouped.sort_values(["val_mse", "val_directional_gain", "lambda"]).iloc[0]["lambda"]
            )
            rule = "fallback_validation_mse"
        else:
            chosen_lambda = float(eligible.iloc[0]["lambda"])
            rule = f"smallest_lambda_val_mse_le_standard_gain_reduction_ge_{gain_target:g}"

        selected = method_rows[np.isclose(method_rows["lambda"], chosen_lambda)].copy()
        selected["dataset_key"] = dataset_key
        selected["selected_lambda"] = chosen_lambda
        selected["selection_rule"] = rule
        frames.append(selected)

    return pd.concat(frames, ignore_index=True)


def _air_quality_dtr_path(air_summary: pd.DataFrame) -> pd.DataFrame:
    standard = air_summary[
        (air_summary["method"] == "standard") & np.isclose(air_summary["lambda"], 0.0)
    ].set_index("seed")
    rows: list[dict[str, object]] = []
    for lambda_value, subset in air_summary[air_summary["method"] == "dtr"].groupby("lambda"):
        selected = subset.set_index("seed")
        out: dict[str, object] = {
            "lambda": float(lambda_value),
            "n_seeds": int(selected.index.nunique()),
            "val_mse_mean": float(selected["val_mse"].mean()),
            "val_directional_gain_mean": float(selected["val_directional_gain"].mean()),
        }
        for metric in METRICS:
            diff = selected[metric] - standard[metric]
            out[f"{metric}_mean"] = float(selected[metric].mean())
            out[f"{metric}_std"] = float(selected[metric].std(ddof=1))
            out[f"{metric}_wins_vs_standard"] = int((diff < 0.0).sum())
            out[f"{metric}_mean_diff_vs_standard"] = float(diff.mean())
        rows.append(out)
    return pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x_valid = x[mask]
    y_valid = y[mask]
    if float(np.std(x_valid)) == 0.0 or float(np.std(y_valid)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def _hazard_records(dataset_key: str, trajectories: pd.DataFrame) -> pd.DataFrame:
    dtr = trajectories[trajectories["method"] == "dtr"].copy()
    dtr["speed_sq"] = dtr["speed"] ** 2
    records: list[dict[str, object]] = []
    for seed, seed_df in dtr.groupby("seed"):
        seed_df = seed_df.sort_values("block").reset_index(drop=True)
        for idx, row in seed_df.iterrows():
            if float(row["speed"]) <= 0.0:
                continue
            for score_name, column in [
                ("drift_only", "speed_sq"),
                ("gain_only", "gain"),
                ("matched", "hazard"),
            ]:
                base = {
                    "dataset_key": dataset_key,
                    "dataset": DISPLAY_DATASETS[dataset_key],
                    "seed": int(seed),
                    "block": row["block"],
                    "score": score_name,
                    "score_value": float(row[column]),
                    "same_block_risk": float(row["risk"]),
                }
                if idx + 1 < len(seed_df):
                    base["next_block_risk"] = float(seed_df.loc[idx + 1, "risk"])
                else:
                    base["next_block_risk"] = float("nan")
                records.append(base)
    return pd.DataFrame(records)


def _hazard_ablation(air_trajectories: pd.DataFrame, tetouan_trajectories: pd.DataFrame) -> pd.DataFrame:
    records = pd.concat(
        [
            _hazard_records("air_quality", air_trajectories),
            _hazard_records("tetouan_zone1_power", tetouan_trajectories),
        ],
        ignore_index=True,
    )
    rows: list[dict[str, object]] = []
    for (dataset_key, score_name), subset in records.groupby(["dataset_key", "score"]):
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": DISPLAY_DATASETS[dataset_key],
                "score": score_name,
                "same_block_corr": _corr(
                    subset["score_value"].to_numpy(dtype=np.float64),
                    subset["same_block_risk"].to_numpy(dtype=np.float64),
                ),
                "next_block_corr": _corr(
                    subset["score_value"].to_numpy(dtype=np.float64),
                    subset["next_block_risk"].to_numpy(dtype=np.float64),
                ),
                "n_same_block": int(subset["same_block_risk"].notna().sum()),
                "n_next_block": int(subset["next_block_risk"].notna().sum()),
            }
        )

    per_dataset = pd.DataFrame(rows)
    aggregate = (
        per_dataset.groupby("score", as_index=False)
        .agg(
            same_block_corr=("same_block_corr", "mean"),
            next_block_corr=("next_block_corr", "mean"),
            n_same_block=("n_same_block", "sum"),
            n_next_block=("n_next_block", "sum"),
        )
        .assign(dataset_key="mean_real_data", dataset="Mean")
    )
    out = pd.concat([per_dataset, aggregate], ignore_index=True)
    score_order = {"drift_only": 0, "gain_only": 1, "matched": 2}
    dataset_order = {"air_quality": 0, "tetouan_zone1_power": 1, "mean_real_data": 2}
    return (
        out.assign(
            _dataset_order=lambda df: [dataset_order[key] for key in df["dataset_key"]],
            _score_order=lambda df: [score_order[key] for key in df["score"]],
        )
        .sort_values(["_dataset_order", "_score_order"])
        .drop(columns=["_dataset_order", "_score_order"])
        .reset_index(drop=True)
    )


def run_suite(base_dir: Path, force: bool = False) -> dict[str, pd.DataFrame]:
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tetouan_dir = base_dir / "benchmark_package" / "tetouan_city_power_consumption"

    air = run_air_quality_experiment(base_dir, force=force)
    tetouan = run_tetouan_benchmark(base_dir, force=force)

    air_selected = _selected_rows(
        air["summary"],
        air["selected_lambdas"],
        "air_quality",
    )
    tetouan_selected = _selected_rows(
        tetouan["sweep_summary"],
        tetouan["selected_lambdas"],
        "tetouan_zone1_power",
    )
    selected_rows = pd.concat([air_selected, tetouan_selected], ignore_index=True)

    summary_stats = _summary_stats(selected_rows)
    paired = _paired_comparisons(selected_rows)
    conservative_rows = pd.concat(
        [
            _conservative_gain_selected_rows(
                air["summary"],
                "air_quality",
            ),
            _conservative_gain_selected_rows(
                tetouan["sweep_summary"],
                "tetouan_zone1_power",
            ),
        ],
        ignore_index=True,
    )
    conservative_stats = _summary_stats(conservative_rows)
    conservative_paired = _paired_comparisons(conservative_rows)
    air_quality_dtr_path = _air_quality_dtr_path(air["summary"])
    hazard = _hazard_ablation(
        air["selected_trajectories_all_seeds"],
        tetouan["selected_trajectories_all_seeds"],
    )

    summary_path = figures_dir / "real_deployment_summary_stats.csv"
    paired_path = figures_dir / "real_deployment_paired_comparisons.csv"
    conservative_summary_path = figures_dir / "real_deployment_conservative_gain_summary.csv"
    conservative_paired_path = figures_dir / "real_deployment_conservative_gain_paired.csv"
    air_quality_path_path = figures_dir / "air_quality_dtr_lambda_path.csv"
    hazard_path = figures_dir / "hazard_score_ablation.csv"
    report_path = figures_dir / "real_deployment_report.json"
    air_subspace_summary_path = figures_dir / "air_quality_subspace_ablation_selected.csv"
    air_subspace_paired_path = figures_dir / "air_quality_subspace_ablation_paired.csv"

    summary_stats.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    conservative_stats.to_csv(conservative_summary_path, index=False)
    conservative_paired.to_csv(conservative_paired_path, index=False)
    air_quality_dtr_path.to_csv(air_quality_path_path, index=False)
    hazard.to_csv(hazard_path, index=False)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selection_rule": (
                    "Per-method lambda selected by validation MSE across matched seeds; "
                    "validation directional gain is used only as a secondary tie-breaker."
                ),
                "real_data_seeds": 10,
                "summary_stats_path": str(summary_path.relative_to(base_dir)),
                "paired_comparisons_path": str(paired_path.relative_to(base_dir)),
                "conservative_gain_selection_rule": (
                    "Secondary ablation: choose the smallest lambda whose mean validation MSE is no worse "
                    f"than standard training and whose validation directional gain is reduced by at least {GAIN_TARGET:.0%}."
                ),
                "conservative_gain_summary_path": str(conservative_summary_path.relative_to(base_dir)),
                "conservative_gain_paired_path": str(conservative_paired_path.relative_to(base_dir)),
                "air_quality_dtr_lambda_path": str(air_quality_path_path.relative_to(base_dir)),
                "air_quality_subspace_ablation_summary_path": str(
                    air_subspace_summary_path.relative_to(base_dir)
                ),
                "air_quality_subspace_ablation_paired_path": str(
                    air_subspace_paired_path.relative_to(base_dir)
                ),
                "hazard_score_ablation_path": str(hazard_path.relative_to(base_dir)),
                "tetouan_output_dir": str(tetouan_dir.relative_to(base_dir)),
            },
            handle,
            indent=2,
        )

    return {
        "summary_stats": summary_stats,
        "paired_comparisons": paired,
        "conservative_gain_summary": conservative_stats,
        "conservative_gain_paired": conservative_paired,
        "air_quality_dtr_lambda_path": air_quality_dtr_path,
        "hazard_ablation": hazard,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    results = run_suite(args.base_dir, force=args.force)
    print(
        "Wrote real deployment reports to "
        f"{args.base_dir / 'figures'} "
        f"({len(results['summary_stats'])} summary rows, "
        f"{len(results['paired_comparisons'])} paired rows)."
    )


if __name__ == "__main__":
    main()
