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
MONITORING_BOOTSTRAP_RESAMPLES = 2000
GAIN_TARGET = 0.05
MONITORING_SCORE_ORDER = [
    ("s2", "Drift $s_t^2$"),
    ("gain", "Gain $g_t$"),
    ("h_raw", "Product $h_t$"),
    ("h_roll2", "Roll-2 $h_t$"),
    ("h_roll3", "Roll-3 $h_t$"),
    ("log_h", "Log product"),
    ("log_h_roll2", "Roll-2 log product"),
    ("log_h_roll3", "Roll-3 log product"),
]
MONITORING_TARGET_ORDER = [
    "sq_change_same",
    "sq_change_next",
    "abs_change_same",
    "abs_change_next",
    "risk",
    "risk_next",
]


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


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 4:
        return float("nan")
    x_valid = x[mask]
    y_valid = y[mask]
    if float(np.std(x_valid)) == 0.0 or float(np.std(y_valid)) == 0.0:
        return float("nan")
    x_rank = pd.Series(x_valid).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y_valid).rank(method="average").to_numpy(dtype=np.float64)
    if float(np.std(x_rank)) == 0.0 or float(np.std(y_rank)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _monitoring_blockwise_records(dataset_key: str, trajectories: pd.DataFrame) -> pd.DataFrame:
    dtr = trajectories[trajectories["method"] == "dtr"].copy()
    dtr = dtr.sort_values(["seed", "block"]).reset_index(drop=True)
    dtr["block_order"] = dtr.groupby("seed").cumcount()
    out = pd.DataFrame(
        {
            "dataset_key": dataset_key,
            "dataset": DISPLAY_DATASETS[dataset_key],
            "seed": dtr["seed"].astype(int),
            "block": dtr["block"],
            "block_order": dtr["block_order"].astype(int),
            "risk": dtr["risk"].astype(float),
            "s2": np.square(dtr["speed"].astype(float)),
            "gain": dtr["gain"].astype(float),
        }
    )
    out["h_raw"] = out["s2"] * out["gain"]
    out.loc[out["block_order"].eq(0), ["s2", "gain", "h_raw"]] = np.nan
    return out


def _add_monitoring_targets(blockwise: pd.DataFrame) -> pd.DataFrame:
    out = blockwise.sort_values(["dataset_key", "seed", "block_order"]).reset_index(drop=True)
    group_cols = ["dataset_key", "seed"]

    out["risk_prev"] = out.groupby(group_cols)["risk"].shift(1)
    out["risk_next"] = out.groupby(group_cols)["risk"].shift(-1)
    out["abs_change_same"] = (out["risk"] - out["risk_prev"]).abs()
    out["sq_change_same"] = (out["risk"] - out["risk_prev"]) ** 2
    out["abs_change_next"] = (out["risk_next"] - out["risk"]).abs()
    out["sq_change_next"] = (out["risk_next"] - out["risk"]) ** 2

    for window in (2, 3):
        out[f"h_roll{window}"] = out.groupby(group_cols)["h_raw"].transform(
            lambda values, w=window: values.rolling(w, min_periods=w).mean()
        )

    eps = 1e-12
    out["log_s2"] = np.log(out["s2"] + eps)
    out["log_gain"] = np.log(out["gain"] + eps)
    out["log_h"] = out["log_s2"] + out["log_gain"]
    for window in (2, 3):
        out[f"log_h_roll{window}"] = out.groupby(group_cols)["log_h"].transform(
            lambda values, w=window: values.rolling(w, min_periods=w).mean()
        )

    return out


def _bootstrap_monitoring_corr(
    blockwise: pd.DataFrame,
    dataset_key: str,
    score: str,
    target: str,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    dataset_df = blockwise[blockwise["dataset_key"] == dataset_key]
    seed_arrays = [
        (
            seed_df[score].to_numpy(dtype=np.float64),
            seed_df[target].to_numpy(dtype=np.float64),
        )
        for _, seed_df in dataset_df.groupby("seed", sort=True)
    ]
    vals: list[float] = []
    for _ in range(MONITORING_BOOTSTRAP_RESAMPLES):
        sampled = rng.integers(0, len(seed_arrays), size=len(seed_arrays))
        x = np.concatenate([seed_arrays[idx][0] for idx in sampled])
        y = np.concatenate([seed_arrays[idx][1] for idx in sampled])
        rho = _spearman_corr(x, y)
        if np.isfinite(rho):
            vals.append(rho)

    point = _spearman_corr(
        dataset_df[score].to_numpy(dtype=np.float64),
        dataset_df[target].to_numpy(dtype=np.float64),
    )
    if not vals:
        return {
            "point_spearman": point,
            "bootstrap_mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_bootstrap_finite": 0,
        }

    values = np.asarray(vals, dtype=np.float64)
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {
        "point_spearman": point,
        "bootstrap_mean": float(values.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_bootstrap_finite": int(len(values)),
    }


def _monitoring_volatility_reports(
    air_trajectories: pd.DataFrame,
    tetouan_trajectories: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blockwise = pd.concat(
        [
            _monitoring_blockwise_records("air_quality", air_trajectories),
            _monitoring_blockwise_records("tetouan_zone1_power", tetouan_trajectories),
        ],
        ignore_index=True,
    )
    blockwise = _add_monitoring_targets(blockwise)

    corr_rows: list[dict[str, object]] = []
    for dataset_key, dataset_df in blockwise.groupby("dataset_key", sort=False):
        for target in MONITORING_TARGET_ORDER:
            for score, score_label in MONITORING_SCORE_ORDER:
                x = dataset_df[score].to_numpy(dtype=np.float64)
                y = dataset_df[target].to_numpy(dtype=np.float64)
                valid = np.isfinite(x) & np.isfinite(y)
                corr_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": DISPLAY_DATASETS[dataset_key],
                        "target": target,
                        "score": score,
                        "score_label": score_label,
                        "spearman": _spearman_corr(x, y),
                        "n": int(valid.sum()),
                    }
                )
    correlations = pd.DataFrame(corr_rows)

    rng = np.random.default_rng(20260424)
    primary_scores = ["s2", "gain", "h_raw", "h_roll2", "log_h", "log_h_roll2"]
    primary_targets = ["sq_change_same", "sq_change_next"]
    score_labels = dict(MONITORING_SCORE_ORDER)
    boot_rows: list[dict[str, object]] = []
    for dataset_key in DISPLAY_DATASETS:
        dataset_df = blockwise[blockwise["dataset_key"] == dataset_key]
        for target in primary_targets:
            for score in primary_scores:
                stats = _bootstrap_monitoring_corr(blockwise, dataset_key, score, target, rng)
                x = dataset_df[score].to_numpy(dtype=np.float64)
                y = dataset_df[target].to_numpy(dtype=np.float64)
                boot_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": DISPLAY_DATASETS[dataset_key],
                        "target": target,
                        "score": score,
                        "score_label": score_labels[score],
                        "n": int((np.isfinite(x) & np.isfinite(y)).sum()),
                        **stats,
                    }
                )
    bootstrap = pd.DataFrame(boot_rows)
    return blockwise, correlations, bootstrap


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
    monitoring_blockwise, monitoring_correlations, monitoring_bootstrap = _monitoring_volatility_reports(
        air["selected_trajectories_all_seeds"],
        tetouan["selected_trajectories_all_seeds"],
    )

    summary_path = figures_dir / "real_deployment_summary_stats.csv"
    paired_path = figures_dir / "real_deployment_paired_comparisons.csv"
    conservative_summary_path = figures_dir / "real_deployment_conservative_gain_summary.csv"
    conservative_paired_path = figures_dir / "real_deployment_conservative_gain_paired.csv"
    air_quality_path_path = figures_dir / "air_quality_dtr_lambda_path.csv"
    monitoring_blockwise_path = figures_dir / "monitoring_blockwise_selected_dtr.csv"
    monitoring_correlations_path = figures_dir / "monitoring_volatility_ablation.csv"
    monitoring_bootstrap_path = figures_dir / "monitoring_volatility_bootstrap.csv"
    report_path = figures_dir / "real_deployment_report.json"
    air_subspace_summary_path = figures_dir / "air_quality_subspace_ablation_selected.csv"
    air_subspace_paired_path = figures_dir / "air_quality_subspace_ablation_paired.csv"

    summary_stats.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    conservative_stats.to_csv(conservative_summary_path, index=False)
    conservative_paired.to_csv(conservative_paired_path, index=False)
    air_quality_dtr_path.to_csv(air_quality_path_path, index=False)
    monitoring_blockwise.to_csv(monitoring_blockwise_path, index=False)
    monitoring_correlations.to_csv(monitoring_correlations_path, index=False)
    monitoring_bootstrap.to_csv(monitoring_bootstrap_path, index=False)
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
                "monitoring_target": "next-block squared risk change, (r_{t+1} - r_t)^2",
                "monitoring_statistic": "Spearman correlation on selected DTR deployment blocks",
                "monitoring_blockwise_path": str(monitoring_blockwise_path.relative_to(base_dir)),
                "monitoring_volatility_ablation_path": str(
                    monitoring_correlations_path.relative_to(base_dir)
                ),
                "monitoring_volatility_bootstrap_path": str(
                    monitoring_bootstrap_path.relative_to(base_dir)
                ),
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
        "monitoring_blockwise": monitoring_blockwise,
        "monitoring_volatility_ablation": monitoring_correlations,
        "monitoring_volatility_bootstrap": monitoring_bootstrap,
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
        f"{len(results['paired_comparisons'])} paired rows, "
        f"{len(results['monitoring_volatility_ablation'])} monitoring-correlation rows)."
    )


if __name__ == "__main__":
    main()
