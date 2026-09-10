from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "benchmark_package" / "gas_sensor_array_drift" / "results"
METRICS = ["deploy_ce", "volatility", "terminal_ce", "deploy_macro_accuracy"]


def bootstrap_paired_interval(
    differences: np.ndarray, rng: np.random.Generator, draws: int = 50000
) -> list[float]:
    samples = rng.choice(differences, size=(draws, len(differences)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def paired_comparison(
    selected: pd.DataFrame,
    candidate_name: str,
    reference_name: str,
    rng: np.random.Generator,
) -> dict[str, object]:
    candidate = selected[selected["basis"] == candidate_name]
    reference = selected[selected["basis"] == reference_name]
    paired = candidate.merge(reference, on="seed", suffixes=("_candidate", "_reference"))
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
        wins = differences < 0.0 if lower_is_better else differences > 0.0
        result[metric] = {
            "mean_difference": float(differences.mean()),
            "bootstrap_95_ci": bootstrap_paired_interval(differences, rng),
            "wins": int(wins.sum()),
        }
    return result


def rank_correlation(first: pd.Series, second: pd.Series) -> float:
    first_rank = first.rank(method="average").to_numpy(dtype=float)
    second_rank = second.rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def mean_block_trajectory(rows: pd.DataFrame) -> list[float]:
    matrix = np.asarray([json.loads(value) for value in rows["block_ce"]], dtype=float)
    return matrix.mean(axis=0).tolist()


def main() -> None:
    selected = pd.read_csv(RESULTS_DIR / "full_selected.csv")
    summary = pd.read_csv(RESULTS_DIR / "full_summary.csv")
    rank_sweep = pd.read_csv(RESULTS_DIR / "full_prospective_rank_sweep.csv")
    prospective_k4 = rank_sweep[
        (rank_sweep["rank"] == 4) & np.isclose(rank_sweep["lambda"], 1.0)
    ].copy()
    prospective_k4["basis"] = "prospective_k4"
    selected = pd.concat([selected, prospective_k4], ignore_index=True)
    rng = np.random.default_rng(8472)

    comparisons = [
        paired_comparison(selected, "prospective", "standard", rng),
        paired_comparison(selected, "prospective", "isotropic", rng),
        paired_comparison(selected, "prospective", "oracle_deployment", rng),
        paired_comparison(selected, "prospective_k4", "standard", rng),
        paired_comparison(selected, "prospective_k4", "isotropic", rng),
        paired_comparison(selected, "isotropic", "standard", rng),
    ]

    main_names = [
        "standard",
        "isotropic",
        "prospective",
        "prospective_k4",
        "oracle_deployment",
    ]
    trajectories = {
        name: mean_block_trajectory(selected[selected["basis"] == name])
        for name in main_names
    }

    prospective = summary[summary["basis"] == "prospective"].iloc[0]
    random_rows = summary[summary["basis"].str.contains("random")].copy()
    random_results: list[dict[str, object]] = []
    for control_type, mask in [
        ("ambient_random", random_rows["basis"].str.startswith("ambient")),
        ("drift_span_random", random_rows["basis"].str.startswith("drift_span")),
    ]:
        subset = random_rows[mask]
        item: dict[str, object] = {"control_type": control_type, "bases": int(len(subset))}
        for metric in [
            "deploy_ce_mean",
            "volatility_mean",
            "terminal_ce_mean",
            "deploy_macro_accuracy_mean",
        ]:
            prospective_value = float(prospective[metric])
            lower_is_better = metric != "deploy_macro_accuracy_mean"
            wins = (
                prospective_value < subset[metric]
                if lower_is_better
                else prospective_value > subset[metric]
            )
            item[metric] = {
                "prospective": prospective_value,
                "control_mean": float(subset[metric].mean()),
                "control_min": float(subset[metric].min()),
                "control_max": float(subset[metric].max()),
                "prospective_wins_over_bases": int(wins.sum()),
            }
        item["spearman_energy_vs_deploy_ce"] = rank_correlation(
            subset["captured_deploy_drift_energy"], subset["deploy_ce_mean"]
        )
        item["spearman_energy_vs_volatility"] = rank_correlation(
            subset["captured_deploy_drift_energy"], subset["volatility_mean"]
        )
        random_results.append(item)

    rank4_controls = pd.read_csv(RESULTS_DIR / "full_rank4_random_summary.csv")
    rank4_reference = {
        "deploy_ce_mean": float(prospective_k4["deploy_ce"].mean()),
        "volatility_mean": float(prospective_k4["volatility"].mean()),
        "terminal_ce_mean": float(prospective_k4["terminal_ce"].mean()),
        "deploy_macro_accuracy_mean": float(
            prospective_k4["deploy_macro_accuracy"].mean()
        ),
    }
    rank4_comparison: dict[str, object] = {"bases": int(len(rank4_controls))}
    for metric, reference_value in rank4_reference.items():
        lower_is_better = metric != "deploy_macro_accuracy_mean"
        wins = (
            reference_value < rank4_controls[metric]
            if lower_is_better
            else reference_value > rank4_controls[metric]
        )
        rank4_comparison[metric] = {
            "prospective_k4": reference_value,
            "control_mean": float(rank4_controls[metric].mean()),
            "control_min": float(rank4_controls[metric].min()),
            "control_max": float(rank4_controls[metric].max()),
            "prospective_wins_over_bases": int(wins.sum()),
        }

    output = {
        "paired_comparisons": comparisons,
        "mean_deployment_batch_ce": {
            "batches": [6, 7, 8, 9, 10],
            **trajectories,
        },
        "random_basis_comparisons": random_results,
        "selected_rank4_vs_ambient_random": rank4_comparison,
    }
    output_path = RESULTS_DIR / "full_analysis.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
