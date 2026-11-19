#!/usr/bin/env python3
"""Aggregate independent GCBD result_row.json files after jobs finish."""

from __future__ import annotations

import argparse
import json
import os
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = [
    "dataset",
    "scenario",
    "teacher",
    "student",
    "method",
    "budget",
    "seed",
    "task_type",
    "student_lm_objective",
    "teacher_lm_objective",
    "logits_distill_possible",
    "gpu_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="trillium_experiments.json")
    parser.add_argument(
        "--output-root",
        help="Exact result tree to scan (for example, .../outputs/epochs_10).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset to aggregate; repeat for more than one. Defaults to config datasets.",
    )
    parser.add_argument(
        "--aggregate-root",
        help="Directory for generated CSV files. A dataset-specific directory is used by default.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero after writing reports if any configured run is missing.",
    )
    return parser.parse_args()


def expand_path(value: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(value))).resolve()


def budgets_for_method(method: str, budgets: List[float]) -> List[float]:
    if method == "GCBD-NoBudget" or not method.startswith("GCBD"):
        return [0.0]
    return budgets


def identity(row: Dict[str, Any]) -> Tuple[str, str, str, int, float]:
    return (
        str(row["dataset"]),
        str(row["scenario"]),
        str(row["method"]),
        int(row["seed"]),
        float(row["budget"]),
    )


def main() -> None:
    args = parse_args()
    config_path = expand_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    output_root = expand_path(args.output_root or config["output_root"])
    configured_datasets = [str(value) for value in config["datasets"]]
    selected_datasets = list(dict.fromkeys(args.dataset or configured_datasets))
    unknown_datasets = sorted(set(selected_datasets) - set(configured_datasets))
    if unknown_datasets:
        raise ValueError(
            "Dataset(s) not enabled in the configuration: "
            + ", ".join(unknown_datasets)
        )

    if args.aggregate_root:
        aggregate_root = expand_path(args.aggregate_root)
    else:
        label = (
            selected_datasets[0]
            if len(selected_datasets) == 1
            else "all_datasets"
        )
        label = label.replace("/", "__").replace(" ", "_")
        aggregate_root = output_root / "aggregates" / label
    aggregate_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    sources: Dict[Tuple[str, str, str, int, float], Path] = {}
    for path in sorted(output_root.rglob("result_row.json")):
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        if str(row.get("dataset")) not in selected_datasets:
            continue
        key = identity(row)
        if key in sources:
            raise RuntimeError(f"Duplicate result identity in {sources[key]} and {path}")
        sources[key] = path
        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No selected result_row.json files found below {output_root} "
            f"for dataset(s): {', '.join(selected_datasets)}"
        )

    raw = pd.DataFrame(rows).sort_values(
        ["dataset", "scenario", "method", "budget", "seed"]
    )

    # Pair every method run with the Student result from the same
    # dataset/scenario/seed. This makes accuracy gain a seed-matched quantity
    # instead of comparing against an unrelated global baseline.
    baseline_keys = ["dataset", "scenario", "seed"]
    student_rows = raw[raw["method"] == "Student"].copy()
    if student_rows.duplicated(baseline_keys).any():
        duplicates = student_rows.loc[
            student_rows.duplicated(baseline_keys, keep=False), baseline_keys
        ]
        raise RuntimeError(
            "Multiple Student baselines found for the same dataset/scenario/seed:\n"
            + duplicates.to_string(index=False)
        )

    baseline_columns = baseline_keys.copy()
    rename_columns: Dict[str, str] = {}
    if "accuracy" in raw.columns:
        baseline_columns.append("accuracy")
        rename_columns["accuracy"] = "student_accuracy"
    if "perplexity" in raw.columns:
        baseline_columns.append("perplexity")
        rename_columns["perplexity"] = "student_perplexity"
    if rename_columns:
        raw = raw.merge(
            student_rows[baseline_columns].rename(columns=rename_columns),
            on=baseline_keys,
            how="left",
            validate="many_to_one",
        )

    query_ratio = pd.to_numeric(
        raw.get("train_query_ratio", pd.Series(np.nan, index=raw.index)),
        errors="coerce",
    )
    prediction_queries = pd.to_numeric(
        raw.get("teacher_prediction_queries", pd.Series(np.nan, index=raw.index)),
        errors="coerce",
    )
    if "accuracy" in raw.columns and "student_accuracy" in raw.columns:
        raw["accuracy_gain_over_student"] = raw["accuracy"] - raw["student_accuracy"]
        raw["teacher_query_efficiency"] = raw["accuracy_gain_over_student"].div(
            query_ratio.where(query_ratio > 0)
        )
        raw["teacher_query_efficiency_per_1k_queries"] = (
            1000.0
            * raw["accuracy_gain_over_student"].div(
                prediction_queries.where(prediction_queries > 0)
            )
        )
    if "perplexity" in raw.columns and "student_perplexity" in raw.columns:
        raw["relative_perplexity_gain_over_student"] = (
            raw["student_perplexity"] - raw["perplexity"]
        ).div(raw["student_perplexity"].where(raw["student_perplexity"] > 0))
        raw["perplexity_gain_per_train_query_ratio"] = raw[
            "relative_perplexity_gain_over_student"
        ].div(query_ratio.where(query_ratio > 0))

    raw_path = aggregate_root / "method_comparison_raw.csv"
    raw.to_csv(raw_path, index=False)

    group_keys = ["dataset", "scenario", "method", "budget"]
    numeric_columns = [
        column
        for column in raw.columns
        if column not in IDENTITY_COLUMNS and pd.api.types.is_numeric_dtype(raw[column])
    ]
    aggregated = raw.groupby(group_keys, dropna=False)[numeric_columns].agg(["mean", "std"])
    aggregated.columns = [f"{metric}_{stat}" for metric, stat in aggregated.columns]
    aggregated = aggregated.reset_index()
    aggregated_path = aggregate_root / "method_comparison_aggregated.csv"
    aggregated.to_csv(aggregated_path, index=False)

    efficiency_metrics = [
        "accuracy_gain_over_student",
        "teacher_query_efficiency",
        "teacher_query_efficiency_per_1k_queries",
        "relative_perplexity_gain_over_student",
        "perplexity_gain_per_train_query_ratio",
        "train_query_ratio",
        "final_train_query_ratio",
        "teacher_prediction_queries",
        "teacher_geometry_queries",
    ]
    efficiency_columns = group_keys + [
        f"{metric}_{stat}"
        for metric in efficiency_metrics
        for stat in ("mean", "std")
        if f"{metric}_{stat}" in aggregated.columns
    ]
    efficiency_path = aggregate_root / "teacher_query_efficiency.csv"
    aggregated[efficiency_columns].to_csv(efficiency_path, index=False)

    expected = {
        (dataset, scenario, method, int(seed), float(budget))
        for dataset, scenario, method, seed in product(
            selected_datasets, config["scenarios"], config["methods"], config["seeds"]
        )
        for budget in budgets_for_method(method, [float(value) for value in config["budgets"]])
    }
    missing = sorted(expected - set(sources))
    missing_path = aggregate_root / "missing_runs.csv"
    pd.DataFrame(
        missing,
        columns=["dataset", "scenario", "method", "seed", "budget"],
    ).to_csv(missing_path, index=False)

    print(f"Input result root: {output_root}")
    print(f"Dataset(s): {', '.join(selected_datasets)}")
    print(f"Run rows: {len(raw)}")
    print(f"Aggregated rows: {len(aggregated)}")
    print(f"Raw results: {raw_path}")
    print(f"Aggregated results: {aggregated_path}")
    print(f"Teacher-query efficiency: {efficiency_path}")
    print(f"Missing-run report: {missing_path} ({len(missing)} missing)")
    if args.strict and missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
