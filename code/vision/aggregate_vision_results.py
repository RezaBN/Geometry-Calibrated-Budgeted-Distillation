#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gcbd_vision.config import load_config
from gcbd_vision.utils import atomic_json, ensure_dir


IDENTITY_COLUMNS = [
    "dataset",
    "scenario",
    "teacher_architecture",
    "student_architecture",
    "method",
    "budget",
    "epochs",
]

METRIC_COLUMNS = [
    "top1",
    "top5",
    "macro_f1",
    "ece",
    "nll",
    "accuracy_gain_over_student",
    "teacher_prediction_query_ratio",
    "teacher_geometry_query_ratio",
    "teacher_forward_ratio",
    "teacher_query_efficiency",
    "teacher_query_efficiency_per_1k_queries",
    "runtime_sec",
    "peak_gpu_memory_gib",
]


def load_rows(input_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_root.rglob("result.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Skipping unreadable result {path}: {exc}")
            continue
        if row.get("mode") != "method":
            continue
        row["result_path"] = str(path)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No method result.json files found under {input_root}")
    return pd.DataFrame(rows)


def add_paired_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    pair_keys = ["dataset", "scenario", "seed", "epochs"]
    students = (
        result[result["method"] == "Student"][pair_keys + ["top1"]]
        .drop_duplicates(pair_keys, keep="last")
        .rename(columns={"top1": "student_top1"})
    )
    result = result.merge(students, on=pair_keys, how="left", validate="many_to_one")
    result["accuracy_gain_over_student"] = result["top1"] - result["student_top1"]
    query_ratio = pd.to_numeric(result["teacher_prediction_query_ratio"], errors="coerce")
    queries = pd.to_numeric(result["teacher_prediction_queries"], errors="coerce")
    gain = pd.to_numeric(result["accuracy_gain_over_student"], errors="coerce")
    result["teacher_query_efficiency"] = np.where(query_ratio > 0, gain / query_ratio, np.nan)
    result["teacher_query_efficiency_per_1k_queries"] = np.where(
        queries > 0, 1000.0 * gain / queries, np.nan
    )
    result["teacher_retention_ratio"] = np.where(
        pd.to_numeric(result.get("teacher_top1"), errors="coerce") > 0,
        result["top1"] / result["teacher_top1"],
        np.nan,
    )
    return result


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    available_metrics = [column for column in METRIC_COLUMNS if column in frame.columns]
    grouped = frame.groupby(IDENTITY_COLUMNS, dropna=False, sort=True)
    parts = []
    for metric in available_metrics:
        summary = grouped[metric].agg(["mean", "std", "count"]).rename(
            columns={
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
                "count": f"{metric}_count",
            }
        )
        parts.append(summary)
    result = pd.concat(parts, axis=1).reset_index()
    result["seeds"] = grouped["seed"].apply(
        lambda values: ",".join(str(int(value)) for value in sorted(set(values)))
    ).values
    return result


def missing_matrix(frame: pd.DataFrame, cfg: dict[str, Any], epochs: int | None) -> list[dict[str, Any]]:
    missing = []
    datasets = sorted(set(frame["dataset"]))
    for dataset in datasets:
        dataset_epochs = [epochs] if epochs is not None else sorted(
            int(value) for value in frame.loc[frame["dataset"] == dataset, "epochs"].unique()
        )
        for selected_epochs in dataset_epochs:
            for scenario in cfg["scenarios"]:
                for seed in cfg["seeds"]:
                    for method in cfg["methods"]:
                        mask = (
                            (frame["dataset"] == dataset)
                            & (frame["scenario"] == scenario)
                            & (frame["seed"] == seed)
                            & (frame["method"] == method)
                            & (frame["epochs"] == selected_epochs)
                        )
                        if not mask.any():
                            missing.append(
                                {
                                    "dataset": dataset,
                                    "scenario": scenario,
                                    "seed": int(seed),
                                    "method": method,
                                    "epochs": int(selected_epochs),
                                }
                            )
    return missing


def save_plots(aggregated: pd.DataFrame, output_dir: Path) -> None:
    for dataset in sorted(aggregated["dataset"].unique()):
        subset = aggregated[aggregated["dataset"] == dataset]
        scenarios = sorted(subset["scenario"].unique())
        figure, axes = plt.subplots(
            len(scenarios), 1, figsize=(11, max(4, 3.5 * len(scenarios))), squeeze=False
        )
        for axis, scenario in zip(axes[:, 0], scenarios):
            rows = subset[subset["scenario"] == scenario].sort_values("top1_mean")
            axis.barh(rows["method"], rows["top1_mean"], xerr=rows["top1_std"].fillna(0.0))
            axis.set_title(f"{dataset} — Scenario {scenario}")
            axis.set_xlabel("Top-1 accuracy")
            axis.grid(axis="x", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / f"{dataset}_top1_comparison.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8, 6))
        for scenario in scenarios:
            rows = subset[subset["scenario"] == scenario]
            axis.scatter(
                rows["teacher_prediction_query_ratio_mean"],
                rows["top1_mean"],
                label=f"Scenario {scenario}",
                alpha=0.85,
            )
            for _, row in rows.iterrows():
                if row["method"].startswith("GCBD"):
                    axis.annotate(
                        row["method"].replace("GCBD-", ""),
                        (row["teacher_prediction_query_ratio_mean"], row["top1_mean"]),
                        fontsize=7,
                    )
        axis.set(xlabel="Teacher prediction-query ratio", ylabel="Top-1 accuracy")
        axis.set_title(f"{dataset}: accuracy–query trade-off")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"{dataset}_accuracy_query_tradeoff.png", dpi=180)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate GCBD vision result.json files")
    parser.add_argument("--config", default="vision_experiments.json")
    parser.add_argument("--input-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--dataset", action="append", choices=("CIFAR100", "ImageNet"))
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    input_root = Path(args.input_root or cfg["output_root"])
    frame = load_rows(input_root)
    if args.dataset:
        frame = frame[frame["dataset"].isin(args.dataset)]
    if args.epochs is not None:
        frame = frame[frame["epochs"] == args.epochs]
    if frame.empty:
        raise RuntimeError("No results remain after applying filters")
    enriched = add_paired_metrics(frame)
    summary = aggregate(enriched)
    suffix = f"epochs_{args.epochs}" if args.epochs is not None else "all_epochs"
    output_dir = ensure_dir(args.output_dir or (input_root / "aggregated" / suffix))
    enriched.to_csv(output_dir / "vision_results_enriched.csv", index=False)
    summary.to_csv(output_dir / "vision_method_comparison_aggregated.csv", index=False)
    for dataset in sorted(summary["dataset"].unique()):
        summary[summary["dataset"] == dataset].to_csv(
            output_dir / f"{dataset}_method_comparison_aggregated.csv", index=False
        )
    missing = missing_matrix(enriched, cfg, args.epochs)
    atomic_json(
        output_dir / "aggregation_report.json",
        {
            "input_root": str(input_root),
            "runs_found": int(len(enriched)),
            "aggregated_rows": int(len(summary)),
            "missing_count": len(missing),
            "missing": missing,
            "teacher_query_efficiency_definition": "(method top1 - paired Student top1) / teacher prediction-query ratio",
            "teacher_query_efficiency_per_1k_queries_definition": "1000 * (method top1 - paired Student top1) / teacher prediction-query count",
        },
    )
    save_plots(summary, output_dir)
    print(f"Runs aggregated: {len(enriched)}")
    print(f"Missing cells: {len(missing)}")
    print(f"Aggregation directory: {output_dir}")


if __name__ == "__main__":
    main()
