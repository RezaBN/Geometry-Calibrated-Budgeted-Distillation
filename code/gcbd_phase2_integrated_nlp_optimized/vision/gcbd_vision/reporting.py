from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import atomic_json, ensure_dir


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_evaluation_diagnostics(
    output_dir: str | Path,
    diagnostics: dict[str, Any],
    classes: list[str],
) -> None:
    output = ensure_dir(output_dir)
    calibration = diagnostics["calibration"]
    write_csv(output / "calibration.csv", calibration)
    class_rows = []
    for index, name in enumerate(classes):
        class_rows.append(
            {
                "class_index": index,
                "class_name": name,
                "examples": diagnostics["per_class_count"][index],
                "accuracy": diagnostics["per_class_accuracy"][index],
            }
        )
    write_csv(output / "per_class_metrics.csv", class_rows)
    confusion_rows = []
    for row in diagnostics["top_confusions"]:
        confusion_rows.append(
            {
                **row,
                "true_class_name": classes[row["true_class"]],
                "predicted_class_name": classes[row["predicted_class"]],
            }
        )
    write_csv(output / "top_confusions.csv", confusion_rows)
    if diagnostics.get("confusion_matrix") is not None:
        write_csv(
            output / "confusion_matrix.csv",
            [
                {"true_class": classes[index], **{classes[j]: value for j, value in enumerate(row)}}
                for index, row in enumerate(diagnostics["confusion_matrix"])
            ],
        )

    populated = [row for row in calibration if row["count"] > 0]
    if populated:
        accuracies = [row["accuracy"] for row in populated]
        confidences = [row["confidence"] for row in populated]
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        axis.plot(confidences, accuracies, marker="o", label="Model")
        axis.set(xlabel="Mean confidence", ylabel="Accuracy", xlim=(0, 1), ylim=(0, 1))
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / "reliability_diagram.png", dpi=180)
        plt.close(figure)


def save_training_history(output_dir: str | Path, history: list[dict[str, Any]]) -> None:
    output = ensure_dir(output_dir)
    write_csv(output / "training_history.csv", history)
    atomic_json(output / "training_history.json", history)
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    if "validation_top1" in history[0]:
        axes[1].plot(epochs, [row.get("validation_top1", math.nan) for row in history], label="Validation top-1")
    else:
        axes[1].plot(epochs, [row.get("train_top1", math.nan) for row in history], label="Train top-1")
    axes[0].set(xlabel="Epoch", ylabel="Loss")
    axes[1].set(xlabel="Epoch", ylabel="Validation top-1")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=180)
    plt.close(figure)

    if "teacher_prediction_query_ratio" in history[0]:
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(
            epochs,
            [row.get("teacher_prediction_query_ratio", 0.0) for row in history],
            marker="o",
            label="Prediction queries",
        )
        axis.plot(
            epochs,
            [row.get("teacher_geometry_query_ratio", 0.0) for row in history],
            marker="s",
            label="Geometry accesses",
        )
        axis.set(xlabel="Epoch", ylabel="Fraction of training examples", ylim=(0, 1.05))
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "teacher_query_ratios.png", dpi=180)
        plt.close(figure)
