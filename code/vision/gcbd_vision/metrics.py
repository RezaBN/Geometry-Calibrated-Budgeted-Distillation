from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    precision: str,
    num_classes: int,
    ece_bins: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    nll_sum = 0.0
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    bin_count = np.zeros(ece_bins, dtype=np.int64)
    bin_confidence = np.zeros(ece_bins, dtype=np.float64)
    bin_correct = np.zeros(ece_bins, dtype=np.float64)

    for index, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device, precision):
            output = model(images)
            logits = output[0] if isinstance(output, tuple) else output
        logits = logits.float()
        probabilities = F.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)
        topk = min(5, num_classes)
        top_predictions = logits.topk(topk, dim=1).indices
        batch_size = labels.shape[0]
        total += batch_size
        correct1 += int((prediction == labels).sum().item())
        correct5 += int((top_predictions == labels[:, None]).any(dim=1).sum().item())
        nll_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())

        confidence_np = confidence.cpu().numpy()
        correct_np = (prediction == labels).float().cpu().numpy()
        indices = np.minimum((confidence_np * ece_bins).astype(np.int64), ece_bins - 1)
        for bin_index in range(ece_bins):
            mask = indices == bin_index
            if mask.any():
                bin_count[bin_index] += int(mask.sum())
                bin_confidence[bin_index] += float(confidence_np[mask].sum())
                bin_correct[bin_index] += float(correct_np[mask].sum())
        labels_all.append(labels.cpu().numpy())
        predictions_all.append(prediction.cpu().numpy())
        if progress_callback is not None:
            progress_callback(index + 1, len(loader))

    if total == 0:
        raise RuntimeError("Evaluation loader produced no examples")
    labels_np = np.concatenate(labels_all)
    predictions_np = np.concatenate(predictions_all)
    macro_f1 = float(f1_score(labels_np, predictions_np, average="macro", zero_division=0))
    ece = 0.0
    calibration_rows = []
    for index in range(ece_bins):
        if bin_count[index] > 0:
            accuracy = bin_correct[index] / bin_count[index]
            confidence = bin_confidence[index] / bin_count[index]
            ece += (bin_count[index] / total) * abs(accuracy - confidence)
        else:
            accuracy = float("nan")
            confidence = float("nan")
        calibration_rows.append(
            {
                "bin": index,
                "lower": index / ece_bins,
                "upper": (index + 1) / ece_bins,
                "count": int(bin_count[index]),
                "accuracy": accuracy,
                "confidence": confidence,
            }
        )

    cm = confusion_matrix(labels_np, predictions_np, labels=np.arange(num_classes))
    true_count = cm.sum(axis=1)
    per_class_accuracy = np.divide(
        np.diag(cm),
        true_count,
        out=np.zeros(num_classes, dtype=np.float64),
        where=true_count > 0,
    )
    confusions: list[tuple[int, int, int]] = []
    for true_class in range(num_classes):
        row = cm[true_class].copy()
        row[true_class] = 0
        if row.max() > 0:
            predicted_class = int(row.argmax())
            confusions.append((true_class, predicted_class, int(row[predicted_class])))
    confusions.sort(key=lambda item: item[2], reverse=True)

    metrics = {
        "top1": correct1 / total,
        "top5": correct5 / total,
        "macro_f1": macro_f1,
        "ece": float(ece),
        "nll": nll_sum / total,
        "examples": total,
    }
    diagnostics = {
        "calibration": calibration_rows,
        "per_class_accuracy": per_class_accuracy.tolist(),
        "per_class_count": true_count.astype(int).tolist(),
        "top_confusions": [
            {"true_class": item[0], "predicted_class": item[1], "count": item[2]}
            for item in confusions[:100]
        ],
        "confusion_matrix": cm.tolist() if num_classes <= 100 else None,
    }
    return metrics, diagnostics
