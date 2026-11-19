"""Strict class-order validation and microbatched semantic transport helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from gcbd_v2.common.transport import entropic_transport_loss, semantic_cost


def load_semantic_cost(path: str | Path, expected_class_names: list[str], device: torch.device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Semantic class embedding asset is required for GCBD UOT: {path}. "
            "Generate it from the staged frozen encoder and the exact ordered class texts."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    names = payload.get("class_names", payload.get("class_texts"))
    embeddings = payload.get("embeddings")
    if names is None or embeddings is None:
        raise ValueError("Class embedding asset must contain class_names/class_texts and embeddings")
    if list(names) != list(expected_class_names):
        raise ValueError("Semantic class embedding order does not match the dataset classifier label order")
    if not torch.is_tensor(embeddings) or embeddings.ndim != 2 or embeddings.shape[0] != len(expected_class_names):
        raise ValueError("Semantic class embeddings have invalid shape")
    return semantic_cost(embeddings).to(device=device), payload.get("metadata", {})


def microbatched_transport_loss(logits_s: torch.Tensor, logits_t: torch.Tensor,
                                cost: torch.Tensor, *, mode: str = "uot",
                                epsilon: float = 0.05, rho_t: float = 1.0,
                                rho_s: float = 1.0, iterations: int = 100,
                                tolerance: float = 1e-5, microbatch_size: int = 8):
    if logits_s.shape != logits_t.shape:
        raise ValueError("Teacher/student logits must have identical shape")
    if logits_s.shape[0] == 0:
        return logits_s.new_zeros(()), {"finite": True, "iterations": 0}
    losses, diagnostics = [], []
    for start in range(0, logits_s.shape[0], max(1, int(microbatch_size))):
        stop = min(logits_s.shape[0], start + max(1, int(microbatch_size)))
        loss, diag = entropic_transport_loss(
            logits_s[start:stop], logits_t[start:stop], cost,
            epsilon=epsilon, mode=mode, rho_t=rho_t, rho_s=rho_s,
            iterations=iterations, tolerance=tolerance,
        )
        losses.append(loss * (stop - start))
        diagnostics.append(diag)
    total = torch.stack(losses).sum() / logits_s.shape[0]
    return total, {
        "finite": all(x["finite"] for x in diagnostics),
        "microbatches": len(diagnostics),
        "max_iterations": max(x["iterations"] for x in diagnostics),
        "max_log_scaling_delta": max(x["max_log_scaling_delta"] for x in diagnostics),
        "mean_transport_cost": sum(x["mean_transport_cost"] for x in diagnostics) / len(diagnostics),
    }
