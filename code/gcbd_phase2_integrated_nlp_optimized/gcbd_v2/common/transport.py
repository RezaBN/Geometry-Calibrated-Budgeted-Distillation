"""Semantic balanced and unbalanced entropic OT with convergence diagnostics."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def semantic_cost(class_embeddings: torch.Tensor) -> torch.Tensor:
    if class_embeddings.ndim != 2 or class_embeddings.shape[0] < 2:
        raise ValueError("class_embeddings must have shape [K,D], K>=2")
    e = F.normalize(class_embeddings.float(), dim=-1)
    cost = (1.0 - e @ e.T).clamp_min(0.0)
    cost = 0.5 * (cost + cost.T)
    cost.fill_diagonal_(0.0)
    if not torch.isfinite(cost).all():
        raise ValueError("semantic cost contains non-finite values")
    return cost


def generalized_kl(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.clamp_min(0.0)
    y = y.clamp_min(0.0)
    return (x * ((x + eps).log() - (y + eps).log()) - x + y).sum(dim=-1)


def entropic_transport_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    cost: torch.Tensor,
    *,
    epsilon: float = 0.05,
    mode: str = "uot",
    rho_t: float = 1.0,
    rho_s: float = 1.0,
    iterations: int = 100,
    tolerance: float = 1e-5,
    eps_kl: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """Per-example exact-support transport; caller should microbatch large K.

    ``cost`` is fixed, semantic and [K,K]. No top-k truncation or synthetic
    'other' class is introduced.
    """
    if mode not in {"uot", "balanced"}:
        raise ValueError("mode must be 'uot' or 'balanced'")
    if epsilon <= 0 or iterations < 1 or rho_t <= 0 or rho_s <= 0:
        raise ValueError("epsilon, iterations, rho_t and rho_s must be positive")
    ps = F.softmax(logits_s.float(), dim=-1).clamp_min(eps_kl)
    pt = F.softmax(logits_t.float(), dim=-1).clamp_min(eps_kl)
    ps = ps / ps.sum(-1, keepdim=True)
    pt = pt / pt.sum(-1, keepdim=True)
    c = cost.to(device=ps.device, dtype=torch.float32)
    k = ps.shape[-1]
    if pt.shape != ps.shape or c.shape != (k, k):
        raise ValueError("teacher/student logits and cost must share exact class support")
    if not torch.isfinite(c).all() or (c < 0).any():
        raise ValueError("cost must be finite and nonnegative")
    log_kernel = -c / float(epsilon)
    loga, logb = pt.log(), ps.log()
    logu, logv = torch.zeros_like(loga), torch.zeros_like(logb)
    tau_t = 1.0 if mode == "balanced" else rho_t / (rho_t + epsilon)
    tau_s = 1.0 if mode == "balanced" else rho_s / (rho_s + epsilon)
    delta = float("inf")
    used_iterations = 0
    for i in range(iterations):
        old_u, old_v = logu, logv
        logu = tau_t * (loga - torch.logsumexp(log_kernel.unsqueeze(0) + logv.unsqueeze(1), dim=2))
        logv = tau_s * (logb - torch.logsumexp(log_kernel.unsqueeze(0) + logu.unsqueeze(2), dim=1))
        delta = torch.maximum((logu - old_u).abs().amax(), (logv - old_v).abs().amax())
        used_iterations = i + 1
        if float(delta.detach()) < tolerance:
            break
    log_pi = logu.unsqueeze(2) + log_kernel.unsqueeze(0) + logv.unsqueeze(1)
    pi = torch.exp(log_pi)
    row, col = pi.sum(2), pi.sum(1)
    transport = (pi * c).sum((1, 2))
    entropy = float(epsilon) * (pi * (log_pi - 1.0)).sum((1, 2))
    if mode == "uot":
        marginal = rho_t * generalized_kl(row, pt, eps_kl) + rho_s * generalized_kl(col, ps, eps_kl)
        per_example = transport + entropy + marginal
    else:
        per_example = transport + entropy
    objective = per_example.mean()
    diagnostics = {
        "finite": bool(torch.isfinite(objective).detach()),
        "iterations": used_iterations,
        "max_log_scaling_delta": float(delta.detach()),
        "mean_transport_cost": float(transport.mean().detach()),
        "mean_plan_mass": float(pi.sum((1, 2)).mean().detach()),
        "mean_row_mass": float(row.sum(-1).mean().detach()),
        "mean_column_mass": float(col.sum(-1).mean().detach()),
        "mean_row_marginal_l1": float((row - pt).abs().sum(-1).mean().detach()),
        "mean_column_marginal_l1": float((col - ps).abs().sum(-1).mean().detach()),
    }
    if not diagnostics["finite"]:
        raise FloatingPointError("Transport objective became non-finite")
    return objective, diagnostics
