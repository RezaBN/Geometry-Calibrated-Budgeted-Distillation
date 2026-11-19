"""Cosine-paced, calibration-buffer selection and rate-matched controls."""
from __future__ import annotations
import math
from collections import deque
import torch
import torch.nn.functional as F


def entropy_from_logits(logits: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    p = F.softmax(logits.float(), dim=-1)
    h = -(p * p.clamp_min(1e-12).log()).sum(dim=-1)
    return h / math.log(logits.shape[-1]) if normalize and logits.shape[-1] > 1 else h


class CalibrationBuffer:
    def __init__(self, capacity: int = 8192, min_values: int = 32):
        if capacity < min_values or min_values < 2:
            raise ValueError("capacity must be >= min_values >= 2")
        self.capacity = int(capacity)
        self.min_values = int(min_values)
        self._values: deque[float] = deque(maxlen=self.capacity)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        vals = values.detach().float().cpu().flatten()
        vals = vals[torch.isfinite(vals)]
        self._values.extend(float(x) for x in vals.tolist())

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "min_values": self.min_values,
                "values": list(self._values)}

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity or int(state["min_values"]) != self.min_values:
            raise ValueError("Calibration-buffer checkpoint configuration mismatch")
        self._values.clear()
        self._values.extend(float(x) for x in state["values"][-self.capacity:])

    def quantile(self, q: float, fallback: torch.Tensor | None = None) -> float:
        if not 0.0 <= q <= 1.0:
            raise ValueError("quantile must be in [0,1]")
        if len(self._values) >= self.min_values:
            values = torch.tensor(list(self._values), dtype=torch.float32)
        elif fallback is not None:
            values = fallback.detach().float().cpu().flatten()
            values = values[torch.isfinite(values)]
        else:
            raise RuntimeError("Calibration buffer is not initialized and no fallback was supplied")
        if values.numel() == 0:
            return 0.0
        return float(torch.quantile(values, q).item())


def cosine_frontier(step: int, total_steps: int, gamma_min: float, gamma_max: float) -> float:
    if total_steps < 1 or gamma_min > gamma_max:
        raise ValueError("Require total_steps >= 1 and gamma_min <= gamma_max")
    progress = 0.0 if total_steps == 1 else min(max(step / (total_steps - 1), 0.0), 1.0)
    return float(gamma_min + (gamma_max - gamma_min) * 0.5 * (1.0 - math.cos(math.pi * progress)))


def select_gcbd(
    entropy: torch.Tensor,
    budget: float,
    frontier: float,
    calibration: CalibrationBuffer,
    rho: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Return hard query mask, soft gate, weights, and auditable diagnostics.

    The calibration buffer stores only curriculum-admissible uncertainties.
    The threshold is the (1-budget) quantile. A hard top-k guard ensures that
    the realized query count cannot exceed ceil(budget * admissible_count).
    """
    if not 0.0 < budget <= 1.0:
        raise ValueError("budget must be in (0,1]")
    if rho <= 0:
        raise ValueError("rho must be positive")
    u = entropy.float()
    eligible = torch.isfinite(u) & (u <= float(frontier))
    eligible_values = u[eligible]
    eta = calibration.quantile(1.0 - budget, fallback=eligible_values) if eligible_values.numel() else float(frontier)
    threshold_mask = eligible & (u > eta)
    target_k = min(int(math.ceil(budget * int(eligible.sum().item()))), int(eligible.sum().item()))
    selected = torch.zeros_like(eligible)
    candidate_idx = threshold_mask.nonzero(as_tuple=False).flatten()
    if candidate_idx.numel() > target_k:
        chosen = candidate_idx[torch.topk(u[candidate_idx], k=target_k, largest=True).indices]
    elif candidate_idx.numel() > 0:
        chosen = candidate_idx
    elif target_k > 0:
        # Quantile ties or a drifting calibration distribution: preserve the
        # intended upper-tail policy with a deterministic within-batch fallback.
        idx = eligible.nonzero(as_tuple=False).flatten()
        chosen = idx[torch.topk(u[idx], k=target_k, largest=True).indices]
    else:
        chosen = candidate_idx
    if chosen.numel():
        selected[chosen] = True
    calibration.update(eligible_values)
    # Eq. (17): differentiable soft band; hard access remains discrete.
    gate = torch.sigmoid((u - eta) / rho) * torch.sigmoid((float(frontier) - u) / rho)
    gate = torch.where(torch.isfinite(gate), gate, torch.zeros_like(gate))
    weights = selected.to(gate.dtype) * gate
    diagnostics = {
        "eta": float(eta), "frontier": float(frontier),
        "eligible": int(eligible.sum().item()), "selected": int(selected.sum().item()),
        "realized_rate": float(selected.float().mean().item()),
        "eligible_rate": float(eligible.float().mean().item()),
        "soft_rate": float(gate.mean().detach().item()),
    }
    return selected, gate, weights, diagnostics


def matched_rate_select(entropy: torch.Tensor, count: int, strategy: str, generator: torch.Generator) -> torch.Tensor:
    n = entropy.numel()
    k = min(n, max(0, int(count)))
    out = torch.zeros(n, dtype=torch.bool, device=entropy.device)
    if k == 0:
        return out
    if strategy == "random":
        scores = torch.rand(n, generator=generator, device=entropy.device)
    elif strategy == "entropy":
        scores = entropy.float()
    else:
        raise ValueError("strategy must be 'random' or 'entropy'")
    out[torch.topk(scores, k=k, largest=True).indices] = True
    return out
