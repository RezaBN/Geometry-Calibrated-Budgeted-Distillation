"""Probability-measure geometry and native-support sliced Wasserstein alignment."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def spatial_grid_support(height: int, width: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return normalized (x,y) cell-centre coordinates with shape [H*W,2]."""
    if height < 1 or width < 1:
        raise ValueError("height and width must be positive")
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def token_offset_support(
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    text_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map fast-tokenizer character offsets to normalized 1-D token supports.

    Special tokens generally have (0,0) offsets and are excluded. The output
    support is [B,L,1] and the validity mask is [B,L].
    """
    if offsets.ndim != 3 or offsets.shape[-1] != 2:
        raise ValueError("offsets must have shape [B,L,2]")
    if attention_mask.shape != offsets.shape[:2]:
        raise ValueError("attention_mask must have shape [B,L]")
    lengths = torch.as_tensor(text_lengths, device=offsets.device, dtype=torch.float32).reshape(-1)
    if lengths.numel() != offsets.shape[0]:
        raise ValueError("text_lengths must contain one length per sample")
    starts = offsets[..., 0].to(torch.float32)
    ends = offsets[..., 1].to(torch.float32)
    valid = attention_mask.to(torch.bool) & (ends > starts)
    denom = lengths.clamp_min(1.0).unsqueeze(1)
    centers = ((starts + ends) * 0.5 / denom).clamp(0.0, 1.0)
    centers = centers.masked_fill(~valid, 0.0)
    if not valid.any(dim=1).all():
        bad = (~valid.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"Samples have no non-special valid tokens: {bad}")
    return centers.unsqueeze(-1), valid


def energy_measure(
    features: torch.Tensor,
    support: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert local descriptors into sum-to-one energy weights.

    Accepts CNN features [B,C,H,W] or local descriptors [B,N,D]. A small
    positive epsilon is added to each valid location, matching the manuscript's
    normalized squared-L2 energy definition. Invalid locations receive zero.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    if features.ndim == 4:
        features = features.flatten(2).transpose(1, 2)
    if features.ndim != 3:
        raise ValueError(f"Expected [B,N,D] or [B,C,H,W], got {tuple(features.shape)}")
    energy = features.float().square().sum(dim=-1) + float(eps)
    if support.ndim == 2:
        support = support.unsqueeze(0).expand(features.shape[0], -1, -1)
    if support.ndim != 3 or support.shape[:2] != features.shape[:2]:
        raise ValueError("support must have shape [N,d] or [B,N,d] matching features")
    if not torch.isfinite(support).all():
        raise ValueError("support contains non-finite coordinates")
    if mask is not None:
        mask = mask.to(device=energy.device, dtype=torch.bool)
        if mask.shape != energy.shape:
            raise ValueError("mask must have shape [B,N]")
        if not mask.any(dim=1).all():
            raise ValueError("Every sample must have at least one valid support location")
        energy = energy.masked_fill(~mask, 0.0)
    denom = energy.sum(dim=-1, keepdim=True)
    if not torch.isfinite(denom).all() or (denom <= 0).any():
        raise ValueError("Energy normalization denominator is invalid")
    weights = energy / denom
    return weights, support.to(device=features.device, dtype=torch.float32)


def _weighted_quantile(values: torch.Tensor, weights: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    x = values[order]
    w = weights[order].clamp_min(0)
    keep = w > 0
    x, w = x[keep], w[keep]
    if x.numel() == 0:
        raise ValueError("A sliced measure has zero total mass")
    w = w / w.sum().clamp_min(1e-12)
    cdf = w.cumsum(0)
    idx = torch.searchsorted(cdf.detach().contiguous(), q.contiguous(), right=False)
    idx = idx.clamp(min=0, max=x.numel() - 1)
    prev_idx = (idx - 1).clamp(min=0)
    prev_cdf = torch.where(idx > 0, cdf[prev_idx], torch.zeros_like(q))
    prev_x = torch.where(idx > 0, x[prev_idx], x[idx])
    curr_x = x[idx]
    frac = ((q - prev_cdf) / w[idx].clamp_min(1e-12)).clamp(0.0, 1.0)
    return prev_x + frac * (curr_x - prev_x)


def swat_loss(
    weights_s: torch.Tensor,
    support_s: torch.Tensor,
    weights_t: torch.Tensor,
    support_t: torch.Tensor,
    n_projections: int = 16,
    seed: int = 0,
    quantile_bins: int = 128,
) -> torch.Tensor:
    """Batched sliced W2^2 between measures on native, commensurate supports."""
    if weights_s.ndim != 2 or weights_t.ndim != 2:
        raise ValueError("weights must have shape [B,N]")
    if weights_s.shape[0] != weights_t.shape[0]:
        raise ValueError("teacher/student batch sizes differ")
    if support_s.ndim == 2:
        support_s = support_s.unsqueeze(0).expand(weights_s.shape[0], -1, -1)
    if support_t.ndim == 2:
        support_t = support_t.unsqueeze(0).expand(weights_t.shape[0], -1, -1)
    if support_s.shape[:2] != weights_s.shape or support_t.shape[:2] != weights_t.shape:
        raise ValueError("support and weight shapes do not match")
    if support_s.shape[-1] != support_t.shape[-1]:
        raise ValueError("teacher/student support dimensions differ")
    if n_projections < 1 or quantile_bins < 2:
        raise ValueError("n_projections >= 1 and quantile_bins >= 2 are required")
    device = weights_s.device
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    directions = F.normalize(
        torch.randn(n_projections, support_s.shape[-1], generator=gen, device=device), dim=-1
    )
    q = torch.linspace(0.0, 1.0, quantile_bins, device=device, dtype=torch.float32)
    sample_losses = []
    for b in range(weights_s.shape[0]):
        if not torch.isclose(weights_s[b].sum(), torch.tensor(1.0, device=device), atol=2e-3):
            raise ValueError("student energy weights must sum to one")
        if not torch.isclose(weights_t[b].sum(), torch.tensor(1.0, device=device), atol=2e-3):
            raise ValueError("teacher energy weights must sum to one")
        directional = []
        for theta in directions:
            xs = support_s[b] @ theta
            xt = support_t[b] @ theta
            qs = _weighted_quantile(xs, weights_s[b], q)
            qt = _weighted_quantile(xt, weights_t[b], q)
            directional.append((qs - qt).square().mean())
        sample_losses.append(torch.stack(directional).mean())
    return torch.stack(sample_losses).mean()
