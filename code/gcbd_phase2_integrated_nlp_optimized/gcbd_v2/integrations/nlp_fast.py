"""GCBD-NLP V3: batched native-support SWAT, preserving the reference objective.

Variable-length character-offset measures are packed into zero-mass padded rows.
The quantile routine ignores zero-mass locations just as the reference does.
No modification to the frozen geometry cache or the teacher-query policy.

The original reference functions remain importable from integrations.nlp as
cached_nlp_swat_loss_reference and queried_nlp_final_swat_loss_reference.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

from gcbd_v2.common.geometry import token_offset_support


@lru_cache(maxsize=64)
def _cache_cap_default() -> int:
    # Only small, completed caches are kept fully resident. The original
    # SST-2 cache is ~126MB; after the first pass this avoids all shard I/O.
    return int(os.environ.get('GCBD_NLP_V3_MAX_RESIDENT_BYTES', str(256 * 1024**2)))


def _configure_reader(cache: Any) -> None:
    if getattr(cache, '_nlp_v3_reader_configured', False):
        return
    total = int(cache.manifest.get('total_bytes', 0))
    shards = len(cache.manifest['completed_shards'])
    if 0 < total <= _cache_cap_default():
        cache.max_open_shards = max(cache.max_open_shards, shards)
    # Keep otherwise the original bounded reader behavior.
    cache._nlp_v3_reader_configured = True


def _student_measure(hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """The original squared-L2 energy, with padded locations having zero mass."""
    energy = hidden.float().square().sum(dim=-1) + 1e-12
    energy = energy.masked_fill(~valid, 0.0)
    return energy / energy.sum(dim=-1, keepdim=True)


def _pad_teacher_layer(records: list[dict], layer: int,
                       device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the *stored FP16* teacher support and mass without approximation."""
    layer_key = str(int(layer))
    entries = [rec['layers'][layer_key] for rec in records]
    length = max(int(rec['weights'].numel()) for rec in entries)
    weights = torch.zeros((len(entries), length), dtype=torch.float32)
    supports = torch.zeros((len(entries), length, 1), dtype=torch.float32)
    for idx, rec in enumerate(entries):
        w = rec['weights'].float().reshape(-1)
        x = rec['support'].float().reshape(-1, 1)
        if w.numel() != x.shape[0]:
            raise ValueError('Mismatched stored teacher support and weights')
        weights[idx, :w.numel()] = w
        supports[idx, :w.numel()] = x
    weights = weights.to(device=device, non_blocking=True)
    supports = supports.to(device=device, non_blocking=True)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12), supports


def _weighted_quantile_batched(
    projected: torch.Tensor, weights: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """Weighted quantiles for [B,P,N], omitting zero-weight padded entries.

    This exactly follows the reference's argsort -> positive filter -> CDF ->
    searchsorted -> interpolation sequence in each row. The preceding node
    is the preceding *positive-mass* node, not the preceding padded entry.
    """
    order = torch.argsort(projected, dim=-1)
    x = projected.gather(-1, order)
    w = weights.unsqueeze(1).expand(-1, projected.size(1), -1).gather(-1, order)
    w = w.clamp_min(0)
    positive = w > 0
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    cdf = w.cumsum(dim=-1)

    queries = q.view(1, 1, -1).expand(w.shape[0], w.shape[1], -1)
    indices = torch.searchsorted(cdf.detach().contiguous(), queries.contiguous(), right=False)
    indices = indices.clamp(max=w.shape[-1] - 1)

    positions = torch.arange(w.shape[-1], device=w.device, dtype=torch.long).view(1, 1, -1)
    first = positive.to(torch.long).argmax(dim=-1, keepdim=True)
    last = torch.where(positive, positions, -1).amax(dim=-1, keepdim=True)
    indices = torch.minimum(torch.maximum(indices, first), last)
    prev_valid = torch.cummax(torch.where(positive, positions, -1), dim=-1).values
    prev = prev_valid.gather(-1, (indices - 1).clamp_min(0))
    has_prev = prev >= 0
    prev_idx = prev.clamp_min(0)

    cdf_prev = torch.where(has_prev, cdf.gather(-1, prev_idx), torch.zeros_like(queries))
    x_curr = x.gather(-1, indices)
    x_prev = torch.where(has_prev, x.gather(-1, prev_idx), x_curr)
    mass = w.gather(-1, indices)
    frac = ((queries - cdf_prev) / mass.clamp_min(1e-12)).clamp(0.0, 1.0)
    return x_prev + frac * (x_curr - x_prev)


def _swat_batched(
    weights_s: torch.Tensor, support_s: torch.Tensor,
    weights_t: torch.Tensor, support_t: torch.Tensor,
    *, projections: int, seed: int, quantile_bins: int = 128,
) -> torch.Tensor:
    if weights_s.ndim != 2 or weights_t.ndim != 2 or weights_s.shape[0] != weights_t.shape[0]:
        raise ValueError('SWAT requires [B,L] teacher/student weights with matching B')
    if support_s.shape != (*weights_s.shape, 1) or support_t.shape != (*weights_t.shape, 1):
        raise ValueError('NLP SWAT requires normalized 1-D character supports [B,L,1]')
    if projections < 1 or quantile_bins < 2:
        raise ValueError('projections >=1 and quantile_bins >=2 are required')
    device = weights_s.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    # Identical projection generator/direction normalization as swat_loss.
    directions = F.normalize(
        torch.randn(projections, 1, generator=generator, device=device), dim=-1,
    )
    q = torch.linspace(0, 1, quantile_bins, device=device, dtype=torch.float32)
    # [B,P,L]. All samples and projections share a small number of GPU launches.
    xs = support_s.float().transpose(1, 2) * directions.view(1, projections, 1)
    xt = support_t.float().transpose(1, 2) * directions.view(1, projections, 1)
    qs = _weighted_quantile_batched(xs, weights_s.float(), q)
    qt = _weighted_quantile_batched(xt, weights_t.float(), q)
    return (qs - qt).square().mean()


def cached_nlp_swat_loss_fast(
    *, student_hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
    student_offsets: torch.Tensor, student_attention_mask: torch.Tensor,
    text_lengths: torch.Tensor, sample_ids: torch.Tensor | list[int],
    cache: Any, student_layer_indices: list[int], teacher_layer_indices: list[int],
    active_layers: int | None = None, projections: int = 8, seed: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if len(student_layer_indices) != len(teacher_layer_indices):
        raise ValueError('Teacher/student NLP geometry layer maps must have equal lengths')
    count = len(student_layer_indices) if active_layers is None else max(
        0, min(int(active_layers), len(student_layer_indices))
    )
    if not count:
        return student_hidden_states[-1].new_zeros(()), {}
    ids = [int(x) for x in torch.as_tensor(sample_ids).detach().cpu().tolist()]
    if len(ids) != student_hidden_states[0].shape[0]:
        raise ValueError('Sample ID count does not match student batch size')
    _configure_reader(cache)
    records = cache.get_many(ids)
    # Only once for all active layers, not for every sample * layer.
    supports, valid = token_offset_support(student_offsets, student_attention_mask, text_lengths)
    losses = []
    diagnostics = {}
    for layer_pos, (s_idx, t_idx) in enumerate(zip(student_layer_indices[:count], teacher_layer_indices[:count])):
        if s_idx <= 0 or s_idx >= len(student_hidden_states):
            raise IndexError(f'Student hidden-state index {s_idx} unavailable')
        student_weights = _student_measure(student_hidden_states[s_idx], valid)
        teacher_weights, teacher_supports = _pad_teacher_layer(records, t_idx, student_weights.device)
        layer_loss = _swat_batched(student_weights, supports, teacher_weights, teacher_supports,
                                  projections=int(projections), seed=int(seed) + layer_pos)
        losses.append(layer_loss)
        # Training loop does not use these diagnostics; avoid GPU synchronization.
        diagnostics[f'layer_{t_idx}_loss'] = layer_loss.detach()
    return torch.stack(losses).mean(), diagnostics


def queried_nlp_final_swat_loss_fast(
    *, student_final_hidden: torch.Tensor,
    teacher_hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
    student_offsets: torch.Tensor, student_attention_mask: torch.Tensor,
    teacher_offsets: torch.Tensor, teacher_attention_mask: torch.Tensor,
    text_lengths: torch.Tensor, teacher_text_lengths: torch.Tensor,
    projections: int = 8, seed: int = 991,
) -> torch.Tensor:
    if student_final_hidden.shape[0] == 0:
        return student_final_hidden.new_zeros(())
    ss, sv = token_offset_support(student_offsets, student_attention_mask, text_lengths)
    ts, tv = token_offset_support(teacher_offsets, teacher_attention_mask, teacher_text_lengths)
    student_weights = _student_measure(student_final_hidden, sv)
    teacher_weights = _student_measure(teacher_hidden_states[-1].detach(), tv)

    # The reference nlp_sample_geometry first stores teacher weights and
    # supports as FP16 on CPU, then _pair_swat restores float32 / unit mass.
    # An FP16 round-trip on-device reproduces this without a CPU transfer.
    teacher_weights = teacher_weights.to(torch.float16).float()
    teacher_weights = teacher_weights / teacher_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    ts = ts.to(torch.float16).float()

    return _swat_batched(student_weights, ss, teacher_weights, ts,
                         projections=int(projections), seed=int(seed))
