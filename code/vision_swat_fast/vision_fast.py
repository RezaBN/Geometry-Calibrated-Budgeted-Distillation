"""Vectorized, native-grid Vision SWAT with bounded-memory batching.

This module changes the execution of SWAT, not its quantile-grid W2 objective.
It is deliberately Vision-specific; NLP's shared reference swat_loss is untouched.
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F

from gcbd_v2.adapters.vision import vision_feature_measure
from gcbd_v2.common.geometry import spatial_grid_support, swat_loss

# Cache only small projected support plans (not teacher features or student tensors).
_PLAN_CACHE: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
_DIRECTION_CACHE: OrderedDict[tuple, torch.Tensor] = OrderedDict()
_PLAN_CAPACITY = 48
_CHUNK_SIZE = 16  # Bound intermediate [chunk, projections, support_size] allocations.


def _remember(cache: OrderedDict, key: tuple, value: Any, limit: int):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


def _directions(n: int, dimension: int, seed: int, device: torch.device) -> torch.Tensor:
    key = (str(device), int(dimension), int(n), int(seed))
    if key in _DIRECTION_CACHE:
        _DIRECTION_CACHE.move_to_end(key)
        return _DIRECTION_CACHE[key]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    directions = F.normalize(
        torch.randn(int(n), int(dimension), generator=generator, device=device),
        dim=-1,
    )
    return _remember(_DIRECTION_CACHE, key, directions, 32)


def _plan(key: tuple, support: torch.Tensor, n: int, seed: int):
    """Return sorted projection values and original-support permutation."""
    if key in _PLAN_CACHE:
        _PLAN_CACHE.move_to_end(key)
        return _PLAN_CACHE[key]
    with torch.no_grad():
        directions = _directions(n, support.shape[-1], seed, support.device)
        # Match the reference's per-direction matvec (not a different GEMM path).
        projected = torch.stack([support @ direction for direction in directions], dim=0)
        permutation = torch.argsort(projected, dim=-1)
        sorted_values = projected.gather(-1, permutation)
    return _remember(_PLAN_CACHE, key, (sorted_values, permutation), _PLAN_CAPACITY)


def _grid_shape(features: torch.Tensor) -> tuple[int, int]:
    if features.ndim == 4:
        return (int(features.shape[-2]), int(features.shape[-1]))
    if features.ndim == 3:
        side = math.isqrt(int(features.shape[1]) - 1)
        if side * side != int(features.shape[1]) - 1:
            raise ValueError('Vision patch tokens must form a square grid')
        return (side, side)
    raise ValueError(f'Unexpected vision feature shape: {tuple(features.shape)}')


def _grid_plan(features: torch.Tensor, n: int, seed: int):
    h, w = _grid_shape(features)
    device = features.device
    key = ('vision-fp32-grid', h, w, str(device), int(n), int(seed))
    if key in _PLAN_CACHE:
        _PLAN_CACHE.move_to_end(key)
        return _PLAN_CACHE[key]
    with torch.no_grad():
        grid = spatial_grid_support(h, w, device=device, dtype=torch.float32)
    return _plan(key, grid, n, seed)


def _cached_teacher_plan(support_cpu: torch.Tensor, n: int, seed: int, device: torch.device):
    # Hash the actual fp16 cached support, not an approximated reconstructed grid.
    cpu = support_cpu.detach().contiguous().to('cpu')
    token = hashlib.blake2b(cpu.numpy().tobytes(), digest_size=16).hexdigest()
    key = ('vision-actual-teacher-support', token, str(cpu.dtype), tuple(cpu.shape),
           str(device), int(n), int(seed))
    if key in _PLAN_CACHE:
        _PLAN_CACHE.move_to_end(key)
        return _PLAN_CACHE[key]
    support = cpu.to(device=device, dtype=torch.float32)
    return _plan(key, support, n, seed)


def _quantiles_from_plan(
    weights: torch.Tensor, plan: tuple[torch.Tensor, torch.Tensor], q: torch.Tensor
) -> torch.Tensor:
    """All samples and directions at once, including exact zero-weight skipping.

    The reference removes zero-mass support locations before computing the CDF.
    Keep the dense layout here, but select the *previous positive-mass* support
    instead of the preceding array entry when interpolating quantiles.
    """
    sorted_x, permutation = plan  # [P,N]
    batch, support_size = weights.shape
    n_projections = permutation.shape[0]
    perm = permutation.unsqueeze(0).expand(batch, -1, -1)
    w = weights.float().clamp_min(0).unsqueeze(1).expand(-1, n_projections, -1)
    w = torch.gather(w, dim=-1, index=perm)
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    cdf = torch.cumsum(w, dim=-1)
    queries = q.view(1, 1, -1).expand(batch, n_projections, -1)
    indices = torch.searchsorted(cdf.detach().contiguous(), queries.contiguous(), right=False)
    indices = indices.clamp(max=support_size - 1)
    positive = w > 0
    positions = torch.arange(support_size, device=w.device, dtype=torch.long).view(1, 1, -1)
    # The positive-mass safeguard is algebraically equivalent to zero filtering.
    first = positive.to(torch.int64).argmax(dim=-1, keepdim=True)
    last = torch.where(positive, positions, -1).amax(dim=-1, keepdim=True)
    indices = torch.minimum(torch.maximum(indices, first), last)
    prev_valid = torch.cummax(torch.where(positive, positions, -1), dim=-1).values
    prev = prev_valid.gather(-1, (indices - 1).clamp_min(0))
    has_prev = prev >= 0
    prev_safe = prev.clamp_min(0)
    cdf_prev = torch.where(has_prev, cdf.gather(-1, prev_safe), torch.zeros_like(queries))
    xs = sorted_x.unsqueeze(0).expand(batch, -1, -1)
    x_curr = xs.gather(-1, indices)
    x_prev = torch.where(has_prev, xs.gather(-1, prev_safe), x_curr)
    current_mass = w.gather(-1, indices)
    fraction = ((queries - cdf_prev) / current_mass.clamp_min(1e-12)).clamp(0.0, 1.0)
    return x_prev + fraction * (x_curr - x_prev)


def _batched_w2(
    weights_s: torch.Tensor, student_plan: tuple[torch.Tensor, torch.Tensor],
    weights_t: torch.Tensor, teacher_plan: tuple[torch.Tensor, torch.Tensor],
    *, quantile_bins: int = 128, chunk_size: int = _CHUNK_SIZE,
) -> torch.Tensor:
    if weights_s.ndim != 2 or weights_t.ndim != 2 or weights_s.shape[0] != weights_t.shape[0]:
        raise ValueError('Vision weights need matching [B,N] batch axes')
    if int(quantile_bins) < 2 or int(chunk_size) < 1:
        raise ValueError('Quantile bins must be >=2 and chunk_size >=1')
    if student_plan[0].shape[0] != teacher_plan[0].shape[0]:
        raise ValueError('Teacher and student projections must match')
    batch_size = weights_s.shape[0]
    if batch_size == 0:
        raise ValueError('Empty SWAT batch')
    # No tensor-to-Python conversions inside the costly image/projection loops.
    q = torch.linspace(0, 1, int(quantile_bins), device=weights_s.device, dtype=torch.float32)
    accumulator = weights_s.new_zeros((), dtype=torch.float32)
    for start in range(0, batch_size, int(chunk_size)):
        stop = min(start + int(chunk_size), batch_size)
        qs = _quantiles_from_plan(weights_s[start:stop], student_plan, q)
        qt = _quantiles_from_plan(weights_t[start:stop], teacher_plan, q)
        accumulator = accumulator + (qs - qt).square().sum()
    return accumulator / (batch_size * student_plan[0].shape[0] * quantile_bins)


def _homogeneous_support(records: list[dict], layer: str) -> torch.Tensor | None:
    """Return exact shared cache support, or None for heterogeneous supports.

    Avoid repeatedly comparing the same shard-shared support tensor.
    """
    first = records[0]['layers'][layer]['support']
    unique: dict[tuple, torch.Tensor] = {}
    for record in records:
        tensor = record['layers'][layer]['support']
        signature = (tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype))
        unique.setdefault(signature, tensor)
    for tensor in unique.values():
        if tensor.shape != first.shape or tensor.dtype != first.dtype or not torch.equal(tensor, first):
            return None
    return first


def cached_vision_swat_loss_fast(*, student_features: dict[str, torch.Tensor],
                                  sample_ids: torch.Tensor | list[int], cache,
                                  layer_pairs: list[tuple[str, str]],
                                  active_layers: int | None = None,
                                  projections: int = 16, seed: int = 0):
    count = len(layer_pairs) if active_layers is None else max(0, min(int(active_layers), len(layer_pairs)))
    if count == 0:
        reference = next(iter(student_features.values()))
        return reference.new_zeros(()), {}
    if int(projections) < 1:
        raise ValueError('At least one projection required')
    ids = [int(i) for i in torch.as_tensor(sample_ids).detach().cpu().tolist()]
    # A full CIFAR100 cache is small relative to node RAM. Retain verified shards
    # across batches/epochs, avoiding SHA checks and torch.load on every step.
    if not getattr(cache, '_vision_fast_shards_configured', False):
        shard_count = len(cache.manifest['completed_shards'])
        cache.max_open_shards = max(cache.max_open_shards, min(shard_count, 256))
        cache._vision_fast_shards_configured = True
    records = cache.get_many(ids)  # grouped shard lookups (patched reader)
    losses, diagnostics = [], {}
    for layer_pos, (teacher_name, student_name) in enumerate(layer_pairs[:count]):
        if student_name not in student_features:
            raise KeyError(f'Student feature hook {student_name!r} is missing')
        student_feature = student_features[student_name]
        student_weights, _ = vision_feature_measure(student_feature)
        if student_weights.shape[0] != len(ids):
            raise ValueError('Sample IDs and student batch length differ')
        teacher_support_cpu = _homogeneous_support(records, teacher_name)
        if teacher_support_cpu is None:
            # Correctness-preserving fallback for unusual variable native grids.
            from gcbd_v2.integrations.vision import _single_swat
            old_losses = [
                _single_swat(student_weights[row],
                             spatial_grid_support(*_grid_shape(student_feature), device=student_weights.device),
                             records[row]['layers'][teacher_name], projections, seed + layer_pos)
                for row in range(len(ids))
            ]
            layer_loss = torch.stack(old_losses).mean()
        else:
            w_cpu = torch.stack([record['layers'][teacher_name]['weights'] for record in records])
            w_teacher = w_cpu.to(device=student_weights.device, dtype=torch.float32, non_blocking=True)
            w_teacher = w_teacher / w_teacher.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            student_plan = _grid_plan(student_feature, int(projections), int(seed) + layer_pos)
            teacher_plan = _cached_teacher_plan(
                teacher_support_cpu, int(projections), int(seed) + layer_pos, student_weights.device)
            layer_loss = _batched_w2(student_weights, student_plan, w_teacher, teacher_plan)
        losses.append(layer_loss)
        # The training loop discards these values. Keep the diagnostics as
        # detached tensors, avoiding a mandatory GPU->CPU sync each layer/step.
        # Call float(value) in offline diagnostics if scalar Python output is needed.
        diagnostics[f'{teacher_name}_to_{student_name}'] = layer_loss.detach()
    return torch.stack(losses).mean(), diagnostics


def queried_vision_final_swat_loss_fast(student_final_features: torch.Tensor,
                                        teacher_final_features: torch.Tensor,
                                        *, projections: int = 16, seed: int = 991):
    weights_s, _ = vision_feature_measure(student_final_features)
    weights_t, _ = vision_feature_measure(teacher_final_features)
    student_plan = _grid_plan(student_final_features, int(projections), int(seed))
    teacher_plan = _grid_plan(teacher_final_features, int(projections), int(seed))
    return _batched_w2(weights_s, student_plan, weights_t, teacher_plan)
