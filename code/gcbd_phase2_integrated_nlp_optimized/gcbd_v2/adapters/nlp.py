"""Heterogeneous-tokenizer NLP geometry adapters based on character supports."""
from __future__ import annotations
import math
import torch
from gcbd_v2.common.geometry import energy_measure, token_offset_support


def layer_indices_by_depth(num_transformer_layers: int, fractions=(0.25, 0.50, 0.75)) -> list[int]:
    """Return 1-based hidden_states indices for ordered interior transformer layers.

    Hugging Face hidden_states[0] is the embedding output; index k is the
    output of transformer block k. The final block is excluded from this
    early-cache mapping and handled by the online final-refinement branch.
    """
    if num_transformer_layers < 2:
        raise ValueError("At least two transformer layers are required")
    indices = []
    for fraction in fractions:
        if not 0.0 < fraction < 1.0:
            raise ValueError("early-layer fractions must lie strictly between 0 and 1")
        idx = max(1, min(num_transformer_layers - 1, int(math.floor(fraction * num_transformer_layers + 0.5))))
        if idx not in indices:
            indices.append(idx)
    return sorted(indices)


def _valid_token_geometry(hidden: torch.Tensor, offsets: torch.Tensor, attention_mask: torch.Tensor,
                          text_length: int) -> dict[str, torch.Tensor]:
    support, valid = token_offset_support(
        offsets.unsqueeze(0), attention_mask.unsqueeze(0),
        torch.tensor([text_length], device=offsets.device),
    )
    valid = valid[0]
    feat = hidden[valid].unsqueeze(0)
    coords = support[0, valid]
    weights, coords = energy_measure(feat, coords)
    return {"weights": weights[0].detach().to(torch.float16).cpu(),
            "support": coords[0].detach().to(torch.float16).cpu()}


def nlp_sample_geometry(
    hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    text_lengths: torch.Tensor,
    layer_indices: list[int],
) -> list[dict[str, torch.Tensor]]:
    """Convert a teacher batch into per-sample, per-layer compact cache records."""
    if offsets.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("invalid batched token offset or attention-mask shape")
    if len(text_lengths) != offsets.shape[0]:
        raise ValueError("text_lengths batch dimension mismatch")
    for idx in layer_indices:
        if idx <= 0 or idx >= len(hidden_states):
            raise IndexError(f"hidden-state layer index {idx} is unavailable")
    records = []
    for b in range(offsets.shape[0]):
        per_layer = {}
        for idx in layer_indices:
            per_layer[str(idx)] = _valid_token_geometry(
                hidden_states[idx][b], offsets[b], attention_mask[b], int(text_lengths[b])
            )
        records.append({"layers": per_layer})
    return records


def nlp_student_measure(
    hidden: torch.Tensor,
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    text_lengths: torch.Tensor,
) -> list[dict[str, torch.Tensor]]:
    """Build student native-support measures for a batch, preserving variable lengths."""
    records = []
    for b in range(hidden.shape[0]):
        records.append(_valid_token_geometry(
            hidden[b], offsets[b], attention_mask[b], int(text_lengths[b])
        ))
    return records
