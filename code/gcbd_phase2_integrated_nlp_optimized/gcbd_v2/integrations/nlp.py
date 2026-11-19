"""NLP runner integration: stable IDs, character supports, one-pass cache and SWAT."""
from __future__ import annotations

import hashlib
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from gcbd_v2.adapters.nlp import layer_indices_by_depth, nlp_sample_geometry
from gcbd_v2.common.cache import GeometryCacheReader
from gcbd_v2.common.extraction import extract_geometry_cache
from gcbd_v2.common.geometry import energy_measure, swat_loss, token_offset_support
from .provenance import (
    cache_directory,
    module_state_sha256,
    open_or_prepare_cache,
    text_manifest_sha256,
    tokenizer_fingerprint,
    write_ledger_json,
    cache_build_lock,
)


class HiddenStateForwarder(torch.nn.Module):
    """Ensure a Hugging Face model returns hidden states in extraction passes."""
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, **inputs):
        return self.model(**inputs, output_hidden_states=True, return_dict=True)


def attach_stable_sample_ids(dataset):
    """Attach deterministic split-local IDs after any sampling/capping operation."""
    if "sample_id" in dataset.column_names:
        ids = [int(x) for x in dataset["sample_id"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Existing sample_id column is not unique")
        return dataset
    return dataset.add_column("sample_id", list(range(len(dataset))))


def _tokenize_offsets(tokenizer, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError("GCBD NLP geometry requires a fast tokenizer with return_offsets_mapping")
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    if "offset_mapping" not in encoded:
        raise RuntimeError("Fast tokenizer did not return offset_mapping")
    return {key: value for key, value in encoded.items()}


class NLPGeometryExtractionCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [str(item["text"]) for item in features]
        encoded = _tokenize_offsets(self.tokenizer, texts, self.max_length)
        model_inputs = {k: v for k, v in encoded.items() if k != "offset_mapping"}
        return {
            "model_inputs": model_inputs,
            "offset_mapping": encoded["offset_mapping"],
            "attention_mask": encoded["attention_mask"],
            "text_lengths": torch.tensor([len(x) for x in texts], dtype=torch.long),
            "sample_id": torch.tensor([int(item["sample_id"]) for item in features], dtype=torch.long),
        }


def _masked_lm_inputs(clean: dict[str, torch.Tensor], tokenizer, mlm_probability: float):
    if tokenizer.mask_token is None:
        raise ValueError("Masked-LM tokenizer must define mask_token")
    training = {key: value.clone() for key, value in clean.items()}
    input_ids = training["input_ids"]
    labels = input_ids.clone()
    special = torch.tensor(
        [tokenizer.get_special_tokens_mask(row.tolist(), already_has_special_tokens=True) for row in input_ids],
        dtype=torch.bool,
    )
    probabilities = torch.full(labels.shape, float(mlm_probability), dtype=torch.float32)
    probabilities.masked_fill_(special, 0.0)
    masked = torch.bernoulli(probabilities).bool()
    labels[~masked] = -100
    replace = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked
    input_ids[replace] = int(tokenizer.mask_token_id)
    random_replace = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked & ~replace
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[random_replace] = random_words[random_replace]
    training["labels"] = labels
    return training


class GCBDClassificationCollator:
    """Dual-tokenizer classification collator retaining character supports."""
    def __init__(self, teacher_tokenizer, student_tokenizer, max_length: int):
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.max_length = int(max_length)
        tokenizer_fingerprint(teacher_tokenizer)
        tokenizer_fingerprint(student_tokenizer)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [str(item["text"]) for item in features]
        teacher = _tokenize_offsets(self.teacher_tokenizer, texts, self.max_length)
        student = _tokenize_offsets(self.student_tokenizer, texts, self.max_length)
        return {
            "teacher": teacher,
            "student": student,
            "labels": torch.tensor([int(item["label"]) for item in features], dtype=torch.long),
            "sample_id": torch.tensor([int(item["sample_id"]) for item in features], dtype=torch.long),
            "text_lengths": torch.tensor([len(x) for x in texts], dtype=torch.long),
            "texts": texts,
        }


class GCBDLMTextCollator:
    """LM collator separating clean geometry views from the MLM training view."""
    def __init__(self, teacher_tokenizer, student_tokenizer, teacher_objective: str,
                 student_objective: str, max_length: int, mlm_probability: float):
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.teacher_objective = teacher_objective
        self.student_objective = student_objective
        self.max_length = int(max_length)
        self.mlm_probability = float(mlm_probability)
        tokenizer_fingerprint(teacher_tokenizer)
        tokenizer_fingerprint(student_tokenizer)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [str(item["text"]) for item in features]
        teacher = _tokenize_offsets(self.teacher_tokenizer, texts, self.max_length)
        student_geometry = _tokenize_offsets(self.student_tokenizer, texts, self.max_length)
        if self.student_objective == "masked":
            student = _masked_lm_inputs(student_geometry, self.student_tokenizer, self.mlm_probability)
        elif self.student_objective == "causal":
            student = {key: value.clone() for key, value in student_geometry.items()}
            labels = student["input_ids"].clone()
            labels[student["attention_mask"] == 0] = -100
            student["labels"] = labels
        else:
            raise ValueError(f"Unsupported student objective: {self.student_objective}")
        if self.teacher_objective not in {"masked", "causal"}:
            raise ValueError(f"Unsupported teacher objective: {self.teacher_objective}")
        return {
            "teacher": teacher,
            "student": student,
            "student_geometry": student_geometry,
            "sample_id": torch.tensor([int(item["sample_id"]) for item in features], dtype=torch.long),
            "text_lengths": torch.tensor([len(x) for x in texts], dtype=torch.long),
            "texts": texts,
        }


@dataclass
class NLPCacheBuild:
    reader: GeometryCacheReader
    root: Path
    teacher_layer_indices: list[int]
    ledger: dict[str, Any]
    cache_hit: bool


def _num_transformer_layers(config: Any) -> int:
    for name in ("num_hidden_layers", "n_layer", "n_layers", "num_layers", "encoder_layers"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError(f"Cannot infer transformer depth from config {type(config).__name__}")


def build_nlp_geometry_cache(
    *,
    teacher: torch.nn.Module,
    dataset,
    tokenizer,
    dataset_name: str,
    scenario: str,
    seed: int,
    base_cache_dir: str | Path,
    batch_size: int,
    max_length: int,
    device: torch.device,
    num_workers: int = 0,
    autocast_context=None,
) -> NLPCacheBuild:
    """Build or validate one complete teacher geometry cache for a train split."""
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError("Cannot build NLP geometry cache with a slow tokenizer")
    if "sample_id" not in dataset.column_names:
        dataset = attach_stable_sample_ids(dataset)
    sample_ids = [int(x) for x in dataset["sample_id"]]
    texts = [str(x) for x in dataset["text"]]
    layer_indices = layer_indices_by_depth(_num_transformer_layers(teacher.config))
    metadata = {
        "domain": "nlp",
        "dataset": dataset_name,
        "scenario": scenario,
        "seed": int(seed),
        "split": "train",
        "sample_count": len(sample_ids),
        "dataset_manifest_sha256": text_manifest_sha256(sample_ids, texts),
        "teacher_state_sha256": module_state_sha256(teacher),
        "teacher_config": teacher.config.to_dict() if hasattr(teacher.config, "to_dict") else str(teacher.config),
        "teacher_tokenizer": tokenizer_fingerprint(tokenizer),
        "max_length": int(max_length),
        "geometry_layer_indices": layer_indices,
        "support": "normalized_character_offset_centres_v1",
        "preprocessing": "raw_unmasked_text_fast_tokenizer_offsets",
    }
    root = cache_directory(base_cache_dir, "nlp", dataset_name, scenario, seed, metadata)
    with cache_build_lock(root):
        if os.environ.get("GCBD_NLP_REQUIRE_CACHE_HIT", "0") == "1":
            # V3 pilot: fail closed if original cache is absent or its
            # full teacher/dataset/tokenizer identity differs. Never rebuild
            # assets inside the shared original geometry-cache directory.
            reader = GeometryCacheReader(root, metadata)
            writer, cache_hit = None, True
        else:
            reader, writer, cache_hit = open_or_prepare_cache(root, metadata)
        if cache_hit:
            ledger_path = root / "ledger.json"
            ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else {
                "geometry_examples": int(reader.manifest.get("sample_count", 0)),
                "geometry_forward_invocations": len(reader.manifest.get("completed_shards", {})),
                "cache_bytes": int(reader.manifest.get("total_bytes", 0)),
                "cache_hit": True,
            }
            ledger["cache_hit"] = True
            return NLPCacheBuild(reader, root, layer_indices, ledger, True)

        collator = NLPGeometryExtractionCollator(tokenizer, max_length)
        loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False,
                            num_workers=int(num_workers), collate_fn=collator,
                            pin_memory=(device.type == "cuda"))
        model = HiddenStateForwarder(teacher).to(device)
        model.eval()

        def batch_to_inputs(batch, target_device):
            return {key: value.to(target_device, non_blocking=True) for key, value in batch["model_inputs"].items()}

        def outputs_to_records(outputs, batch, target_device):
            return nlp_sample_geometry(
                outputs.hidden_states,
                batch["offset_mapping"].to(target_device),
                batch["attention_mask"].to(target_device),
                batch["text_lengths"].to(target_device),
                layer_indices,
            )

        ledger = extract_geometry_cache(
            model, loader, writer, sample_ids, device,
            batch_to_teacher_inputs=batch_to_inputs,
            outputs_to_records=outputs_to_records,
            autocast_context=autocast_context,
        )
        ledger_payload = ledger.as_dict()
        ledger_payload["cache_hit"] = False
        write_ledger_json(root / "ledger.json", ledger_payload)
        reader = GeometryCacheReader(root, metadata)
        return NLPCacheBuild(reader, root, layer_indices, ledger_payload, False)


def differentiable_token_measure(hidden: torch.Tensor, offsets: torch.Tensor,
                                  attention_mask: torch.Tensor, text_lengths: torch.Tensor):
    """Student energy measures preserving autograd (unlike cache-side records)."""
    supports, valid = token_offset_support(offsets, attention_mask, text_lengths)
    records = []
    for b in range(hidden.shape[0]):
        keep = valid[b]
        features = hidden[b, keep].unsqueeze(0)
        coords = supports[b, keep]
        weights, coords = energy_measure(features, coords)
        records.append({"weights": weights[0], "support": coords[0]})
    return records


def _pair_swat(student_record: dict[str, torch.Tensor], teacher_record: dict[str, torch.Tensor],
               *, projections: int, seed: int) -> torch.Tensor:
    device = student_record["weights"].device
    wt = teacher_record["weights"].to(device=device, dtype=torch.float32)
    wt = wt / wt.sum().clamp_min(1e-12)  # Restore exact unit mass after fp16 cache storage.
    rt = teacher_record["support"].to(device=device, dtype=torch.float32)
    return swat_loss(
        student_record["weights"].float().unsqueeze(0),
        student_record["support"].float().unsqueeze(0),
        wt.unsqueeze(0), rt.unsqueeze(0),
        n_projections=int(projections), seed=int(seed),
    )


def cached_nlp_swat_loss(
    *,
    student_hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
    student_offsets: torch.Tensor,
    student_attention_mask: torch.Tensor,
    text_lengths: torch.Tensor,
    sample_ids: torch.Tensor | list[int],
    cache: GeometryCacheReader,
    student_layer_indices: list[int],
    teacher_layer_indices: list[int],
    active_layers: int | None = None,
    projections: int = 8,
    seed: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(student_layer_indices) != len(teacher_layer_indices):
        raise ValueError("Teacher/student NLP geometry layer maps must have equal lengths")
    count = len(student_layer_indices) if active_layers is None else max(0, min(int(active_layers), len(student_layer_indices)))
    if count == 0:
        reference = student_hidden_states[-1]
        return reference.new_zeros(()), {}
    ids = [int(x) for x in torch.as_tensor(sample_ids).detach().cpu().tolist()]
    if len(ids) != student_hidden_states[0].shape[0]:
        raise ValueError("sample ID count does not match student batch size")
    losses = []
    diagnostics = {}
    for layer_pos, (s_idx, t_idx) in enumerate(zip(student_layer_indices[:count], teacher_layer_indices[:count])):
        if s_idx <= 0 or s_idx >= len(student_hidden_states):
            raise IndexError(f"Student hidden-state index {s_idx} unavailable")
        student_records = differentiable_token_measure(
            student_hidden_states[s_idx], student_offsets, student_attention_mask, text_lengths
        )
        layer_losses = []
        for row, sample_id in enumerate(ids):
            teacher_record = cache.get(sample_id)["layers"][str(t_idx)]
            layer_losses.append(_pair_swat(student_records[row], teacher_record,
                                           projections=projections, seed=seed + layer_pos))
        layer_loss = torch.stack(layer_losses).mean()
        losses.append(layer_loss)
        diagnostics[f"layer_{t_idx}_loss"] = float(layer_loss.detach().cpu())
    return torch.stack(losses).mean(), diagnostics


def queried_nlp_final_swat_loss(
    *,
    student_final_hidden: torch.Tensor,
    teacher_hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
    student_offsets: torch.Tensor,
    student_attention_mask: torch.Tensor,
    teacher_offsets: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    text_lengths: torch.Tensor,
    teacher_text_lengths: torch.Tensor,
    projections: int = 8,
    seed: int = 991,
) -> torch.Tensor:
    student_records = differentiable_token_measure(
        student_final_hidden, student_offsets, student_attention_mask, text_lengths
    )
    teacher_final_idx = len(teacher_hidden_states) - 1
    teacher_records = nlp_sample_geometry(
        teacher_hidden_states,
        teacher_offsets,
        teacher_attention_mask,
        teacher_text_lengths,
        [teacher_final_idx],
    )
    losses = []
    for student_record, teacher_record in zip(student_records, teacher_records):
        losses.append(_pair_swat(student_record, teacher_record["layers"][str(teacher_final_idx)],
                                 projections=projections, seed=seed))
    if not losses:
        return student_final_hidden.new_zeros(())
    return torch.stack(losses).mean()


# V3 execution aliases: reference functions retained for equivalence testing.
cached_nlp_swat_loss_reference = cached_nlp_swat_loss
queried_nlp_final_swat_loss_reference = queried_nlp_final_swat_loss
from .nlp_fast import (
    cached_nlp_swat_loss_fast as cached_nlp_swat_loss,
    queried_nlp_final_swat_loss_fast as queried_nlp_final_swat_loss,
)
