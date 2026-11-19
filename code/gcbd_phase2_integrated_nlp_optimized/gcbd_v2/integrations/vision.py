"""Vision runner integration: deterministic indexed views, offline cache and SWAT."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from gcbd_v2.adapters.vision import vision_feature_measure, vision_sample_geometry
from gcbd_v2.common.cache import GeometryCacheReader
from gcbd_v2.common.extraction import extract_geometry_cache
from gcbd_v2.common.geometry import swat_loss
from .provenance import cache_directory, module_state_sha256, open_or_prepare_cache, vision_manifest_sha256, write_ledger_json, cache_build_lock


class IndexedDataset(Dataset):
    """Wrap an image dataset so stable split-local IDs survive shuffling."""
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[int(index)]
        return image, int(label), int(index)


@dataclass
class VisionCacheBuild:
    reader: GeometryCacheReader
    root: Path
    teacher_layer_names: list[str]
    ledger: dict[str, Any]
    cache_hit: bool


def build_vision_geometry_cache(
    *,
    teacher: torch.nn.Module,
    indexed_dataset: Dataset,
    dataset_name: str,
    scenario: str,
    seed: int,
    teacher_architecture: str,
    teacher_layer_names: list[str],
    base_cache_dir: str | Path,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    autocast_context=None,
    view_fingerprint: str = "deterministic_center_crop_v1",
    class_names: list[str] | None = None,
) -> VisionCacheBuild:
    """Build or validate a one-pass teacher cache on a deterministic image view."""
    base_dataset = getattr(indexed_dataset, "dataset", indexed_dataset)
    if not hasattr(indexed_dataset, "__getitem__"):
        raise TypeError("indexed_dataset must be a Dataset")
    if len(indexed_dataset) != len(base_dataset):
        raise RuntimeError("Indexed dataset length changed")
    if not teacher_layer_names:
        raise ValueError("At least one teacher geometry layer is required")
    metadata = {
        "domain": "vision",
        "dataset": dataset_name,
        "scenario": scenario,
        "seed": int(seed),
        "split": "train",
        "sample_count": len(indexed_dataset),
        "dataset_manifest_sha256": vision_manifest_sha256(base_dataset),
        "teacher_state_sha256": module_state_sha256(teacher),
        "teacher_architecture": teacher_architecture,
        "teacher_geometry_layers": list(teacher_layer_names),
        "support": "normalized_native_spatial_cell_centres_v1",
        "view_fingerprint": str(view_fingerprint),
        "class_names_sha256": __import__("hashlib").sha256(
            "\n".join(class_names or []).encode("utf-8")
        ).hexdigest(),
    }
    root = cache_directory(base_cache_dir, "vision", dataset_name, scenario, seed, metadata)
    with cache_build_lock(root):
        reader, writer, cache_hit = open_or_prepare_cache(root, metadata)
        if cache_hit:
            ledger_path = root / "ledger.json"
            import json
            ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else {
                "geometry_examples": int(reader.manifest.get("sample_count", 0)),
                "geometry_forward_invocations": len(reader.manifest.get("completed_shards", {})),
                "cache_bytes": int(reader.manifest.get("total_bytes", 0)),
                "cache_hit": True,
            }
            ledger["cache_hit"] = True
            return VisionCacheBuild(reader, root, list(teacher_layer_names), ledger, True)

        kwargs: dict[str, Any] = {
            "batch_size": int(batch_size),
            "shuffle": False,
            "drop_last": False,
            "num_workers": int(num_workers),
            "pin_memory": bool(pin_memory and device.type == "cuda"),
            "persistent_workers": bool(persistent_workers and num_workers > 0),
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(prefetch_factor)
        loader = DataLoader(indexed_dataset, **kwargs)

        def batch_to_teacher_inputs(batch, target_device):
            images, _labels, _sample_ids = batch
            return images.to(target_device, non_blocking=True)

        def outputs_to_records(outputs, _batch, _target_device):
            _logits, features = outputs
            return vision_sample_geometry(features, teacher_layer_names)

        ledger = extract_geometry_cache(
            teacher, loader, writer, range(len(indexed_dataset)), device,
            batch_to_teacher_inputs=batch_to_teacher_inputs,
            outputs_to_records=outputs_to_records,
            autocast_context=autocast_context,
        )
        ledger_payload = ledger.as_dict()
        ledger_payload["cache_hit"] = False
        write_ledger_json(root / "ledger.json", ledger_payload)
        reader = GeometryCacheReader(root, metadata)
        return VisionCacheBuild(reader, root, list(teacher_layer_names), ledger_payload, False)


def _single_swat(student_weights: torch.Tensor, student_support: torch.Tensor,
                 teacher_record: dict[str, torch.Tensor], projections: int, seed: int):
    device = student_weights.device
    teacher_weights = teacher_record["weights"].to(device=device, dtype=torch.float32)
    teacher_weights = teacher_weights / teacher_weights.sum().clamp_min(1e-12)
    teacher_support = teacher_record["support"].to(device=device, dtype=torch.float32)
    return swat_loss(
        student_weights.float().unsqueeze(0), student_support.float().unsqueeze(0),
        teacher_weights.unsqueeze(0), teacher_support.unsqueeze(0),
        n_projections=int(projections), seed=int(seed),
    )


def cached_vision_swat_loss(
    *,
    student_features: dict[str, torch.Tensor],
    sample_ids: torch.Tensor | list[int],
    cache: GeometryCacheReader,
    layer_pairs: list[tuple[str, str]],
    active_layers: int | None = None,
    projections: int = 16,
    seed: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    count = len(layer_pairs) if active_layers is None else max(0, min(int(active_layers), len(layer_pairs)))
    if count == 0:
        reference = next(iter(student_features.values()))
        return reference.new_zeros(()), {}
    ids = [int(x) for x in torch.as_tensor(sample_ids).detach().cpu().tolist()]
    losses, diagnostics = [], {}
    for layer_pos, (teacher_name, student_name) in enumerate(layer_pairs[:count]):
        if student_name not in student_features:
            raise KeyError(f"Student feature hook {student_name!r} is missing")
        weights_s, support_s = vision_feature_measure(student_features[student_name])
        if weights_s.shape[0] != len(ids):
            raise ValueError("sample ID count does not match student feature batch")
        per_sample = []
        for row, sample_id in enumerate(ids):
            record = cache.get(sample_id)["layers"][teacher_name]
            per_sample.append(_single_swat(weights_s[row], support_s[row], record,
                                            projections, seed + layer_pos))
        layer_loss = torch.stack(per_sample).mean()
        losses.append(layer_loss)
        diagnostics[f"{teacher_name}_to_{student_name}"] = float(layer_loss.detach().cpu())
    return torch.stack(losses).mean(), diagnostics


def queried_vision_final_swat_loss(
    student_final_features: torch.Tensor,
    teacher_final_features: torch.Tensor,
    *,
    projections: int = 16,
    seed: int = 991,
) -> torch.Tensor:
    weights_s, support_s = vision_feature_measure(student_final_features)
    weights_t, support_t = vision_feature_measure(teacher_final_features)
    return swat_loss(weights_s, support_s, weights_t, support_t,
                     n_projections=int(projections), seed=int(seed))


# Vision-only vectorized implementation. Original reference functions remain
# above for loss/gradient equivalence testing and reproducibility.
from .vision_fast import (
    cached_vision_swat_loss_fast as cached_vision_swat_loss,
    queried_vision_final_swat_loss_fast as queried_vision_final_swat_loss,
)
