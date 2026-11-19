"""Runner-facing provenance and cache lifecycle helpers."""
from __future__ import annotations

import hashlib
import json
import shutil
import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import torch

from gcbd_v2.common.cache import GeometryCacheReader, GeometryCacheWriter, SCHEMA_VERSION
from gcbd_v2.common.fingerprints import canonical_hash


def module_state_sha256(module: torch.nn.Module) -> str:
    """Hash a model's exact state tensors without serializing a checkpoint file."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def text_manifest_sha256(sample_ids: Iterable[int], texts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    ids, values = list(sample_ids), list(texts)
    if len(ids) != len(values) or len(ids) != len(set(int(x) for x in ids)):
        raise ValueError("text manifest requires aligned, unique sample IDs")
    for sample_id, value in zip(ids, values):
        digest.update(str(int(sample_id)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def vision_manifest_sha256(dataset: Any) -> str:
    """Hash ordered image paths/labels or CIFAR data bytes and labels."""
    digest = hashlib.sha256()
    if hasattr(dataset, "samples"):
        import os
        for path, label in dataset.samples:
            stat = os.stat(path)
            digest.update(str(path).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(int(label)).encode("ascii"))
            digest.update(b"\0")
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
            digest.update(b"\n")
    elif hasattr(dataset, "data") and hasattr(dataset, "targets"):
        data = dataset.data
        if hasattr(data, "tobytes"):
            digest.update(data.tobytes())
        else:
            digest.update(bytes(data))
        digest.update(json.dumps([int(x) for x in dataset.targets]).encode("ascii"))
    else:
        raise TypeError("Unsupported dataset type for deterministic vision manifest")
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer: Any) -> dict[str, Any]:
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError(
            "GCBD cached NLP geometry requires a fast tokenizer with character offsets; "
            f"received {type(tokenizer).__name__} (is_fast={getattr(tokenizer, 'is_fast', None)})"
        )
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", type(tokenizer).__name__)),
        "vocab_size": int(getattr(tokenizer, "vocab_size", len(tokenizer))),
        "model_max_length": int(getattr(tokenizer, "model_max_length", 0)),
        "special_tokens_map": dict(getattr(tokenizer, "special_tokens_map", {})),
        "padding_side": str(getattr(tokenizer, "padding_side", "right")),
        "truncation_side": str(getattr(tokenizer, "truncation_side", "right")),
    }


def cache_identity(metadata: dict[str, Any]) -> str:
    return canonical_hash({**metadata, "schema_version": SCHEMA_VERSION})


def cache_directory(base: str | Path, namespace: str, dataset: str, scenario: str,
                    seed: int, metadata: dict[str, Any]) -> Path:
    identity = cache_identity(metadata)
    return Path(base) / namespace / dataset / scenario / f"seed_{int(seed)}" / identity[:20]




@contextmanager
def cache_build_lock(root: str | Path):
    """Serialize builders for one provenance identity across concurrent jobs."""
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{root.name}.build.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def open_or_prepare_cache(root: str | Path, metadata: dict[str, Any]):
    """Return a validated reader on cache hit, or a fresh writer on cache miss.

    Incomplete same-identity caches are discarded and rebuilt. Complete caches
    are never overwritten. Provenance is also validated by the reader.
    """
    root = Path(root)
    manifest = root / "manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("complete"):
            return GeometryCacheReader(root, metadata), None, True
        shutil.rmtree(root)
    writer = GeometryCacheWriter(root, metadata)
    writer.initialize()
    return None, writer, False


def write_ledger_json(path: str | Path, ledger: Any, extra: dict[str, Any] | None = None) -> None:
    payload = ledger.as_dict() if hasattr(ledger, "as_dict") else dict(ledger)
    if extra:
        payload.update(extra)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
