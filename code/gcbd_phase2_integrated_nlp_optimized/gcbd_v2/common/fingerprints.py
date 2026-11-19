"""Deterministic provenance fingerprints for data, models and preprocessing."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    p = Path(path).resolve()
    stat = p.stat()
    return {"path": str(p), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(p)}
