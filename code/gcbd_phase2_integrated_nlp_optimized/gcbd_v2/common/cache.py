"""Provenance-locked, atomic, sharded geometry cache with lazy indexed reads."""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import torch

from .fingerprints import canonical_hash, sha256_file

SCHEMA_VERSION = 2


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _validate_tree(value: Any, key: str = "") -> None:
    if torch.is_tensor(value):
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite tensor stored under key {key!r}")
        if key in {"weights", "energy_weights"}:
            x = value.float()
            if (x < -1e-7).any():
                raise ValueError("Energy weights must be nonnegative")
            if x.ndim >= 1 and not torch.allclose(
                x.sum(dim=-1), torch.ones_like(x.sum(dim=-1)), atol=2e-3, rtol=2e-3
            ):
                raise ValueError("Energy weights must sum to one along the last dimension")
        if key in {"support", "coordinates"} and value.numel():
            x = value.float()
            if (x < -1e-5).any() or (x > 1.00001).any():
                raise ValueError("Support coordinates must be normalized to [0,1]")
    elif isinstance(value, dict):
        for k, v in value.items():
            _validate_tree(v, str(k))
    elif isinstance(value, (list, tuple)):
        for v in value:
            _validate_tree(v, key)


class GeometryCacheWriter:
    """Single logical writer; manifest updates are protected against concurrent jobs."""
    def __init__(self, root: str | Path, metadata: dict[str, Any]):
        self.root = Path(root)
        self.metadata = dict(metadata)
        self.metadata["schema_version"] = SCHEMA_VERSION
        self.identity = canonical_hash(self.metadata)
        self.manifest_path = self.root / "manifest.json"
        self.lock_path = self.root / ".manifest.lock"
        self.root.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.manifest_path.exists():
                manifest = json.loads(self.manifest_path.read_text())
                if manifest.get("identity") != self.identity:
                    raise RuntimeError("Cache provenance mismatch; refusing stale cache reuse")
            else:
                _atomic_json(self.manifest_path, {
                    "schema_version": SCHEMA_VERSION,
                    "identity": self.identity,
                    "metadata": self.metadata,
                    "created_unix": time.time(),
                    "completed_shards": {},
                    "complete": False,
                })
            fcntl.flock(lock, fcntl.LOCK_UN)

    def write_shard(self, shard_id: int, sample_ids: Iterable[int], records: list[dict[str, Any]]) -> None:
        self.initialize()
        ids = [int(x) for x in sample_ids]
        if not ids or len(ids) != len(records):
            raise ValueError("Each nonempty shard must have one record per sample ID")
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate sample IDs within shard")
        _validate_tree(records)
        path = self.root / f"shard_{int(shard_id):06d}.pt"
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=self.root)
        os.close(fd)
        try:
            # Image spatial supports are fixed within a layer and batch. Store
            # identical support grids once per shard; NLP character supports
            # vary by sample and therefore remain inline.
            shared_supports: dict[str, torch.Tensor] = {}
            if records and all(isinstance(record, dict) and isinstance(record.get("layers"), dict) for record in records):
                for layer_name in records[0]["layers"]:
                    values = [record["layers"].get(layer_name, {}).get("support") for record in records]
                    if values and all(torch.is_tensor(value) for value in values):
                        reference = values[0]
                        if all(value.shape == reference.shape and torch.equal(value, reference) for value in values[1:]):
                            shared_supports[layer_name] = reference
            packed_records = []
            for record in records:
                packed = dict(record)
                if shared_supports and isinstance(record.get("layers"), dict):
                    packed_layers = {}
                    for layer_name, layer_record in record["layers"].items():
                        layer_packed = dict(layer_record)
                        if layer_name in shared_supports:
                            layer_packed.pop("support", None)
                            layer_packed["support_ref"] = layer_name
                        packed_layers[layer_name] = layer_packed
                    packed["layers"] = packed_layers
                packed_records.append(packed)
            torch.save({
                "identity": self.identity,
                "sample_ids": torch.tensor(ids, dtype=torch.int64),
                "records": packed_records,
                "shared_supports": shared_supports,
            }, tmp_name)
            with open(tmp_name, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        digest = sha256_file(path)
        info = {"file": path.name, "sha256": digest, "samples": len(ids),
                "bytes": path.stat().st_size, "first_id": min(ids), "last_id": max(ids)}
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            manifest = json.loads(self.manifest_path.read_text())
            if manifest.get("identity") != self.identity or manifest.get("complete"):
                raise RuntimeError("Cache identity changed or cache already finalized")
            key = str(int(shard_id))
            if key in manifest["completed_shards"]:
                raise RuntimeError(f"Shard {shard_id} already exists; refusing overwrite")
            manifest["completed_shards"][key] = info
            _atomic_json(self.manifest_path, manifest)
            fcntl.flock(lock, fcntl.LOCK_UN)

    def finalize(self, expected_sample_ids: Iterable[int]) -> None:
        expected = set(int(x) for x in expected_sample_ids)
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("identity") != self.identity:
            raise RuntimeError("Cache identity mismatch during finalize")
        seen: set[int] = set()
        total_bytes = 0
        for shard_key, info in sorted(manifest["completed_shards"].items(), key=lambda x: int(x[0])):
            path = self.root / info["file"]
            if sha256_file(path) != info["sha256"]:
                raise RuntimeError(f"Checksum failure in {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("identity") != self.identity:
                raise RuntimeError(f"Identity mismatch in {path}")
            ids = [int(x) for x in payload["sample_ids"].tolist()]
            if len(ids) != len(payload["records"]):
                raise RuntimeError(f"Record count mismatch in {path}")
            if seen.intersection(ids):
                raise RuntimeError("Duplicate sample IDs across cache shards")
            seen.update(ids)
            _validate_tree(payload["records"])
            total_bytes += int(info["bytes"])
        if seen != expected:
            raise RuntimeError(f"Cache coverage mismatch: missing={len(expected-seen)}, extra={len(seen-expected)}")
        # Persist a compact sample -> (shard,row) index once, rather than
        # rescanning every tensor shard at every training-process startup.
        entries = []
        for shard_key, info in sorted(manifest["completed_shards"].items(), key=lambda x: int(x[0])):
            payload = torch.load(self.root / info["file"], map_location="cpu", weights_only=False)
            for row, sample_id in enumerate(payload["sample_ids"].tolist()):
                entries.append((int(sample_id), int(shard_key), int(row)))
        entries.sort(key=lambda item: item[0])
        index_path = self.root / "sample_index.pt"
        fd, tmp_name = tempfile.mkstemp(prefix=index_path.name + ".", suffix=".tmp", dir=self.root)
        os.close(fd)
        try:
            torch.save({
                "sample_ids": torch.tensor([x[0] for x in entries], dtype=torch.int64),
                "shard_ids": torch.tensor([x[1] for x in entries], dtype=torch.int32),
                "rows": torch.tensor([x[2] for x in entries], dtype=torch.int32),
            }, tmp_name)
            os.replace(tmp_name, index_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        manifest.update({"complete": True, "sample_count": len(seen), "total_bytes": total_bytes,
                         "index_file": index_path.name, "index_sha256": sha256_file(index_path),
                         "finalized_unix": time.time()})
        _atomic_json(self.manifest_path, manifest)


class GeometryCacheReader:
    """Lazy sample-ID lookup; shards are checksum-verified and LRU-cached."""
    def __init__(self, root: str | Path, metadata: dict[str, Any] | None = None, max_open_shards: int = 4):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        if not self.manifest.get("complete"):
            raise RuntimeError("Geometry cache is not finalized")
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("Unsupported geometry cache schema")
        if metadata is not None and canonical_hash({**metadata, "schema_version": SCHEMA_VERSION}) != self.manifest.get("identity"):
            raise RuntimeError("Requested metadata does not match cache identity")
        self.max_open_shards = max(1, int(max_open_shards))
        self._location_by_sample: dict[int, tuple[int, int]] = {}
        self._lru: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.io_metrics = {
            "shard_loads": 0,
            "shard_cache_hits": 0,
            "bytes_read": 0,
            "read_seconds": 0.0,
            "index_bytes_read": 0,
            "index_read_seconds": 0.0,
        }
        index_path = self.root / str(self.manifest.get("index_file", "sample_index.pt"))
        index_started = time.perf_counter()
        if not index_path.is_file() or sha256_file(index_path) != self.manifest.get("index_sha256"):
            raise RuntimeError("Geometry cache sample index is missing or corrupt")
        index = torch.load(index_path, map_location="cpu", weights_only=False)
        self.io_metrics["index_bytes_read"] = int(index_path.stat().st_size)
        self.io_metrics["index_read_seconds"] = time.perf_counter() - index_started
        sample_ids = index["sample_ids"].tolist()
        shard_ids = index["shard_ids"].tolist()
        rows = index["rows"].tolist()
        if not (len(sample_ids) == len(shard_ids) == len(rows) == int(self.manifest["sample_count"])):
            raise RuntimeError("Sample index length does not match manifest sample_count")
        for sample_id, shard_id, row in zip(sample_ids, shard_ids, rows):
            sample_id, shard_id, row = int(sample_id), int(shard_id), int(row)
            if sample_id in self._location_by_sample:
                raise RuntimeError(f"Duplicate cached sample ID {sample_id}")
            if str(shard_id) not in self.manifest["completed_shards"]:
                raise RuntimeError(f"Sample index refers to unknown shard {shard_id}")
            self._location_by_sample[sample_id] = (shard_id, row)

    def _load(self, shard_id: int) -> dict[str, Any]:
        if shard_id in self._lru:
            self.io_metrics["shard_cache_hits"] += 1
            self._lru.move_to_end(shard_id)
            return self._lru[shard_id]
        started = time.perf_counter()
        info = self.manifest["completed_shards"][str(shard_id)]
        path = self.root / info["file"]
        if sha256_file(path) != info["sha256"]:
            raise RuntimeError(f"Checksum failure in {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.io_metrics["shard_loads"] += 1
        self.io_metrics["bytes_read"] += int(path.stat().st_size)
        self.io_metrics["read_seconds"] += time.perf_counter() - started
        self._lru[shard_id] = payload
        self._lru.move_to_end(shard_id)
        while len(self._lru) > self.max_open_shards:
            self._lru.popitem(last=False)
        return payload

    def get(self, sample_id: int) -> dict[str, Any]:
        sample_id = int(sample_id)
        if sample_id not in self._location_by_sample:
            raise KeyError(f"Sample ID {sample_id} is absent from geometry cache")
        shard_id, row = self._location_by_sample[sample_id]
        payload = self._load(shard_id)
        if row >= len(payload["records"]) or int(payload["sample_ids"][row]) != sample_id:
            raise RuntimeError("Sample index points to an inconsistent shard record")
        record = payload["records"][row]
        shared_supports = payload.get("shared_supports", {})
        if shared_supports and isinstance(record, dict) and isinstance(record.get("layers"), dict):
            materialized = dict(record)
            materialized_layers = {}
            for layer_name, layer_record in record["layers"].items():
                if "support_ref" in layer_record:
                    support_name = layer_record["support_ref"]
                    if support_name not in shared_supports:
                        raise RuntimeError(f"Missing shared support {support_name!r} in cache shard {shard_id}")
                    materialized_layers[layer_name] = {**layer_record, "support": shared_supports[support_name]}
                else:
                    materialized_layers[layer_name] = layer_record
            materialized["layers"] = materialized_layers
            return materialized
        return record

    def get_many(self, sample_ids: Iterable[int]) -> list[dict[str, Any]]:
        """Fetch a minibatch in shard groups, preserving requested sample order.

        Loading each shard once per call avoids pathological LRU churn when a
        shuffled minibatch spans many disk shards. Checksum verification and
        shared-support reconstruction are unchanged from :meth:`get`.
        """
        ids = [int(x) for x in sample_ids]
        groups: dict[int, list[tuple[int, int, int]]] = {}
        for position, sample_id in enumerate(ids):
            if sample_id not in self._location_by_sample:
                raise KeyError(f"Sample ID {sample_id} is absent from geometry cache")
            shard_id, row = self._location_by_sample[sample_id]
            groups.setdefault(shard_id, []).append((position, sample_id, row))
        results: list[dict[str, Any] | None] = [None] * len(ids)
        for shard_id, items in groups.items():
            payload = self._load(shard_id)
            shared_supports = payload.get("shared_supports", {})
            for position, sample_id, row in items:
                if row >= len(payload["records"]) or int(payload["sample_ids"][row]) != sample_id:
                    raise RuntimeError("Sample index points to an inconsistent shard record")
                record = payload["records"][row]
                if shared_supports and isinstance(record, dict) and isinstance(record.get("layers"), dict):
                    materialized_layers = {}
                    for layer_name, layer_record in record["layers"].items():
                        if "support_ref" in layer_record:
                            ref = layer_record["support_ref"]
                            if ref not in shared_supports:
                                raise RuntimeError(f"Missing shared support {ref!r} in cache shard {shard_id}")
                            materialized_layers[layer_name] = {**layer_record, "support": shared_supports[ref]}
                        else:
                            materialized_layers[layer_name] = layer_record
                    record = {**record, "layers": materialized_layers}
                results[position] = record
        return results
