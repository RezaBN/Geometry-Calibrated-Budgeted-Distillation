"""Framework-neutral one-pass teacher geometry extraction loop."""
from __future__ import annotations
import time
from typing import Callable, Iterable
import torch

from .accounting import TeacherLedger
from .cache import GeometryCacheWriter


def extract_geometry_cache(
    teacher: torch.nn.Module,
    dataloader,
    writer: GeometryCacheWriter,
    expected_sample_ids: Iterable[int],
    device: torch.device,
    batch_to_teacher_inputs: Callable,
    outputs_to_records: Callable,
    *,
    autocast_context: Callable | None = None,
    max_batches: int | None = None,
) -> TeacherLedger:
    """Run exactly one teacher forward per loader batch and finalize cache.

    The caller owns deterministic preprocessing and stable sample IDs. This
    function intentionally refuses to infer IDs from batch order.
    """
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    ledger = TeacherLedger()
    expected = [int(x) for x in expected_sample_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("expected_sample_ids contains duplicates")
    start = time.perf_counter()
    writer.initialize()
    with torch.inference_mode():
        iterator = iter(dataloader)
        shard_id = 0
        while True:
            wait_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            ledger.dataloader_wait_seconds += time.perf_counter() - wait_start
            if max_batches is not None and shard_id >= max_batches:
                raise RuntimeError("max_batches interrupted extraction; cache will not be finalized")
            if isinstance(batch, dict):
                sample_ids = batch.get("sample_id")
            elif isinstance(batch, (tuple, list)) and len(batch) >= 3:
                sample_ids = batch[2]
            else:
                sample_ids = None
            if sample_ids is None:
                raise KeyError("Every extraction batch must carry stable sample_id values")
            sample_ids = [int(x) for x in torch.as_tensor(sample_ids).tolist()]
            inputs = batch_to_teacher_inputs(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            teacher_start = time.perf_counter()
            if autocast_context is None:
                outputs = teacher(**inputs) if isinstance(inputs, dict) else teacher(inputs)
            else:
                with autocast_context():
                    outputs = teacher(**inputs) if isinstance(inputs, dict) else teacher(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            teacher_seconds = time.perf_counter() - teacher_start
            records = outputs_to_records(outputs, batch, device)
            if len(records) != len(sample_ids):
                raise RuntimeError("Geometry record count does not match sample IDs")
            write_start = time.perf_counter()
            writer.write_shard(shard_id, sample_ids, records)
            ledger.cache_write_seconds += time.perf_counter() - write_start
            ledger.record_geometry(len(sample_ids), 1, teacher_seconds)
            ledger.cache_bytes += sum(p.stat().st_size for p in writer.root.glob(f"shard_{shard_id:06d}.pt"))
            shard_id += 1
    write_start = time.perf_counter()
    writer.finalize(expected)
    ledger.cache_write_seconds += time.perf_counter() - write_start
    index_path = writer.root / "sample_index.pt"
    if index_path.is_file():
        ledger.cache_bytes += int(index_path.stat().st_size)
    ledger.total_wall_seconds = time.perf_counter() - start
    return ledger
