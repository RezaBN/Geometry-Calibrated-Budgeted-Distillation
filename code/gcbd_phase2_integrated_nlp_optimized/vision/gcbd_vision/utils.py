from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def atomic_json(path: str | os.PathLike[str], payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(target)


def atomic_text(path: str | os.PathLike[str], text: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(target)


def sha256_file(path: str | os.PathLike[str], block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int, deterministic: bool = False) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def dataloader_generator(seed: int):
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def normalize_state_dict(state):
    for prefix in ("module.model.", "module.", "model."):
        if state and all(key.startswith(prefix) for key in state):
            return {key[len(prefix) :]: value for key, value in state.items()}
    return state


def torch_load(path: str | os.PathLike[str], map_location: Any = "cpu") -> Any:
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def slurm_metadata() -> dict[str, str | None]:
    keys = ["SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURMD_NODENAME", "HOSTNAME"]
    return {key.lower(): os.environ.get(key) for key in keys}
