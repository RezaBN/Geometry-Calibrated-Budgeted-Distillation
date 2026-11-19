"""Frozen text-encoder class prototypes for semantic transport costs."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import torch
import torch.nn.functional as F


def encode_class_texts(
    class_texts: list[str],
    encoder_path: str,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    local_files_only: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Mean-pool a frozen AutoModel encoder and return normalized [K,D] vectors.

    `encoder_path` must point to a staged local checkpoint. This helper never
    downloads weights when local_files_only=True (the default).
    """
    if not class_texts or any(not isinstance(x, str) or not x.strip() for x in class_texts):
        raise ValueError("class_texts must be a nonempty list of nonblank strings")
    if len(set(class_texts)) != len(class_texts):
        raise ValueError("class_texts must be unique and ordered exactly like the classifier labels")
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required to encode semantic class prototypes") from exc
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=local_files_only, use_fast=True)
    model = AutoModel.from_pretrained(encoder_path, local_files_only=local_files_only)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(class_texts), batch_size):
            texts = class_texts[start:start + batch_size]
            batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            hidden = model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            chunks.append(F.normalize(pooled, dim=-1).cpu())
    embeddings = torch.cat(chunks, dim=0)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(class_texts) or not torch.isfinite(embeddings).all():
        raise RuntimeError("Class embedding generation returned invalid values")
    payload = json.dumps(class_texts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    metadata = {
        "encoder_path": str(Path(encoder_path).resolve()),
        "encoder_config_name": getattr(model.config, "_name_or_path", None),
        "embedding_dimension": int(embeddings.shape[1]),
        "class_count": int(embeddings.shape[0]),
        "class_texts_sha256": hashlib.sha256(payload).hexdigest(),
        "pooling": "attention_masked_mean_of_last_hidden_state_then_l2_normalize",
    }
    return embeddings, metadata


def save_class_embeddings(path: str | Path, embeddings: torch.Tensor, class_texts: list[str], metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if embeddings.shape[0] != len(class_texts):
        raise ValueError("embedding count and class text count differ")
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"embeddings": embeddings.detach().float().cpu(), "class_texts": list(class_texts),
                "metadata": dict(metadata)}, tmp)
    tmp.replace(path)
