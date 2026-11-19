#!/usr/bin/env python3
"""Build ordered semantic class prototypes from a local frozen text encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from gcbd_v2.common.class_embeddings import encode_class_texts
from gcbd_v2.common.fingerprints import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSON with ordered class_names and class_texts arrays")
    parser.add_argument("--encoder", required=True, help="Local frozen encoder checkpoint directory")
    parser.add_argument("--output", required=True, help="Output .pt path")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    class_names = manifest.get("class_names")
    class_texts = manifest.get("class_texts")
    if not isinstance(class_names, list) or not isinstance(class_texts, list):
        raise ValueError("Manifest must contain class_names and class_texts arrays")
    if len(class_names) < 2 or len(class_names) != len(class_texts):
        raise ValueError("class_names and class_texts must have equal length >= 2")
    if len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be unique and preserve classifier output order")

    embeddings, encoder_metadata = encode_class_texts(
        class_texts,
        args.encoder,
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=True,
    )
    payload = {
        "class_names": list(class_names),
        "class_texts": list(class_texts),
        "embeddings": embeddings.float().cpu(),
        "metadata": {
            **encoder_metadata,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "class_names_sha256": hashlib.sha256(
                json.dumps(class_names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(json.dumps({
        "output": str(output.resolve()),
        "class_count": len(class_names),
        "embedding_shape": list(embeddings.shape),
        "metadata": payload["metadata"],
    }, indent=2))


if __name__ == "__main__":
    main()
