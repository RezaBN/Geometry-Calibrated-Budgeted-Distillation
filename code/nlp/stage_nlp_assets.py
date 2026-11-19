#!/usr/bin/env python3
"""Stage GCBD NLP Hugging Face models and DatasetDicts into configured RBN paths.

Run from an internet-enabled Trillium login/Open OnDemand session after loading
StdEnv/2023, gcc/12.3, arrow/25.0.0, cuda/12.6, python/3.11.5 and activating
the existing NLP venv. This script writes explicit local directories consumed
by GCBD-NLP_RUN.py and verify_local_assets.py; it does not merely fill HF cache.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

MODEL_IDS = {
    "bert-base-uncased",
    "microsoft/deberta-v3-base",
    "gpt2",
    "google/bert_uncased_L-4_H-256_A-4",
    "distilbert-base-uncased",
    "distilgpt2",
}


def expand(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def dataset_loader(load_dataset, name: str):
    if name == "SST-2":
        return load_dataset("glue", "sst2")
    if name == "IMDb":
        return load_dataset("imdb")
    if name == "DBPedia":
        try:
            return load_dataset("fancyzhx/dbpedia_14")
        except Exception:
            return load_dataset("dbpedia_14")
    if name == "WikiText-2":
        return load_dataset("wikitext", "wikitext-2-raw-v1")
    raise ValueError(f"Unsupported dataset: {name}")


def has_model_payload(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def prepare_destination(destination: Path, overwrite: bool) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {destination}. Inspect it, or rerun with --overwrite."
            )
        backup = destination.with_name(destination.name + f".backup-{int(time.time())}")
        destination.rename(backup)
        print(f"Existing destination moved to: {backup}", flush=True)
    staging = destination.with_name(destination.name + f".staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    return staging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Active RBN trillium_experiments.json")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--datasets-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.models_only and args.datasets_only:
        parser.error("Use at most one of --models-only and --datasets-only")

    config_path = expand(args.config)
    cfg: dict[str, Any] = json.loads(config_path.read_text())
    hf_home = expand(cfg["hf_home"])
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model_paths = cfg.get("model_paths", {})
    dataset_paths = cfg.get("dataset_paths", {})
    selected_datasets = list(cfg.get("datasets", []))
    if not model_paths:
        raise ValueError("config.model_paths is empty")
    missing_models = sorted(set(model_paths) - MODEL_IDS)
    if missing_models:
        raise ValueError(f"No Hugging Face repository mapping for model(s): {missing_models}")
    missing_datasets = sorted(set(selected_datasets) - set(dataset_paths))
    if missing_datasets:
        raise ValueError(f"No dataset_paths mapping for active dataset(s): {missing_datasets}")
    unused_dataset_paths = sorted(set(dataset_paths) - set(selected_datasets))
    if unused_dataset_paths:
        raise ValueError(
            "dataset_paths contains inactive datasets that verify_local_assets.py will still "
            f"require: {unused_dataset_paths}. Trim dataset_paths to config.datasets before staging."
        )

    from huggingface_hub import snapshot_download
    from datasets import load_dataset

    if not args.datasets_only:
        for repo_id, raw_destination in model_paths.items():
            destination = expand(raw_destination)
            if destination.exists() and has_model_payload(destination) and not args.overwrite:
                print(f"MODEL REUSE | {repo_id} | {destination}", flush=True)
                continue
            staging = prepare_destination(destination, args.overwrite)
            try:
                print(f"MODEL DOWNLOAD | {repo_id} -> {destination}", flush=True)
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="model",
                    local_dir=str(staging),
                    ignore_patterns=[
                        "*.h5", "*.msgpack", "*.onnx", "*.tflite",
                        "tf_model*", "flax_model*",
                    ],
                )
                if not has_model_payload(staging):
                    raise RuntimeError(f"Downloaded model has no config/weights: {staging}")
                staging.rename(destination)
                print(f"MODEL READY | {repo_id} | {destination}", flush=True)
            except Exception:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)
                raise

    if not args.models_only:
        for name in selected_datasets:
            destination = expand(dataset_paths[name])
            if destination.exists() and (destination / "dataset_dict.json").is_file() and not args.overwrite:
                print(f"DATASET REUSE | {name} | {destination}", flush=True)
                continue
            staging = prepare_destination(destination, args.overwrite)
            try:
                print(f"DATASET DOWNLOAD | {name} -> {destination}", flush=True)
                dataset = dataset_loader(load_dataset, name)
                dataset.save_to_disk(str(staging))
                if not (staging / "dataset_dict.json").is_file():
                    raise RuntimeError(f"DatasetDict save marker missing: {staging}")
                staging.rename(destination)
                print(
                    "DATASET READY | " + name + " | " +
                    json.dumps({split: len(ds) for split, ds in dataset.items()}) +
                    f" | {destination}", flush=True
                )
            except Exception:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)
                raise

    marker = {
        "config": str(config_path),
        "models": {} if args.datasets_only else {k: str(expand(v)) for k, v in model_paths.items()},
        "datasets": {} if args.models_only else {k: str(expand(dataset_paths[k])) for k in selected_datasets},
        "completed_unix": time.time(),
    }
    marker_path = hf_home / "gcbd_explicit_assets_staged.json"
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    print(f"STAGING COMPLETE | manifest={marker_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
