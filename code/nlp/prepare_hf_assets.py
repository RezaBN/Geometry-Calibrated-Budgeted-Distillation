#!/usr/bin/env python3
"""Prefetch all selected Hugging Face models and datasets for offline jobs."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple


SCENARIOS: Dict[str, Tuple[str, str]] = {
    "Setup_1_BERT_TinyBERT": ("bert-base-uncased", "google/bert_uncased_L-4_H-256_A-4"),
    "Setup_2_DeBERTa_DistilBERT": ("microsoft/deberta-v3-base", "distilbert-base-uncased"),
    "Setup_3_GPT2_DistilGPT2": ("gpt2", "distilgpt2"),
    "Setup_4_BERT_DistilGPT2": ("bert-base-uncased", "distilgpt2"),
    "Setup_5_GPT2_DistilBERT": ("gpt2", "distilbert-base-uncased"),
}
ALL_DATASETS = ["SST-2", "IMDb", "DBPedia", "WikiText-2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="trillium_experiments.json")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--datasets-only", action="store_true")
    return parser.parse_args()


def expand_path(value: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(value))).resolve()


def prefetch_dataset(load_dataset, name: str):
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
    raise ValueError(f"Unknown dataset: {name}")


def main() -> None:
    args = parse_args()
    if args.models_only and args.datasets_only:
        raise ValueError("Choose at most one of --models-only and --datasets-only.")
    config_path = expand_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    prefetch_config = config.get("prefetch", {})
    if not isinstance(prefetch_config, dict):
        raise ValueError("prefetch must be a JSON object.")
    selected_scenarios = prefetch_config.get("scenarios", config["scenarios"])
    selected_datasets = prefetch_config.get("datasets", config["datasets"])
    unknown_scenarios = sorted(set(selected_scenarios) - set(SCENARIOS))
    unknown_datasets = sorted(set(selected_datasets) - set(ALL_DATASETS))
    if unknown_scenarios:
        raise ValueError(f"Unknown prefetch scenario(s): {', '.join(unknown_scenarios)}")
    if unknown_datasets:
        raise ValueError(f"Unknown prefetch dataset(s): {', '.join(unknown_datasets)}")

    hf_home = expand_path(config["hf_home"])
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    # Import after setting cache variables so every library resolves the same cache.
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    completed = {"models": [], "datasets": []}
    if not args.datasets_only:
        selected_models = sorted(
            {
                model
                for scenario in selected_scenarios
                for model in SCENARIOS[scenario]
            }
        )
        for model in selected_models:
            print(f"Prefetching model: {model}", flush=True)
            snapshot_download(
                repo_id=model,
                repo_type="model",
                ignore_patterns=[
                    "*.h5",
                    "*.msgpack",
                    "*.onnx",
                    "*.tflite",
                    "tf_model*",
                    "flax_model*",
                ],
            )
            completed["models"].append(model)

    if not args.models_only:
        for dataset in selected_datasets:
            print(f"Prefetching dataset: {dataset}", flush=True)
            prefetch_dataset(load_dataset, dataset)
            completed["datasets"].append(dataset)

    completed["completed_utc"] = datetime.now(timezone.utc).isoformat()
    marker = hf_home / "prefetch_complete.json"
    with marker.open("w", encoding="utf-8") as handle:
        json.dump(completed, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Offline assets ready in {hf_home}")


if __name__ == "__main__":
    main()
