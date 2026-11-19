#!/usr/bin/env python3
"""Verify every configured model and dataset using local files only."""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


DATASET_SCHEMA = {
    "SST-2": {
        "train": {"sentence", "label"},
        "validation": {"sentence", "label"},
    },
    "IMDb": {
        "train": {"text", "label"},
        "test": {"text", "label"},
    },
    "DBPedia": {
        "train": {"title", "content", "label"},
        "test": {"title", "content", "label"},
    },
    "WikiText-2": {
        "train": {"text"},
        "validation": {"text"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="trillium_experiments.json")
    parser.add_argument(
        "--report",
        help="Output JSON path; defaults to local_assets_verified.json beside the config.",
    )
    return parser.parse_args()


def expand_path(value: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(value))).resolve()


def require_mapping(config: Dict[str, Any], key: str) -> Dict[str, str]:
    value = config.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{key} must be a non-empty JSON object.")
    return {str(name): str(path) for name, path in value.items()}


def validate_dataset_schema(canonical_name: str, dataset: Any) -> None:
    if canonical_name not in DATASET_SCHEMA:
        raise ValueError(f"No dataset schema specification for {canonical_name!r}.")
    for split_name, required_columns in DATASET_SCHEMA[canonical_name].items():
        if split_name not in dataset:
            raise ValueError(
                f"Dataset {canonical_name} is missing required split {split_name!r}."
            )
        missing_columns = required_columns - set(dataset[split_name].column_names)
        if missing_columns:
            raise ValueError(
                f"Dataset {canonical_name} split {split_name!r} is missing "
                f"column(s): {', '.join(sorted(missing_columns))}."
            )


def main() -> None:
    args = parse_args()
    config_path = expand_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    # These are set before importing Hugging Face libraries. Model calls also
    # use local_files_only=True, while datasets are loaded with load_from_disk.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from datasets import DatasetDict, load_from_disk
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    report: Dict[str, Any] = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "offline_mode": True,
        "models": {},
        "datasets": {},
    }

    for canonical_name, raw_path in require_mapping(config, "model_paths").items():
        path = expand_path(raw_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Model directory is missing for {canonical_name}: {path}")
        weight_files = sorted(
            {
                file.name
                for pattern in ("*.safetensors", "pytorch_model*.bin")
                for file in path.glob(pattern)
            }
        )
        if not weight_files:
            raise FileNotFoundError(f"No PyTorch or safetensors weights found in {path}")

        model_config = AutoConfig.from_pretrained(path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=True,
            local_files_only=True,
        )

        # Deserialize the same task-head classes used by GCBD-NLP_RUN.py. A
        # file-presence check alone cannot detect corrupt or incompatible
        # checkpoint shards. Models are loaded sequentially and released so
        # peak host memory is bounded by the largest individual model.
        classification_model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=2,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        classification_report = {
            "class": classification_model.__class__.__name__,
            "parameters": int(classification_model.num_parameters()),
        }
        del classification_model
        gc.collect()

        lm_objective = "causal" if "gpt2" in canonical_name.lower() else "masked"
        lm_model_class = (
            AutoModelForCausalLM if lm_objective == "causal" else AutoModelForMaskedLM
        )
        lm_model = lm_model_class.from_pretrained(
            path,
            local_files_only=True,
        )
        lm_report = {
            "class": lm_model.__class__.__name__,
            "objective": lm_objective,
            "parameters": int(lm_model.num_parameters()),
        }
        del lm_model
        gc.collect()

        report["models"][canonical_name] = {
            "path": str(path),
            "model_type": model_config.model_type,
            "tokenizer_class": tokenizer.__class__.__name__,
            "vocabulary_size": len(tokenizer),
            "weight_files": weight_files,
            "verified_loads": {
                "sequence_classification": classification_report,
                "language_modeling": lm_report,
            },
        }
        print(
            f"MODEL OK | {canonical_name} | {model_config.model_type} | "
            f"vocab={len(tokenizer)} | classification={classification_report['class']} | "
            f"lm={lm_report['class']} | {path}",
            flush=True,
        )

    for canonical_name, raw_path in require_mapping(config, "dataset_paths").items():
        path = expand_path(raw_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory is missing for {canonical_name}: {path}")
        dataset = load_from_disk(path)
        if not isinstance(dataset, DatasetDict):
            raise TypeError(
                f"Expected DatasetDict for {canonical_name}, found {type(dataset).__name__}."
            )
        validate_dataset_schema(canonical_name, dataset)
        split_report = {
            split_name: {
                "rows": len(split),
                "columns": split.column_names,
                "fingerprint": split._fingerprint,
            }
            for split_name, split in dataset.items()
        }
        report["datasets"][canonical_name] = {
            "path": str(path),
            "splits": split_report,
        }
        sizes = {name: value["rows"] for name, value in split_report.items()}
        print(f"DATASET OK | {canonical_name} | {sizes} | {path}", flush=True)

    report_path = (
        expand_path(args.report)
        if args.report
        else config_path.parent / "local_assets_verified.json"
    )
    temporary = report_path.with_suffix(report_path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, report_path)
    print(f"LOCAL ASSET VERIFICATION PASSED | report={report_path}")


if __name__ == "__main__":
    main()
