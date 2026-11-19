# -*- coding: utf-8 -*-
"""
Geometry-Calibrated Budgeted Distillation (GCBD) for NLP
========================================================
Single-GPU NVIDIA H100 implementation for Trillium using Hugging Face
Accelerate + PyTorch.

Supported datasets:
  - SST-2       : sequence classification, binary sentiment
  - IMDb        : sequence classification, binary sentiment
  - DBPedia     : sequence classification, 14-way topic classification
  - WikiText-2  : language modeling; causal LM for GPT-style models, masked LM for BERT-style models

Supported teacher-student scenarios:
  Setup_1_BERT_TinyBERT       : bert-base-uncased -> google/bert_uncased_L-4_H-256_A-4
  Setup_2_DeBERTa_DistilBERT  : microsoft/deberta-v3-base -> distilbert-base-uncased
  Setup_3_GPT2_DistilGPT2     : gpt2 -> distilgpt2
  Setup_4_BERT_DistilGPT2     : bert-base-uncased -> distilgpt2
  Setup_5_GPT2_DistilBERT     : gpt2 -> distilbert-base-uncased

This entry point intentionally runs exactly one experiment per process:
  - teacher mode: one dataset x scenario x seed teacher preparation job;
  - method mode: one dataset x scenario x method x seed x budget job.

Use ``submit_trillium.py`` to expand an experiment matrix into independent
SLURM jobs and to attach method jobs to their teacher job with ``afterok``.

Notes:
  - For classification, all scenarios are supported via AutoModelForSequenceClassification.
  - For WikiText-2, token-logit KD is enabled only when teacher/student LM objectives and vocabularies are compatible.
    Cross-objective setups still train with supervised student loss and optional hidden-state geometry alignment.
  - UOT is applied to low-dimensional classification label distributions. For LM vocabulary distributions,
    the script uses budgeted token KL because full UOT over a 30k-50k vocabulary is computationally infeasible.
  - Baselines include Student, KD, FitNet, PKD, RKD, CRD, MiniLM, TinyBERT-style task distillation,
    DKD, and DIST. Some logits-only baselines automatically fall back to compatible token-KD on LM tasks.
"""

import argparse
import os
import json
import math
import time
import random
import warnings
from dataclasses import dataclass, asdict, fields
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import load_from_disk, DatasetDict
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    get_cosine_schedule_with_warmup,
)
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from gcbd_v2.integrations.nlp import (
    attach_stable_sample_ids, GCBDClassificationCollator, GCBDLMTextCollator,
    build_nlp_geometry_cache, cached_nlp_swat_loss, queried_nlp_final_swat_loss,
)
from gcbd_v2.integrations.transport import load_semantic_cost, microbatched_transport_loss
from gcbd_v2.adapters.nlp import layer_indices_by_depth
from gcbd_v2.common.selection import CalibrationBuffer, select_gcbd, cosine_frontier
from gcbd_v2.common.sampling import ShardAwareBatchSampler
from gcbd_v2.integrations.timing import TeacherForwardTimer

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ============================================================
# High-level experiment matrix
# ============================================================

SCENARIOS = {
    "Setup_1_BERT_TinyBERT": ("bert-base-uncased", "google/bert_uncased_L-4_H-256_A-4"),
    "Setup_2_DeBERTa_DistilBERT": ("microsoft/deberta-v3-base", "distilbert-base-uncased"),
    "Setup_3_GPT2_DistilGPT2": ("gpt2", "distilgpt2"),
    "Setup_4_BERT_DistilGPT2": ("bert-base-uncased", "distilgpt2"),
    "Setup_5_GPT2_DistilBERT": ("gpt2", "distilbert-base-uncased"),
}
# Baseline and SOTA-style distillation methods.
# For fast debugging, temporarily reduce this list to ["Student", "KD", "GCBD-Full"].
BASELINE_METHODS = [
    "Student",     # supervised student only
    "KD",          # Hinton soft-logit KD
    "FitNet",      # last hidden representation matching
    "PKD",         # Patient KD: multi-layer hidden-state matching
    "RKD",         # Relational KD: distance/angle geometry matching
    "CRD",         # Contrastive representation distillation
    "MiniLM",      # attention-relation distillation when attentions are compatible
    "TinyBERT",    # task KD + hidden + attention distillation
    "DKD",         # decoupled KD for classification logits
    "DIST",        # correlation-based logits distillation
    "GCBD-Full",
]

GCBD_METHODS = [
    "GCBD-Full",
    "GCBD-NoCurriculum",
    "GCBD-NoBudget",
    "GCBD-NoStage1",
    "GCBD-NoUOT",
]

# ============================================================
# Runtime / training config
# ============================================================

@dataclass
class NLPConfig:
    output_root: str = os.path.join(os.environ.get("SCRATCH", os.getcwd()), "gcbd-nlp", "outputs")
    teacher_cache_root: str = os.path.join(os.environ.get("SCRATCH", os.getcwd()), "gcbd-nlp", "models", "teachers_finetuned")
    geometry_cache_root: str = os.path.join(os.environ.get("SCRATCH", os.getcwd()), "gcbd-nlp", "geometry_cache")
    semantic_class_embeddings_path: Dict[str, str] = None
    semantic_class_names: Dict[str, List[str]] = None
    model_paths: Dict[str, str] = None
    dataset_paths: Dict[str, str] = None

    # Conservative single-H100 starting point. Sequence length is also a
    # scientific design choice; do not increase it only because memory exists.
    max_length_cls: int = 256
    max_length_lm: int = 256
    train_batch_size_cls: int = 32
    eval_batch_size_cls: int = 64
    train_batch_size_lm: int = 16
    eval_batch_size_lm: int = 32
    epochs_cls: int = 10
    epochs_lm: int = 3
    teacher_epochs_cls: int = 3
    teacher_epochs_lm: int = 3
    learning_rate_student: float = 2e-5
    learning_rate_teacher: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_accumulation_steps: int = 2
    mixed_precision: str = "bf16"      # H100-native; trainable weights remain FP32
    allow_tf32: bool = True

    # GCBD losses
    temperature: float = 4.0
    alpha: float = 1.0                  # prediction KD/UOT weight
    beta: float = 0.25                  # hidden geometry weight
    gamma_budget: float = 0.05          # soft budget penalty weight
    sinkhorn_eps: float = 0.05
    sinkhorn_iters: int = 20
    lm_max_kd_tokens_per_batch: int = 1024
    mlm_probability: float = 0.15

    # Budget / curriculum
    curriculum_start_ratio: float = 0.2
    rho_gate: Optional[float] = None  # Lock from pilot; do not silently guess.
    calibration_buffer_capacity: int = 8192
    calibration_buffer_min_values: int = 32
    progressive_layer_activation_fractions: List[float] = None
    gcbd_swd_projections: int = 8
    uot_microbatch_size: int = 8

    # Sampling caps. Set values to None for full datasets.
    # DBPedia is large; a cap is useful while testing the full 4x5 scenario matrix.
    max_train_samples: Dict[str, Optional[int]] = None
    max_eval_samples: Dict[str, Optional[int]] = None

    # Teacher handling
    train_teacher_if_missing: bool = True
    reuse_finetuned_teacher: bool = True
    require_finetuned_teacher: bool = False

    # DataLoader
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Evaluation reports. Arrays are bounded so diagnostics do not dominate an
    # H100 job's memory or walltime.
    generate_evaluation_plots: bool = True
    calibration_bins: int = 15
    report_max_examples: int = 2000
    report_tsne_max_points: int = 1000
    report_max_lm_tokens: int = 10000
    report_lm_confusion_top_tokens: int = 20
    report_max_weight_values: int = 1000000
    report_seed: int = 42

    def __post_init__(self):
        if self.model_paths is None:
            self.model_paths = {}
        if self.dataset_paths is None:
            self.dataset_paths = {}
        if self.semantic_class_embeddings_path is None:
            self.semantic_class_embeddings_path = {}
        if self.semantic_class_names is None:
            self.semantic_class_names = {}
        if self.progressive_layer_activation_fractions is None:
            self.progressive_layer_activation_fractions = [0.0, 0.33, 0.67]
        if self.max_train_samples is None:
            self.max_train_samples = {
                "SST-2": None,
                "IMDb": None,
                "DBPedia": 50000,
                "WikiText-2": 50000,
            }
        if self.max_eval_samples is None:
            self.max_eval_samples = {
                "SST-2": None,
                "IMDb": None,
                "DBPedia": 10000,
                "WikiText-2": 5000,
            }

CFG = NLPConfig()

ALL_DATASETS = ["SST-2", "IMDb", "DBPedia", "WikiText-2"]
ALL_METHODS = list(dict.fromkeys(BASELINE_METHODS + GCBD_METHODS))


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one GCBD-NLP teacher or method experiment on one Trillium H100."
    )
    parser.add_argument("--mode", choices=["teacher", "method"], required=True)
    parser.add_argument("--dataset", choices=ALL_DATASETS, required=True)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--method", choices=ALL_METHODS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=float, default=0.0)
    parser.add_argument("--config", help="Experiment JSON containing a top-level runtime object and paths.")
    parser.add_argument("--output-root")
    parser.add_argument("--teacher-cache-root")
    parser.add_argument("--epochs", type=int, help="Student epochs for the selected task type.")
    parser.add_argument("--teacher-epochs", type=int, help="Teacher epochs for the selected task type.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
        help="Override any NLPConfig field; repeat as needed (for example --set train_batch_size_cls=16).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Run even when a DONE marker exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the resolved run without loading data or models.")
    return parser.parse_args()


def _parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _expand_path(value: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(value)))


def apply_runtime_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve the JSON/CLI configuration and mutate the process-local CFG."""
    payload: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

    valid_fields = {field.name for field in fields(NLPConfig)}
    overrides = dict(payload.get("runtime", {}))
    for key in ("output_root", "teacher_cache_root", "geometry_cache_root", "model_paths", "dataset_paths", "semantic_class_embeddings_path", "semantic_class_names"):
        if key in payload:
            overrides[key] = payload[key]
    if args.output_root:
        overrides["output_root"] = args.output_root
    if args.teacher_cache_root:
        overrides["teacher_cache_root"] = args.teacher_cache_root

    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set value {item!r}; expected KEY=JSON_VALUE.")
        key, raw = item.split("=", 1)
        overrides[key] = _parse_override_value(raw)

    unknown = sorted(set(overrides) - valid_fields)
    if unknown:
        raise ValueError(f"Unknown runtime configuration field(s): {', '.join(unknown)}")
    for key, value in overrides.items():
        if key in {"output_root", "teacher_cache_root", "geometry_cache_root"}:
            value = _expand_path(str(value))
        elif key in {"model_paths", "dataset_paths", "semantic_class_embeddings_path"}:
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a JSON object mapping canonical names to local paths.")
            value = {str(name): _expand_path(str(path)) for name, path in value.items()}
        setattr(CFG, key, value)

    task_type = get_task_type(args.dataset)
    if args.epochs is not None:
        setattr(CFG, "epochs_cls" if task_type == "classification" else "epochs_lm", args.epochs)
    if args.teacher_epochs is not None:
        setattr(
            CFG,
            "teacher_epochs_cls" if task_type == "classification" else "teacher_epochs_lm",
            args.teacher_epochs,
        )

    # Restore default cap dictionaries when a JSON file explicitly uses null.
    CFG.__post_init__()
    validate_resolved_config(args)
    return payload


def validate_resolved_config(args: argparse.Namespace) -> None:
    if args.mode == "method" and not args.method:
        raise ValueError("--method is required in method mode.")
    if args.mode == "teacher" and args.method:
        raise ValueError("--method is not valid in teacher mode.")
    if not 0.0 <= args.budget <= 1.0:
        raise ValueError("--budget must be in [0, 1].")
    if args.mode == "method" and args.method and not args.method.startswith("GCBD") and args.budget != 0.0:
        raise ValueError("Non-GCBD baselines must use --budget 0.0.")

    positive_ints = [
        "max_length_cls",
        "max_length_lm",
        "train_batch_size_cls",
        "eval_batch_size_cls",
        "train_batch_size_lm",
        "eval_batch_size_lm",
        "epochs_cls",
        "epochs_lm",
        "teacher_epochs_cls",
        "teacher_epochs_lm",
        "gradient_accumulation_steps",
        "calibration_bins",
        "report_max_examples",
        "report_tsne_max_points",
        "report_max_lm_tokens",
        "report_lm_confusion_top_tokens",
        "report_max_weight_values",
    ]
    for name in positive_ints:
        if int(getattr(CFG, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    if CFG.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if CFG.mixed_precision not in {"bf16", "fp16", "no"}:
        raise ValueError("mixed_precision must be bf16, fp16, or no.")

    teacher_name, student_name = SCENARIOS[args.scenario]
    for model_name in (teacher_name, student_name):
        model_path = resolve_model_source(model_name)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Local model directory for {model_name!r} does not exist: {model_path}"
            )
    dataset_path = resolve_dataset_source(args.dataset)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Local dataset directory for {args.dataset!r} does not exist: {dataset_path}"
        )

    scratch = os.environ.get("SCRATCH")
    if scratch:
        scratch = _expand_path(scratch)
        for name in ("output_root", "teacher_cache_root", "geometry_cache_root"):
            path = _expand_path(str(getattr(CFG, name)))
            if os.path.commonpath([scratch, path]) != scratch:
                raise ValueError(f"{name} must be under $SCRATCH on Trillium; resolved to {path}")
        for label, paths in (("model_paths", CFG.model_paths), ("dataset_paths", CFG.dataset_paths),
                             ("semantic_class_embeddings_path", CFG.semantic_class_embeddings_path)):
            for canonical_name, path in paths.items():
                if os.path.commonpath([scratch, path]) != scratch:
                    raise ValueError(
                        f"{label}[{canonical_name!r}] must be under $SCRATCH; resolved to {path}"
                    )


def resolved_run_spec(args: argparse.Namespace) -> Dict[str, Any]:
    teacher_name, student_name = SCENARIOS[args.scenario]
    return {
        "mode": args.mode,
        "dataset": args.dataset,
        "scenario": args.scenario,
        "teacher": teacher_name,
        "teacher_source": resolve_model_source(teacher_name),
        "student": student_name,
        "student_source": resolve_model_source(student_name),
        "dataset_source": resolve_dataset_source(args.dataset),
        "method": args.method,
        "seed": args.seed,
        "budget": args.budget,
        "runtime": asdict(CFG),
    }

# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_json_atomic(path: str, value: Any) -> None:
    ensure_dir(os.path.dirname(path))
    temporary = f"{path}.tmp-{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_text_atomic(path: str, value: str) -> None:
    ensure_dir(os.path.dirname(path))
    temporary = f"{path}.tmp-{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(value)
    os.replace(temporary, path)


def sanitize_name(x: str) -> str:
    return x.replace("/", "__").replace(" ", "_").replace(":", "_")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_model_source(model_name: str) -> str:
    """Resolve a canonical Hugging Face model ID to a required local snapshot."""
    if model_name not in CFG.model_paths:
        raise KeyError(
            f"No local model path configured for {model_name!r}. "
            "Remote model resolution is disabled for Trillium jobs."
        )
    return _expand_path(CFG.model_paths[model_name])


def resolve_dataset_source(dataset_name: str) -> str:
    """Resolve a canonical experiment dataset name to a saved DatasetDict."""
    if dataset_name not in CFG.dataset_paths:
        raise KeyError(
            f"No local dataset path configured for {dataset_name!r}. "
            "Remote dataset resolution is disabled for Trillium jobs."
        )
    return _expand_path(CFG.dataset_paths[dataset_name])


def seed_dataloader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_dataloader(
    dataset, *, batch_size: int, shuffle: bool, collate_fn,
    block_shuffle: bool = False, seed: int = 0,
) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "collate_fn": collate_fn,
        "num_workers": CFG.num_workers,
        "pin_memory": bool(CFG.pin_memory and torch.cuda.is_available()),
        "worker_init_fn": seed_dataloader_worker,
    }
    if CFG.num_workers > 0:
        kwargs["persistent_workers"] = bool(CFG.persistent_workers)
        kwargs["prefetch_factor"] = int(CFG.prefetch_factor)
    if block_shuffle:
        kwargs["batch_sampler"] = ShardAwareBatchSampler(
            len(dataset), int(batch_size), seed=int(seed), drop_last=False
        )
    else:
        kwargs["batch_size"] = int(batch_size)
        kwargs["shuffle"] = bool(shuffle)
    return DataLoader(dataset, **kwargs)


def set_dataloader_epoch(dataloader: DataLoader, epoch: int) -> None:
    if hasattr(dataloader, "set_epoch"):
        dataloader.set_epoch(int(epoch))
        return
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(int(epoch))


def is_main(accelerator: Accelerator) -> bool:
    return accelerator.is_main_process


def main_print(accelerator: Accelerator, *args, **kwargs):
    if is_main(accelerator):
        print(*args, **kwargs)


def infer_lm_objective(model_name: str) -> str:
    low = model_name.lower()
    if "gpt2" in low:
        return "causal"
    return "masked"


def is_gpt_like(model_name: str) -> bool:
    return "gpt2" in model_name.lower()


def prepare_tokenizer(model_name_or_path: str):
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        local_files_only=True,
    )
    if tok.pad_token is None:
        # GPT-2-style models do not define a padding token by default.
        tok.pad_token = tok.eos_token
    return tok


def maybe_set_pad_token(model, tokenizer):
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


def unwrap_model_if_wrapped(model):
    """Return the underlying module for DDP/Accelerate-wrapped models."""
    return getattr(model, "module", model)


def force_trainable_parameters_fp32(model):
    """Keep trainable parameters in FP32 while still allowing AMP autocast.

    PyTorch GradScaler cannot unscale gradients attached to FP16 trainable
    parameters. Some Transformer checkpoints/configurations can materialize
    trainable weights as float16 under mixed-precision notebook states. This
    function makes the invariant explicit: trainable weights stay FP32; AMP
    controls activation/operation precision during forward passes.
    """
    base = unwrap_model_if_wrapped(model)
    for p in base.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            if p.grad is not None:
                p.grad = None
    return model


def trainable_dtype_summary(model) -> Dict[str, int]:
    base = unwrap_model_if_wrapped(model)
    out: Dict[str, int] = {}
    for p in base.parameters():
        if p.requires_grad:
            key = str(p.dtype).replace("torch.", "")
            out[key] = out.get(key, 0) + int(p.numel())
    return out


def has_fp16_trainable_params(model) -> bool:
    base = unwrap_model_if_wrapped(model)
    return any(p.requires_grad and p.dtype == torch.float16 for p in base.parameters())


def get_task_type(dataset_name: str) -> str:
    return "language_modeling" if dataset_name == "WikiText-2" else "classification"


def get_num_labels(dataset_name: str) -> int:
    if dataset_name in ["SST-2", "IMDb"]:
        return 2
    if dataset_name == "DBPedia":
        return 14
    raise ValueError(f"No num_labels for dataset={dataset_name}")

# ============================================================
# Dataset loading
# ============================================================

def load_standardized_dataset(dataset_name: str, seed: int) -> Tuple[DatasetDict, str]:
    """Return DatasetDict with standardized keys.

    Classification datasets contain: text, label
    WikiText-2 contains: text
    """
    local_path = resolve_dataset_source(dataset_name)
    loaded = load_from_disk(local_path)
    if not isinstance(loaded, DatasetDict):
        raise TypeError(f"Expected DatasetDict at {local_path}, found {type(loaded).__name__}.")
    train_cap = CFG.max_train_samples.get(dataset_name)
    eval_cap = CFG.max_eval_samples.get(dataset_name)
    caps_applied = False

    if dataset_name == "SST-2":
        raw = loaded
        raw = DatasetDict({"train": raw["train"], "validation": raw["validation"]})
        raw = raw.rename_column("sentence", "text")
        keep_cols = ["text", "label"]
        raw = raw.remove_columns([c for c in raw["train"].column_names if c not in keep_cols])
        eval_split = "validation"

    elif dataset_name == "IMDb":
        raw = loaded
        raw = DatasetDict({"train": raw["train"], "validation": raw["test"]})
        keep_cols = ["text", "label"]
        raw = raw.remove_columns([c for c in raw["train"].column_names if c not in keep_cols])
        eval_split = "validation"

    elif dataset_name == "DBPedia":
        raw = loaded
        raw = DatasetDict({"train": raw["train"], "validation": raw["test"]})
        # Apply the configured cap before materializing joined title/content
        # strings. This avoids transforming all 630k records for a capped run.
        if train_cap is not None and len(raw["train"]) > train_cap:
            raw["train"] = raw["train"].shuffle(seed=seed).select(range(train_cap))
        if eval_cap is not None and len(raw["validation"]) > eval_cap:
            raw["validation"] = raw["validation"].shuffle(seed=seed).select(range(eval_cap))
        caps_applied = True
        def join_text(ex):
            title = ex.get("title", "") or ""
            content = ex.get("content", "") or ""
            return {"text": (title + "\n" + content).strip(), "label": int(ex["label"])}
        raw = raw.map(
            join_text,
            remove_columns=raw["train"].column_names,
            keep_in_memory=True,
        )
        eval_split = "validation"

    elif dataset_name == "WikiText-2":
        raw = loaded
        raw = DatasetDict({"train": raw["train"], "validation": raw["validation"]})
        raw = raw.filter(
            lambda x: x["text"] is not None and len(x["text"].strip()) > 0,
            keep_in_memory=True,
        )
        eval_split = "validation"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if not caps_applied:
        if train_cap is not None and len(raw["train"]) > train_cap:
            raw["train"] = raw["train"].shuffle(seed=seed).select(range(train_cap))
        if eval_cap is not None and len(raw[eval_split]) > eval_cap:
            raw[eval_split] = raw[eval_split].shuffle(seed=seed).select(range(eval_cap))

    raw["train"] = attach_stable_sample_ids(raw["train"])
    raw[eval_split] = attach_stable_sample_ids(raw[eval_split])
    return raw, eval_split

# ============================================================
# Collators
# ============================================================

class DualClassificationCollator:
    def __init__(self, teacher_tokenizer, student_tokenizer, max_length: int):
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [f["text"] for f in features]
        labels = torch.tensor([int(f["label"]) for f in features], dtype=torch.long)
        tb = self.teacher_tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        sb = self.student_tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return {"teacher": tb, "student": sb, "labels": labels, "texts": texts}


class TeacherOnlyClassificationCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [f["text"] for f in features]
        labels = torch.tensor([int(f["label"]) for f in features], dtype=torch.long)
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        batch["labels"] = labels
        return batch


def torch_mask_tokens(inputs: torch.Tensor, tokenizer, mlm_probability: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Minimal BERT-style MLM masking, equivalent in spirit to DataCollatorForLanguageModeling."""
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True)
        for val in labels
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    return inputs, labels


class DualLMTextCollator:
    def __init__(
        self,
        teacher_tokenizer,
        student_tokenizer,
        teacher_objective: str,
        student_objective: str,
        max_length: int,
        share_tokenization: bool,
        mlm_probability: float,
    ):
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.teacher_objective = teacher_objective
        self.student_objective = student_objective
        self.max_length = max_length
        self.share_tokenization = share_tokenization
        self.mlm_probability = mlm_probability

    def _make_lm_batch(self, tokenizer, texts, objective: str) -> Dict[str, torch.Tensor]:
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if objective == "causal":
            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = -100
            batch["labels"] = labels
        elif objective == "masked":
            if tokenizer.mask_token is None:
                raise ValueError("Masked LM tokenizer must define mask_token.")
            masked_ids, labels = torch_mask_tokens(batch["input_ids"].clone(), tokenizer, self.mlm_probability)
            batch["input_ids"] = masked_ids
            batch["labels"] = labels
        else:
            raise ValueError(objective)
        return batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [f["text"] for f in features]
        if self.share_tokenization and self.teacher_objective == self.student_objective:
            sb = self._make_lm_batch(self.student_tokenizer, texts, self.student_objective)
            tb = {k: v.clone() for k, v in sb.items() if k != "labels"}
            return {"teacher": tb, "student": sb, "texts": texts}

        sb = self._make_lm_batch(self.student_tokenizer, texts, self.student_objective)
        # Teacher batch is mainly for hidden/logit distillation, so labels are unnecessary.
        tb = self.teacher_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {"teacher": tb, "student": sb, "texts": texts}


class TeacherOnlyLMTextCollator:
    def __init__(self, tokenizer, objective: str, max_length: int, mlm_probability: float):
        self.tokenizer = tokenizer
        self.objective = objective
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [f["text"] for f in features]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if self.objective == "causal":
            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = -100
            batch["labels"] = labels
        else:
            masked_ids, labels = torch_mask_tokens(batch["input_ids"].clone(), self.tokenizer, self.mlm_probability)
            batch["input_ids"] = masked_ids
            batch["labels"] = labels
        return batch

# ============================================================
# Model builders
# ============================================================

def build_model_for_task(
    model_name_or_path: str,
    task_type: str,
    tokenizer,
    num_labels: Optional[int] = None,
    lm_objective: Optional[str] = None,
):
    if task_type == "classification":
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            output_hidden_states=True,
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=config,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
    else:
        if lm_objective == "causal":
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                output_hidden_states=True,
                local_files_only=True,
            )
        elif lm_objective == "masked":
            model = AutoModelForMaskedLM.from_pretrained(
                model_name_or_path,
                output_hidden_states=True,
                local_files_only=True,
            )
        else:
            raise ValueError(lm_objective)
    maybe_set_pad_token(model, tokenizer)
    force_trainable_parameters_fp32(model)
    return model

# ============================================================
# Loss functions
# ============================================================

def kd_kl_loss(logits_s: torch.Tensor, logits_t: torch.Tensor, temperature: float) -> torch.Tensor:
    log_p_s = F.log_softmax(logits_s / temperature, dim=-1)
    p_t = F.softmax(logits_t / temperature, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature ** 2)


def compute_uot_label_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    eps: float,
    iters: int,
) -> torch.Tensor:
    p_s = F.softmax(logits_s, dim=-1)
    p_t = F.softmax(logits_t, dim=-1)
    B, K = p_s.shape
    C = (torch.ones(K, K, device=logits_s.device) - torch.eye(K, device=logits_s.device))
    C = C.unsqueeze(0).expand(B, -1, -1)
    f = torch.zeros(B, K, device=logits_s.device)
    g = torch.zeros(B, K, device=logits_s.device)
    for _ in range(iters):
        f = eps * (torch.log(p_t + 1e-12) - torch.logsumexp((g.unsqueeze(1) - C) / eps, dim=2))
        g = eps * (torch.log(p_s + 1e-12) - torch.logsumexp((f.unsqueeze(2) - C) / eps, dim=1))
    pi = torch.exp((f.unsqueeze(2) + g.unsqueeze(1) - C) / eps)
    return (pi * C).sum(dim=(1, 2)).mean()


def mean_pool_hidden(last_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if last_hidden.dim() != 3:
        return last_hidden.reshape(last_hidden.size(0), -1)
    if attention_mask is None:
        return last_hidden.mean(dim=1)
    mask = attention_mask.to(last_hidden.device).float().unsqueeze(-1)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def hidden_geometry_loss(outputs_s, outputs_t, s_attention_mask=None, t_attention_mask=None) -> torch.Tensor:
    hs = outputs_s.hidden_states[-1]
    ht = outputs_t.hidden_states[-1]
    ps = mean_pool_hidden(hs, s_attention_mask)
    pt = mean_pool_hidden(ht, t_attention_mask)
    d = min(ps.size(-1), pt.size(-1))
    ps = F.normalize(ps[:, :d], p=2, dim=-1)
    pt = F.normalize(pt[:, :d], p=2, dim=-1)
    return F.mse_loss(ps, pt)


def _safe_zero_like(logits_or_hidden: torch.Tensor) -> torch.Tensor:
    return torch.tensor(0.0, device=logits_or_hidden.device)


def _crop_batch(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n = min(a.size(0), b.size(0))
    return a[:n], b[:n]


def pooled_last_hidden(outputs, attention_mask=None) -> torch.Tensor:
    return mean_pool_hidden(outputs.hidden_states[-1], attention_mask)


def hidden_fitnet_loss(outputs_s, outputs_t, s_attention_mask=None, t_attention_mask=None) -> torch.Tensor:
    """FitNet-style final representation regression.

    For heterogeneous tokenizer/model pairs, we pool over tokens and match the
    common hidden dimension, which keeps the baseline stable for BERT/GPT crosses.
    """
    return hidden_geometry_loss(outputs_s, outputs_t, s_attention_mask, t_attention_mask)


def pkd_hidden_loss(outputs_s, outputs_t, s_attention_mask=None, t_attention_mask=None) -> torch.Tensor:
    """Patient KD: match several student hidden layers to proportionally spaced teacher layers.

    This implements the practical task-distillation variant used for sequence models:
    each selected hidden state is mean-pooled, L2-normalized, dimension-cropped, and
    matched by MSE.
    """
    hs_s = list(outputs_s.hidden_states or [])
    hs_t = list(outputs_t.hidden_states or [])
    if len(hs_s) == 0 or len(hs_t) == 0:
        ref = outputs_s.logits if hasattr(outputs_s, "logits") else hs_s[-1]
        return _safe_zero_like(ref)

    # Exclude embedding layer when possible.
    idx_s = list(range(1, len(hs_s))) if len(hs_s) > 1 else [0]
    losses = []
    for i_s in idx_s:
        # Proportional teacher layer mapping.
        rel = i_s / max(1, len(hs_s) - 1)
        i_t = int(round(rel * (len(hs_t) - 1)))
        i_t = max(0, min(i_t, len(hs_t) - 1))
        ps = mean_pool_hidden(hs_s[i_s], s_attention_mask)
        pt = mean_pool_hidden(hs_t[i_t], t_attention_mask)
        ps, pt = _crop_batch(ps, pt)
        d = min(ps.size(-1), pt.size(-1))
        ps = F.normalize(ps[:, :d], p=2, dim=-1)
        pt = F.normalize(pt[:, :d], p=2, dim=-1)
        losses.append(F.mse_loss(ps, pt))
    return torch.stack(losses).mean() if losses else _safe_zero_like(outputs_s.logits)


def _pdist_no_diag(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = x.size(0)
    if n <= 1:
        return torch.empty(0, device=x.device, dtype=x.dtype)
    dist = torch.cdist(x, x, p=2)
    mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
    return dist[mask].clamp_min(eps)


def rkd_loss(outputs_s, outputs_t, s_attention_mask=None, t_attention_mask=None) -> torch.Tensor:
    """Relational KD using distance-wise and angle-wise geometry on pooled features."""
    fs = pooled_last_hidden(outputs_s, s_attention_mask)
    ft = pooled_last_hidden(outputs_t, t_attention_mask)
    fs, ft = _crop_batch(fs, ft)
    if fs.size(0) < 2:
        return _safe_zero_like(fs)
    d = min(fs.size(-1), ft.size(-1))
    fs = F.normalize(fs[:, :d], p=2, dim=-1)
    ft = F.normalize(ft[:, :d], p=2, dim=-1)

    ds = _pdist_no_diag(fs)
    dt = _pdist_no_diag(ft)
    ds = ds / ds.mean().clamp_min(1e-12)
    dt = dt / dt.mean().clamp_min(1e-12)
    distance_loss = F.smooth_l1_loss(ds, dt)

    diff_s = fs.unsqueeze(0) - fs.unsqueeze(1)
    diff_t = ft.unsqueeze(0) - ft.unsqueeze(1)
    diff_s = F.normalize(diff_s, p=2, dim=-1)
    diff_t = F.normalize(diff_t, p=2, dim=-1)
    angle_s = torch.bmm(diff_s, diff_s.transpose(1, 2)).reshape(-1)
    angle_t = torch.bmm(diff_t, diff_t.transpose(1, 2)).reshape(-1)
    angle_loss = F.smooth_l1_loss(angle_s, angle_t)
    return distance_loss + angle_loss


def crd_loss(outputs_s, outputs_t, s_attention_mask=None, t_attention_mask=None, temperature: float = 0.07) -> torch.Tensor:
    """In-batch contrastive representation distillation."""
    fs = pooled_last_hidden(outputs_s, s_attention_mask)
    ft = pooled_last_hidden(outputs_t, t_attention_mask)
    fs, ft = _crop_batch(fs, ft)
    if fs.size(0) < 2:
        return _safe_zero_like(fs)
    d = min(fs.size(-1), ft.size(-1))
    fs = F.normalize(fs[:, :d], p=2, dim=-1)
    ft = F.normalize(ft[:, :d], p=2, dim=-1)
    logits_st = fs @ ft.t() / temperature
    labels = torch.arange(fs.size(0), device=fs.device)
    return F.cross_entropy(logits_st, labels)


def minilm_attention_loss(outputs_s, outputs_t) -> torch.Tensor:
    """MiniLM-style attention-relation distillation.

    Uses the final-layer attention relation averaged over heads. If attention tensors
    are unavailable or sequence lengths cannot be aligned, returns zero so the caller
    can combine it with hidden or logit losses safely.
    """
    att_s = getattr(outputs_s, "attentions", None)
    att_t = getattr(outputs_t, "attentions", None)
    if not att_s or not att_t or att_s[-1] is None or att_t[-1] is None:
        ref = outputs_s.logits if hasattr(outputs_s, "logits") else outputs_s.hidden_states[-1]
        return _safe_zero_like(ref)
    a_s = att_s[-1].mean(dim=1)  # [B, L, L]
    a_t = att_t[-1].mean(dim=1)
    b = min(a_s.size(0), a_t.size(0))
    l = min(a_s.size(-1), a_t.size(-1), a_s.size(-2), a_t.size(-2))
    if b == 0 or l == 0:
        return _safe_zero_like(a_s)
    a_s = a_s[:b, :l, :l]
    a_t = a_t[:b, :l, :l]
    a_s = F.normalize(a_s.reshape(b, -1), p=2, dim=-1)
    a_t = F.normalize(a_t.reshape(b, -1), p=2, dim=-1)
    return F.mse_loss(a_s, a_t)


def dkd_classification_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    alpha: float = 1.0,
    beta: float = 8.0,
) -> torch.Tensor:
    """Decoupled KD: separate target-class and non-target-class knowledge."""
    if logits_s.size(-1) != logits_t.size(-1):
        k = min(logits_s.size(-1), logits_t.size(-1))
        logits_s = logits_s[:, :k]
        logits_t = logits_t[:, :k]
        target = target.clamp(max=k - 1)

    gt_mask = torch.zeros_like(logits_s, dtype=torch.bool).scatter_(1, target.view(-1, 1), True)
    other_mask = ~gt_mask

    pred_s = F.softmax(logits_s / temperature, dim=1)
    pred_t = F.softmax(logits_t / temperature, dim=1)
    pred_s_two = torch.cat([(pred_s * gt_mask).sum(1, keepdim=True), (pred_s * other_mask).sum(1, keepdim=True)], dim=1)
    pred_t_two = torch.cat([(pred_t * gt_mask).sum(1, keepdim=True), (pred_t * other_mask).sum(1, keepdim=True)], dim=1)
    tckd = F.kl_div(torch.log(pred_s_two.clamp_min(1e-12)), pred_t_two, reduction="batchmean") * (temperature ** 2)

    logits_s_nt = logits_s / temperature - 1000.0 * gt_mask.float()
    logits_t_nt = logits_t / temperature - 1000.0 * gt_mask.float()
    nckd = F.kl_div(F.log_softmax(logits_s_nt, dim=1), F.softmax(logits_t_nt, dim=1), reduction="batchmean") * (temperature ** 2)
    return alpha * tckd + beta * nckd


def _pearson_corr_loss(x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
    x_s = x_s - x_s.mean(dim=1, keepdim=True)
    x_t = x_t - x_t.mean(dim=1, keepdim=True)
    x_s = F.normalize(x_s, p=2, dim=1)
    x_t = F.normalize(x_t, p=2, dim=1)
    return 1.0 - (x_s * x_t).sum(dim=1).mean()


def dist_logits_loss(logits_s: torch.Tensor, logits_t: torch.Tensor, temperature: float) -> torch.Tensor:
    """DIST-style inter-class and inter-sample correlation distillation."""
    k = min(logits_s.size(-1), logits_t.size(-1))
    ps = F.softmax(logits_s[:, :k] / temperature, dim=-1)
    pt = F.softmax(logits_t[:, :k] / temperature, dim=-1)
    inter_class = _pearson_corr_loss(ps, pt)
    if ps.size(0) > 1:
        inter_sample = _pearson_corr_loss(ps.t(), pt.t())
    else:
        inter_sample = _safe_zero_like(ps)
    return inter_class + inter_sample


def token_dist_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    labels: torch.Tensor,
    objective: str,
    temperature: float,
    max_tokens: int,
) -> torch.Tensor:
    if objective == "causal":
        logits_s = logits_s[:, :-1, :].contiguous()
        logits_t = logits_t[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
    mask = labels != -100
    if mask.sum().item() == 0:
        return _safe_zero_like(logits_s)
    flat_s = logits_s[mask]
    flat_t = logits_t[mask]
    if flat_s.size(0) > max_tokens:
        idx = torch.randperm(flat_s.size(0), device=flat_s.device)[:max_tokens]
        flat_s = flat_s[idx]
        flat_t = flat_t[idx]
    return dist_logits_loss(flat_s, flat_t, temperature)


def method_needs_attention(method: str) -> bool:
    return method in {"MiniLM", "TinyBERT"}


def compute_dense_baseline_losses(
    method: str,
    task_type: str,
    outputs_s,
    outputs_t,
    s_batch: Dict[str, torch.Tensor],
    t_batch: Dict[str, torch.Tensor],
    labels_cls: Optional[torch.Tensor],
    logits_distill_possible: bool,
    lm_objective_s: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (logit_loss, representation_loss) for a dense baseline method."""
    kd_loss = _safe_zero_like(outputs_s.logits)
    hid_loss = _safe_zero_like(outputs_s.logits)

    if method == "KD":
        if task_type == "classification":
            kd_loss = kd_kl_loss(outputs_s.logits, outputs_t.logits, CFG.temperature)
        elif logits_distill_possible:
            kd_loss = token_kd_loss(outputs_s.logits, outputs_t.logits, s_batch["labels"], lm_objective_s, CFG.temperature, CFG.lm_max_kd_tokens_per_batch)

    elif method == "FitNet":
        hid_loss = hidden_fitnet_loss(outputs_s, outputs_t, s_batch.get("attention_mask"), t_batch.get("attention_mask"))

    elif method == "PKD":
        hid_loss = pkd_hidden_loss(outputs_s, outputs_t, s_batch.get("attention_mask"), t_batch.get("attention_mask"))

    elif method == "RKD":
        hid_loss = rkd_loss(outputs_s, outputs_t, s_batch.get("attention_mask"), t_batch.get("attention_mask"))

    elif method == "CRD":
        hid_loss = crd_loss(outputs_s, outputs_t, s_batch.get("attention_mask"), t_batch.get("attention_mask"))

    elif method == "MiniLM":
        # Primary MiniLM signal is attention relation. Hidden relation fallback keeps
        # the baseline meaningful for model families where attentions are unavailable.
        att_loss = minilm_attention_loss(outputs_s, outputs_t)
        hid_loss = att_loss + 0.25 * rkd_loss(outputs_s, outputs_t, s_batch.get("attention_mask"), t_batch.get("attention_mask"))

    elif method == "TinyBERT":
        if task_type == "classification":
            kd_loss = kd_kl_loss(outputs_s.logits, outputs_t.logits, CFG.temperature)
        elif logits_distill_possible:
            kd_loss = token_kd_loss(outputs_s.logits, outputs_t.logits, s_batch["labels"], lm_objective_s, CFG.temperature, CFG.lm_max_kd_tokens_per_batch)
        hid_loss = pkd_hidden_loss(outputs_s, outputs_t, s_batch.get("attention_mask"), t_batch.get("attention_mask"))
        hid_loss = hid_loss + minilm_attention_loss(outputs_s, outputs_t)

    elif method == "DKD":
        if task_type == "classification" and labels_cls is not None:
            kd_loss = dkd_classification_loss(outputs_s.logits, outputs_t.logits, labels_cls, CFG.temperature)
        elif logits_distill_possible:
            kd_loss = token_kd_loss(outputs_s.logits, outputs_t.logits, s_batch["labels"], lm_objective_s, CFG.temperature, CFG.lm_max_kd_tokens_per_batch)

    elif method == "DIST":
        if task_type == "classification":
            kd_loss = dist_logits_loss(outputs_s.logits, outputs_t.logits, CFG.temperature)
        elif logits_distill_possible:
            kd_loss = token_dist_loss(outputs_s.logits, outputs_t.logits, s_batch["labels"], lm_objective_s, CFG.temperature, CFG.lm_max_kd_tokens_per_batch)

    else:
        raise ValueError(f"Unknown dense baseline method: {method}")

    return kd_loss, hid_loss


def classification_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)


def token_supervised_loss_per_sample(logits: torch.Tensor, labels: torch.Tensor, objective: str) -> torch.Tensor:
    """Approximate per-example uncertainty for LM active selection."""
    if objective == "causal":
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
    B, L, V = logits.shape
    loss_flat = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100, reduction="none")
    loss_tok = loss_flat.view(B, L)
    valid = (labels != -100).float()
    return (loss_tok * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def token_kd_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    labels: torch.Tensor,
    objective: str,
    temperature: float,
    max_tokens: int,
) -> torch.Tensor:
    if objective == "causal":
        logits_s = logits_s[:, :-1, :].contiguous()
        logits_t = logits_t[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

    mask = labels != -100
    if mask.sum().item() == 0:
        return torch.tensor(0.0, device=logits_s.device)

    flat_s = logits_s[mask]
    flat_t = logits_t[mask]
    if flat_s.size(0) > max_tokens:
        idx = torch.randperm(flat_s.size(0), device=flat_s.device)[:max_tokens]
        flat_s = flat_s[idx]
        flat_t = flat_t[idx]

    V = min(flat_s.size(-1), flat_t.size(-1))
    return kd_kl_loss(flat_s[:, :V], flat_t[:, :V], temperature)


def compute_curriculum_frontier(epoch: int, total_epochs: int, max_uncertainty: float) -> float:
    start = CFG.curriculum_start_ratio * max_uncertainty
    end = max_uncertainty
    progress = epoch / max(1, total_epochs - 1)
    return start + progress * (end - start)


def select_gcbd_active_mask(
    uncertainty: torch.Tensor,
    budget: float,
    gamma_t: float,
    curriculum_enabled: bool,
    no_budget: bool,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if curriculum_enabled:
        candidate = uncertainty <= gamma_t
    else:
        candidate = torch.ones_like(uncertainty, dtype=torch.bool)

    if no_budget:
        # Budget relaxation removes the query cap, but retains curriculum eligibility.
        return candidate.clone(), candidate, float("-inf")

    active = torch.zeros_like(uncertainty, dtype=torch.bool)
    idx = candidate.nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0 or budget <= 0:
        return active, candidate, float("inf")

    k = int(math.ceil(float(budget) * idx.numel()))
    k = max(1, min(k, idx.numel()))
    selected_local = torch.topk(uncertainty[idx], k=k, largest=True).indices
    selected_idx = idx[selected_local]
    active[selected_idx] = True
    eta = float(uncertainty[selected_idx].min().detach().float().cpu().item())
    return active, candidate, eta

# ============================================================
# Batch helpers
# ============================================================

def to_device_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device) if torch.is_tensor(vv) else vv for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def subset_model_inputs(inputs: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    idx = mask.nonzero(as_tuple=False).squeeze(1)
    return {k: v[idx] for k, v in inputs.items() if torch.is_tensor(v)}


def get_model_inputs(batch_part: Dict[str, torch.Tensor], include_labels: bool = False) -> Dict[str, torch.Tensor]:
    if include_labels:
        return {k: v for k, v in batch_part.items() if k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
    return {k: v for k, v in batch_part.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}

# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_model(
    accelerator: Accelerator,
    model,
    dataloader: DataLoader,
    task_type: str,
    lm_objective: Optional[str],
) -> Dict[str, float]:
    model.eval()
    losses = []

    if task_type == "classification":
        all_preds, all_labels = [], []
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process, desc="Eval"):
            batch = to_device_batch(batch, accelerator.device)
            labels = batch["labels"] if "labels" in batch else batch["student"]["labels"]
            inputs = batch["student"] if "student" in batch else batch
            outputs = model(**get_model_inputs(inputs, include_labels=False), labels=labels, output_hidden_states=True, return_dict=True)
            preds = outputs.logits.argmax(dim=-1)
            gathered_preds = accelerator.gather_for_metrics(preds)
            gathered_labels = accelerator.gather_for_metrics(labels)
            all_preds.append(gathered_preds.cpu())
            all_labels.append(gathered_labels.cpu())
            losses.append(accelerator.gather_for_metrics(outputs.loss.detach()).float().cpu())
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "eval_loss": float(torch.cat([x.view(-1) for x in losses]).mean().item()) if losses else 0.0,
        }

    # Language modeling
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process, desc="Eval"):
        batch = to_device_batch(batch, accelerator.device)
        inputs = batch["student"] if "student" in batch else batch
        outputs = model(**get_model_inputs(inputs, include_labels=True), output_hidden_states=True, return_dict=True)
        gathered_loss = accelerator.gather_for_metrics(outputs.loss.detach()).float().cpu()
        losses.append(gathered_loss.view(-1))
    eval_loss = float(torch.cat(losses).mean().item()) if losses else 0.0
    ppl = float(math.exp(min(20.0, eval_loss)))
    return {"eval_loss": eval_loss, "perplexity": ppl}


def classification_label_names(dataset_name: str) -> List[str]:
    if dataset_name in {"SST-2", "IMDb"}:
        return ["negative", "positive"]
    if dataset_name == "DBPedia":
        return [
            "Company", "EducationalInstitution", "Artist", "Athlete",
            "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
            "Village", "Animal", "Plant", "Album", "Film", "WrittenWork",
        ]
    return [str(index) for index in range(get_num_labels(dataset_name))]


def compute_ece_from_confidence(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int,
) -> Tuple[float, pd.DataFrame]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    n = max(1, len(confidences))
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidences >= lower) & (confidences < upper)
        if index == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else float("nan")
        confidence = float(confidences[mask].mean()) if count else float("nan")
        if count:
            ece += (count / n) * abs(accuracy - confidence)
        rows.append({
            "bin": index,
            "lower": float(lower),
            "upper": float(upper),
            "count": count,
            "accuracy": accuracy,
            "confidence": confidence,
        })
    return float(ece), pd.DataFrame(rows)


def align_numpy_features(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    d = min(a.shape[1], b.shape[1])
    return a[:n, :d], b[:n, :d]


@torch.no_grad()
def collect_evaluation_artifacts(
    accelerator: Accelerator,
    teacher_model,
    student_model,
    dataloader: DataLoader,
    task_type: str,
    lm_objective_s: Optional[str],
) -> Dict[str, np.ndarray]:
    """Collect bounded, CPU-resident diagnostics after training.

    Teacher forwards here support representation plots only and are deliberately
    excluded from training-time query accounting.
    """
    teacher_model.eval()
    student_model.eval()
    rng = np.random.default_rng(CFG.report_seed)

    confidences: List[np.ndarray] = []
    correct: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    preds_all: List[np.ndarray] = []
    true_nll: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    features_s: List[np.ndarray] = []
    features_t: List[np.ndarray] = []
    example_count = 0
    token_count = 0

    for batch in tqdm(
        dataloader,
        disable=not accelerator.is_local_main_process,
        desc="Evaluation reports",
    ):
        batch = to_device_batch(batch, accelerator.device)
        s_batch = batch["student"] if "student" in batch else batch
        t_batch = batch["teacher"] if "teacher" in batch else batch
        outputs_s = student_model(
            **get_model_inputs(s_batch, include_labels=False),
            output_hidden_states=True,
            return_dict=True,
        )
        outputs_t = teacher_model(
            **get_model_inputs(t_batch, include_labels=False),
            output_hidden_states=True,
            return_dict=True,
        )

        remaining_examples = CFG.report_max_examples - example_count
        if remaining_examples > 0:
            pooled_s = pooled_last_hidden(outputs_s, s_batch.get("attention_mask"))
            pooled_t = pooled_last_hidden(outputs_t, t_batch.get("attention_mask"))
            take = min(remaining_examples, pooled_s.size(0), pooled_t.size(0))
            features_s.append(pooled_s[:take].detach().float().cpu().numpy())
            features_t.append(pooled_t[:take].detach().float().cpu().numpy())
            example_count += take

        if task_type == "classification":
            labels = batch["labels"] if "labels" in batch else s_batch["labels"]
            log_probs = F.log_softmax(outputs_s.logits.float(), dim=-1)
            probs = log_probs.exp()
            conf, preds = probs.max(dim=-1)
            nll = -log_probs.gather(1, labels.view(-1, 1)).squeeze(1)
            probabilities.append(probs.detach().cpu().numpy())
            confidences.append(conf.detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())
            preds_all.append(preds.detach().cpu().numpy())
            correct.append(preds.eq(labels).detach().cpu().numpy())
            true_nll.append(nll.detach().cpu().numpy())
        else:
            remaining_tokens = CFG.report_max_lm_tokens - token_count
            if remaining_tokens > 0:
                logits = outputs_s.logits
                labels = s_batch["labels"]
                if lm_objective_s == "causal":
                    logits = logits[:, :-1, :].contiguous()
                    labels = labels[:, 1:].contiguous()
                mask = labels != -100
                flat_logits = logits[mask]
                flat_labels = labels[mask]
                if flat_labels.numel() > 0:
                    take = min(remaining_tokens, flat_labels.numel())
                    if flat_labels.numel() > take:
                        indices_np = rng.choice(flat_labels.numel(), size=take, replace=False)
                        indices = torch.as_tensor(indices_np, device=flat_labels.device)
                        flat_logits = flat_logits[indices]
                        flat_labels = flat_labels[indices]
                    else:
                        flat_logits = flat_logits[:take]
                        flat_labels = flat_labels[:take]
                    max_logits, preds = flat_logits.float().max(dim=-1)
                    log_normalizer = torch.logsumexp(flat_logits.float(), dim=-1)
                    conf = torch.exp(max_logits - log_normalizer)
                    nll = log_normalizer - flat_logits.float().gather(
                        1, flat_labels.view(-1, 1)
                    ).squeeze(1)
                    confidences.append(conf.detach().cpu().numpy())
                    labels_all.append(flat_labels.detach().cpu().numpy())
                    preds_all.append(preds.detach().cpu().numpy())
                    correct.append(preds.eq(flat_labels).detach().cpu().numpy())
                    true_nll.append(nll.detach().cpu().numpy())
                    token_count += take

        if task_type == "classification":
            diagnostics_complete = example_count >= CFG.report_max_examples
        else:
            diagnostics_complete = (
                example_count >= CFG.report_max_examples
                and token_count >= CFG.report_max_lm_tokens
            )
        if diagnostics_complete:
            break

    artifacts = {
        "confidences": np.concatenate(confidences) if confidences else np.empty(0),
        "correct": np.concatenate(correct).astype(bool) if correct else np.empty(0, dtype=bool),
        "labels": np.concatenate(labels_all).astype(np.int64) if labels_all else np.empty(0, dtype=np.int64),
        "preds": np.concatenate(preds_all).astype(np.int64) if preds_all else np.empty(0, dtype=np.int64),
        "true_nll": np.concatenate(true_nll) if true_nll else np.empty(0),
        "feat_s_last": np.concatenate(features_s) if features_s else np.empty((0, 0)),
        "feat_t_last": np.concatenate(features_t) if features_t else np.empty((0, 0)),
    }
    if probabilities:
        artifacts["probs"] = np.concatenate(probabilities)
    return artifacts


def plot_training_curves(log_df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(8, 5))
    for column, label in [
        ("train_loss", "Total"),
        ("supervised_loss", "Supervised"),
        ("kd_or_uot_loss", "KD/UOT"),
        ("hidden_geometry_loss", "Hidden geometry"),
    ]:
        if column in log_df:
            plt.plot(log_df["epoch"], log_df[column], marker="o", label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training losses")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    for column, label in [
        ("train_query_ratio", "Prediction-query ratio"),
        ("active_query_ratio", "GCBD active ratio"),
        ("candidate_ratio", "Candidate ratio"),
        ("teacher_geometry_forward_ratio", "Geometry-forward ratio"),
    ]:
        if column in log_df:
            plt.plot(log_df["epoch"], log_df[column], marker="o", label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Ratio")
    plt.ylim(bottom=0.0)
    plt.title("Teacher-query diagnostics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "query_ratio.png"), dpi=160)
    plt.close()


def plot_confidence_and_calibration(
    confidences: np.ndarray,
    correct: np.ndarray,
    calibration_df: pd.DataFrame,
    out_dir: str,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(confidences, bins=20, alpha=0.75, color="C0")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confidence_hist.png"), dpi=160)
    plt.close()

    valid = calibration_df[calibration_df["count"] > 0]
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(valid["confidence"], valid["accuracy"], marker="o", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability diagram")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reliability_diagram.png"), dpi=160)
    plt.close()


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: List[str],
    out_dir: str,
) -> None:
    size = max(6.0, min(12.0, 0.55 * len(labels)))
    plt.figure(figsize=(size, size))
    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=90, fontsize=8)
    plt.yticks(ticks, labels, fontsize=8)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=180)
    plt.close()


def plot_tsne(
    feat_s: np.ndarray,
    feat_t: np.ndarray,
    out_dir: str,
) -> Optional[str]:
    feat_s, feat_t = align_numpy_features(feat_s, feat_t)
    n = len(feat_s)
    if n < 3:
        return "t-SNE skipped because fewer than three paired examples were collected."
    if n > CFG.report_tsne_max_points:
        rng = np.random.default_rng(CFG.report_seed)
        indices = rng.choice(n, CFG.report_tsne_max_points, replace=False)
        feat_s, feat_t = feat_s[indices], feat_t[indices]
    combined = np.concatenate([feat_s, feat_t], axis=0).astype(np.float32)
    combined = (combined - combined.mean(axis=0, keepdims=True)) / (
        combined.std(axis=0, keepdims=True) + 1e-6
    )
    domain = np.array(["student"] * len(feat_s) + ["teacher"] * len(feat_t))
    perplexity = min(30.0, max(2.0, (len(combined) - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=CFG.report_seed,
        perplexity=perplexity,
    ).fit_transform(combined)
    pd.DataFrame({"x": embedding[:, 0], "y": embedding[:, 1], "domain": domain}).to_csv(
        os.path.join(out_dir, "tsne_embedding.csv"), index=False
    )
    plt.figure(figsize=(7, 6))
    for name, color in [("student", "C0"), ("teacher", "C1")]:
        mask = domain == name
        plt.scatter(embedding[mask, 0], embedding[mask, 1], s=9, alpha=0.55, label=name.title(), color=color)
    plt.title("t-SNE: teacher and student representations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tsne_teacher_student.png"), dpi=170)
    plt.close()
    return None


def plot_kd_gap_distribution(feat_s: np.ndarray, feat_t: np.ndarray, out_dir: str) -> Dict[str, float]:
    feat_s, feat_t = align_numpy_features(feat_s, feat_t)
    if not len(feat_s):
        return {"kd_gap_mean": float("nan"), "kd_gap_std": float("nan")}
    gaps = np.linalg.norm(feat_s - feat_t, axis=1)
    pd.DataFrame({"feature_l2_gap": gaps}).to_csv(
        os.path.join(out_dir, "kd_feature_gaps.csv"), index=False
    )
    plt.figure(figsize=(6, 4))
    plt.hist(gaps, bins=30, alpha=0.75, color="C1")
    plt.xlabel("Teacher–student representation L2 gap")
    plt.ylabel("Count")
    plt.title("KD representation-gap distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kd_gap_hist.png"), dpi=160)
    plt.close()
    return {"kd_gap_mean": float(gaps.mean()), "kd_gap_std": float(gaps.std())}


def plot_weight_histograms(model, out_dir: str) -> Dict[str, float]:
    rng = np.random.default_rng(CFG.report_seed)
    sampled = []
    remaining = CFG.report_max_weight_values
    for parameter in model.parameters():
        if parameter.dim() < 2 or remaining <= 0:
            continue
        values = parameter.detach().float().cpu().numpy().ravel()
        take = min(remaining, len(values))
        if take < len(values):
            values = values[rng.choice(len(values), take, replace=False)]
        sampled.append(values)
        remaining -= take
    weights = np.concatenate(sampled) if sampled else np.empty(0)
    if not len(weights):
        return {"sampled_weight_mean": float("nan"), "sampled_weight_std": float("nan")}
    plt.figure(figsize=(6, 4))
    plt.hist(weights, bins=60, alpha=0.75, color="C2")
    plt.xlabel("Weight value")
    plt.ylabel("Count")
    plt.title(f"Student weight distribution (sample n={len(weights):,})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "weight_hist.png"), dpi=160)
    plt.close()
    return {
        "sampled_weight_count": int(len(weights)),
        "sampled_weight_mean": float(weights.mean()),
        "sampled_weight_std": float(weights.std()),
    }


def build_lm_confusion(
    labels: np.ndarray,
    preds: np.ndarray,
    tokenizer,
) -> Tuple[np.ndarray, List[str]]:
    unique, counts = np.unique(labels, return_counts=True)
    top = unique[np.argsort(-counts)[:CFG.report_lm_confusion_top_tokens]]
    top_set = set(int(value) for value in top)
    other_index = len(top)
    mapping = {int(token_id): index for index, token_id in enumerate(top)}
    mapped_labels = np.array([mapping.get(int(value), other_index) for value in labels])
    mapped_preds = np.array([
        mapping.get(int(value), other_index) if int(value) in top_set else other_index
        for value in preds
    ])
    display = [str(tokenizer.convert_ids_to_tokens(int(value))) for value in top] + ["<other>"]
    matrix = confusion_matrix(mapped_labels, mapped_preds, labels=list(range(len(display))))
    return matrix, display


def generate_evaluation_reports(
    out_dir: str,
    dataset_name: str,
    task_type: str,
    tokenizer,
    log_df: pd.DataFrame,
    artifacts: Dict[str, np.ndarray],
    student_model,
    run_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    warnings_list: List[str] = []
    report: Dict[str, Any] = {
        "report_example_count": int(len(artifacts["feat_s_last"])),
        "calibration_observation_count": int(len(artifacts["confidences"])),
    }
    if run_summary:
        report.update(run_summary)
    plot_training_curves(log_df, out_dir)

    if len(artifacts["confidences"]):
        ece, calibration_df = compute_ece_from_confidence(
            artifacts["confidences"], artifacts["correct"], CFG.calibration_bins
        )
        calibration_df.to_csv(os.path.join(out_dir, "calibration_bins.csv"), index=False)
        plot_confidence_and_calibration(
            artifacts["confidences"], artifacts["correct"], calibration_df, out_dir
        )
        if task_type == "classification":
            report["ece"] = ece
            report["nll"] = float(artifacts["true_nll"].mean())
            report["report_accuracy"] = float(artifacts["correct"].mean())
        else:
            # These are computed on the bounded token sample used for the
            # diagnostics, not on the full language-model evaluation split.
            report["sampled_token_ece"] = ece
            report["sampled_token_nll"] = float(artifacts["true_nll"].mean())
            report["sampled_token_accuracy"] = float(artifacts["correct"].mean())
    else:
        warnings_list.append("Confidence and calibration plots skipped: no valid observations.")

    if len(artifacts["labels"]):
        if task_type == "classification":
            display = classification_label_names(dataset_name)
            matrix = confusion_matrix(
                artifacts["labels"], artifacts["preds"], labels=list(range(len(display)))
            )
        else:
            matrix, display = build_lm_confusion(
                artifacts["labels"], artifacts["preds"], tokenizer
            )
        pd.DataFrame(matrix, index=display, columns=display).to_csv(
            os.path.join(out_dir, "confusion_matrix.csv")
        )
        plot_confusion_matrix(matrix, display, out_dir)
    else:
        warnings_list.append("Confusion matrix skipped: no valid labels.")

    tsne_warning = plot_tsne(
        artifacts["feat_s_last"], artifacts["feat_t_last"], out_dir
    )
    if tsne_warning:
        warnings_list.append(tsne_warning)
    report.update(plot_kd_gap_distribution(
        artifacts["feat_s_last"], artifacts["feat_t_last"], out_dir
    ))
    report.update(plot_weight_histograms(student_model, out_dir))
    report["report_warnings"] = warnings_list
    write_json_atomic(os.path.join(out_dir, "evaluation_report.json"), report)
    return report

# ============================================================
# Teacher training
# ============================================================

def make_teacher_cache_dir(dataset_name: str, scenario_name: str, teacher_name: str, task_type: str, seed: int) -> str:
    return os.path.join(
        CFG.teacher_cache_root,
        sanitize_name(dataset_name),
        sanitize_name(scenario_name),
        sanitize_name(teacher_name),
        task_type,
        f"seed_{seed}",
    )


def train_or_load_teacher(
    accelerator: Accelerator,
    dataset_name: str,
    scenario_name: str,
    teacher_name: str,
    teacher_tokenizer,
    raw: DatasetDict,
    eval_split: str,
    task_type: str,
    seed: int,
) -> Any:
    cache_dir = make_teacher_cache_dir(dataset_name, scenario_name, teacher_name, task_type, seed)
    teacher_source = resolve_model_source(teacher_name)
    done_file = os.path.join(cache_dir, "DONE")
    if CFG.reuse_finetuned_teacher and os.path.isfile(done_file):
        main_print(accelerator, f"Loading cached teacher: {cache_dir}")
        if task_type == "classification":
            model = build_model_for_task(cache_dir, task_type, teacher_tokenizer, num_labels=get_num_labels(dataset_name))
        else:
            obj = infer_lm_objective(teacher_name)
            model = build_model_for_task(cache_dir, task_type, teacher_tokenizer, lm_objective=obj)
        return model

    if CFG.require_finetuned_teacher:
        raise FileNotFoundError(
            f"Required fine-tuned teacher is missing or incomplete: {cache_dir}. "
            "Submit the teacher job first and use an afterok dependency."
        )

    if not CFG.train_teacher_if_missing:
        main_print(accelerator, f"Using pretrained teacher without task fine-tuning: {teacher_name}")
        if task_type == "classification":
            return build_model_for_task(teacher_source, task_type, teacher_tokenizer, num_labels=get_num_labels(dataset_name))
        obj = infer_lm_objective(teacher_name)
        return build_model_for_task(teacher_source, task_type, teacher_tokenizer, lm_objective=obj)

    main_print(accelerator, f"Fine-tuning teacher for {dataset_name} | {scenario_name}: {teacher_name}")
    if task_type == "classification":
        model = build_model_for_task(teacher_source, task_type, teacher_tokenizer, num_labels=get_num_labels(dataset_name))
        collator = TeacherOnlyClassificationCollator(teacher_tokenizer, CFG.max_length_cls)
        train_bs = CFG.train_batch_size_cls
        eval_bs = CFG.eval_batch_size_cls
        epochs = CFG.teacher_epochs_cls
    else:
        obj = infer_lm_objective(teacher_name)
        model = build_model_for_task(teacher_source, task_type, teacher_tokenizer, lm_objective=obj)
        collator = TeacherOnlyLMTextCollator(teacher_tokenizer, obj, CFG.max_length_lm, CFG.mlm_probability)
        train_bs = CFG.train_batch_size_lm
        eval_bs = CFG.eval_batch_size_lm
        epochs = CFG.teacher_epochs_lm

    force_trainable_parameters_fp32(model)
    main_print(accelerator, f"Teacher trainable dtype summary before prepare: {trainable_dtype_summary(model)}")

    train_loader = make_dataloader(
        raw["train"], batch_size=train_bs, shuffle=True, collate_fn=collator,
        block_shuffle=True, seed=seed,
    )
    eval_loader = make_dataloader(raw[eval_split], batch_size=eval_bs, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate_teacher, weight_decay=CFG.weight_decay)
    steps = max(1, math.ceil(len(train_loader) / CFG.gradient_accumulation_steps) * epochs)
    warmup = int(CFG.warmup_ratio * steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, steps)

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, eval_loader, scheduler)
    force_trainable_parameters_fp32(model)
    if has_fp16_trainable_params(model):
        raise RuntimeError("FP16 trainable parameters remain after Accelerate preparation; set CFG.mixed_precision='no'.")
    main_print(accelerator, f"Teacher trainable dtype summary after prepare: {trainable_dtype_summary(model)}")
    model.train()
    global_step = 0
    for epoch in range(epochs):
        pbar = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Teacher epoch {epoch}")
        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                outputs = model(**batch, output_hidden_states=True, return_dict=True)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if accelerator.is_local_main_process:
                pbar.set_postfix(loss=f"{loss.detach().float().item():.4f}")

    metrics = evaluate_model(accelerator, model, eval_loader, task_type, infer_lm_objective(teacher_name) if task_type != "classification" else None)
    main_print(accelerator, f"Teacher metrics: {metrics}")

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        ensure_dir(cache_dir)
        unwrapped.save_pretrained(cache_dir)
        teacher_tokenizer.save_pretrained(cache_dir)
        write_json_atomic(os.path.join(cache_dir, "teacher_metrics.json"), metrics)
        write_text_atomic(done_file, "ok\n")
    accelerator.wait_for_everyone()

    # Important: free wrapped objects before loading a clean non-DDP teacher for student distillation.
    accelerator.free_memory(model, optimizer, train_loader, eval_loader, scheduler)
    if task_type == "classification":
        return build_model_for_task(cache_dir, task_type, teacher_tokenizer, num_labels=get_num_labels(dataset_name))
    obj = infer_lm_objective(teacher_name)
    return build_model_for_task(cache_dir, task_type, teacher_tokenizer, lm_objective=obj)

# ============================================================
# Student distillation training
# ============================================================

def make_method_output_dir(
    dataset_name: str,
    scenario_name: str,
    method: str,
    budget: float,
    seed: int,
) -> str:
    is_gcbd = method.startswith("GCBD")
    method_dir = method + (f"_B{budget:g}" if is_gcbd else "")
    return os.path.join(
        CFG.output_root,
        sanitize_name(dataset_name),
        sanitize_name(scenario_name),
        sanitize_name(method_dir),
        f"seed_{seed}",
    )

def _nlp_num_transformer_layers(config) -> int:
    for key in ("num_hidden_layers", "n_layer", "n_layers", "num_layers", "encoder_layers"):
        value = getattr(config, key, None)
        if value is not None:
            return int(value)
    raise ValueError(f"Cannot infer transformer depth from {type(config).__name__}")


def train_student_method(
    accelerator: Accelerator,
    dataset_name: str,
    scenario_name: str,
    teacher_name: str,
    student_name: str,
    method: str,
    budget: float,
    seed: int,
    raw: DatasetDict,
    eval_split: str,
    teacher_model,
    teacher_tokenizer,
    student_tokenizer,
) -> Dict[str, Any]:
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    task_type = get_task_type(dataset_name)
    is_gcbd = method.startswith("GCBD")
    no_budget = method == "GCBD-NoBudget"
    curriculum_enabled = method != "GCBD-NoCurriculum"
    stage1_enabled = method != "GCBD-NoStage1"
    no_uot = method == "GCBD-NoUOT"
    requires_attention = method_needs_attention(method)
    student_source = resolve_model_source(student_name)

    if task_type == "classification":
        num_labels = get_num_labels(dataset_name)
        student_model = build_model_for_task(student_source, task_type, student_tokenizer, num_labels=num_labels)
        collator = (
            GCBDClassificationCollator(teacher_tokenizer, student_tokenizer, CFG.max_length_cls)
            if is_gcbd else DualClassificationCollator(teacher_tokenizer, student_tokenizer, CFG.max_length_cls)
        )
        train_bs = CFG.train_batch_size_cls
        eval_bs = CFG.eval_batch_size_cls
        epochs = CFG.epochs_cls
        max_uncertainty = math.log(num_labels)
        lm_objective_s = None
        lm_objective_t = None
        logits_distill_possible = True
    else:
        lm_objective_t = infer_lm_objective(teacher_name)
        lm_objective_s = infer_lm_objective(student_name)
        student_model = build_model_for_task(student_source, task_type, student_tokenizer, lm_objective=lm_objective_s)
        same_vocab = len(teacher_tokenizer) == len(student_tokenizer)
        same_objective = lm_objective_t == lm_objective_s
        share_tokenization = same_vocab and same_objective
        logits_distill_possible = bool(same_vocab and same_objective)
        collator = (
            GCBDLMTextCollator(
                teacher_tokenizer=teacher_tokenizer,
                student_tokenizer=student_tokenizer,
                teacher_objective=lm_objective_t,
                student_objective=lm_objective_s,
                max_length=CFG.max_length_lm,
                mlm_probability=CFG.mlm_probability,
            )
            if is_gcbd else DualLMTextCollator(
                teacher_tokenizer=teacher_tokenizer,
                student_tokenizer=student_tokenizer,
                teacher_objective=lm_objective_t,
                student_objective=lm_objective_s,
                max_length=CFG.max_length_lm,
                share_tokenization=share_tokenization,
                mlm_probability=CFG.mlm_probability,
            )
        )
        train_bs = CFG.train_batch_size_lm
        eval_bs = CFG.eval_batch_size_lm
        epochs = CFG.epochs_lm
        # LM uncertainty is approximate per-sample CE; log(vocab) is a reasonable entropy-like upper scale.
        max_uncertainty = math.log(max(2, len(student_tokenizer)))

    # Freeze teacher.
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    geometry_cache_build = None
    geometry_cache = None
    geometry_ledger: Dict[str, Any] = {
        "geometry_examples": 0, "geometry_forward_invocations": 0,
        "teacher_seconds": 0.0, "total_wall_seconds": 0.0,
        "cache_bytes": 0, "cache_hit": False,
    }
    teacher_layer_indices: List[int] = []
    student_layer_indices: List[int] = []
    semantic_cost_matrix = None
    semantic_embedding_metadata = None
    if is_gcbd and stage1_enabled:
        if accelerator.num_processes != 1:
            raise RuntimeError("Phase 2 geometry cache currently requires one process / one GPU")
        teacher_model.to(accelerator.device)
        geometry_cache_build = build_nlp_geometry_cache(
            teacher=teacher_model,
            dataset=raw["train"],
            tokenizer=teacher_tokenizer,
            dataset_name=dataset_name,
            scenario=teacher_name,  # share cache across student scenarios using the same teacher
            seed=seed,
            base_cache_dir=CFG.geometry_cache_root,
            batch_size=train_bs,
            max_length=CFG.max_length_cls if task_type == "classification" else CFG.max_length_lm,
            device=accelerator.device,
            num_workers=CFG.num_workers,
            autocast_context=accelerator.autocast,
        )
        geometry_cache = geometry_cache_build.reader
        teacher_layer_indices = list(geometry_cache_build.teacher_layer_indices)
        student_layer_indices = layer_indices_by_depth(_nlp_num_transformer_layers(student_model.config))
        geometry_ledger = dict(geometry_cache_build.ledger)
        main_print(
            accelerator,
            f"GCBD NLP geometry cache: {geometry_cache_build.root} | hit={geometry_cache_build.cache_hit} "
            f"| examples={geometry_ledger.get('geometry_examples', 0)} "
            f"| teacher_forwards={geometry_ledger.get('geometry_forward_invocations', 0)} "
            f"| bytes={geometry_ledger.get('cache_bytes', 0)}",
        )
    if is_gcbd and not no_uot and task_type == "classification":
        semantic_path = CFG.semantic_class_embeddings_path.get(dataset_name)
        feature = raw["train"].features.get("label") if hasattr(raw["train"], "features") else None
        feature_names = getattr(feature, "names", None)
        expected_names = CFG.semantic_class_names.get(dataset_name) or (
            [str(x) for x in feature_names] if feature_names else classification_label_names(dataset_name)
        )
        if not semantic_path:
            raise RuntimeError(
                f"GCBD UOT for {dataset_name} requires semantic_class_embeddings_path in the runtime config. "
                "Legacy identity cost is disabled."
            )
        if len(expected_names) != get_num_labels(dataset_name):
            raise ValueError("Ordered semantic class names do not match the classifier label count")
        if feature_names and CFG.semantic_class_names.get(dataset_name):
            if list(CFG.semantic_class_names[dataset_name]) != [str(x) for x in feature_names]:
                raise ValueError("Configured semantic class order differs from the saved dataset ClassLabel order")
        semantic_cost_matrix, semantic_embedding_metadata = load_semantic_cost(
            semantic_path, expected_names, accelerator.device
        )
    cache_current_job_wall = (
        float(geometry_ledger.get("total_wall_seconds", 0.0))
        if geometry_cache_build is not None and not geometry_cache_build.cache_hit else 0.0
    )

    force_trainable_parameters_fp32(student_model)
    main_print(accelerator, f"Student trainable dtype summary before prepare [{method}]: {trainable_dtype_summary(student_model)}")

    # V2: shard-aware batches for GCBD geometry-cache locality.
    # All other methods retain the original random minibatch sampling.
    use_shard_aware = (method == "GCBD-Full")

    train_loader = make_dataloader(
        raw["train"],
        batch_size=train_bs,
        shuffle=True,
        collate_fn=collator,
        block_shuffle=use_shard_aware,
        seed=seed,
    )
    eval_loader = make_dataloader(raw[eval_split], batch_size=eval_bs, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=CFG.learning_rate_student, weight_decay=CFG.weight_decay)
    steps = max(1, math.ceil(len(train_loader) / CFG.gradient_accumulation_steps) * epochs)
    warmup = int(CFG.warmup_ratio * steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, steps)

    teacher_model, student_model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        teacher_model, student_model, optimizer, train_loader, eval_loader, scheduler
    )
    force_trainable_parameters_fp32(student_model)
    if has_fp16_trainable_params(student_model):
        raise RuntimeError("FP16 trainable student parameters remain after Accelerate preparation; set CFG.mixed_precision='no'.")
    main_print(accelerator, f"Student trainable dtype summary after prepare [{method}]: {trainable_dtype_summary(student_model)}")

    metrics_log = []
    train_start = time.perf_counter()
    calibration_buffer = CalibrationBuffer(
        capacity=CFG.calibration_buffer_capacity,
        min_values=CFG.calibration_buffer_min_values,
    )

    for epoch in range(epochs):
        set_dataloader_epoch(train_loader, epoch)
        student_model.train()
        teacher_model.eval()
        epoch_loss = 0.0
        epoch_sup = 0.0
        epoch_kd = 0.0
        epoch_hidden = 0.0
        epoch_budget_penalty = 0.0
        total_seen = 0
        total_active = 0
        total_candidate = 0
        teacher_prediction_examples = 0
        teacher_prediction_forward_invocations = 0
        teacher_geometry_forwards = 0  # No broad teacher geometry calls occur in training.
        teacher_timer = TeacherForwardTimer(accelerator.device)
        eta_values = []
        gamma_min = CFG.curriculum_start_ratio * max_uncertainty
        gamma_t = cosine_frontier(epoch, epochs, gamma_min, max_uncertainty)

        pbar = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"{method} epoch {epoch} B={budget}")
        for batch in pbar:
            batch = to_device_batch(batch, accelerator.device)
            labels_cls = batch.get("labels", None)
            s_batch = batch["student"]
            t_batch = batch["teacher"]
            bs = s_batch["input_ids"].size(0)

            with accelerator.accumulate(student_model):
                optimizer.zero_grad(set_to_none=True)
                if task_type == "classification":
                    outputs_s = student_model(
                        **get_model_inputs(s_batch, include_labels=False),
                        labels=labels_cls,
                        output_hidden_states=True,
                        output_attentions=requires_attention,
                        return_dict=True,
                    )
                    supervised_loss = outputs_s.loss
                    uncertainty = classification_entropy(outputs_s.logits)
                else:
                    outputs_s = student_model(
                        **get_model_inputs(s_batch, include_labels=True),
                        output_hidden_states=True,
                        output_attentions=requires_attention,
                        return_dict=True,
                    )
                    supervised_loss = outputs_s.loss
                    uncertainty = token_supervised_loss_per_sample(outputs_s.logits, s_batch["labels"], lm_objective_s)

                geometry_outputs_s = outputs_s
                student_geometry_batch = s_batch
                if is_gcbd and task_type != "classification":
                    student_geometry_batch = batch["student_geometry"]
                    # Geometry is computed on the clean text view, not stochastic MLM corruption.
                    geometry_outputs_s = student_model(
                        **get_model_inputs(student_geometry_batch, include_labels=False),
                        output_hidden_states=True,
                        return_dict=True,
                    )

                kd_loss = torch.tensor(0.0, device=accelerator.device)
                hid_loss = torch.tensor(0.0, device=accelerator.device)
                budget_penalty = torch.tensor(0.0, device=accelerator.device)

                if method == "Student":
                    loss = supervised_loss

                elif not is_gcbd:
                    # Dense baselines: teacher is queried for every training example.
                    with torch.no_grad():
                        outputs_t = teacher_timer.run(
                            lambda: teacher_model(
                                **get_model_inputs(t_batch, include_labels=False),
                                output_hidden_states=True,
                                output_attentions=requires_attention,
                                return_dict=True,
                            )
                        )
                    teacher_prediction_examples += bs
                    teacher_prediction_forward_invocations += 1
                    kd_loss, hid_loss = compute_dense_baseline_losses(
                        method=method,
                        task_type=task_type,
                        outputs_s=outputs_s,
                        outputs_t=outputs_t,
                        s_batch=s_batch,
                        t_batch=t_batch,
                        labels_cls=labels_cls,
                        logits_distill_possible=logits_distill_possible,
                        lm_objective_s=lm_objective_s,
                    )
                    loss = supervised_loss + CFG.alpha * kd_loss + CFG.beta * hid_loss

                else:
                    if no_budget:
                        active_mask, candidate_mask, eta = select_gcbd_active_mask(
                            uncertainty=uncertainty,
                            budget=budget,
                            gamma_t=gamma_t,
                            curriculum_enabled=curriculum_enabled,
                            no_budget=True,
                        )
                        gate = active_mask.to(dtype=torch.float32)
                    else:
                        if CFG.rho_gate is None or float(CFG.rho_gate) <= 0:
                            raise RuntimeError(
                                "CFG.rho_gate is not locked. Set it from the revision pilot before GCBD training; do not guess."
                            )
                        frontier = gamma_t if curriculum_enabled else float(uncertainty.detach().max().cpu()) + 1e-6
                        active_mask, gate, _gate_weights, selection_diag = select_gcbd(
                            uncertainty,
                            float(budget),
                            float(frontier),
                            calibration_buffer,
                            rho=float(CFG.rho_gate),
                        )
                        candidate_mask = uncertainty.detach() <= float(frontier)
                        eta = float(selection_diag["eta"])
                    eta_values.append(eta)
                    total_active += int(active_mask.sum().detach().cpu().item())
                    total_candidate += int(candidate_mask.sum().detach().cpu().item())

                    if stage1_enabled:
                        if geometry_cache is None:
                            raise RuntimeError("GCBD Stage 1 is enabled but no validated offline geometry cache is open")
                        activation = list(CFG.progressive_layer_activation_fractions)
                        if len(activation) != len(teacher_layer_indices):
                            raise ValueError("progressive_layer_activation_fractions must match cached geometry layers")
                        progress = epoch / max(1, epochs - 1)
                        active_layers = sum(progress >= float(value) for value in activation)
                        cached_loss, _cache_diag = cached_nlp_swat_loss(
                            student_hidden_states=geometry_outputs_s.hidden_states,
                            student_offsets=student_geometry_batch["offset_mapping"],
                            student_attention_mask=student_geometry_batch["attention_mask"],
                            text_lengths=batch["text_lengths"],
                            sample_ids=batch["sample_id"],
                            cache=geometry_cache,
                            student_layer_indices=student_layer_indices,
                            teacher_layer_indices=teacher_layer_indices,
                            active_layers=active_layers,
                            projections=CFG.gcbd_swd_projections,
                        )
                        hid_loss = hid_loss + cached_loss

                    if active_mask.any():
                        s_sel = subset_model_inputs(student_geometry_batch, active_mask)
                        t_sel = subset_model_inputs(t_batch, active_mask)
                        with torch.no_grad():
                            outputs_t_sel = teacher_timer.run(
                                lambda: teacher_model(
                                    **get_model_inputs(t_sel, include_labels=False),
                                    output_hidden_states=True,
                                    return_dict=True,
                                )
                            )
                        selected_count = int(active_mask.sum().detach().cpu().item())
                        teacher_prediction_examples += selected_count
                        teacher_prediction_forward_invocations += 1
                        if task_type == "classification":
                            logits_s_sel = outputs_s.logits[active_mask]
                            if no_uot:
                                kd_loss = kd_kl_loss(logits_s_sel, outputs_t_sel.logits, CFG.temperature)
                            else:
                                kd_loss, _transport_diag = microbatched_transport_loss(
                                    logits_s_sel / CFG.temperature,
                                    outputs_t_sel.logits / CFG.temperature,
                                    semantic_cost_matrix,
                                    mode="uot",
                                    epsilon=CFG.sinkhorn_eps,
                                    rho_t=1.0,
                                    rho_s=1.0,
                                    iterations=100,
                                    tolerance=1e-5,
                                    microbatch_size=CFG.uot_microbatch_size,
                                )
                        else:
                            if logits_distill_possible:
                                kd_loss = token_kd_loss(
                                    outputs_s.logits[active_mask],
                                    outputs_t_sel.logits,
                                    s_batch["labels"][active_mask],
                                    lm_objective_s,
                                    CFG.temperature,
                                    CFG.lm_max_kd_tokens_per_batch,
                                )
                        final_swat = queried_nlp_final_swat_loss(
                            student_final_hidden=geometry_outputs_s.hidden_states[-1][active_mask],
                            teacher_hidden_states=outputs_t_sel.hidden_states,
                            student_offsets=student_geometry_batch["offset_mapping"][active_mask],
                            student_attention_mask=student_geometry_batch["attention_mask"][active_mask],
                            teacher_offsets=t_sel["offset_mapping"],
                            teacher_attention_mask=t_sel["attention_mask"],
                            text_lengths=batch["text_lengths"][active_mask],
                            teacher_text_lengths=batch["text_lengths"][active_mask],
                            projections=CFG.gcbd_swd_projections,
                        )
                        gate_scale = gate[active_mask].mean() if gate.numel() else 1.0
                        kd_loss = kd_loss * gate_scale
                        hid_loss = hid_loss + final_swat * gate_scale

                    if not no_budget:
                        query_ratio_batch = active_mask.float().mean()
                        budget_penalty = CFG.gamma_budget * torch.clamp(query_ratio_batch - float(budget), min=0.0)

                    loss = supervised_loss + CFG.alpha * kd_loss + CFG.beta * hid_loss + budget_penalty

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

            total_seen += bs
            epoch_loss += float(loss.detach().float().item()) * bs
            epoch_sup += float(supervised_loss.detach().float().item()) * bs
            epoch_kd += float(kd_loss.detach().float().item()) * bs
            epoch_hidden += float(hid_loss.detach().float().item()) * bs
            epoch_budget_penalty += float(budget_penalty.detach().float().item()) * bs
            if accelerator.is_local_main_process:
                pbar.set_postfix(loss=f"{float(loss.detach().float().item()):.4f}", q=f"{total_active/max(1,total_seen):.3f}")

        # Reduce scalar logs across processes.
        teacher_prediction_seconds = teacher_timer.elapsed_seconds()
        device = accelerator.device
        log_tensor = torch.tensor([
            total_seen,
            epoch_loss,
            epoch_sup,
            epoch_kd,
            epoch_hidden,
            epoch_budget_penalty,
            total_active,
            total_candidate,
            teacher_prediction_examples,
            teacher_prediction_forward_invocations,
            teacher_geometry_forwards,
            teacher_prediction_seconds,
        ], dtype=torch.float64, device=device)
        reduced = accelerator.reduce(log_tensor, reduction="sum").detach().cpu().numpy()
        seen = max(1.0, reduced[0])
        epoch_log = {
            "epoch": epoch,
            "train_loss": float(reduced[1] / seen),
            "supervised_loss": float(reduced[2] / seen),
            "kd_or_uot_loss": float(reduced[3] / seen),
            "hidden_geometry_loss": float(reduced[4] / seen),
            "budget_penalty": float(reduced[5] / seen),
            # Match the Vision convention for an online teacher query: dense
            # distillation baselines query every example, GCBD counts only its
            # active prediction branch, and Student makes no teacher queries.
            "train_query_ratio": float(reduced[8] / seen),
            "active_query_ratio": float(reduced[6] / seen),
            "candidate_ratio": float(reduced[7] / seen),
            "train_candidate_ratio": float(reduced[7] / seen),
            "teacher_prediction_forward_ratio": float(reduced[9] / max(1, len(train_loader) * accelerator.num_processes)),
            "teacher_prediction_examples": int(reduced[8]),
            "teacher_prediction_forward_invocations": int(reduced[9]),
            "teacher_prediction_seconds": float(reduced[11]),
            "teacher_geometry_forward_ratio_during_training": 0.0,
            "teacher_geometry_examples_offline": int(geometry_ledger.get("geometry_examples", 0)),
            "teacher_geometry_forward_invocations_offline": int(geometry_ledger.get("geometry_forward_invocations", 0)),
            "train_samples_seen": int(reduced[0]),
            "active_queries": int(reduced[6]),
            "candidate_queries": int(reduced[7]),
            "teacher_prediction_queries": int(reduced[8]),
            "teacher_geometry_queries": 0,
            "gamma_t": float(gamma_t),
            "eta_t_mean_local": float(np.mean([x for x in eta_values if math.isfinite(x)])) if len(eta_values) else 0.0,
        }
        metrics_log.append(epoch_log)
        main_print(accelerator, f"Epoch log: {epoch_log}")

    train_runtime = time.perf_counter() - train_start
    eval_metrics = evaluate_model(accelerator, student_model, eval_loader, task_type, lm_objective_s)
    log_df = pd.DataFrame(metrics_log)
    total_train_samples = int(log_df["train_samples_seen"].sum()) if not log_df.empty else 0
    total_prediction_queries = int(log_df["teacher_prediction_queries"].sum()) if not log_df.empty else 0
    total_geometry_queries = int(geometry_ledger.get("geometry_examples", 0))
    total_prediction_forward_invocations = int(log_df["teacher_prediction_forward_invocations"].sum()) if not log_df.empty else 0
    total_prediction_teacher_seconds = float(log_df["teacher_prediction_seconds"].sum()) if not log_df.empty else 0.0
    total_active_queries = int(log_df["active_queries"].sum()) if not log_df.empty else 0
    total_candidate_queries = int(log_df["candidate_queries"].sum()) if not log_df.empty else 0
    train_query_ratio = total_prediction_queries / max(1, total_train_samples)
    active_query_ratio = total_active_queries / max(1, total_train_samples)
    train_candidate_ratio = total_candidate_queries / max(1, total_train_samples)

    eval_metrics["train_runtime_sec"] = float(train_runtime)
    eval_metrics["train_runtime_min"] = float(train_runtime / 60.0)
    eval_metrics["end_to_end_method_runtime_sec"] = float(train_runtime + cache_current_job_wall)
    eval_metrics["query_ratio"] = float(train_query_ratio)
    eval_metrics["train_query_ratio"] = float(train_query_ratio)
    eval_metrics["mean_train_query_ratio"] = float(train_query_ratio)
    eval_metrics["final_train_query_ratio"] = float(metrics_log[-1]["train_query_ratio"] if metrics_log else 0.0)
    eval_metrics["active_query_ratio"] = float(active_query_ratio)
    eval_metrics["final_active_query_ratio"] = float(metrics_log[-1]["active_query_ratio"] if metrics_log else 0.0)
    eval_metrics["train_candidate_ratio"] = float(train_candidate_ratio)
    eval_metrics["final_train_candidate_ratio"] = float(
        metrics_log[-1]["train_candidate_ratio"] if metrics_log else 0.0
    )
    eval_metrics["train_samples_seen"] = total_train_samples
    eval_metrics["teacher_prediction_queries"] = total_prediction_queries
    eval_metrics["teacher_geometry_queries"] = total_geometry_queries
    eval_metrics["teacher_geometry_examples"] = int(geometry_ledger.get("geometry_examples", 0))
    eval_metrics["teacher_geometry_forward_invocations"] = int(geometry_ledger.get("geometry_forward_invocations", 0))
    eval_metrics["teacher_geometry_forward_invocations_current_job"] = (
        int(geometry_ledger.get("geometry_forward_invocations", 0))
        if geometry_cache_build is not None and not geometry_cache_build.cache_hit else 0
    )
    eval_metrics["teacher_prediction_forward_invocations"] = total_prediction_forward_invocations
    eval_metrics["teacher_prediction_seconds"] = total_prediction_teacher_seconds
    eval_metrics["teacher_total_seconds"] = float(geometry_ledger.get("teacher_seconds", 0.0)) + total_prediction_teacher_seconds
    eval_metrics["teacher_total_forward_invocations"] = (
        int(geometry_ledger.get("geometry_forward_invocations", 0)) + total_prediction_forward_invocations
    )
    eval_metrics["teacher_total_examples"] = total_geometry_queries + total_prediction_queries
    eval_metrics["geometry_cache_dir"] = str(geometry_cache_build.root) if geometry_cache_build is not None else None
    eval_metrics["geometry_cache_bytes"] = int(geometry_ledger.get("cache_bytes", 0))
    eval_metrics["geometry_cache_hit"] = bool(geometry_ledger.get("cache_hit", False))
    eval_metrics["geometry_cache_teacher_seconds"] = float(geometry_ledger.get("teacher_seconds", 0.0))
    eval_metrics["geometry_cache_write_seconds"] = float(geometry_ledger.get("cache_write_seconds", 0.0))
    eval_metrics["geometry_cache_dataloader_wait_seconds"] = float(geometry_ledger.get("dataloader_wait_seconds", 0.0))
    eval_metrics["geometry_cache_build_wall_seconds"] = float(geometry_ledger.get("total_wall_seconds", 0.0))
    eval_metrics["geometry_cache_current_job_wall_seconds"] = float(cache_current_job_wall)
    eval_metrics["geometry_cache_read_metrics"] = dict(geometry_cache.io_metrics) if geometry_cache is not None else {}
    eval_metrics["semantic_embedding_metadata"] = semantic_embedding_metadata
    eval_metrics["teacher_geometry_forward_ratio"] = total_geometry_queries / max(1, total_train_samples)
    eval_metrics["teacher_geometry_forward_invocation_ratio"] = (
        int(geometry_ledger.get("geometry_forward_invocations", 0)) / max(1, len(train_loader) * epochs)
    )
    eval_metrics["teacher_prediction_forward_invocation_ratio"] = (
        total_prediction_forward_invocations / max(1, len(train_loader) * epochs)
    )
    eval_metrics["teacher_total_forward_invocation_ratio"] = (
        eval_metrics["teacher_total_forward_invocations"] / max(1, len(train_loader) * epochs)
    )
    eval_metrics["teacher_total_example_ratio"] = (total_geometry_queries + total_prediction_queries) / max(1, total_train_samples)
    # A single method job cannot calculate Teacher-Query Efficiency because
    # the seed-matched Student baseline may still be running independently.
    # Record only the method's raw performance-normalized quantity here.
    # aggregate_results.py later pairs this run with the matching Student run
    # and computes the scientifically defined AccGain/TrainQueries metric.
    if train_query_ratio > 0.0:
        if task_type == "classification":
            eval_metrics["performance_per_train_query_ratio"] = float(
                eval_metrics["accuracy"] / train_query_ratio
            )
            eval_metrics["performance_per_train_query_ratio_metric"] = "accuracy"
        else:
            eval_metrics["performance_per_train_query_ratio"] = float(
                (1.0 / eval_metrics["perplexity"]) / train_query_ratio
            )
            eval_metrics["performance_per_train_query_ratio_metric"] = "inverse_perplexity"
    else:
        eval_metrics["performance_per_train_query_ratio"] = None
        eval_metrics["performance_per_train_query_ratio_metric"] = (
            "accuracy" if task_type == "classification" else "inverse_perplexity"
        )
    eval_metrics["teacher_query_efficiency"] = None
    eval_metrics["teacher_query_efficiency_status"] = (
        "Deferred to aggregate_results.py, which pairs this run with the "
        "seed-matched Student baseline."
    )
    eval_metrics["logits_distill_possible"] = bool(logits_distill_possible)
    eval_metrics["task_type"] = task_type
    eval_metrics["student_lm_objective"] = lm_objective_s
    eval_metrics["teacher_lm_objective"] = lm_objective_t
    eval_metrics["per_device_train_batch_size"] = int(train_bs)
    eval_metrics["per_device_eval_batch_size"] = int(eval_bs)
    eval_metrics["gradient_accumulation_steps"] = int(CFG.gradient_accumulation_steps)
    eval_metrics["effective_train_batch_size"] = int(
        train_bs * CFG.gradient_accumulation_steps * accelerator.num_processes
    )
    if torch.cuda.is_available():
        eval_metrics["gpu_name"] = torch.cuda.get_device_name(accelerator.device)
        eval_metrics["peak_gpu_memory_allocated_gib"] = float(
            torch.cuda.max_memory_allocated(accelerator.device) / (1024 ** 3)
        )
        eval_metrics["peak_gpu_memory_reserved_gib"] = float(
            torch.cuda.max_memory_reserved(accelerator.device) / (1024 ** 3)
        )

    out_dir = make_method_output_dir(dataset_name, scenario_name, method, budget, seed)
    if CFG.generate_evaluation_plots:
        report_start = time.perf_counter()
        artifacts = collect_evaluation_artifacts(
            accelerator=accelerator,
            teacher_model=teacher_model,
            student_model=student_model,
            dataloader=eval_loader,
            task_type=task_type,
            lm_objective_s=lm_objective_s,
        )
        report_metrics = generate_evaluation_reports(
            out_dir=out_dir,
            dataset_name=dataset_name,
            task_type=task_type,
            tokenizer=student_tokenizer,
            log_df=log_df,
            artifacts=artifacts,
            student_model=accelerator.unwrap_model(student_model),
            run_summary={
                key: eval_metrics[key]
                for key in [
                    "accuracy",
                    "macro_f1",
                    "eval_loss",
                    "perplexity",
                    "query_ratio",
                    "train_query_ratio",
                    "final_train_query_ratio",
                    "active_query_ratio",
                    "final_active_query_ratio",
                    "train_candidate_ratio",
                    "final_train_candidate_ratio",
                    "train_samples_seen",
                    "teacher_prediction_queries",
                    "teacher_geometry_queries",
                    "performance_per_train_query_ratio",
                    "teacher_query_efficiency",
                ]
                if key in eval_metrics
            },
        )
        eval_metrics.update(report_metrics)
        eval_metrics["report_runtime_sec"] = float(time.perf_counter() - report_start)

    if accelerator.is_main_process:
        ensure_dir(out_dir)
        log_df.to_csv(os.path.join(out_dir, "train_log.csv"), index=False)
        write_json_atomic(os.path.join(out_dir, "eval_metrics.json"), eval_metrics)
        # Save final student checkpoint.
        accelerator.unwrap_model(student_model).save_pretrained(os.path.join(out_dir, "student_model"))
        student_tokenizer.save_pretrained(os.path.join(out_dir, "student_model"))

    accelerator.wait_for_everyone()
    accelerator.free_memory(teacher_model, student_model, optimizer, train_loader, eval_loader, scheduler)
    return eval_metrics

# ============================================================
# Single-run entry point
# ============================================================

def build_accelerator() -> Accelerator:
    if CFG.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    accelerator = Accelerator(
        mixed_precision=CFG.mixed_precision,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        log_with=None,
    )
    if accelerator.num_processes != 1:
        raise RuntimeError(
            "This Trillium entry point expects one process and one GPU per SLURM job. "
            "Run it with python, not accelerate launch --multi_gpu."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Submit this program through the one-H100 SLURM script.")
    if CFG.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable on this GPU; set mixed_precision to fp16 or no.")
    return accelerator


def run_single_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    accelerator = build_accelerator()
    ensure_dir(CFG.output_root)
    ensure_dir(CFG.teacher_cache_root)

    dataset_name = args.dataset
    scenario_name = args.scenario
    teacher_name, student_name = SCENARIOS[scenario_name]
    teacher_source = resolve_model_source(teacher_name)
    student_source = resolve_model_source(student_name)
    task_type = get_task_type(dataset_name)

    main_print(accelerator, "=" * 72)
    main_print(accelerator, "GCBD-NLP | Trillium single-H100 run")
    main_print(accelerator, f"Mode: {args.mode}")
    main_print(accelerator, f"Dataset: {dataset_name} | Scenario: {scenario_name} | Seed: {args.seed}")
    main_print(accelerator, f"Method: {args.method} | Budget: {args.budget}")
    main_print(accelerator, f"Device: {torch.cuda.get_device_name(accelerator.device)}")
    main_print(accelerator, f"Mixed precision: {CFG.mixed_precision} | TF32: {CFG.allow_tf32}")
    main_print(accelerator, f"SLURM job: {os.environ.get('SLURM_JOB_ID', 'not-set')}")
    main_print(accelerator, "=" * 72)
    if "H100" not in torch.cuda.get_device_name(accelerator.device).upper():
        main_print(
            accelerator,
            "WARNING: the resolved GPU is not an H100; re-evaluate batch and precision settings.",
        )

    if args.mode == "teacher":
        cache_dir = make_teacher_cache_dir(
            dataset_name, scenario_name, teacher_name, task_type, args.seed
        )
        done_file = os.path.join(cache_dir, "DONE")
        if os.path.isfile(done_file) and not args.overwrite:
            main_print(accelerator, f"Teacher already complete; skipping: {cache_dir}")
            return
        CFG.require_finetuned_teacher = False
        CFG.train_teacher_if_missing = True
        CFG.reuse_finetuned_teacher = not args.overwrite
        lock_dir = cache_dir + ".LOCK"
        # The lock is a sibling of the final seed cache directory. On the first
        # run, its parent hierarchy does not exist yet, so create that hierarchy
        # before the atomic mkdir used to acquire the lock.
        ensure_dir(os.path.dirname(cache_dir))
        try:
            os.mkdir(lock_dir)
        except FileExistsError as exc:
            raise RuntimeError(
                f"Teacher cache lock already exists: {lock_dir}. Another teacher job may be running. "
                "If no such job exists, inspect and remove the stale lock before resubmitting."
            ) from exc
        owner_file = os.path.join(lock_dir, "OWNER")
        write_text_atomic(
            owner_file,
            f"slurm_job_id={os.environ.get('SLURM_JOB_ID', 'not-set')}\npid={os.getpid()}\n",
        )
        try:
            if accelerator.is_main_process:
                ensure_dir(cache_dir)
                write_json_atomic(os.path.join(cache_dir, "run_config.json"), resolved_run_spec(args))

            raw, eval_split = load_standardized_dataset(dataset_name, args.seed)
            teacher_tokenizer = prepare_tokenizer(teacher_source)
            teacher_model = train_or_load_teacher(
                accelerator=accelerator,
                dataset_name=dataset_name,
                scenario_name=scenario_name,
                teacher_name=teacher_name,
                teacher_tokenizer=teacher_tokenizer,
                raw=raw,
                eval_split=eval_split,
                task_type=task_type,
                seed=args.seed,
            )
            accelerator.free_memory(teacher_model)
            del teacher_model
            torch.cuda.empty_cache()
        finally:
            if os.path.isfile(owner_file):
                os.unlink(owner_file)
            if os.path.isdir(lock_dir):
                os.rmdir(lock_dir)
        main_print(accelerator, f"Teacher preparation complete: {cache_dir}")
        return

    out_dir = make_method_output_dir(
        dataset_name, scenario_name, args.method, args.budget, args.seed
    )
    done_file = os.path.join(out_dir, "DONE")
    if os.path.isfile(done_file) and not args.overwrite:
        main_print(accelerator, f"Method run already complete; skipping: {out_dir}")
        return

    CFG.require_finetuned_teacher = True
    CFG.train_teacher_if_missing = False
    CFG.reuse_finetuned_teacher = True
    if accelerator.is_main_process:
        ensure_dir(out_dir)
        write_json_atomic(os.path.join(out_dir, "run_config.json"), resolved_run_spec(args))

    raw, eval_split = load_standardized_dataset(dataset_name, args.seed)
    main_print(
        accelerator,
        f"Dataset task={task_type} | train={len(raw['train'])} | eval={len(raw[eval_split])}",
    )
    teacher_tokenizer = prepare_tokenizer(teacher_source)
    student_tokenizer = prepare_tokenizer(student_source)
    teacher_model = train_or_load_teacher(
        accelerator=accelerator,
        dataset_name=dataset_name,
        scenario_name=scenario_name,
        teacher_name=teacher_name,
        teacher_tokenizer=teacher_tokenizer,
        raw=raw,
        eval_split=eval_split,
        task_type=task_type,
        seed=args.seed,
    )
    metrics = train_student_method(
        accelerator=accelerator,
        dataset_name=dataset_name,
        scenario_name=scenario_name,
        teacher_name=teacher_name,
        student_name=student_name,
        method=args.method,
        budget=args.budget,
        seed=args.seed,
        raw=raw,
        eval_split=eval_split,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
    )
    row = {
        "dataset": dataset_name,
        "scenario": scenario_name,
        "teacher": teacher_name,
        "student": student_name,
        "method": args.method,
        "budget": args.budget,
        "seed": args.seed,
        **metrics,
    }
    if accelerator.is_main_process:
        write_json_atomic(os.path.join(out_dir, "result_row.json"), row)
        write_text_atomic(done_file, "ok\n")
    accelerator.wait_for_everyone()
    main_print(accelerator, f"Method run complete: {out_dir}")


def main() -> None:
    args = parse_cli_args()
    apply_runtime_overrides(args)
    if args.dry_run:
        print(json.dumps(resolved_run_spec(args), indent=2, sort_keys=True))
        return
    run_single_experiment(args)


if __name__ == "__main__":
    main()
