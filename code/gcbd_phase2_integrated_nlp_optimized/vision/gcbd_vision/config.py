from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


SCENARIOS = {
    "A": {"teacher": "resnet50", "student": "resnet18"},
    "B": {"teacher": "resnet50", "student": "mobilenet_v2"},
    "C": {"teacher": "vit_b_16", "student": "deit_tiny_patch16_224"},
    "D": {"teacher": "convnext_base", "student": "resnet18"},
}

METHODS = [
    "Student",
    "KD",
    "FitNet",
    "AT",
    "CRD",
    "DKD",
    "ReviewKD",
    "DIST",
    "OTKD",
    "GCBD-Full",
    "GCBD-NoCurriculum",
    "GCBD-NoBudget",
    "GCBD-NoStage1",
    "GCBD-NoUOT",
]


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    def expand(value: Any) -> Any:
        if isinstance(value, str):
            return os.path.expandvars(os.path.expanduser(value))
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    cfg = expand(cfg)
    cfg["_config_path"] = str(config_path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = [
        "project_root",
        "venv",
        "dataset_paths",
        "model_assets",
        "teacher_cache_root",
        "output_root",
        "logs_root",
        "datasets",
        "scenarios",
        "methods",
        "seeds",
        "runtime",
        "slurm",
    ]
    missing = [name for name in required if name not in cfg]
    if missing:
        raise ValueError(f"Configuration is missing keys: {missing}")

    unknown_scenarios = sorted(set(cfg["scenarios"]) - set(SCENARIOS))
    unknown_methods = sorted(set(cfg["methods"]) - set(METHODS))
    if unknown_scenarios:
        raise ValueError(f"Unknown scenarios: {unknown_scenarios}")
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")
    if set(cfg["datasets"]) - {"CIFAR100", "ImageNet"}:
        raise ValueError("Only CIFAR100 and ImageNet are supported by this package")
    if not 0.0 < float(cfg["runtime"]["gcbd_budget"]) <= 1.0:
        raise ValueError("runtime.gcbd_budget must be in (0, 1]")


def scenario_spec(scenario: str) -> dict[str, str]:
    key = scenario.upper()
    if key not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose A, B, C, or D")
    return dict(SCENARIOS[key])


def teacher_cache_dir(
    cfg: dict[str, Any], dataset: str, architecture: str, seed: int
) -> Path:
    return Path(cfg["teacher_cache_root"]) / dataset / architecture / f"seed_{seed}"


def method_output_dir(
    output_root: str,
    dataset: str,
    scenario: str,
    method: str,
    budget: float,
    seed: int,
) -> Path:
    budget_token = f"{budget:g}".replace(".", "p")
    method_token = method if not method.startswith("GCBD") else f"{method}_B{budget_token}"
    return Path(output_root) / dataset / f"Scenario_{scenario}" / method_token / f"seed_{seed}"


def dataset_value(mapping: dict[str, Any], dataset: str, default: Any = None) -> Any:
    if dataset in mapping:
        return mapping[dataset]
    if "default" in mapping:
        return mapping["default"]
    if default is not None:
        return default
    raise KeyError(f"No value configured for dataset {dataset}")


def batch_size_for(cfg: dict[str, Any], dataset: str, scenario: str, mode: str) -> int:
    runtime = cfg["runtime"]
    rules = runtime.get("batch_size_rules", [])
    candidates: list[tuple[int, int]] = []
    for rule in rules:
        if rule.get("dataset", dataset) != dataset:
            continue
        if rule.get("scenario", scenario) != scenario:
            continue
        if rule.get("mode", mode) != mode:
            continue
        specificity = sum(k in rule for k in ("dataset", "scenario", "mode"))
        candidates.append((specificity, int(rule["batch_size"])))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    key = "teacher_batch_size" if mode == "teacher" else "method_batch_size"
    return int(dataset_value(runtime[key], dataset))


def epochs_for(cfg: dict[str, Any], dataset: str, mode: str) -> int:
    key = "teacher_epochs" if mode == "teacher" else "method_epochs"
    return int(dataset_value(cfg["runtime"][key], dataset))


def slurm_time_for(
    cfg: dict[str, Any],
    mode: str,
    dataset: str,
    scenario: str | None = None,
    method: str | None = None,
) -> str:
    slurm = cfg["slurm"]
    default_key = "teacher_time" if mode == "teacher" else "method_time"
    selected = str(dataset_value(slurm[default_key], dataset))
    best_specificity = -1
    for rule in slurm.get("time_rules", []):
        if rule.get("mode", mode) != mode:
            continue
        if rule.get("dataset", dataset) != dataset:
            continue
        if scenario is not None and rule.get("scenario", scenario) != scenario:
            continue
        if method is not None and rule.get("method", method) != method:
            continue
        if scenario is None and "scenario" in rule:
            continue
        if method is None and "method" in rule:
            continue
        specificity = sum(k in rule for k in ("mode", "dataset", "scenario", "method"))
        if specificity >= best_specificity:
            selected = str(rule["time"])
            best_specificity = specificity
    return selected
