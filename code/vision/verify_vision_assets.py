#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torchvision
import timm

from gcbd_vision.config import SCENARIOS, load_config
from gcbd_vision.data import verify_dataset
from gcbd_vision.models import FeatureModel, create_model, layer_names, load_local_pretrained_teacher
from gcbd_vision.utils import atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly verify all local vision assets offline")
    parser.add_argument("--config", default="vision_experiments.json")
    parser.add_argument("--skip-imagenet", action="store_true")
    parser.add_argument("--forward", action="store_true", help="Also run a CPU forward pass")
    args = parser.parse_args()
    cfg = load_config(args.config)
    os.environ["TORCH_HOME"] = cfg["torch_home"]
    report: dict[str, object] = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "models": [],
        "datasets": [],
    }

    teacher_architectures = sorted({spec["teacher"] for spec in SCENARIOS.values()})
    for architecture in teacher_architectures:
        path = Path(cfg["model_assets"][architecture])
        if not path.is_file():
            raise FileNotFoundError(f"Missing model asset: {path}")
        model = load_local_pretrained_teacher(architecture, str(path), 1000)
        model = FeatureModel(model, layer_names(architecture)).eval()
        if args.forward:
            with torch.inference_mode():
                output, features = model(torch.zeros(1, 3, 224, 224))
            if tuple(output.shape) != (1, 1000):
                raise RuntimeError(f"Unexpected {architecture} output shape: {tuple(output.shape)}")
            if len(features) != len(layer_names(architecture)):
                raise RuntimeError(f"Not all feature hooks fired for {architecture}")
        parameters = sum(parameter.numel() for parameter in model.parameters())
        entry = {
            "architecture": architecture,
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "parameters": parameters,
            "forward_checked": args.forward,
        }
        report["models"].append(entry)
        print(f"MODEL OK: {architecture} ({parameters:,} parameters)")
        del model

    student_architectures = sorted({spec["student"] for spec in SCENARIOS.values()})
    for architecture in student_architectures:
        model = FeatureModel(create_model(architecture, 1000), layer_names(architecture)).eval()
        if args.forward:
            with torch.inference_mode():
                output, features = model(torch.zeros(1, 3, 224, 224))
            if tuple(output.shape) != (1, 1000):
                raise RuntimeError(f"Unexpected {architecture} output shape: {tuple(output.shape)}")
            if len(features) != len(layer_names(architecture)):
                raise RuntimeError(f"Not all feature hooks fired for {architecture}")
        parameters = sum(parameter.numel() for parameter in model.parameters())
        report["models"].append(
            {
                "architecture": architecture,
                "role": "student-architecture",
                "parameters": parameters,
            }
        )
        print(f"MODEL OK: {architecture} scratch architecture ({parameters:,} parameters)")
        del model

    for dataset in cfg["datasets"]:
        if dataset == "ImageNet" and args.skip_imagenet:
            continue
        entry = verify_dataset(dataset, cfg["dataset_paths"][dataset])
        report["datasets"].append(entry)
        print(
            f"DATASET OK: {dataset} train={entry['train_examples']:,} "
            f"validation={entry['validation_examples']:,} classes={entry['num_classes']}"
        )

    report_path = Path(cfg["project_root"]) / "vision_assets_verified.json"
    atomic_json(report_path, report)
    print(f"Report: {report_path}")
    print("LOCAL VISION ASSET VERIFICATION PASSED")


if __name__ == "__main__":
    main()
