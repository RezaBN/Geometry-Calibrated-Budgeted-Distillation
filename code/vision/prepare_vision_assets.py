#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from torchvision import datasets, models

from gcbd_vision.config import SCENARIOS, load_config
from gcbd_vision.utils import atomic_json, ensure_dir, sha256_file


WEIGHTS = {
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
    "vit_b_16": models.ViT_B_16_Weights.IMAGENET1K_V1,
    "convnext_base": models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage public vision assets once before the Trillium experiment campaign"
    )
    parser.add_argument("--config", default="vision_experiments.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-cifar100", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    os.environ["TORCH_HOME"] = cfg["torch_home"]
    ensure_dir(cfg["torch_home"])

    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "models": [],
        "students": [],
        "datasets": [],
    }
    for architecture, weights in WEIGHTS.items():
        destination = Path(cfg["model_assets"][architecture])
        ensure_dir(destination.parent)
        if destination.exists() and not args.overwrite:
            print(f"Reusing {destination}")
        else:
            print(f"Downloading {architecture}: {weights.url}")
            state_dict = weights.get_state_dict(progress=True, check_hash=True)
            torch.save(
                {
                    "state_dict": state_dict,
                    "architecture": architecture,
                    "weights_enum": str(weights),
                    "source_url": weights.url,
                    "num_classes": 1000,
                },
                destination,
            )
        manifest["models"].append(
            {
                "architecture": architecture,
                "path": str(destination),
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
                "weights_enum": str(weights),
            }
        )

    # Student models intentionally start from random initialization. Their architecture
    # implementations are provided by torchvision/timm, so no pretrained student weight
    # files are downloaded or cached here. Record that design explicitly in the manifest.
    for architecture in sorted({spec["student"] for spec in SCENARIOS.values()}):
        manifest["students"].append(
            {
                "architecture": architecture,
                "pretrained": False,
                "downloaded_weights": False,
                "initialization": "random per experiment seed",
                "implementation_source": (
                    "timm" if architecture == "deit_tiny_patch16_224" else "torchvision"
                ),
            }
        )

    if not args.skip_cifar100:
        cifar_root = Path(cfg["dataset_paths"]["CIFAR100"])
        ensure_dir(cifar_root)
        train = datasets.CIFAR100(root=str(cifar_root), train=True, download=True)
        test = datasets.CIFAR100(root=str(cifar_root), train=False, download=True)
        manifest["datasets"].append(
            {
                "dataset": "CIFAR100",
                "path": str(cifar_root),
                "train_examples": len(train),
                "validation_examples": len(test),
                "classes": len(train.classes),
            }
        )

    imagenet_root = Path(cfg["dataset_paths"]["ImageNet"])
    manifest["datasets"].append(
        {
            "dataset": "ImageNet",
            "path": str(imagenet_root),
            "status": "user-staged" if (imagenet_root / "train").is_dir() else "not-staged",
            "note": "ImageNet is licensed and is not downloaded by this script.",
        }
    )
    manifest_path = Path(cfg["project_root"]) / "vision_asset_manifest.json"
    atomic_json(manifest_path, manifest)
    print(f"Asset manifest: {manifest_path}")


if __name__ == "__main__":
    main()
