from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .utils import dataloader_generator, seed_worker


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def dataset_pair(
    dataset_name: str,
    root: str,
    image_size: int = 224,
) -> tuple[Any, Any, int, list[str]]:
    train_transform, eval_transform = build_transforms(image_size)
    root_path = Path(root)
    if dataset_name == "CIFAR100":
        train_set = datasets.CIFAR100(
            root=str(root_path), train=True, transform=train_transform, download=False
        )
        val_set = datasets.CIFAR100(
            root=str(root_path), train=False, transform=eval_transform, download=False
        )
        classes = list(train_set.classes)
    elif dataset_name == "ImageNet":
        train_dir = root_path / "train"
        val_dir = root_path / "val"
        if not val_dir.is_dir() and (root_path / "validation").is_dir():
            val_dir = root_path / "validation"
        if not train_dir.is_dir() or not val_dir.is_dir():
            raise FileNotFoundError(
                f"ImageNet must use ImageFolder layout: {train_dir}/<class>/... and "
                f"{root_path}/val/<class>/..."
            )
        train_set = datasets.ImageFolder(str(train_dir), transform=train_transform)
        val_set = datasets.ImageFolder(str(val_dir), transform=eval_transform)
        if train_set.class_to_idx != val_set.class_to_idx:
            raise ValueError("ImageNet train and validation class mappings do not match")
        classes = list(train_set.classes)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    num_classes = len(classes)
    expected = 100 if dataset_name == "CIFAR100" else 1000
    if num_classes != expected:
        raise ValueError(f"{dataset_name} has {num_classes} classes; expected {expected}")
    return train_set, val_set, num_classes, classes


def build_loaders(
    dataset_name: str,
    root: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    image_size: int = 224,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> tuple[DataLoader, DataLoader, int, list[str]]:
    train_set, val_set, num_classes, classes = dataset_pair(dataset_name, root, image_size)
    common: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        common["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=dataloader_generator(seed),
        **common,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        generator=dataloader_generator(seed + 1),
        **common,
    )
    return train_loader, val_loader, num_classes, classes


def verify_dataset(dataset_name: str, root: str) -> dict[str, Any]:
    train_set, val_set, num_classes, classes = dataset_pair(dataset_name, root, 224)
    train_image, train_label = train_set[0]
    val_image, val_label = val_set[0]
    return {
        "dataset": dataset_name,
        "root": str(Path(root).resolve()),
        "train_examples": len(train_set),
        "validation_examples": len(val_set),
        "num_classes": num_classes,
        "first_class": classes[0],
        "last_class": classes[-1],
        "train_tensor_shape": list(train_image.shape),
        "validation_tensor_shape": list(val_image.shape),
        "sample_labels": [int(train_label), int(val_label)],
    }
