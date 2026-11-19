from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

from .config import scenario_spec
from .utils import normalize_state_dict, torch_load


ARCH_FEATURE_LAYERS: dict[str, dict[str, list[str]]] = {
    "resnet18": {
        "early": ["layer1", "layer2", "layer3"],
        "final": ["layer4"],
    },
    "resnet50": {
        "early": ["layer1", "layer2", "layer3"],
        "final": ["layer4"],
    },
    "mobilenet_v2": {
        "early": ["features.3", "features.6", "features.13"],
        "final": ["features.18"],
    },
    "vit_b_16": {
        "early": [
            "encoder.layers.encoder_layer_2",
            "encoder.layers.encoder_layer_5",
            "encoder.layers.encoder_layer_8",
        ],
        "final": ["encoder.layers.encoder_layer_11"],
    },
    "deit_tiny_patch16_224": {
        "early": ["blocks.2", "blocks.5", "blocks.8"],
        "final": ["blocks.11"],
    },
    "convnext_base": {
        "early": ["features.1", "features.3", "features.5"],
        "final": ["features.7"],
    },
}


def create_model(architecture: str, num_classes: int) -> nn.Module:
    """Create an architecture without any implicit pretrained-weight lookup."""
    if architecture == "resnet18":
        model = models.resnet18(weights=None, num_classes=num_classes)
    elif architecture == "resnet50":
        model = models.resnet50(weights=None, num_classes=num_classes)
    elif architecture == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None, num_classes=num_classes)
    elif architecture == "vit_b_16":
        model = models.vit_b_16(weights=None, num_classes=num_classes, image_size=224)
    elif architecture == "convnext_base":
        model = models.convnext_base(weights=None, num_classes=num_classes)
    elif architecture == "deit_tiny_patch16_224":
        try:
            import timm
        except ImportError as exc:
            raise ImportError("DeiT-Tiny requires the timm package") from exc
        model = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=False,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    setattr(model, "_gcbd_architecture", architecture)
    return model


def replace_classifier(model: nn.Module, architecture: str, num_classes: int) -> None:
    if architecture in {"resnet18", "resnet50"}:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "mobilenet_v2":
        final = model.classifier[-1]
        model.classifier[-1] = nn.Linear(final.in_features, num_classes)
    elif architecture == "vit_b_16":
        final = model.heads.head
        model.heads.head = nn.Linear(final.in_features, num_classes)
    elif architecture == "convnext_base":
        final = model.classifier[-1]
        model.classifier[-1] = nn.Linear(final.in_features, num_classes)
    elif architecture == "deit_tiny_patch16_224":
        model.reset_classifier(num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    setattr(model, "_gcbd_architecture", architecture)


def load_local_pretrained_teacher(
    architecture: str,
    asset_path: str,
    num_classes: int,
) -> nn.Module:
    """Load a local ImageNet-1K state dict and replace its head when needed."""
    model = create_model(architecture, 1000)
    payload = torch_load(asset_path, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dict in {asset_path}")
    missing, unexpected = model.load_state_dict(normalize_state_dict(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Pretrained asset mismatch for {architecture}: missing={missing}, unexpected={unexpected}"
        )
    if num_classes != 1000:
        replace_classifier(model, architecture, num_classes)
    return model


def load_cached_teacher(
    architecture: str,
    checkpoint_path: str,
    num_classes: int,
) -> nn.Module:
    model = create_model(architecture, num_classes)
    payload = torch_load(checkpoint_path, map_location="cpu")
    state = payload.get("model_state", payload.get("state_dict", payload))
    missing, unexpected = model.load_state_dict(normalize_state_dict(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Fine-tuned teacher mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model


def build_scenario_models(
    scenario: str,
    num_classes: int,
    teacher_checkpoint: str,
) -> tuple[nn.Module, nn.Module, str, str]:
    spec = scenario_spec(scenario)
    teacher_arch = spec["teacher"]
    student_arch = spec["student"]
    teacher = load_cached_teacher(teacher_arch, teacher_checkpoint, num_classes)
    student = create_model(student_arch, num_classes)
    return teacher, student, teacher_arch, student_arch


def layer_names(architecture: str, include_early: bool = True) -> list[str]:
    mapping = ARCH_FEATURE_LAYERS[architecture]
    names = list(mapping["final"])
    if include_early:
        names = list(mapping["early"]) + names
    return list(dict.fromkeys(names))


def early_layer_names(architecture: str) -> list[str]:
    return list(ARCH_FEATURE_LAYERS[architecture]["early"])


def final_layer_name(architecture: str) -> str:
    return ARCH_FEATURE_LAYERS[architecture]["final"][0]


class FeatureModel(nn.Module):
    """Return logits plus selected hook outputs from a single-GPU model."""

    def __init__(self, model: nn.Module, layers: Sequence[str]):
        super().__init__()
        self.model = model
        self.layers = list(dict.fromkeys(layers))
        available = dict(model.named_modules())
        missing = [name for name in self.layers if name not in available]
        if missing:
            raise ValueError(
                f"Feature layers {missing} not found in {type(model).__name__}; "
                f"this usually means torchvision/timm is outside the tested range"
            )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        captured: dict[str, torch.Tensor] = {}
        handles: list[Any] = []
        modules = dict(self.model.named_modules())

        def make_hook(name: str):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                if isinstance(output, (tuple, list)):
                    output = output[0]
                if not torch.is_tensor(output):
                    raise TypeError(f"Hook {name} returned {type(output).__name__}, not a tensor")
                captured[name] = output

            return hook

        for name in self.layers:
            handles.append(modules[name].register_forward_hook(make_hook(name)))
        try:
            logits = self.model(inputs)
        finally:
            for handle in handles:
                handle.remove()
        return logits, captured


def freeze(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def total_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
