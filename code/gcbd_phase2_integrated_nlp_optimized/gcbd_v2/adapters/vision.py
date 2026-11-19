"""CNN/ViT geometry adapters with normalized native spatial supports."""
from __future__ import annotations
import math
import torch
from gcbd_v2.common.geometry import energy_measure, spatial_grid_support


def vision_feature_measure(feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-image energy measures and normalized (x,y) support.

    CNN maps [B,C,H,W] use their native HxW cell centres. ViT sequences
    [B,1+P,D] are interpreted as [CLS, patch tokens] and require square P.
    The CLS token is excluded because it has no spatial coordinate.
    """
    if feature.ndim == 4:
        _, _, h, w = feature.shape
        support = spatial_grid_support(h, w, device=feature.device)
        return energy_measure(feature, support)
    if feature.ndim == 3:
        b, n, d = feature.shape
        patch_count = n - 1
        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise ValueError(
                f"ViT feature sequence must contain one CLS token plus a square patch grid; got N={n}"
            )
        patch_features = feature[:, 1:, :]
        support = spatial_grid_support(side, side, device=feature.device)
        return energy_measure(patch_features, support)
    raise ValueError(f"Unsupported vision feature shape: {tuple(feature.shape)}")


def vision_sample_geometry(features: dict[str, torch.Tensor], layer_names: list[str]) -> list[dict]:
    """Convert hooked teacher features to per-sample cache records."""
    if not layer_names:
        raise ValueError("At least one geometry layer is required")
    batch_sizes = {int(features[name].shape[0]) for name in layer_names}
    if len(batch_sizes) != 1:
        raise ValueError("Teacher feature layers have inconsistent batch sizes")
    batch_size = batch_sizes.pop()
    records = [{"layers": {}} for _ in range(batch_size)]
    for name in layer_names:
        weights, support = vision_feature_measure(features[name])
        for b in range(batch_size):
            records[b]["layers"][name] = {
                "weights": weights[b].detach().to(torch.float16).cpu(),
                "support": support[b].detach().to(torch.float16).cpu(),
            }
    return records
