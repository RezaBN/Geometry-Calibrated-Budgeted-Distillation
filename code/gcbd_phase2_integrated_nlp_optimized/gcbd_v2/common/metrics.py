"""Representation diagnostics used in the geometry-validation experiments."""
from __future__ import annotations
import torch


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Linear CKA for paired observations [N,Dx] and [N,Dy]."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("CKA requires paired [N,D] matrices with N>=2")
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    y = y.float() - y.float().mean(dim=0, keepdim=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    numerator = cross.square().sum()
    denominator = torch.sqrt(xx.square().sum() * yy.square().sum()).clamp_min(eps)
    return numerator / denominator
