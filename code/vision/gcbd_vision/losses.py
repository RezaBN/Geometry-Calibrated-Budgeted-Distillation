from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def feature_vector(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim == 4:
        return F.adaptive_avg_pool2d(feature, 1).flatten(1)
    if feature.ndim == 3:
        return feature.mean(dim=1)
    if feature.ndim == 2:
        return feature
    return feature.reshape(feature.shape[0], -1)


def align_vectors(
    student: torch.Tensor, teacher: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    student = feature_vector(student)
    teacher = feature_vector(teacher)
    count = min(student.shape[0], teacher.shape[0])
    width = min(student.shape[1], teacher.shape[1])
    return student[:count, :width], teacher[:count, :width]


def attention_vector(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim == 4:
        attention = feature.float().pow(2).mean(dim=1).flatten(1)
    elif feature.ndim == 3:
        attention = feature.float().pow(2).mean(dim=-1)
    else:
        attention = feature_vector(feature).float().pow(2)
    return F.normalize(attention, p=2, dim=1)


def align_attention(
    student: torch.Tensor, teacher: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    student = attention_vector(student)
    teacher = attention_vector(teacher)
    count = min(student.shape[0], teacher.shape[0])
    width = min(student.shape[1], teacher.shape[1])
    return student[:count, :width], teacher[:count, :width]


def kd_loss(logits_s: torch.Tensor, logits_t: torch.Tensor, temperature: float) -> torch.Tensor:
    log_s = F.log_softmax(logits_s.float() / temperature, dim=1)
    prob_t = F.softmax(logits_t.float() / temperature, dim=1)
    return F.kl_div(log_s, prob_t, reduction="batchmean") * temperature**2


def fitnet_loss(feature_s: torch.Tensor, feature_t: torch.Tensor) -> torch.Tensor:
    student, teacher = align_vectors(feature_s, feature_t)
    return F.mse_loss(student, teacher)


def attention_transfer_loss(feature_s: torch.Tensor, feature_t: torch.Tensor) -> torch.Tensor:
    student, teacher = align_attention(feature_s, feature_t)
    return F.mse_loss(student, teacher)


def crd_loss(
    feature_s: torch.Tensor,
    feature_t: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    student, teacher = align_vectors(feature_s, feature_t)
    student = F.normalize(student, p=2, dim=1)
    teacher = F.normalize(teacher, p=2, dim=1)
    logits = student @ teacher.t() / temperature
    labels = torch.arange(student.shape[0], device=student.device)
    return F.cross_entropy(logits, labels)


def dkd_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Decoupled KD with target-class and non-target-class components."""
    target_mask = torch.zeros_like(logits_s, dtype=torch.bool).scatter_(
        1, target.reshape(-1, 1), True
    )
    prob_s = F.softmax(logits_s.float() / temperature, dim=1)
    prob_t = F.softmax(logits_t.float() / temperature, dim=1)

    target_s = (prob_s * target_mask).sum(dim=1, keepdim=True)
    other_s = (prob_s * ~target_mask).sum(dim=1, keepdim=True)
    target_t = (prob_t * target_mask).sum(dim=1, keepdim=True)
    other_t = (prob_t * ~target_mask).sum(dim=1, keepdim=True)
    binary_s = torch.cat([target_s, other_s], dim=1).clamp_min(1e-12)
    binary_t = torch.cat([target_t, other_t], dim=1).clamp_min(1e-12)
    target_component = F.kl_div(binary_s.log(), binary_t, reduction="batchmean")

    masked_s = logits_s.float() / temperature - target_mask.float() * 1000.0
    masked_t = logits_t.float() / temperature - target_mask.float() * 1000.0
    non_target_component = F.kl_div(
        F.log_softmax(masked_s, dim=1),
        F.softmax(masked_t, dim=1),
        reduction="batchmean",
    )
    return (alpha * target_component + beta * non_target_component) * temperature**2


def reviewkd_loss(
    features_s: list[torch.Tensor], features_t: list[torch.Tensor]
) -> torch.Tensor:
    losses = [fitnet_loss(student, teacher) for student, teacher in zip(features_s, features_t)]
    if not losses:
        raise ValueError("ReviewKD requires at least one feature pair")
    return torch.stack(losses).mean()


def dist_loss(feature_s: torch.Tensor, feature_t: torch.Tensor) -> torch.Tensor:
    student, teacher = align_vectors(feature_s, feature_t)
    student = F.normalize(student, p=2, dim=1)
    teacher = F.normalize(teacher, p=2, dim=1)
    return F.mse_loss(student @ student.t(), teacher @ teacher.t())


def _coarsen_probabilities(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    max_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the shared top classes and aggregate the tail into an 'other' bin.

    This bounds Sinkhorn memory on ImageNet while retaining probability mass.
    CIFAR-100 is exact whenever max_classes >= 100.
    """
    prob_s = F.softmax(logits_s.float(), dim=1)
    prob_t = F.softmax(logits_t.float(), dim=1)
    class_count = prob_s.shape[1]
    if class_count <= max_classes:
        return prob_s, prob_t
    importance = 0.5 * (prob_s + prob_t)
    indices = importance.topk(max_classes, dim=1).indices
    selected_s = prob_s.gather(1, indices)
    selected_t = prob_t.gather(1, indices)
    tail_s = (1.0 - selected_s.sum(dim=1, keepdim=True)).clamp_min(0.0)
    tail_t = (1.0 - selected_t.sum(dim=1, keepdim=True)).clamp_min(0.0)
    return torch.cat([selected_s, tail_s], dim=1), torch.cat([selected_t, tail_t], dim=1)


def _sinkhorn_cost(
    prob_s: torch.Tensor,
    prob_t: torch.Tensor,
    epsilon: float,
    iterations: int,
    tau_teacher: float = 1.0,
    tau_student: float = 1.0,
) -> torch.Tensor:
    batch_size, class_count = prob_s.shape
    dtype = prob_s.dtype
    device = prob_s.device
    cost = 1.0 - torch.eye(class_count, device=device, dtype=dtype)
    cost = cost.unsqueeze(0).expand(batch_size, -1, -1)
    f = torch.zeros_like(prob_t)
    g = torch.zeros_like(prob_s)
    for _ in range(iterations):
        f = tau_teacher * epsilon * (
            torch.log(prob_t.clamp_min(1e-12))
            - torch.logsumexp((g.unsqueeze(1) - cost) / epsilon, dim=2)
        )
        g = tau_student * epsilon * (
            torch.log(prob_s.clamp_min(1e-12))
            - torch.logsumexp((f.unsqueeze(2) - cost) / epsilon, dim=1)
        )
    plan = torch.exp((f.unsqueeze(2) + g.unsqueeze(1) - cost) / epsilon)
    return (plan * cost).sum(dim=(1, 2)).mean()


def otkd_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    epsilon: float,
    iterations: int,
    max_classes: int,
) -> torch.Tensor:
    prob_s, prob_t = _coarsen_probabilities(logits_s, logits_t, max_classes)
    return _sinkhorn_cost(prob_s, prob_t, epsilon, iterations)


def uot_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    epsilon: float,
    iterations: int,
    tau_teacher: float,
    tau_student: float,
    max_classes: int,
) -> torch.Tensor:
    prob_s, prob_t = _coarsen_probabilities(logits_s, logits_t, max_classes)
    return _sinkhorn_cost(
        prob_s,
        prob_t,
        epsilon,
        iterations,
        tau_teacher=tau_teacher,
        tau_student=tau_student,
    )


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits.float(), dim=1)
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)


def curriculum_frontier(epoch: int, total_epochs: int, num_classes: int) -> float:
    max_entropy = math.log(num_classes)
    start = 0.2 * max_entropy
    if total_epochs <= 1:
        return max_entropy
    progress = min(max(epoch / (total_epochs - 1), 0.0), 1.0)
    return start + progress * (max_entropy - start)


def select_active_mask(
    entropy: torch.Tensor,
    budget: float,
    gamma: float,
    curriculum_enabled: bool,
    no_budget: bool,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if curriculum_enabled:
        candidate = entropy <= gamma
    else:
        candidate = torch.ones_like(entropy, dtype=torch.bool)
    candidate_indices = candidate.nonzero(as_tuple=False).flatten()
    active = torch.zeros_like(candidate)
    if candidate_indices.numel() == 0:
        return active, candidate, float("nan")
    if no_budget:
        active[candidate_indices] = True
    else:
        count = max(1, math.ceil(float(budget) * candidate_indices.numel()))
        values = entropy[candidate_indices]
        selected_local = torch.topk(values, min(count, values.numel()), largest=True).indices
        active[candidate_indices[selected_local]] = True
    eta = float(entropy[active].min().item()) if active.any() else float("nan")
    return active, candidate, eta


def sliced_wasserstein_attention_loss(
    feature_s: torch.Tensor,
    feature_t: torch.Tensor,
    projections: int,
) -> torch.Tensor:
    student, teacher = align_attention(feature_s, feature_t)
    width = student.shape[1]
    directions = torch.randn(projections, width, device=student.device, dtype=student.dtype)
    directions = F.normalize(directions, p=2, dim=1)
    student_projection = student @ directions.t()
    teacher_projection = teacher @ directions.t()
    student_projection = student_projection.sort(dim=0).values
    teacher_projection = teacher_projection.sort(dim=0).values
    return F.mse_loss(student_projection, teacher_projection)
