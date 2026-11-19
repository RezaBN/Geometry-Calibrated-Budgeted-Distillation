from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import (
    batch_size_for,
    dataset_value,
    epochs_for,
    method_output_dir,
    scenario_spec,
    teacher_cache_dir,
)
from .data import build_loaders
from .losses import (
    attention_transfer_loss,
    crd_loss,
    curriculum_frontier,
    dist_loss,
    dkd_loss,
    fitnet_loss,
    kd_loss,
    otkd_loss,
    reviewkd_loss,
    select_active_mask,
    sliced_wasserstein_attention_loss,
    uot_loss,
)
from .metrics import autocast_context, evaluate_classifier
from .models import (
    FeatureModel,
    build_scenario_models,
    create_model,
    early_layer_names,
    final_layer_name,
    freeze,
    layer_names,
    load_local_pretrained_teacher,
    total_parameter_count,
    trainable_parameter_count,
)
from .reporting import save_evaluation_diagnostics, save_training_history
from .utils import atomic_json, atomic_text, ensure_dir, set_seed, slurm_metadata, torch_load


class RunLock:
    def __init__(self, output_dir: Path, clear_stale: bool = False):
        self.output_dir = output_dir
        self.path = output_dir / ".LOCK"
        self.clear_stale = clear_stale

    def __enter__(self):
        ensure_dir(self.output_dir)
        if self.path.exists():
            active, details = self._is_active()
            if active and not self.clear_stale:
                raise RuntimeError(
                    f"Run lock already exists at {self.path}. If the prior job is no longer "
                    f"running, resubmit with --clear-stale-lock. Lock details: {details}"
                )
            print(f"Removing stale run lock: {self.path}", flush=True)
            self.path.unlink()
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({**slurm_metadata(), "pid": os.getpid(), "created_unix": time.time()}, handle)
            handle.write("\n")
        return self

    def _is_active(self) -> tuple[bool, str]:
        raw = self.path.read_text(encoding="utf-8", errors="replace")
        try:
            details = json.loads(raw)
        except json.JSONDecodeError:
            return True, raw
        job_id = details.get("slurm_job_id")
        if job_id:
            try:
                completed = subprocess.run(
                    ["squeue", "--noheader", "--jobs", str(job_id), "--format=%i"],
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired):
                return True, raw
            if completed.returncode != 0:
                return True, raw
            return bool(completed.stdout.strip()), raw
        lock_host = details.get("hostname")
        pid = details.get("pid")
        if lock_host == os.environ.get("HOSTNAME") and isinstance(pid, int):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False, raw
            except PermissionError:
                return True, raw
            return True, raw
        return True, raw

    def __exit__(self, exc_type, exc_value, traceback):
        if self.path.exists():
            self.path.unlink()


def _save_torch_atomic(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _device_and_runtime(cfg: dict[str, Any]) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; this runner is intended for a Trillium H100 job")
    device = torch.device("cuda", 0)
    runtime = cfg["runtime"]
    if runtime.get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    precision = str(runtime.get("mixed_precision", "bf16"))
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("mixed_precision=bf16, but this GPU/PyTorch build lacks BF16 support")
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("mixed_precision must be bf16, fp16, or fp32")
    return device, precision


def _software_versions() -> dict[str, str]:
    import torchvision

    try:
        import timm

        timm_version = str(timm.__version__)
    except ImportError:
        timm_version = "not-installed"
    return {
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "timm": timm_version,
    }


def _make_scaler(precision: str):
    enabled = precision == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _cosine_scheduler(
    optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def schedule(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return max(1e-6, (epoch + 1) / warmup_epochs)
        denominator = max(1, total_epochs - warmup_epochs)
        progress = min(max((epoch - warmup_epochs) / denominator, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _teacher_optimizer(
    cfg: dict[str, Any],
    model: nn.Module,
    architecture: str,
    dataset: str,
    batch_size: int,
) -> torch.optim.Optimizer:
    runtime = cfg["runtime"]
    learning_rate = float(dataset_value(runtime["teacher_lr"][architecture], dataset))
    if architecture == "resnet50":
        learning_rate *= batch_size / float(runtime.get("cnn_lr_reference_batch", 256))
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=float(runtime.get("momentum", 0.9)),
            weight_decay=float(runtime["weight_decay_cnn"]),
            nesterov=True,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(runtime["weight_decay_transformer"]),
    )


def _loader_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    runtime = cfg["runtime"]
    return {
        "num_workers": int(runtime["num_workers"]),
        "image_size": int(runtime["image_size"]),
        "pin_memory": bool(runtime.get("pin_memory", True)),
        "persistent_workers": bool(runtime.get("persistent_workers", True)),
        "prefetch_factor": int(runtime.get("prefetch_factor", 2)),
    }


def _print_header(mode: str, dataset: str, scenario: str, seed: int, extra: str = "") -> None:
    print("=" * 80, flush=True)
    print("GCBD Vision | Trillium single-H100 run", flush=True)
    print(f"Mode: {mode} | Dataset: {dataset} | Scenario: {scenario} | Seed: {seed}", flush=True)
    if extra:
        print(extra, flush=True)
    print(f"SLURM job: {os.environ.get('SLURM_JOB_ID', 'not-under-slurm')}", flush=True)
    print("=" * 80, flush=True)


def _remove_completed_marker(output_dir: Path) -> None:
    marker = output_dir / "DONE"
    if marker.exists():
        marker.unlink()


def run_teacher(
    cfg: dict[str, Any],
    dataset: str,
    scenario: str,
    seed: int,
    epochs: int | None = None,
    overwrite: bool = False,
    resume: bool = True,
    clear_stale_lock: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    spec = scenario_spec(scenario)
    architecture = spec["teacher"]
    output_dir = teacher_cache_dir(cfg, dataset, architecture, seed)
    done_path = output_dir / "DONE"
    configured_epochs = epochs_for(cfg, dataset, "teacher") if epochs is None else int(epochs)
    batch_size = batch_size_for(cfg, dataset, scenario, "teacher")
    plan = {
        "mode": "teacher",
        "dataset": dataset,
        "scenario": scenario,
        "teacher_architecture": architecture,
        "seed": seed,
        "epochs": configured_epochs,
        "batch_size": batch_size,
        "pretrained_asset": cfg["model_assets"][architecture],
        "output_dir": str(output_dir),
        "teacher_checkpoint": str(output_dir / "teacher_state.pt"),
    }
    if dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    if done_path.exists() and not overwrite:
        print(f"Teacher already complete: {done_path}", flush=True)
        return json.loads((output_dir / "teacher_metrics.json").read_text(encoding="utf-8"))
    if overwrite:
        _remove_completed_marker(output_dir)

    _print_header("teacher", dataset, scenario, seed, f"Teacher: {architecture}")
    with RunLock(output_dir, clear_stale=clear_stale_lock):
        set_seed(seed, bool(cfg["runtime"].get("deterministic", False)))
        device, precision = _device_and_runtime(cfg)
        train_loader, val_loader, num_classes, classes = build_loaders(
            dataset,
            cfg["dataset_paths"][dataset],
            batch_size,
            seed=seed,
            **_loader_kwargs(cfg),
        )
        asset_path = Path(cfg["model_assets"][architecture])
        if not asset_path.is_file():
            raise FileNotFoundError(
                f"Local pretrained asset missing: {asset_path}. Run prepare_vision_assets.py first."
            )
        model = load_local_pretrained_teacher(architecture, str(asset_path), num_classes).to(device)
        optimizer = _teacher_optimizer(cfg, model, architecture, dataset, batch_size)
        scheduler = _cosine_scheduler(
            optimizer, configured_epochs, int(cfg["runtime"].get("warmup_epochs", 0))
        )
        scaler = _make_scaler(precision)
        checkpoint_path = output_dir / "checkpoint.pt"
        start_epoch = 0
        history: list[dict[str, Any]] = []
        if resume and checkpoint_path.is_file() and not overwrite:
            checkpoint = torch_load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = int(checkpoint["next_epoch"])
            history = list(checkpoint.get("history", []))
            print(f"Resuming teacher at epoch {start_epoch}", flush=True)

        run_start = time.perf_counter()
        for epoch in range(start_epoch, configured_epochs):
            model.train()
            loss_sum = 0.0
            correct = 0
            examples = 0
            epoch_start = time.perf_counter()
            for step, (images, labels) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, precision):
                    logits = model(images)
                    loss = F.cross_entropy(
                        logits,
                        labels,
                        label_smoothing=float(cfg["runtime"].get("label_smoothing", 0.0)),
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip = float(cfg["runtime"].get("gradient_clip_norm", 0.0))
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                batch = labels.shape[0]
                loss_sum += float(loss.detach().item()) * batch
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                examples += batch
                if step % 100 == 0:
                    print(
                        f"teacher epoch {epoch + 1}/{configured_epochs} step {step}/{len(train_loader)} "
                        f"loss={loss.item():.4f}",
                        flush=True,
                    )
            scheduler.step()
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": loss_sum / max(1, examples),
                    "train_top1": correct / max(1, examples),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_runtime_sec": time.perf_counter() - epoch_start,
                }
            )
            _save_torch_atomic(
                checkpoint_path,
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "next_epoch": epoch + 1,
                    "history": history,
                    "plan": plan,
                },
            )

        evaluation_start = time.perf_counter()
        evaluation, diagnostics = evaluate_classifier(
            model,
            val_loader,
            device,
            precision,
            num_classes,
            int(cfg["runtime"].get("ece_bins", 15)),
        )
        evaluation_runtime = time.perf_counter() - evaluation_start
        training_runtime = sum(float(row.get("epoch_runtime_sec", 0.0)) for row in history)
        result = {
            **plan,
            **evaluation,
            "training_runtime_sec": training_runtime,
            "evaluation_runtime_sec": evaluation_runtime,
            "runtime_sec": training_runtime + evaluation_runtime,
            "session_runtime_sec": time.perf_counter() - run_start,
            "parameter_count": total_parameter_count(model),
            "precision": precision,
            "device": torch.cuda.get_device_name(device),
            "software_versions": _software_versions(),
            **slurm_metadata(),
        }
        _save_torch_atomic(
            output_dir / "teacher_state.pt",
            {
                "model_state": model.state_dict(),
                "architecture": architecture,
                "num_classes": num_classes,
                "dataset": dataset,
                "seed": seed,
            },
        )
        atomic_json(output_dir / "teacher_metrics.json", result)
        atomic_json(output_dir / "config_snapshot.json", cfg)
        save_training_history(output_dir, history)
        save_evaluation_diagnostics(output_dir, diagnostics, classes)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        atomic_text(done_path, "completed\n")
        print(f"TEACHER COMPLETE: {output_dir}", flush=True)
        return result


def _student_optimizer(
    cfg: dict[str, Any], model: nn.Module, architecture: str, batch_size: int
) -> torch.optim.Optimizer:
    runtime = cfg["runtime"]
    if architecture == "deit_tiny_patch16_224":
        learning_rate = float(runtime["transformer_student_lr"]) * batch_size / float(
            runtime.get("transformer_lr_reference_batch", 512)
        )
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=float(runtime["weight_decay_transformer"]),
        )
    learning_rate = float(runtime["cnn_student_lr"]) * batch_size / float(
        runtime.get("cnn_lr_reference_batch", 256)
    )
    return torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=float(runtime.get("momentum", 0.9)),
        weight_decay=float(runtime["weight_decay_cnn"]),
        nesterov=True,
    )


def _method_loss(
    method: str,
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    labels: torch.Tensor,
    features_s: dict[str, torch.Tensor],
    features_t: dict[str, torch.Tensor],
    early_s: list[str],
    early_t: list[str],
    final_s: str,
    final_t: str,
    runtime: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    temperature = float(runtime["kd_temperature"])
    if method == "KD":
        return kd_loss(logits_s, logits_t, temperature), logits_s.new_zeros(())
    if method == "FitNet":
        return logits_s.new_zeros(()), fitnet_loss(features_s[final_s], features_t[final_t])
    if method == "AT":
        losses = [
            attention_transfer_loss(features_s[s_name], features_t[t_name])
            for s_name, t_name in zip(early_s, early_t)
        ]
        return logits_s.new_zeros(()), torch.stack(losses).mean()
    if method == "CRD":
        return logits_s.new_zeros(()), crd_loss(features_s[early_s[-1]], features_t[early_t[-1]])
    if method == "DKD":
        return dkd_loss(logits_s, logits_t, labels, temperature), logits_s.new_zeros(())
    if method == "ReviewKD":
        return logits_s.new_zeros(()), reviewkd_loss(
            [features_s[name] for name in early_s], [features_t[name] for name in early_t]
        )
    if method == "DIST":
        return logits_s.new_zeros(()), dist_loss(
            features_s[early_s[-1]], features_t[early_t[-1]]
        )
    if method == "OTKD":
        return (
            otkd_loss(
                logits_s,
                logits_t,
                float(runtime["gcbd_uot_epsilon"]),
                int(runtime["gcbd_uot_iterations"]),
                int(runtime["ot_topk_classes"]),
            ),
            logits_s.new_zeros(()),
        )
    raise ValueError(f"Unknown method: {method}")


def run_method(
    cfg: dict[str, Any],
    dataset: str,
    scenario: str,
    method: str,
    seed: int,
    budget: float | None = None,
    epochs: int | None = None,
    output_root: str | None = None,
    overwrite: bool = False,
    resume: bool = True,
    clear_stale_lock: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    spec = scenario_spec(scenario)
    teacher_arch = spec["teacher"]
    student_arch = spec["student"]
    selected_budget = float(cfg["runtime"]["gcbd_budget"] if budget is None else budget)
    if not method.startswith("GCBD"):
        selected_budget = 0.0
    configured_epochs = epochs_for(cfg, dataset, "method") if epochs is None else int(epochs)
    selected_output_root = output_root or str(Path(cfg["output_root"]) / f"epochs_{configured_epochs}")
    output_dir = method_output_dir(
        selected_output_root, dataset, scenario, method, selected_budget, seed
    )
    teacher_dir = teacher_cache_dir(cfg, dataset, teacher_arch, seed)
    teacher_checkpoint = teacher_dir / "teacher_state.pt"
    batch_size = batch_size_for(cfg, dataset, scenario, "method")
    plan = {
        "mode": "method",
        "dataset": dataset,
        "scenario": scenario,
        "teacher_architecture": teacher_arch,
        "student_architecture": student_arch,
        "method": method,
        "seed": seed,
        "budget": selected_budget,
        "epochs": configured_epochs,
        "batch_size": batch_size,
        "teacher_checkpoint": str(teacher_checkpoint),
        "output_dir": str(output_dir),
    }
    if dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    if not (teacher_dir / "DONE").is_file() or not teacher_checkpoint.is_file():
        raise FileNotFoundError(
            f"Required fine-tuned teacher is incomplete: {teacher_dir}. "
            "Run and validate the teacher job before method jobs."
        )
    done_path = output_dir / "DONE"
    if done_path.exists() and not overwrite:
        print(f"Method already complete: {done_path}", flush=True)
        return json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    if overwrite:
        _remove_completed_marker(output_dir)

    _print_header(
        "method",
        dataset,
        scenario,
        seed,
        f"Method: {method} | Budget: {selected_budget:g} | Epochs: {configured_epochs}",
    )
    with RunLock(output_dir, clear_stale=clear_stale_lock):
        set_seed(seed, bool(cfg["runtime"].get("deterministic", False)))
        device, precision = _device_and_runtime(cfg)
        torch.cuda.reset_peak_memory_stats(device)
        train_loader, val_loader, num_classes, classes = build_loaders(
            dataset,
            cfg["dataset_paths"][dataset],
            batch_size,
            seed=seed,
            **_loader_kwargs(cfg),
        )
        teacher_base, student_base, _, _ = build_scenario_models(
            scenario, num_classes, str(teacher_checkpoint)
        )
        early_t = early_layer_names(teacher_arch)
        early_s = early_layer_names(student_arch)
        final_t = final_layer_name(teacher_arch)
        final_s = final_layer_name(student_arch)
        teacher = FeatureModel(teacher_base, layer_names(teacher_arch)).to(device)
        student = FeatureModel(student_base, layer_names(student_arch)).to(device)
        freeze(teacher)
        optimizer = _student_optimizer(cfg, student, student_arch, batch_size)
        scheduler = _cosine_scheduler(
            optimizer, configured_epochs, int(cfg["runtime"].get("warmup_epochs", 0))
        )
        scaler = _make_scaler(precision)
        checkpoint_path = output_dir / "checkpoint.pt"
        history: list[dict[str, Any]] = []
        start_epoch = 0
        cumulative = {
            "training_examples": 0,
            "teacher_prediction_queries": 0,
            "teacher_geometry_queries": 0,
            "teacher_forward_examples": 0,
        }
        if resume and checkpoint_path.is_file() and not overwrite:
            checkpoint = torch_load(checkpoint_path, map_location="cpu")
            student.load_state_dict(checkpoint["student_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = int(checkpoint["next_epoch"])
            history = list(checkpoint.get("history", []))
            cumulative.update(checkpoint.get("cumulative", {}))
            print(f"Resuming method at epoch {start_epoch}", flush=True)

        runtime = cfg["runtime"]
        is_gcbd = method.startswith("GCBD")
        run_start = time.perf_counter()
        for epoch in range(start_epoch, configured_epochs):
            student.train()
            teacher.eval()
            sums = {
                "loss": 0.0,
                "ce": 0.0,
                "kd": 0.0,
                "feature": 0.0,
                "budget_penalty": 0.0,
            }
            correct = 0
            examples = 0
            prediction_queries = 0
            geometry_queries = 0
            forward_examples = 0
            candidate_examples = 0
            epoch_start = time.perf_counter()
            gamma = curriculum_frontier(epoch, configured_epochs, num_classes)

            for step, (images, labels) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, precision):
                    logits_s, features_s = student(images)
                    ce = F.cross_entropy(
                        logits_s,
                        labels,
                        label_smoothing=float(runtime.get("label_smoothing", 0.0)),
                    )
                    kd_component = logits_s.new_zeros(())
                    feature_component = logits_s.new_zeros(())
                    budget_penalty = logits_s.new_zeros(())

                    if method == "Student":
                        loss = ce
                    elif not is_gcbd:
                        with torch.no_grad():
                            logits_t, features_t = teacher(images)
                        prediction_queries += labels.shape[0]
                        forward_examples += labels.shape[0]
                        kd_component, feature_component = _method_loss(
                            method,
                            logits_s,
                            logits_t,
                            labels,
                            features_s,
                            features_t,
                            early_s,
                            early_t,
                            final_s,
                            final_t,
                            runtime,
                        )
                        loss = (
                            float(runtime["supervised_weight"]) * ce
                            + float(runtime["distillation_weight"])
                            * (kd_component + feature_component)
                        )
                    else:
                        no_curriculum = method == "GCBD-NoCurriculum"
                        no_budget = method == "GCBD-NoBudget"
                        no_stage1 = method == "GCBD-NoStage1"
                        no_uot = method == "GCBD-NoUOT"
                        with torch.no_grad():
                            entropy = -(
                                F.softmax(logits_s.float(), dim=1)
                                * F.log_softmax(logits_s.float(), dim=1)
                            ).sum(dim=1)
                            active, candidate, _eta = select_active_mask(
                                entropy,
                                selected_budget,
                                gamma,
                                curriculum_enabled=not no_curriculum,
                                no_budget=no_budget,
                            )
                        candidate_examples += int(candidate.sum().item())

                        if not no_stage1:
                            with torch.no_grad():
                                _broad_logits_t, broad_features_t = teacher(images)
                            geometry_queries += labels.shape[0]
                            forward_examples += labels.shape[0]
                            broad_losses = [
                                sliced_wasserstein_attention_loss(
                                    features_s[s_name],
                                    broad_features_t[t_name],
                                    int(runtime["gcbd_swd_projections"]),
                                )
                                for s_name, t_name in zip(early_s, early_t)
                            ]
                            feature_component = feature_component + torch.stack(broad_losses).mean()

                        if active.any():
                            indices = active.nonzero(as_tuple=False).flatten()
                            with torch.no_grad():
                                logits_t_selected, features_t_selected = teacher(images[indices])
                            logits_s_selected = logits_s[indices]
                            prediction_queries += indices.numel()
                            forward_examples += indices.numel()
                            if no_uot:
                                kd_component = kd_loss(
                                    logits_s_selected,
                                    logits_t_selected,
                                    float(runtime["kd_temperature"]),
                                )
                            else:
                                kd_component = uot_loss(
                                    logits_s_selected,
                                    logits_t_selected,
                                    float(runtime["gcbd_uot_epsilon"]),
                                    int(runtime["gcbd_uot_iterations"]),
                                    float(runtime["gcbd_uot_tau_teacher"]),
                                    float(runtime["gcbd_uot_tau_student"]),
                                    int(runtime["ot_topk_classes"]),
                                )
                            feature_component = feature_component + sliced_wasserstein_attention_loss(
                                features_s[final_s][indices],
                                features_t_selected[final_t],
                                int(runtime["gcbd_swd_projections"]),
                            )
                        if not no_budget:
                            query_ratio = active.float().mean()
                            budget_penalty = float(runtime["gcbd_lambda_budget"]) * torch.relu(
                                query_ratio - selected_budget
                            )
                        loss = (
                            float(runtime["supervised_weight"]) * ce
                            + float(runtime["gcbd_lambda_uot"]) * kd_component
                            + float(runtime["gcbd_lambda_broad"]) * feature_component
                            + budget_penalty
                        )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip = float(runtime.get("gradient_clip_norm", 0.0))
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                batch = labels.shape[0]
                examples += batch
                correct += int((logits_s.detach().argmax(dim=1) == labels).sum().item())
                sums["loss"] += float(loss.detach().item()) * batch
                sums["ce"] += float(ce.detach().item()) * batch
                sums["kd"] += float(kd_component.detach().item()) * batch
                sums["feature"] += float(feature_component.detach().item()) * batch
                sums["budget_penalty"] += float(budget_penalty.detach().item()) * batch
                if step % 100 == 0:
                    print(
                        f"method epoch {epoch + 1}/{configured_epochs} step {step}/{len(train_loader)} "
                        f"loss={loss.item():.4f} queries={prediction_queries}",
                        flush=True,
                    )
            scheduler.step()
            cumulative["training_examples"] += examples
            cumulative["teacher_prediction_queries"] += prediction_queries
            cumulative["teacher_geometry_queries"] += geometry_queries
            cumulative["teacher_forward_examples"] += forward_examples
            row = {
                "epoch": epoch + 1,
                "train_loss": sums["loss"] / max(1, examples),
                "train_ce": sums["ce"] / max(1, examples),
                "train_kd": sums["kd"] / max(1, examples),
                "train_feature": sums["feature"] / max(1, examples),
                "train_budget_penalty": sums["budget_penalty"] / max(1, examples),
                "train_top1": correct / max(1, examples),
                "teacher_prediction_query_ratio": prediction_queries / max(1, examples),
                "teacher_geometry_query_ratio": geometry_queries / max(1, examples),
                "teacher_forward_ratio": forward_examples / max(1, examples),
                "candidate_ratio": candidate_examples / max(1, examples) if is_gcbd else 0.0,
                "curriculum_gamma": gamma if is_gcbd else None,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_runtime_sec": time.perf_counter() - epoch_start,
            }
            history.append(row)
            _save_torch_atomic(
                checkpoint_path,
                {
                    "student_state": student.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "next_epoch": epoch + 1,
                    "history": history,
                    "cumulative": cumulative,
                    "plan": plan,
                },
            )

        evaluation_start = time.perf_counter()
        evaluation, diagnostics = evaluate_classifier(
            student,
            val_loader,
            device,
            precision,
            num_classes,
            int(runtime.get("ece_bins", 15)),
        )
        evaluation_runtime = time.perf_counter() - evaluation_start
        training_runtime = sum(float(row.get("epoch_runtime_sec", 0.0)) for row in history)
        training_examples = max(1, int(cumulative["training_examples"]))
        prediction_ratio = cumulative["teacher_prediction_queries"] / training_examples
        geometry_ratio = cumulative["teacher_geometry_queries"] / training_examples
        forward_ratio = cumulative["teacher_forward_examples"] / training_examples
        teacher_metrics = json.loads((teacher_dir / "teacher_metrics.json").read_text(encoding="utf-8"))
        result = {
            **plan,
            **evaluation,
            "teacher_top1": teacher_metrics.get("top1"),
            "teacher_top5": teacher_metrics.get("top5"),
            "teacher_prediction_queries": cumulative["teacher_prediction_queries"],
            "teacher_geometry_queries": cumulative["teacher_geometry_queries"],
            "teacher_forward_examples": cumulative["teacher_forward_examples"],
            "training_examples": cumulative["training_examples"],
            "teacher_prediction_query_ratio": prediction_ratio,
            "teacher_geometry_query_ratio": geometry_ratio,
            "teacher_forward_ratio": forward_ratio,
            "performance_per_prediction_query_ratio": (
                evaluation["top1"] / prediction_ratio if prediction_ratio > 0 else None
            ),
            "accuracy_gain_over_student": None,
            "teacher_query_efficiency": None,
            "teacher_query_efficiency_per_1k_queries": None,
            "training_runtime_sec": training_runtime,
            "evaluation_runtime_sec": evaluation_runtime,
            "runtime_sec": training_runtime + evaluation_runtime,
            "session_runtime_sec": time.perf_counter() - run_start,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "student_parameter_count": total_parameter_count(student),
            "student_trainable_parameter_count": trainable_parameter_count(student),
            "teacher_parameter_count": teacher_metrics.get("parameter_count"),
            "precision": precision,
            "device": torch.cuda.get_device_name(device),
            "software_versions": _software_versions(),
            **slurm_metadata(),
        }
        _save_torch_atomic(
            output_dir / "student_state.pt",
            {
                "model_state": student.model.state_dict(),
                "architecture": student_arch,
                "num_classes": num_classes,
                "dataset": dataset,
                "scenario": scenario,
                "method": method,
                "seed": seed,
            },
        )
        atomic_json(output_dir / "result.json", result)
        atomic_json(output_dir / "config_snapshot.json", cfg)
        save_training_history(output_dir, history)
        save_evaluation_diagnostics(output_dir, diagnostics, classes)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        atomic_text(done_path, "completed\n")
        print(f"METHOD COMPLETE: {output_dir}", flush=True)
        return result
