#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gcbd_vision.config import (
    METHODS,
    epochs_for,
    load_config,
    method_output_dir,
    scenario_spec,
    slurm_time_for,
    teacher_cache_dir,
)
from gcbd_vision.utils import atomic_json, ensure_dir


@dataclass
class Job:
    kind: str
    key: tuple[Any, ...]
    name: str
    time_limit: str
    arguments: list[str]
    dependency_key: tuple[Any, ...] | None = None


def token(value: str, maximum: int = 24) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", value)
    return cleaned[:maximum] or "job"


def sbatch_prefix(cfg: dict[str, Any], name: str, time_limit: str) -> list[str]:
    """Build a Trillium single-GPU sbatch command.

    Trillium GPU rules:
    - single GPU: --gpus-per-node=1
    - no explicit GPU partition; scheduler chooses it
    - --mem is ignored and intentionally omitted
    - one GPU corresponds to a quarter node (24 CPU cores, ~188 GiB RAM)
    """
    slurm = cfg["slurm"]
    return [
        "sbatch",
        "--parsable",
        f"--account={cfg['account']}",
        f"--nodes={int(slurm.get('nodes', 1))}",
        f"--ntasks={int(slurm.get('ntasks', 1))}",
        f"--gpus-per-node={int(slurm.get('gpus_per_node', 1))}",
        f"--cpus-per-task={int(slurm.get('cpus_per_task', 24))}",
        f"--job-name={name}",
        f"--time={time_limit}",
        f"--chdir={cfg['project_root']}",
        f"--output={cfg['logs_root']}/%x-%j.out",
        f"--mail-type={slurm.get('mail_type', 'FAIL')}",
        "--export=ALL,"
        f"GCBD_VISION_PROJECT_ROOT={cfg['project_root']},"
        f"GCBD_VISION_CONFIG={cfg['_config_path']},"
        f"GCBD_VISION_VENV={cfg['venv']},"
        f"GCBD_VISION_TORCH_HOME={cfg['torch_home']}",
    ]


def representative_scenario(cfg: dict[str, Any], teacher_architecture: str) -> str:
    for scenario in cfg["scenarios"]:
        if scenario_spec(scenario)["teacher"] == teacher_architecture:
            return scenario
    raise ValueError(f"No configured scenario uses teacher {teacher_architecture}")


def runtime_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    if args.overwrite:
        flags.append("--overwrite")
    if args.no_resume:
        flags.append("--no-resume")
    if args.clear_stale_lock:
        flags.append("--clear-stale-lock")
    return flags


def build_plan(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[list[Job], list[Job]]:
    datasets = args.only_dataset or list(cfg["datasets"])
    scenarios = args.only_scenario or list(cfg["scenarios"])
    seeds = args.only_seed or [int(seed) for seed in cfg["seeds"]]
    methods = args.only_method or list(cfg["methods"])
    flags = runtime_flags(args)

    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")

    teacher_jobs: list[Job] = []
    required_teacher_keys: dict[tuple[Any, ...], str] = {}
    for dataset in datasets:
        for scenario in scenarios:
            architecture = scenario_spec(scenario)["teacher"]
            for seed in seeds:
                required_teacher_keys[(dataset, architecture, seed)] = representative_scenario(
                    cfg, architecture
                )

    for key, scenario in sorted(required_teacher_keys.items()):
        dataset, architecture, seed = key
        cache = teacher_cache_dir(cfg, dataset, architecture, seed)
        complete = (cache / "DONE").is_file() and (cache / "teacher_state.pt").is_file()
        if complete and not args.include_completed_teachers:
            continue
        if args.skip_teacher_jobs:
            if not complete:
                raise FileNotFoundError(
                    f"--skip-teacher-jobs was used, but teacher is incomplete: {cache}"
                )
            continue
        selected_epochs = (
            int(args.teacher_epochs)
            if args.teacher_epochs is not None
            else epochs_for(cfg, dataset, "teacher")
        )
        name = f"TV-{token(dataset, 8)}-{token(architecture, 10)}-s{seed}"
        teacher_jobs.append(
            Job(
                kind="teacher",
                key=key,
                name=name,
                time_limit=args.teacher_time
                or slurm_time_for(cfg, "teacher", dataset, scenario=scenario),
                arguments=[
                    "--mode", "teacher",
                    "--dataset", dataset,
                    "--scenario", scenario,
                    "--seed", str(seed),
                    "--teacher-epochs", str(selected_epochs),
                    *flags,
                ],
            )
        )

    if args.teacher_only:
        return teacher_jobs, []

    method_jobs: list[Job] = []
    for dataset in datasets:
        selected_epochs = int(args.epochs) if args.epochs is not None else epochs_for(
            cfg, dataset, "method"
        )
        output_root = args.output_root or str(Path(cfg["output_root"]) / f"epochs_{selected_epochs}")
        for scenario in scenarios:
            architecture = scenario_spec(scenario)["teacher"]
            for seed in seeds:
                dependency_key = (dataset, architecture, seed)
                cache = teacher_cache_dir(cfg, dataset, architecture, seed)
                teacher_complete = (cache / "DONE").is_file() and (cache / "teacher_state.pt").is_file()
                for method in methods:
                    budget = float(args.budget) if args.budget is not None else float(
                        cfg["runtime"]["gcbd_budget"]
                    )
                    if not method.startswith("GCBD"):
                        budget = 0.0
                    output = method_output_dir(
                        output_root, dataset, scenario, method, budget, seed
                    )
                    if (output / "DONE").is_file() and not args.include_completed_methods:
                        continue
                    name = (
                        f"MV-{token(dataset, 7)}-{scenario}-{token(method, 12)}-"
                        f"s{seed}-e{selected_epochs}"
                    )
                    method_jobs.append(
                        Job(
                            kind="method",
                            key=(dataset, scenario, method, seed, budget, selected_epochs),
                            name=name,
                            time_limit=args.time or slurm_time_for(
                                cfg, "method", dataset, scenario=scenario, method=method
                            ),
                            arguments=[
                                "--mode", "method",
                                "--dataset", dataset,
                                "--scenario", scenario,
                                "--method", method,
                                "--seed", str(seed),
                                "--budget", str(budget),
                                "--epochs", str(selected_epochs),
                                "--output-root", output_root,
                                *flags,
                            ],
                            dependency_key=None if teacher_complete else dependency_key,
                        )
                    )
    return teacher_jobs, method_jobs


def submit(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    job_id = completed.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse sbatch job ID from: {completed.stdout!r}")
    return job_id


def assert_trillium_submission_context(cfg: dict[str, Any], submitting: bool) -> None:
    if submitting and os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "Trillium does not allow job submission from compute jobs. Run this submitter "
            "from a Trillium login node (preferably the GPU login node)."
        )

    scratch = os.environ.get("SCRATCH")
    if not scratch:
        return
    scratch_root = Path(scratch).expanduser().resolve()
    write_paths = {
        "project_root": Path(cfg["project_root"]),
        "venv": Path(cfg["venv"]),
        "torch_home": Path(cfg["torch_home"]),
        "teacher_cache_root": Path(cfg["teacher_cache_root"]),
        "output_root": Path(cfg["output_root"]),
        "logs_root": Path(cfg["logs_root"]),
    }
    outside = []
    for label, path in write_paths.items():
        try:
            path.expanduser().resolve().relative_to(scratch_root)
        except ValueError:
            outside.append(f"{label}={path}")
    if outside:
        raise ValueError(
            "Trillium compute nodes require job output/writes in $SCRATCH. "
            "Move these configured write paths under $SCRATCH: " + ", ".join(outside)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or submit the GCBD vision teacher/method matrix on Trillium"
    )
    parser.add_argument("--config", default="vision_experiments.json")
    parser.add_argument("--account")
    parser.add_argument("--only-dataset", action="append", choices=("CIFAR100", "ImageNet"))
    parser.add_argument("--only-scenario", action="append", choices=("A", "B", "C", "D"))
    parser.add_argument("--only-seed", action="append", type=int)
    parser.add_argument("--only-method", action="append", choices=METHODS)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--teacher-epochs", type=int)
    parser.add_argument("--time", help="Override all selected method walltimes (Trillium max: 24h)")
    parser.add_argument("--teacher-time", help="Override all selected teacher walltimes")
    parser.add_argument("--output-root")
    parser.add_argument("--teacher-only", action="store_true")
    parser.add_argument("--skip-teacher-jobs", action="store_true")
    parser.add_argument("--include-completed-teachers", action="store_true")
    parser.add_argument("--include-completed-methods", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--clear-stale-lock",
        action="store_true",
        help="Safely remove a verified stale run lock before resuming",
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.account:
        cfg["account"] = args.account
    assert_trillium_submission_context(cfg, args.submit)
    ensure_dir(cfg["logs_root"])

    batch_script = str(Path(cfg["project_root"]) / "trillium_vision_h100_phase2.sbatch")
    teacher_jobs, method_jobs = build_plan(cfg, args)
    print(
        f"Plan: {len(teacher_jobs)} teacher job(s) + {len(method_jobs)} method job(s) "
        f"= {len(teacher_jobs) + len(method_jobs)} total"
    )

    submitted_teachers: dict[tuple[Any, ...], str] = {}
    manifest: dict[str, Any] = {
        "config": cfg["_config_path"],
        "cluster": "Trillium",
        "submitted": args.submit,
        "teachers": [],
        "methods": [],
    }

    for job in teacher_jobs:
        command = sbatch_prefix(cfg, job.name, job.time_limit) + [batch_script] + job.arguments
        print(shlex.join(command))
        job_id = submit(command) if args.submit else None
        if job_id:
            submitted_teachers[job.key] = job_id
            print(f"Submitted teacher {job.name}: {job_id}")
        manifest["teachers"].append(
            {"key": list(job.key), "name": job.name, "job_id": job_id, "command": command}
        )

    for job in method_jobs:
        command = sbatch_prefix(cfg, job.name, job.time_limit)
        dependency_id = submitted_teachers.get(job.dependency_key) if job.dependency_key else None
        if dependency_id:
            command.append(f"--dependency=afterok:{dependency_id}")
        elif job.dependency_key is not None and args.submit:
            cache = teacher_cache_dir(cfg, *job.dependency_key)
            if not (cache / "DONE").is_file():
                raise RuntimeError(
                    f"No completed/submitted teacher available for method {job.name}: {cache}"
                )
        command += [batch_script] + job.arguments
        print(shlex.join(command))
        job_id = submit(command) if args.submit else None
        if job_id:
            print(f"Submitted method {job.name}: {job_id}")
        manifest["methods"].append(
            {
                "key": list(job.key),
                "name": job.name,
                "job_id": job_id,
                "teacher_dependency": dependency_id,
                "command": command,
            }
        )

    manifest_path = Path(cfg["logs_root"]) / "last_submission_plan.json"
    atomic_json(manifest_path, manifest)
    print(f"Submission manifest: {manifest_path}")


if __name__ == "__main__":
    main()
