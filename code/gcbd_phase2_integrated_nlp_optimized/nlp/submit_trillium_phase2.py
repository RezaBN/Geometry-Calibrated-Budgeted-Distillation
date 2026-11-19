#!/usr/bin/env python3
"""Expand the GCBD experiment matrix into independent Trillium GPU jobs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ALL_DATASETS = {"SST-2", "IMDb", "DBPedia", "WikiText-2"}
ALL_SCENARIOS = {
    "Setup_1_BERT_TinyBERT",
    "Setup_2_DeBERTa_DistilBERT",
    "Setup_3_GPT2_DistilGPT2",
    "Setup_4_BERT_DistilGPT2",
    "Setup_5_GPT2_DistilBERT",
}
ALL_METHODS = {
    "Student",
    "KD",
    "FitNet",
    "PKD",
    "RKD",
    "CRD",
    "MiniLM",
    "TinyBERT",
    "DKD",
    "DIST",
    "GCBD-Full",
    "GCBD-NoCurriculum",
    "GCBD-NoBudget",
    "GCBD-NoStage1",
    "GCBD-NoUOT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one Trillium job per dataset, scenario, method, seed, and applicable budget."
    )
    parser.add_argument("--config", default="trillium_experiments.json")
    parser.add_argument("--account", help="Override the SLURM account in the JSON configuration.")
    parser.add_argument("--submit", action="store_true", help="Submit jobs; otherwise print a dry-run plan.")
    parser.add_argument("--skip-teacher-jobs", action="store_true", help="Assume all required teacher caches already exist.")
    parser.add_argument("--teacher-only", action="store_true")
    parser.add_argument("--only-dataset", action="append", choices=sorted(ALL_DATASETS))
    parser.add_argument("--only-scenario", action="append", choices=sorted(ALL_SCENARIOS))
    parser.add_argument("--only-method", action="append", choices=sorted(ALL_METHODS))
    parser.add_argument("--only-seed", action="append", type=int)
    parser.add_argument("--only-budget", action="append", type=float)
    parser.add_argument("--overwrite", action="store_true", help="Pass --overwrite to teacher and method jobs.")
    return parser.parse_args()


def expand_path(value: str, base: Optional[Path] = None) -> Path:
    expanded = Path(os.path.expanduser(os.path.expandvars(value)))
    if not expanded.is_absolute() and base is not None:
        expanded = base / expanded
    return expanded.resolve()


def read_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_list(config: Dict[str, Any], key: str) -> List[Any]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty JSON array.")
    return value


def check_allowed(values: Iterable[str], allowed: set[str], label: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}")


def selected(values: List[Any], filters: Optional[List[Any]]) -> List[Any]:
    return values if not filters else [value for value in values if value in filters]


def budgets_for_method(method: str, budgets: List[float]) -> List[float]:
    if method == "GCBD-NoBudget" or not method.startswith("GCBD"):
        return [0.0]
    return budgets


def slug(value: str, limit: int = 34) -> str:
    value = re.sub(r"^Setup_(\d+)_.*$", r"S\1", value)
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")
    return value[:limit]


def budget_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def validate_walltime(value: Any, label: str) -> str:
    """Validate Trillium's maximum 24-hour HH:MM:SS walltime format."""
    raw = str(value)
    match = re.fullmatch(r"(\d{1,2}):([0-5]\d):([0-5]\d)", raw)
    if not match:
        raise ValueError(f"{label} must use HH:MM:SS, for example 08:00:00.")
    hours, minutes, seconds = (int(part) for part in match.groups())
    total_seconds = hours * 3600 + minutes * 60 + seconds
    if total_seconds <= 0 or total_seconds > 24 * 3600:
        raise ValueError(f"{label} must be greater than zero and no more than 24:00:00.")
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_method_time_rules(slurm: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_rules = slurm.get("method_time_rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError("slurm.method_time_rules must be a JSON array.")

    allowed_keys = {"dataset", "scenario", "method", "budget", "time"}
    selector_keys = allowed_keys - {"time"}
    rules: List[Dict[str, Any]] = []
    for index, raw_rule in enumerate(raw_rules):
        label = f"slurm.method_time_rules[{index}]"
        if not isinstance(raw_rule, dict):
            raise ValueError(f"{label} must be a JSON object.")
        unknown = sorted(set(raw_rule) - allowed_keys)
        if unknown:
            raise ValueError(f"{label} has unknown field(s): {', '.join(unknown)}")
        if "time" not in raw_rule:
            raise ValueError(f"{label} must define time.")
        if not any(key in raw_rule for key in selector_keys):
            raise ValueError(f"{label} must select at least one dataset, scenario, method, or budget.")

        rule = dict(raw_rule)
        if "dataset" in rule and rule["dataset"] not in ALL_DATASETS:
            raise ValueError(f"{label} has unknown dataset: {rule['dataset']}")
        if "scenario" in rule and rule["scenario"] not in ALL_SCENARIOS:
            raise ValueError(f"{label} has unknown scenario: {rule['scenario']}")
        if "method" in rule and rule["method"] not in ALL_METHODS:
            raise ValueError(f"{label} has unknown method: {rule['method']}")
        if "budget" in rule:
            if isinstance(rule["budget"], bool) or not isinstance(rule["budget"], (int, float)):
                raise ValueError(f"{label}.budget must be numeric.")
            rule["budget"] = float(rule["budget"])
            if not 0.0 <= rule["budget"] <= 1.0:
                raise ValueError(f"{label}.budget must lie in [0, 1].")
        rule["time"] = validate_walltime(rule["time"], f"{label}.time")
        rules.append(rule)
    return rules


def resolve_method_time(
    default_time: str,
    rules: List[Dict[str, Any]],
    dataset: str,
    scenario: str,
    method: str,
    budget: float,
) -> str:
    """Return the most-specific matching time; later rules break ties."""
    selected_time = default_time
    selected_specificity = 0
    for rule in rules:
        matches = (
            ("dataset" not in rule or rule["dataset"] == dataset)
            and ("scenario" not in rule or rule["scenario"] == scenario)
            and ("method" not in rule or rule["method"] == method)
            and (
                "budget" not in rule
                or math.isclose(rule["budget"], budget, rel_tol=0.0, abs_tol=1e-12)
            )
        )
        if not matches:
            continue
        specificity = sum(
            key in rule for key in ("dataset", "scenario", "method", "budget")
        )
        if specificity >= selected_specificity:
            selected_time = rule["time"]
            selected_specificity = specificity
    return selected_time


def export_argument(values: Dict[str, Path]) -> str:
    parts = ["ALL"]
    for key, path in values.items():
        raw = str(path)
        if any(char in raw for char in [",", "\n", "\r"]):
            raise ValueError(f"{key} contains a character unsupported by sbatch --export: {raw}")
        parts.append(f"{key}={raw}")
    return ",".join(parts)


def submit_command(command: List[str], do_submit: bool) -> str:
    if not do_submit:
        print(shlex.join(command))
        return "DRY-RUN"
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse sbatch job ID from: {completed.stdout!r}")
    return job_id


def write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.teacher_only and args.skip_teacher_jobs:
        raise ValueError("--teacher-only and --skip-teacher-jobs cannot be combined.")
    config_path = expand_path(args.config, Path.cwd())
    config = read_config(config_path)
    config_dir = config_path.parent

    project_root = expand_path(str(config["project_root"]), config_dir)
    venv = expand_path(str(config["venv"]), config_dir)
    hf_home = expand_path(str(config["hf_home"]), config_dir)
    logs_dir = expand_path(str(config.get("logs_root", project_root / "logs")), config_dir)
    sbatch_script = expand_path(str(config.get("sbatch_script", "trillium_h100.sbatch")), config_dir)
    account = str(args.account or config.get("account", "")).strip()
    if not account or account.startswith("REPLACE_"):
        raise ValueError("Set account in trillium_experiments.json (for example def-yourpi or rrg-yourpi).")
    if not project_root.is_dir():
        raise FileNotFoundError(f"project_root does not exist: {project_root}")
    if not sbatch_script.is_file():
        raise FileNotFoundError(f"SLURM script does not exist: {sbatch_script}")
    if args.submit and not (venv / "bin" / "python").is_file():
        raise FileNotFoundError(f"Python environment is missing: {venv}")

    datasets = selected(require_list(config, "datasets"), args.only_dataset)
    scenarios = selected(require_list(config, "scenarios"), args.only_scenario)
    methods = selected(require_list(config, "methods"), args.only_method)
    seeds = selected([int(value) for value in require_list(config, "seeds")], args.only_seed)
    budgets = selected([float(value) for value in require_list(config, "budgets")], args.only_budget)
    check_allowed(datasets, ALL_DATASETS, "dataset(s)")
    check_allowed(scenarios, ALL_SCENARIOS, "scenario(s)")
    check_allowed(methods, ALL_METHODS, "method(s)")
    if not all([datasets, scenarios, seeds]) or (not args.teacher_only and not all([methods, budgets])):
        raise ValueError("The selected filters produced an empty experiment dimension.")
    if any(not 0.0 <= value <= 1.0 for value in budgets):
        raise ValueError("Every budget must lie in [0, 1].")

    slurm = config.get("slurm", {})
    if not isinstance(slurm, dict):
        raise ValueError("slurm must be a JSON object.")
    teacher_time = validate_walltime(
        slurm.get("teacher_time", "24:00:00"), "slurm.teacher_time"
    )
    default_method_time = validate_walltime(
        slurm.get("method_time", "24:00:00"), "slurm.method_time"
    )
    method_time_rules = parse_method_time_rules(slurm)
    max_jobs = int(slurm.get("max_jobs_per_submission", 450))
    if max_jobs <= 0:
        raise ValueError("slurm.max_jobs_per_submission must be positive.")
    logs_dir.mkdir(parents=True, exist_ok=True)
    export_env = export_argument(
        {
            "GCBD_CONFIG": config_path,
            "GCBD_PROJECT_ROOT": project_root,
            "GCBD_VENV": venv,
            "GCBD_HF_HOME": hf_home,
        }
    )

    teacher_count = 0 if args.skip_teacher_jobs else len(datasets) * len(scenarios) * len(seeds)
    method_count = 0
    if not args.teacher_only:
        method_count = sum(
            len(budgets_for_method(method, budgets))
            for _dataset in datasets
            for _scenario in scenarios
            for _seed in seeds
            for method in methods
        )
    total_jobs = teacher_count + method_count
    if total_jobs > max_jobs:
        raise ValueError(
            f"Plan contains {total_jobs} jobs, above max_jobs_per_submission={max_jobs}. "
            "Submit a filtered subset or raise the safeguard deliberately."
        )

    print(
        f"Plan: {teacher_count} teacher job(s) + {method_count} method job(s) = {total_jobs} total; "
        f"mode={'SUBMIT' if args.submit else 'DRY-RUN'}"
    )
    common = [
        "sbatch",
        "--parsable",
        f"--account={account}",
        "--nodes=1",
        "--gpus-per-node=1",
        f"--chdir={project_root}",
        f"--export={export_env}",
    ]
    overwrite_arg = ["--overwrite"] if args.overwrite else []
    rows: List[Dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = logs_dir / f"submission_manifest_{timestamp}.csv"

    for dataset in datasets:
        for scenario in scenarios:
            for seed in seeds:
                teacher_job_id: Optional[str] = None
                if not args.skip_teacher_jobs:
                    teacher_name = f"T-{slug(dataset, 12)}-{slug(scenario, 10)}-s{seed}"
                    teacher_cmd = common + [
                        f"--job-name={teacher_name}",
                        f"--time={teacher_time}",
                        f"--output={logs_dir}/%x-%j.out",
                        str(sbatch_script),
                        "--mode", "teacher",
                        "--dataset", dataset,
                        "--scenario", scenario,
                        "--seed", str(seed),
                        *overwrite_arg,
                    ]
                    teacher_job_id = submit_command(teacher_cmd, args.submit)
                    rows.append({
                        "job_id": teacher_job_id,
                        "kind": "teacher",
                        "dataset": dataset,
                        "scenario": scenario,
                        "method": "",
                        "seed": seed,
                        "budget": "",
                        "requested_time": teacher_time,
                        "dependency": "",
                    })
                    if args.submit:
                        write_manifest(manifest, rows)

                if args.teacher_only:
                    continue
                for method in methods:
                    for budget in budgets_for_method(method, budgets):
                        method_time = resolve_method_time(
                            default_time=default_method_time,
                            rules=method_time_rules,
                            dataset=dataset,
                            scenario=scenario,
                            method=method,
                            budget=budget,
                        )
                        method_name = (
                            f"M-{slug(dataset, 10)}-{slug(scenario, 8)}-"
                            f"{slug(method, 20)}-s{seed}-b{budget_slug(budget)}"
                        )
                        method_cmd = common + [
                            f"--job-name={method_name}",
                            f"--time={method_time}",
                            f"--output={logs_dir}/%x-%j.out",
                        ]
                        if teacher_job_id:
                            dependency_id = teacher_job_id if teacher_job_id != "DRY-RUN" else "<teacher-job-id>"
                            method_cmd.append(f"--dependency=afterok:{dependency_id}")
                        method_cmd += [
                            str(sbatch_script),
                            "--mode", "method",
                            "--dataset", dataset,
                            "--scenario", scenario,
                            "--method", method,
                            "--seed", str(seed),
                            "--budget", str(budget),
                            *overwrite_arg,
                        ]
                        method_job_id = submit_command(method_cmd, args.submit)
                        rows.append({
                            "job_id": method_job_id,
                            "kind": "method",
                            "dataset": dataset,
                            "scenario": scenario,
                            "method": method,
                            "seed": seed,
                            "budget": budget,
                            "requested_time": method_time,
                            "dependency": teacher_job_id or "pre-existing teacher",
                        })
                        if args.submit:
                            write_manifest(manifest, rows)

    if args.submit:
        print(f"Submitted {len(rows)} jobs. Manifest: {manifest}")
    else:
        print("Dry run only. Re-run with --submit after reviewing the commands.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
