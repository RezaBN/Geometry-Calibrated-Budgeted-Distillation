#!/bin/bash
# Submit all 15 GCBD-NLP methods for one dataset x seed x scenario cell.
#
# Usage:
#   ./submit_method_cell.sh DATASET SEED SCENARIO_NUMBER EPOCHS
#   ./submit_method_cell.sh DATASET SEED SCENARIO_NUMBER EPOCHS --submit
#
# Examples:
#   ./submit_method_cell.sh IMDb 18 1 50
#   ./submit_method_cell.sh IMDb 18 1 50 --submit
#   ./submit_method_cell.sh SST-2 64 5 50 --submit
#
# The default is a dry run. Each method is submitted as an independent H100
# SLURM job only when the final argument is --submit.

set -euo pipefail

GCBD_PROJECT_ROOT="${GCBD_PROJECT_ROOT:-/scratch/koohi/gcbd-nlp/code}"
GCBD_CONFIG="${GCBD_CONFIG:-$GCBD_PROJECT_ROOT/trillium_experiments.json}"
GCBD_VENV="${GCBD_VENV:-/scratch/koohi/gcbd-nlp/env/venv}"
GCBD_HF_HOME="${GCBD_HF_HOME:-/scratch/koohi/gcbd-nlp/cache/downloads}"
GCBD_TEACHER_ROOT="${GCBD_TEACHER_ROOT:-/scratch/koohi/gcbd-nlp/models/teachers_finetuned}"
GCBD_OUTPUT_BASE="${GCBD_OUTPUT_BASE:-/scratch/koohi/gcbd-nlp/outputs}"
GCBD_LOG_ROOT="${GCBD_LOG_ROOT:-/scratch/koohi/gcbd-nlp/logs}"
GCBD_ACCOUNT="${GCBD_ACCOUNT:-def-koohi}"
GCBD_BATCH_SCRIPT="$GCBD_PROJECT_ROOT/trillium_h100.sbatch"
GCBD_PYTHON="$GCBD_VENV/bin/python"

usage() {
    sed -n '2,14p' "$0"
}

if (( $# < 4 || $# > 5 )); then
    usage >&2
    exit 2
fi

dataset_input="$1"
seed="$2"
scenario_number="$3"
epochs="$4"
action="${5:-}"

case "$dataset_input" in
    IMDb)
        dataset=IMDb
        ;;
    SST-2|SST2)
        dataset=SST-2
        ;;
    *)
        echo "ERROR: DATASET must be IMDb or SST-2; received: $dataset_input" >&2
        exit 2
        ;;
esac

case "$seed" in
    18|64) ;;
    *)
        echo "ERROR: SEED must be 18 or 64; received: $seed" >&2
        exit 2
        ;;
esac

case "$scenario_number" in
    1)
        scenario=Setup_1_BERT_TinyBERT
        teacher_cache=bert-base-uncased
        ;;
    2)
        scenario=Setup_2_DeBERTa_DistilBERT
        teacher_cache=microsoft__deberta-v3-base
        ;;
    3)
        scenario=Setup_3_GPT2_DistilGPT2
        teacher_cache=gpt2
        ;;
    4)
        scenario=Setup_4_BERT_DistilGPT2
        teacher_cache=bert-base-uncased
        ;;
    5)
        scenario=Setup_5_GPT2_DistilBERT
        teacher_cache=gpt2
        ;;
    *)
        echo "ERROR: SCENARIO_NUMBER must be 1, 2, 3, 4, or 5; received: $scenario_number" >&2
        exit 2
        ;;
esac

case "$epochs" in
    ''|*[!0-9]*|0)
        echo "ERROR: EPOCHS must be a positive integer; received: $epochs" >&2
        exit 2
        ;;
esac

if [[ -n "$action" && "$action" != "--submit" ]]; then
    echo "ERROR: the optional final argument must be --submit; received: $action" >&2
    exit 2
fi

for required_file in "$GCBD_CONFIG" "$GCBD_BATCH_SCRIPT" "$GCBD_PYTHON"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file is missing: $required_file" >&2
        exit 2
    fi
done

teacher_done="$GCBD_TEACHER_ROOT/$dataset/$scenario/$teacher_cache/classification/seed_${seed}/DONE"
if [[ ! -f "$teacher_done" ]]; then
    echo "ERROR: the required fine-tuned teacher is incomplete:" >&2
    echo "  $teacher_done" >&2
    exit 3
fi

# Keep experiments with different epoch counts in distinct output trees. This
# prevents an earlier 10-epoch DONE marker from suppressing a requested
# 50-epoch run and preserves provenance for later aggregation.
cell_output_root="$GCBD_OUTPUT_BASE/epochs_${epochs}"
mkdir -p "$GCBD_LOG_ROOT" "$cell_output_root"

export GCBD_PROJECT_ROOT GCBD_CONFIG GCBD_VENV GCBD_HF_HOME

# Generate method, budget, walltime, job-name, and expected DONE path using the
# same walltime resolver as submit_trillium.py. The current experiment config
# must contain exactly one budget (0.3).
spec_file=$(mktemp "${TMPDIR:-/tmp}/gcbd_method_specs.XXXXXX")
trap 'rm -f "$spec_file"' EXIT

"$GCBD_PYTHON" - \
        "$GCBD_PROJECT_ROOT" \
        "$GCBD_CONFIG" \
        "$cell_output_root" \
        "$dataset" \
        "$scenario" \
        "$seed" \
        "$epochs" >"$spec_file" <<'PY'
import json
import math
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
config_path = Path(sys.argv[2])
output_root = Path(sys.argv[3])
dataset = sys.argv[4]
scenario = sys.argv[5]
seed = int(sys.argv[6])
epochs = int(sys.argv[7])

sys.path.insert(0, str(project_root))
from submit_trillium import (  # noqa: E402
    ALL_METHODS,
    budget_slug,
    budgets_for_method,
    parse_method_time_rules,
    resolve_method_time,
    slug,
    validate_walltime,
)

config = json.loads(config_path.read_text(encoding="utf-8"))
methods = list(config.get("methods", []))
if len(methods) != 15 or set(methods) != ALL_METHODS:
    raise SystemExit(
        "ERROR: trillium_experiments.json must contain the expected 15 methods"
    )

budgets = [float(value) for value in config.get("budgets", [])]
if len(budgets) != 1 or not math.isclose(budgets[0], 0.3, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("ERROR: trillium_experiments.json budgets must be [0.3]")

slurm = config.get("slurm", {})
default_time = validate_walltime(
    slurm.get("method_time", "24:00:00"), "slurm.method_time"
)
rules = parse_method_time_rules(slurm)

def sanitize(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_").replace(":", "_")

for method in methods:
    method_budgets = budgets_for_method(method, budgets)
    if len(method_budgets) != 1:
        raise SystemExit(f"ERROR: expected one budget for {method}")
    budget = method_budgets[0]
    walltime = resolve_method_time(
        default_time=default_time,
        rules=rules,
        dataset=dataset,
        scenario=scenario,
        method=method,
        budget=budget,
    )
    job_name = (
        f"M-{slug(dataset, 10)}-{slug(scenario, 8)}-"
        f"{slug(method, 20)}-s{seed}-b{budget_slug(budget)}-e{epochs}"
    )
    method_dir = method + (f"_B{budget:g}" if method.startswith("GCBD") else "")
    done_file = (
        output_root
        / sanitize(dataset)
        / sanitize(scenario)
        / sanitize(method_dir)
        / f"seed_{seed}"
        / "DONE"
    )
    print(f"{method}\t{budget:g}\t{walltime}\t{job_name}\t{done_file}")
PY

mapfile -t job_specs <"$spec_file"

if (( ${#job_specs[@]} != 15 )); then
    echo "ERROR: expected 15 method specifications; received ${#job_specs[@]}" >&2
    exit 4
fi

export_argument="ALL,GCBD_CONFIG=$GCBD_CONFIG,GCBD_PROJECT_ROOT=$GCBD_PROJECT_ROOT,GCBD_VENV=$GCBD_VENV,GCBD_HF_HOME=$GCBD_HF_HOME"

if [[ "$action" == "--submit" ]]; then
    action_label=SUBMIT
    timestamp=$(date -u +%Y%m%dT%H%M%SZ)
    dataset_slug=${dataset//-/}
    manifest="$GCBD_LOG_ROOT/method_cell_${dataset_slug}_S${scenario_number}_s${seed}_e${epochs}_${timestamp}.tsv"
    printf 'job_id\tdataset\tscenario\tseed\tepochs\tmethod\tbudget\trequested_time\tstatus\n' >"$manifest"
else
    action_label=DRY-RUN
    manifest=
fi

echo "================================================================"
echo "GCBD-NLP METHOD CELL"
echo "Dataset: $dataset"
echo "Scenario: $scenario"
echo "Seed: $seed"
echo "Epochs: $epochs"
echo "Output root: $cell_output_root"
echo "Action: $action_label"
echo "Methods: 15 independent SLURM jobs"
echo "================================================================"

submitted=0
skipped=0
for spec in "${job_specs[@]}"; do
    IFS=$'\t' read -r method budget walltime job_name done_file <<<"$spec"

    if [[ -f "$done_file" ]]; then
        echo "SKIP DONE | $method | $done_file"
        skipped=$((skipped + 1))
        if [[ "$action" == "--submit" ]]; then
            printf -- '-\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tSKIPPED_DONE\n' \
                "$dataset" "$scenario" "$seed" "$epochs" "$method" "$budget" "$walltime" \
                >>"$manifest"
        fi
        continue
    fi

    if [[ "$action" == "--submit" ]]; then
        if ! command -v squeue >/dev/null 2>&1; then
            echo "ERROR: squeue is required when --submit is used" >&2
            exit 5
        fi
        active_job=$(squeue -h -u "$USER" --name="$job_name" --format='%i' | head -n 1 || true)
        if [[ -n "$active_job" ]]; then
            echo "SKIP ACTIVE | $active_job | $method | $job_name"
            skipped=$((skipped + 1))
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tSKIPPED_ACTIVE\n' \
                "$active_job" "$dataset" "$scenario" "$seed" "$epochs" "$method" "$budget" "$walltime" \
                >>"$manifest"
            continue
        fi
    fi

    command=(
        sbatch
        --parsable
        --export="$export_argument"
        --account="$GCBD_ACCOUNT"
        --chdir="$GCBD_PROJECT_ROOT"
        --job-name="$job_name"
        --time="$walltime"
        --output="$GCBD_LOG_ROOT/%x-%j.out"
        "$GCBD_BATCH_SCRIPT"
        --mode method
        --dataset "$dataset"
        --scenario "$scenario"
        --method "$method"
        --seed "$seed"
        --budget "$budget"
        --epochs "$epochs"
        --output-root "$cell_output_root"
    )

    if [[ "$action" == "--submit" ]]; then
        job_id=$("${command[@]}")
        job_id="${job_id%%;*}"
        if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
            echo "ERROR: could not parse SLURM job ID for $method: $job_id" >&2
            exit 5
        fi
        echo "SUBMITTED $job_id | $method | budget=$budget | time=$walltime"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tSUBMITTED\n' \
            "$job_id" "$dataset" "$scenario" "$seed" "$epochs" "$method" "$budget" "$walltime" \
            >>"$manifest"
        submitted=$((submitted + 1))
    else
        printf 'DRY-RUN | %s | budget=%s | time=%s | command:' "$method" "$budget" "$walltime"
        printf ' %q' "${command[@]}"
        printf '\n'
    fi
done

if [[ "$action" == "--submit" ]]; then
    echo "Submitted: $submitted | skipped DONE: $skipped | manifest: $manifest"
else
    echo "Dry run complete. Re-run the same command with --submit after review."
fi
