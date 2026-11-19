#!/bin/bash
# Submit one GCBD-Full timing pilot for a selected dataset x seed x scenario.
#
# Usage:
#   ./submit_gcbd_full_pilot.sh DATASET SEED SCENARIO_NUMBER EPOCHS WALLTIME
#   ./submit_gcbd_full_pilot.sh DATASET SEED SCENARIO_NUMBER EPOCHS WALLTIME --submit
#
# Recommended conservative pilot:
#   ./submit_gcbd_full_pilot.sh IMDb 18 2 50 24:00:00
#   ./submit_gcbd_full_pilot.sh IMDb 18 2 50 24:00:00 --submit

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
    sed -n '2,11p' "$0"
}

if (( $# < 5 || $# > 6 )); then
    usage >&2
    exit 2
fi

dataset_input="$1"
seed="$2"
scenario_number="$3"
epochs="$4"
walltime="$5"
action="${6:-}"

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

if [[ ! "$walltime" =~ ^([0-9]{1,2}):([0-5][0-9]):([0-5][0-9])$ ]]; then
    echo "ERROR: WALLTIME must use HH:MM:SS; received: $walltime" >&2
    exit 2
fi
hours=$((10#${BASH_REMATCH[1]}))
minutes=$((10#${BASH_REMATCH[2]}))
seconds=$((10#${BASH_REMATCH[3]}))
if (( (hours == 0 && minutes == 0 && seconds == 0) || hours > 24 || (hours == 24 && (minutes != 0 || seconds != 0)) )); then
    echo "ERROR: WALLTIME must be greater than zero and cannot exceed 24:00:00" >&2
    exit 2
fi

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

output_root="$GCBD_OUTPUT_BASE/epochs_${epochs}"
method_done="$output_root/$dataset/$scenario/GCBD-Full_B0.3/seed_${seed}/DONE"
if [[ -f "$method_done" ]]; then
    echo "GCBD-Full is already complete for this cell:" >&2
    echo "  $method_done" >&2
    exit 0
fi

mkdir -p "$GCBD_LOG_ROOT" "$output_root"

dataset_slug=${dataset//-/}
job_name="M-${dataset_slug}-S${scenario_number}-GCBD-Full-s${seed}-b0p3-e${epochs}-pilot"
export_argument="ALL,GCBD_CONFIG=$GCBD_CONFIG,GCBD_PROJECT_ROOT=$GCBD_PROJECT_ROOT,GCBD_VENV=$GCBD_VENV,GCBD_HF_HOME=$GCBD_HF_HOME"

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
    --method GCBD-Full
    --seed "$seed"
    --budget 0.3
    --epochs "$epochs"
    --output-root "$output_root"
)

echo "================================================================"
echo "GCBD-Full TIMING PILOT"
echo "Dataset: $dataset"
echo "Scenario: $scenario"
echo "Seed: $seed"
echo "Epochs: $epochs"
echo "Requested walltime: $walltime"
echo "Teacher: $teacher_done"
echo "Output: $output_root"
echo "Action: $([[ "$action" == "--submit" ]] && echo SUBMIT || echo DRY-RUN)"
echo "================================================================"

if [[ "$action" != "--submit" ]]; then
    printf 'DRY-RUN command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    echo "Re-run the same command with --submit after review."
    exit 0
fi

for required_command in sbatch squeue; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "ERROR: $required_command is required when --submit is used" >&2
        exit 4
    fi
done

active_job=$(squeue -h -u "${USER:?USER is not set}" --name="$job_name" --format='%i' | head -n 1 || true)
if [[ -n "$active_job" ]]; then
    echo "An identical pilot is already active: job $active_job"
    exit 0
fi

job_id=$("${command[@]}")
job_id="${job_id%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse the SLURM job ID: $job_id" >&2
    exit 5
fi

echo "Submitted GCBD-Full pilot job: $job_id"
echo "Log: $GCBD_LOG_ROOT/$job_name-$job_id.out"
echo "Accounting: sacct -X -j $job_id --format=JobID,JobName,State,Elapsed,Timelimit,ExitCode"
