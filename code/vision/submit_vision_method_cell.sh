#!/bin/bash
set -euo pipefail

if (( $# < 4 )); then
  echo "Usage: $0 DATASET SEED SCENARIO EPOCHS [--time HH:MM:SS] [--clear-stale-lock] [--submit]" >&2
  echo "Example: $0 ImageNet 18 A 100 --time 24:00:00 --submit" >&2
  exit 2
fi

DATASET=$1
SEED=$2
SCENARIO=$3
EPOCHS=$4
shift 4

case "$SCENARIO" in
  1|a|A) SCENARIO=A ;;
  2|b|B) SCENARIO=B ;;
  3|c|C) SCENARIO=C ;;
  4|d|D) SCENARIO=D ;;
  *) echo "Scenario must be A-D or 1-4" >&2; exit 2 ;;
esac

case "$DATASET" in
  CIFAR100|ImageNet) ;;
  cifar100) DATASET=CIFAR100 ;;
  imagenet|IMAGENET) DATASET=ImageNet ;;
  *) echo "Dataset must be CIFAR100 or ImageNet" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG=${GCBD_VISION_CONFIG:-$SCRIPT_DIR/vision_experiments.json}

module --force purge
module load StdEnv/2023
module load python/3.11.5

if [[ -n "${GCBD_VISION_PYTHON:-}" ]]; then
  PYTHON=$GCBD_VISION_PYTHON
else
  VENV=$(python -c 'import json,os,sys; c=json.load(open(sys.argv[1])); print(os.path.expandvars(os.path.expanduser(c["venv"])))' "$CONFIG")
  PYTHON="$VENV/bin/python"
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  echo "Create the Trillium virtual environment first or set GCBD_VISION_PYTHON." >&2
  exit 2
fi

"$PYTHON" "$SCRIPT_DIR/submit_vision_trillium.py" \
  --config "$CONFIG" \
  --skip-teacher-jobs \
  --only-dataset "$DATASET" \
  --only-scenario "$SCENARIO" \
  --only-seed "$SEED" \
  --epochs "$EPOCHS" \
  "$@"
