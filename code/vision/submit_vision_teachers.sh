#!/bin/bash
set -euo pipefail

if (( $# < 1 )); then
  echo "Usage: $0 CIFAR100|ImageNet|ALL [submit_vision_trillium options]" >&2
  echo "Example: $0 ImageNet --only-seed 18 --teacher-epochs 0 --teacher-time 01:00:00 --submit" >&2
  exit 2
fi

DATASET=$1
shift
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

ARGS=(--config "$CONFIG" --teacher-only)
case "$DATASET" in
  CIFAR100|ImageNet) ARGS+=(--only-dataset "$DATASET") ;;
  ALL|all) ;;
  *) echo "Dataset must be CIFAR100, ImageNet, or ALL" >&2; exit 2 ;;
esac

"$PYTHON" "$SCRIPT_DIR/submit_vision_trillium.py" "${ARGS[@]}" "$@"
