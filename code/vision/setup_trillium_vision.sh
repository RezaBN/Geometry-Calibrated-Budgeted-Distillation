#!/bin/bash
# One-time Trillium bootstrap for GCBD Vision.
# Run from a Trillium login node, preferably trillium-gpu.scinet.utoronto.ca.
# Experiment compute jobs remain offline and use only staged assets.
set -euo pipefail

if [[ -z "${SCRATCH:-}" ]]; then
  echo "SCRATCH is not defined. Log in to Trillium and run this there." >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${GCBD_VISION_ROOT:-$SCRATCH/gcbd-vision}
CODE_DIR=${GCBD_VISION_CODE_DIR:-$ROOT/code}
VENV=${GCBD_VISION_VENV:-$ROOT/env/venv}
CONFIG=${GCBD_VISION_CONFIG:-$CODE_DIR/vision_experiments.json}
TORCH_HOME_DIR=${GCBD_VISION_TORCH_HOME:-$ROOT/cache/torch}

TORCH_VERSION=${GCBD_TORCH_VERSION:-2.10.0}
TORCHVISION_VERSION=${GCBD_TORCHVISION_VERSION:-0.25.0}
PYTORCH_INDEX=${GCBD_PYTORCH_INDEX:-https://download.pytorch.org/whl/cu126}

RECREATE=0
SKIP_ASSETS=0
SKIP_CIFAR100=0

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  --recreate-venv   Remove and recreate the Python virtual environment.
  --skip-assets     Install software only; do not stage teacher/CIFAR assets.
  --skip-cifar100   Stage teacher weights but do not download CIFAR-100.
  -h, --help        Show this help.

The package is designed for:
  code:       $SCRATCH/gcbd-vision/code
  venv:       $SCRATCH/gcbd-vision/env/venv
  outputs:    $SCRATCH/gcbd-vision/outputs
USAGE
}

while (($#)); do
  case "$1" in
    --recreate-venv) RECREATE=1 ;;
    --skip-assets) SKIP_ASSETS=1 ;;
    --skip-cifar100) SKIP_CIFAR100=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ ! -f "$CODE_DIR/requirements-vision-trillium.txt" || ! -f "$CONFIG" ]]; then
  echo "GCBD Vision code was not found at $CODE_DIR" >&2
  echo "Extract/copy this package into $SCRATCH/gcbd-vision/code first." >&2
  exit 2
fi

mkdir -p \
  "$ROOT/cache/torch" \
  "$ROOT/datasets/CIFAR100" \
  "$ROOT/datasets/ImageNet" \
  "$ROOT/env" \
  "$ROOT/logs" \
  "$ROOT/models/pretrained" \
  "$ROOT/models/teachers_finetuned" \
  "$ROOT/outputs" \
  "$ROOT/tmp"

module --force purge
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if ((RECREATE)) && [[ -d "$VENV" ]]; then
  echo "Removing existing environment: $VENV"
  rm -rf "$VENV"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creating virtual environment: $VENV"
  python -m venv "$VENV"
else
  echo "Reusing virtual environment: $VENV"
fi

source "$VENV/bin/activate"
export TORCH_HOME="$TORCH_HOME_DIR"
export PIP_DISABLE_PIP_VERSION_CHECK=1

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "$PYTORCH_INDEX"
python -m pip install --upgrade -r "$CODE_DIR/requirements-vision-trillium.txt"
python -m pip check

python - <<'PY'
import platform
import torch
import torchvision
import timm
print("Python:", platform.python_version())
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("timm:", timm.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA visible on this login node:", torch.cuda.is_available())
if torch.version.cuda is None:
    raise SystemExit("ERROR: CPU-only PyTorch was installed")
PY

cd "$CODE_DIR"

if (( ! SKIP_ASSETS )); then
  PREP_ARGS=(--config "$CONFIG")
  if ((SKIP_CIFAR100)); then
    PREP_ARGS+=(--skip-cifar100)
  fi
  echo "Staging public teacher weights and CIFAR-100."
  echo "If login-node I/O policy or site load makes this inappropriate, Ctrl-C and rerun with --skip-assets; transfer the assets separately into $SCRATCH."
  python prepare_vision_assets.py "${PREP_ARGS[@]}"

  if [[ -d "$ROOT/datasets/ImageNet/train" ]] && \
     { [[ -d "$ROOT/datasets/ImageNet/val" ]] || [[ -d "$ROOT/datasets/ImageNet/validation" ]]; }; then
    python verify_vision_assets.py --config "$CONFIG"
  else
    python verify_vision_assets.py --config "$CONFIG" --skip-imagenet
  fi
fi

python -m pip freeze | sort > "$CODE_DIR/environment-trillium.freeze.txt"
module list 2> "$CODE_DIR/environment-trillium.modules.txt" || true

cat <<SUMMARY

GCBD Vision Trillium bootstrap completed.
Environment: $VENV
Code:        $CODE_DIR
Freeze:      $CODE_DIR/environment-trillium.freeze.txt

ImageNet is not downloaded automatically. Stage an authorized ImageFolder tree at:
  $ROOT/datasets/ImageNet/train/<class>/...
  $ROOT/datasets/ImageNet/val/<class>/...

Before training, from the GPU login node and from $CODE_DIR:
  mkdir -p logs
  sbatch trillium_gpu_preflight.sbatch
SUMMARY
