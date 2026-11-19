#!/bin/bash
#SBATCH --account=def-koohi
#SBATCH --job-name=gcbd-trillium-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -z "${SCRATCH:-}" ]]; then
  echo "SCRATCH is not defined. This runtime is intended for Trillium." >&2
  exit 2
fi

GCBD_VISION_PROJECT_ROOT=${GCBD_VISION_PROJECT_ROOT:-$SCRIPT_DIR}
GCBD_VISION_CONFIG=${GCBD_VISION_CONFIG:-$GCBD_VISION_PROJECT_ROOT/vision_experiments.json}
GCBD_VISION_VENV=${GCBD_VISION_VENV:-$SCRATCH/gcbd-vision/env/venv}
GCBD_VISION_TORCH_HOME=${GCBD_VISION_TORCH_HOME:-$SCRATCH/gcbd-vision/cache/torch}

module --force purge
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if [[ ! -x "$GCBD_VISION_VENV/bin/python" ]]; then
  echo "Missing Python environment: $GCBD_VISION_VENV/bin/python" >&2
  exit 2
fi
if [[ ! -f "$GCBD_VISION_PROJECT_ROOT/GCBD-Vision_RUN.py" ]]; then
  echo "Missing runner: $GCBD_VISION_PROJECT_ROOT/GCBD-Vision_RUN.py" >&2
  exit 2
fi
if [[ ! -f "$GCBD_VISION_CONFIG" ]]; then
  echo "Missing configuration: $GCBD_VISION_CONFIG" >&2
  exit 2
fi

source "$GCBD_VISION_VENV/bin/activate"

export TORCH_HOME="$GCBD_VISION_TORCH_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TIMM_FUSED_ATTN=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTHONPATH="$GCBD_VISION_PROJECT_ROOT:$GCBD_VISION_PROJECT_ROOT/..${PYTHONPATH:+:$PYTHONPATH}"

cd "$GCBD_VISION_PROJECT_ROOT"

case "$PWD" in
  "$SCRATCH"/*) ;;
  *) echo "Trillium jobs must run/write from SCRATCH; current directory is $PWD" >&2; exit 2 ;;
esac

mkdir -p "$GCBD_VISION_TORCH_HOME" "$SCRATCH/gcbd-vision/logs"

echo "Host: $(hostname)"
echo "SLURM job: ${SLURM_JOB_ID:-not-under-slurm}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "Python: $GCBD_VISION_VENV/bin/python"
"$GCBD_VISION_VENV/bin/python" --version
nvidia-smi -L

"$GCBD_VISION_VENV/bin/python" - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA runtime seen by torch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "PyTorch cannot see the allocated Trillium GPU"
print("GPU:", torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())
assert torch.cuda.is_bf16_supported(), "Expected BF16 support on Trillium H100"
PY

"$GCBD_VISION_VENV/bin/python" -u "$GCBD_VISION_PROJECT_ROOT/GCBD-Vision_RUN.py" \
  --config "$GCBD_VISION_CONFIG" \
  "$@"
