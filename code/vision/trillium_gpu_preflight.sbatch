#!/bin/bash
#SBATCH --account=def-koohi
#SBATCH --job-name=gcbd-h100-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/koohi/RBN/logs/vision/%x-%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail

# ==========================================
# Explicit RBN PROJECT PATHS
# ==========================================

PROJECT_ROOT="/scratch/koohi/RBN"
CODE_DIR="$PROJECT_ROOT/code/vision"
VENV="$PROJECT_ROOT/envs/vision/venv"
TORCH_HOME="$PROJECT_ROOT/cache/vision/torch"

# ==========================================
# Validate project files
# ==========================================

echo "=========================================="
echo "GCBD H100 PREFLIGHT"
echo "=========================================="

test -f "$CODE_DIR/gcbd_vision/__init__.py"
test -f "$CODE_DIR/gcbd_vision/models.py"
test -x "$VENV/bin/python"

mkdir -p "$TORCH_HOME"

# ==========================================
# Load Trillium modules
# ==========================================

module --force purge
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

source "$VENV/bin/activate"

# ==========================================
# Explicit Python paths
# ==========================================

cd "$CODE_DIR"

export PYTHONPATH="$CODE_DIR"
export TORCH_HOME="$TORCH_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-unknown}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unset}"
echo "Project: $PROJECT_ROOT"
echo "Code: $CODE_DIR"
echo "Python: $(which python)"

echo "=========================================="
echo "GPU DETECTION"
echo "=========================================="

nvidia-smi

# ==========================================
# PYTORCH AND MODEL VALIDATION
# ==========================================

python -u - <<'PY'
import os
import sys
from pathlib import Path

PROJECT = Path("/scratch/koohi/RBN/code/vision")
sys.path.insert(0, str(PROJECT))

print("Python executable:", sys.executable, flush=True)
print("Working directory:", os.getcwd(), flush=True)
print("Project exists:", PROJECT.exists(), flush=True)
print("Package exists:", (PROJECT / "gcbd_vision/__init__.py").exists(), flush=True)
print("Python path:", sys.path[:4], flush=True)

# Imports
import torch
import torchvision
import timm

from gcbd_vision.config import SCENARIOS
from gcbd_vision.models import (
    FeatureModel,
    create_model,
    layer_names,
)

print("PyTorch:", torch.__version__, flush=True)
print("Torchvision:", torchvision.__version__, flush=True)
print("timm:", timm.__version__, flush=True)
print("CUDA runtime:", torch.version.cuda, flush=True)

# GPU checks
assert torch.cuda.is_available(), "CUDA is unavailable"

device = torch.device("cuda:0")

print("GPU:", torch.cuda.get_device_name(0), flush=True)

properties = torch.cuda.get_device_properties(0)

print(
    "GPU memory:",
    round(properties.total_memory / 1024**3, 2),
    "GiB",
    flush=True,
)

assert torch.cuda.is_bf16_supported(), \
    "H100 BF16 support is unavailable"

print("BF16 supported: PASS", flush=True)

# Basic BF16 matrix test
a = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
b = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)

c = a @ b

torch.cuda.synchronize()

assert torch.isfinite(c).all(), "BF16 matrix multiplication failed"

print("BF16 matrix multiplication: PASS", flush=True)

del a, b, c
torch.cuda.empty_cache()

# ==========================================
# STUDENT ARCHITECTURE VALIDATION
# ==========================================

architectures = sorted({
    spec["student"] for spec in SCENARIOS.values()
})

print("Student architectures:", architectures, flush=True)

for architecture in architectures:

    print(
        f"Testing architecture: {architecture}",
        flush=True,
    )

    model = FeatureModel(
        create_model(architecture, 1000),
        layer_names(architecture),
    ).eval().to(device)

    x = torch.zeros(
        1, 3, 224, 224,
        device=device,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            logits, features = model(x)

    assert logits.shape == (1, 1000), \
        f"Invalid logits shape for {architecture}: {logits.shape}"

    assert len(features) == len(layer_names(architecture)), \
        f"Missing feature layers for {architecture}"

    torch.cuda.synchronize()

    print(
        f"GPU FORWARD OK: {architecture}",
        flush=True,
    )

    del model, x, logits, features
    torch.cuda.empty_cache()

# ==========================================
# FINAL VALIDATION
# ==========================================

torch.cuda.synchronize()

print(
    "Peak GPU memory allocated:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    "GiB",
    flush=True,
)

print("==========================================", flush=True)
print("TRILLIUM H100 PREFLIGHT PASSED", flush=True)
print("==========================================", flush=True)

PY
