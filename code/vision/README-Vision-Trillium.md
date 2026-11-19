# GCBD Vision on Trillium H100

This package runs the GCBD vision teacher and distillation experiments as independent **single-H100 Slurm jobs** on Trillium. The scientific method/model implementation is preserved from the revised GCBD codebase; the runtime, paths, setup, and submission layer are specialized for Trillium.

## Trillium assumptions built into this package

- Submit GPU jobs from the Trillium GPU login node, not from a compute job.
- A single-GPU job requests `--gpus-per-node=1` and therefore receives a quarter GPU node: **1 H100 80 GB, 24 CPU cores, and about 188 GiB host RAM**.
- The package does **not** request `--mem`; Trillium ignores memory requests for these jobs.
- The package does **not** force the GPU `compute` partition; Trillium requires the scheduler to select the appropriate GPU partition.
- All writable campaign paths are under `$SCRATCH/gcbd-vision`, because Trillium compute nodes require job output/writes on scratch while home/project are read-only.
- Ordinary GPU jobs have a maximum 24-hour walltime. The runner checkpoints after each completed epoch and supports safe stale-lock clearing for time-limited resubmissions.
- Modules loaded in GPU jobs are only `StdEnv/2023`, `cuda/12.6`, and `python/3.11.5`.
- Production experiment jobs run offline after assets have been staged.

## Experimental matrix

| Scenario | Teacher | Student |
|---|---|---|
| A | ResNet-50 | ResNet-18 |
| B | ResNet-50 | MobileNetV2 |
| C | ViT-B/16 | DeiT-Tiny/16 |
| D | ConvNeXt-Base | ResNet-18 |

Methods: Student, KD, FitNet, AT, CRD, DKD, ReviewKD, DIST, OTKD, GCBD-Full, GCBD-NoCurriculum, GCBD-NoBudget, GCBD-NoStage1, GCBD-NoUOT.

Default seeds: `18`, `64`.

Teacher identity is `dataset × teacher architecture × seed`, so A/B share the same ResNet-50 teacher. The complete default matrix contains 12 unique teachers and 224 method jobs.

## Current resource and batch settings

Each experiment is one independent Trillium single-GPU job:

```text
nodes             = 1
ntasks            = 1
gpus-per-node     = 1
cpus-per-task     = 24
host memory        = Trillium-managed (~188 GiB for a 1-GPU allocation)
precision          = BF16
TF32               = enabled
DataLoader workers = 20
```

Method batch sizes:

| Dataset | A | B | C | D |
|---|---:|---:|---:|---:|
| CIFAR-100 | 256 | 256 | 128 | 128 |
| ImageNet | 256 | 256 | 128 | 128 |

Teacher batches remain 128 by default, with scenarios C/D capped at 64.

Default method epochs are 100 for both CIFAR-100 and ImageNet. ImageNet teacher epochs remain 0 by default, meaning the staged ImageNet-pretrained teacher is evaluated/cached without additional ImageNet fine-tuning unless `--teacher-epochs` is explicitly provided.

## 1. Place the package in scratch

From Trillium:

```bash
mkdir -p "$SCRATCH/gcbd-vision/code"
cd "$SCRATCH/gcbd-vision/code"
```

Extract/copy the contents of this package directly into that directory. All active write paths in `vision_experiments.json` use `$SCRATCH/gcbd-vision/...`.

## Scheduler account

The supplied default account is `def-koohi`. Trillium allocations are system-specific, so if your valid Trillium Slurm account differs, edit `account` in `vision_experiments.json` and the two standalone `.sbatch` files, or pass `--account YOUR_ACCOUNT` to `submit_vision_trillium.py`.

## 2. Create the virtual environment and dependencies

A helper is supplied:

```bash
cd "$SCRATCH/gcbd-vision/code"
chmod +x setup_trillium_vision.sh
./setup_trillium_vision.sh
```

It loads:

```bash
module --force purge
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
```

and creates:

```text
$SCRATCH/gcbd-vision/env/venv
```

The setup pins:

```text
torch==2.10.0
torchvision==0.25.0
CUDA wheel family: cu126
```

then installs `requirements-vision-trillium.txt` and writes:

```text
environment-trillium.freeze.txt
environment-trillium.modules.txt
```

Trillium login nodes are for preparation rather than sustained I/O/compute. If you prefer to stage datasets/model assets separately, use:

```bash
./setup_trillium_vision.sh --skip-assets
```

## 3. Public teacher assets and CIFAR-100

The three teacher initialization assets are:

```text
ResNet-50       IMAGENET1K_V2
ViT-B/16        IMAGENET1K_V1
ConvNeXt-Base   IMAGENET1K_V1
```

They are stored under:

```text
$SCRATCH/gcbd-vision/models/pretrained/
```

CIFAR-100 is staged at:

```text
$SCRATCH/gcbd-vision/datasets/CIFAR100
```

Students deliberately start from random initialization. No pretrained ResNet-18, MobileNetV2, or DeiT-Tiny student checkpoints are downloaded.

To stage assets manually after software installation:

```bash
source "$SCRATCH/gcbd-vision/env/venv/bin/activate"
cd "$SCRATCH/gcbd-vision/code"
python prepare_vision_assets.py --config vision_experiments.json
```

## 4. ImageNet

ImageNet is not downloaded automatically. Stage an authorized ImageFolder-compatible copy at:

```text
$SCRATCH/gcbd-vision/datasets/ImageNet/
├── train/
│   ├── n01440764/
│   └── ... 1000 class directories
└── val/
    ├── n01440764/
    └── ... the same 1000 class directories
```

`validation/` is also accepted in place of `val/`. Train and validation class mappings must match exactly.

If you already have an authorized ImageNet tree at another Trillium-readable location, you may point `dataset_paths.ImageNet` in `vision_experiments.json` at that read-only location; outputs/checkpoints must still remain in `$SCRATCH`.

## 5. Verify all assets

Without ImageNet:

```bash
source "$SCRATCH/gcbd-vision/env/venv/bin/activate"
cd "$SCRATCH/gcbd-vision/code"
python verify_vision_assets.py --config vision_experiments.json --skip-imagenet
```

With ImageNet staged:

```bash
python verify_vision_assets.py --config vision_experiments.json
```

A stronger CPU model/hook check is available with `--forward`.

Success ends with:

```text
LOCAL VISION ASSET VERIFICATION PASSED
```

## 6. Trillium H100 preflight

From the GPU login node, while in the scratch code directory:

```bash
cd "$SCRATCH/gcbd-vision/code"
mkdir -p logs
sbatch trillium_gpu_preflight.sbatch
```

The preflight checks the H100, CUDA, BF16, torchvision/timm compatibility, and a GPU forward pass for every student architecture.

Success ends with:

```text
TRILLIUM H100 PREFLIGHT PASSED
```

## 7. Teacher jobs

Preview ImageNet teachers:

```bash
./submit_vision_teachers.sh ImageNet
```

Submit:

```bash
./submit_vision_teachers.sh ImageNet --submit
```

Specify seed, epochs, and per-job walltime at submission time:

```bash
./submit_vision_teachers.sh ImageNet \
  --only-seed 18 \
  --teacher-epochs 0 \
  --teacher-time 01:00:00 \
  --submit
```

Each unique teacher/seed pair is a separate one-H100 Slurm job.

For all ImageNet teachers with the two default seeds, the planner creates six jobs: ResNet-50 × 2 seeds, ViT-B/16 × 2 seeds, ConvNeXt-Base × 2 seeds.

## 8. Method pilot

Before launching a cell, run one method. Example:

```bash
python submit_vision_trillium.py \
  --config vision_experiments.json \
  --skip-teacher-jobs \
  --only-dataset ImageNet \
  --only-scenario A \
  --only-seed 18 \
  --only-method GCBD-Full \
  --epochs 10 \
  --time 08:00:00
```

Append `--submit` after inspecting the generated `sbatch` command.

The generated command should contain:

```text
--nodes=1
--ntasks=1
--gpus-per-node=1
--cpus-per-task=24
```

It should **not** contain `--mem` or an explicit `--partition`.

## 9. Submit one method cell

A cell is one dataset × scenario × seed and creates 14 independent method jobs:

```bash
./submit_vision_method_cell.sh ImageNet 18 A 100 --time 24:00:00
```

Submit after preview:

```bash
./submit_vision_method_cell.sh ImageNet 18 A 100 --time 24:00:00 --submit
```

## 10. Resume after the 24-hour limit

Trillium's regular GPU jobs are limited to 24 hours. The runner saves `checkpoint.pt` after every completed epoch. If Slurm kills a run at the walltime, wait until the old job has left `squeue`, then resubmit the same logical run with:

```bash
./submit_vision_method_cell.sh ImageNet 18 A 100 \
  --time 24:00:00 \
  --clear-stale-lock \
  --submit
```

Completed methods are skipped. Incomplete methods resume from their checkpoint. `--clear-stale-lock` removes a lock only after the runner verifies that its prior Slurm job is no longer active.

Do not delete `checkpoint.pt` from incomplete runs.

## 11. Full matrix

Preview:

```bash
python submit_vision_trillium.py \
  --config vision_experiments.json \
  --skip-teacher-jobs
```

Submit:

```bash
python submit_vision_trillium.py \
  --config vision_experiments.json \
  --skip-teacher-jobs \
  --submit
```

The submitter must be run from a login node. It explicitly refuses `--submit` if it detects that it is running inside a Slurm compute job, because Trillium prohibits job submission from jobs.

## 12. Monitoring

Useful commands:

```bash
squeue -u "$USER"
scontrol show job JOBID
jobperf JOBID
sacct -j JOBID
scancel JOBID
```

The campaign logs are in:

```text
$SCRATCH/gcbd-vision/logs
```

## 13. Aggregation

For the 100-epoch campaign:

```bash
python aggregate_vision_results.py \
  --config vision_experiments.json \
  --epochs 100
```

The aggregator pairs each method with Student at the same dataset, scenario, seed, and epoch count, then writes enriched run data, aggregated mean/std/count tables, missing-cell diagnostics, and plots.

## Main files

| File | Role |
|---|---|
| `GCBD-Vision_RUN.py` | Run one teacher or method experiment |
| `vision_experiments.json` | Scientific/runtime/resource configuration |
| `trillium_vision_h100.sbatch` | Single-H100 Trillium runtime |
| `trillium_gpu_preflight.sbatch` | Short H100/CUDA/BF16 preflight |
| `submit_vision_trillium.py` | Login-node Slurm matrix planner/submitter |
| `submit_vision_teachers.sh` | Teacher-only helper |
| `submit_vision_method_cell.sh` | One dataset × scenario × seed helper |
| `setup_trillium_vision.sh` | Environment/dependency/asset bootstrap |
| `prepare_vision_assets.py` | Public teacher/CIFAR asset staging |
| `verify_vision_assets.py` | Strict local/offline verification |
| `aggregate_vision_results.py` | Paired result aggregation |
| `gcbd_vision/` | Scientific implementation |

## Reproducibility notes

- BF16 and TF32 remain enabled on H100.
- CNN and DeiT learning rates are automatically scaled from their configured reference batch sizes.
- Every method within a scenario uses the same configured batch size.
- Method outputs are separated by dataset/scenario/method/budget/seed/epoch root.
- Teacher prediction-query ratio, geometry-query ratio, and total teacher-forward ratio are reported separately.
- Production jobs set Hugging Face/Transformers/Datasets offline flags to prevent accidental downloads during experiments.
