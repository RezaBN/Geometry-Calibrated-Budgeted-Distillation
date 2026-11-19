# Trillium adaptation notes

This package is derived from the revised GCBD single-H100 codebase and adapts the orchestration layer to Trillium.

Key changes:

1. All writable paths use `$SCRATCH/gcbd-vision/...`.
2. Single-GPU jobs request `--gpus-per-node=1`.
3. Jobs use one task and 24 CPU cores (`--ntasks=1 --cpus-per-task=24`).
4. `--mem` is deliberately omitted because Trillium ignores memory requests; one GPU receives roughly 188 GiB host RAM automatically.
5. No GPU partition is explicitly requested; Trillium selects it.
6. The submitter refuses nested `sbatch` submission from a Slurm job.
7. `--clear-stale-lock`, `--overwrite`, and `--no-resume` are exposed by the planner.
8. Current balanced method batches are A/B=256 and C/D=128 for both datasets.
9. DataLoader workers are set to 20 for the 24-core single-GPU allocation.
10. Default ImageNet method horizon is 100 epochs; long jobs can resume across the 24-hour Trillium walltime cap.
11. Scientific losses, model mappings, metrics, and GCBD logic are otherwise preserved.
