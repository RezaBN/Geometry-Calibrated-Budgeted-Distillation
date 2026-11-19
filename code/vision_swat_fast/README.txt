GCBD VISION SWAT FAST-V1  (2026-11-03)
======================================
Purpose: optimize how Vision sliced Wasserstein attention transport (SWAT)
executes, preserving the existing projected weighted-quantile objective.

Changes:
  1. Vectorized batched samples and projection quantiles (chunked to bound
     GPU memory); exact zero-weight support skipping is retained.
  2. Seeded projection directions and sorted projected supports cached in
     bounded process-local LRU maps (actual fp16 teacher cache supports hashed).
  3. GeometryCacheReader.get_many groups reads by shard, preserves ID order;
     Vision reader caches up to 256 verified shards (CIFAR100 ~196 shards).
     Teacher weights for each layer transfer in one batched CPU->GPU copy.
  4. Removes per-sample GPU/Python sum checks in SWAT and per-layer forced
     CPU scalar diagnostic transfer. Vision diagnostics are now detached
     tensors; use float(value) if scalar values are explicitly required.

No changes to:
  - gcbd_v2/common/geometry.py (original reference remains for NLP)
  - student or teacher model weights
  - GCBD selection, UOT, loss coefficients, projection count, data splits
  - cache file format, cache identity or experimental configuration
  - existing teacher checkpoints, student checkpoint or saved results

Bundle contents:
  apply_vision_swat_fast.py     guarded install; copies backups; CPU checks
  rollback_vision_swat_fast.py  restores original files using saved backups
  vision_fast.py               optimized Vision-only operations
  verify_fast_swat.py          numerical loss/gradient + cache order tests
  benchmark_vision_swat.py     synthetic reference-vs-fast forward/backward
  verify_on_h100.sbatch        H100 regression and short benchmark

Install (LOGIN NODE, when no Vision jobs are active):
  export GCBD_HOME=/scratch/koohi/RBN
  export PH2="$GCBD_HOME/code/gcbd_phase2_integrated"
  mkdir -p "$GCBD_HOME/jobs/vision_swat_fast"
  unzip -o "$GCBD_HOME/jobs/gcbd_vision_swat_fast_v1.zip" -d "$GCBD_HOME/jobs/vision_swat_fast"
  module --force purge
  module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11.5
  source "$GCBD_HOME/envs/vision/venv/bin/activate"
  export PYTHONPATH="$PH2/vision:$PH2${PYTHONPATH:+:$PYTHONPATH}"
  python "$GCBD_HOME/jobs/vision_swat_fast/apply_vision_swat_fast.py" "$PH2"

GPU validation (LOGIN NODE; submit, don't execute GPU benchmark on login):
  mkdir -p "$GCBD_HOME/logs/vision/swat_fast"
  sbatch --parsable \
    --output="$GCBD_HOME/logs/vision/swat_fast/%x-%j.out" \
    "$GCBD_HOME/jobs/vision_swat_fast/verify_on_h100.sbatch"
  # Wait for COMPLETED 0:0, then grep log for:
  # VISION SWAT LOSS + GRADIENT REGRESSION: PASS
  # H100 SWAT BENCHMARK + GRADIENT CHECK: PASS

Rollback (LOGIN NODE, when no Vision jobs are active):
  python "$GCBD_HOME/jobs/vision_swat_fast/rollback_vision_swat_fast.py" "$PH2"

Evidence from local tests:
  - 15 upstream Phase1/Phase2 tests: PASS (PyTorch 2.10 CPU).
  - SWAT forward+backward regression on CNN/CNN, ViT/CNN,
    heterogeneous support, fp16 cached support and zero-mass locations: PASS.
  - Shard-grouped get_many supports original order/duplicate IDs: PASS.
  - CPU synthetic minibatch: B=4, 56x56, P=8: ~3.46x faster; this is NOT
    an H100 or full-CIFAR100 speed claim.

Scientific caveat:
  GPU loss/gradient agreement and actual H100 speed MUST be checked before
  resuming the 50-epoch job. A resumed job after source change uses two
  execution versions and is not a bitwise reproduction of uninterrupted
  training. Preserve original job log and epoch ledger for reproducibility.
