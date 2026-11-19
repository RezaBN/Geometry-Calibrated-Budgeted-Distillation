# GCBD-NLP on Trillium

This package runs one experiment in each GPU job and uses one NVIDIA H100 per job. The submission program expands the configured matrix as:

`dataset × scenario × seed × method × applicable budget`

It first submits one teacher-preparation job for every dataset–scenario–seed combination. Each corresponding method job receives an `afterok` dependency on that teacher job. This prevents concurrent teacher-cache corruption and ensures that all methods for a seed use the same fine-tuned teacher.

## 1. Package contents

| File | Purpose |
|---|---|
| `GCBD-NLP_RUN.py` | Runs exactly one teacher or one method experiment |
| `trillium_experiments.json` | Experiment matrix, epochs, budgets, paths, and H100 parameters |
| `submit_trillium.py` | Creates and submits the independent SLURM jobs |
| `trillium_h100.sbatch` | One-H100 job environment |
| `prepare_hf_assets.py` | Optional cache-based Hugging Face prefetch helper |
| `verify_local_assets.py` | Verifies explicit model and dataset directories with offline mode enforced |
| `aggregate_results.py` | Combines per-job results and reports missing runs |
| `requirements-trillium.txt` | Python environment requirements |

## 2. Place the package in scratch

From the GPU login node:

```bash
mkdir -p "$SCRATCH/gcbd-nlp/code"
cd "$SCRATCH/gcbd-nlp/code"
```

Transfer all package files into this directory. Submit GPU jobs only from `trillium-gpu.alliancecan.ca`.

## 3. Create the Python environment once

Choose the intended project path explicitly if you belong to several projects; `$PROJECT` may not identify the project you expect.

```bash
module --force purge
module load StdEnv/2023
module load gcc/12.3
module load arrow/25.0.0
module load cuda/12.6
module load python/3.11.5

mkdir -p "$SCRATCH/gcbd-nlp/env"
virtualenv --no-download "$SCRATCH/gcbd-nlp/env/venv"
source "$SCRATCH/gcbd-nlp/env/venv/bin/activate"

pip install --no-index -r "$SCRATCH/gcbd-nlp/code/requirements-trillium.txt"
pip freeze > "$SCRATCH/gcbd-nlp/code/environment.lock.txt"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
PY
```

Use `module spider python` and `module spider cuda` if the documented module versions have changed. Create the environment on a login node or Open OnDemand—not inside a compute job.

## 4. Edit the experiment configuration

Open `trillium_experiments.json` and replace:

```json
"account": "REPLACE_WITH_ACCOUNT"
```

with the applicable `def-...`, `rrg-...`, or `rpp-...` account. Also verify `project_root`, `venv`, and `hf_home`.

For a temporary override without editing JSON, pass `--account ACCOUNT_NAME` to `submit_trillium.py`.

The principal experiment controls are:

```json
"datasets": ["IMDb"],
"scenarios": ["Setup_1_BERT_TinyBERT"],
"methods": ["Student", "KD", "GCBD-Full"],
"seeds": [18, 64],
"budgets": [0.1, 0.3, 0.5]
```

Epochs are under `runtime`:

```json
"epochs_cls": 10,
"epochs_lm": 3,
"teacher_epochs_cls": 3,
"teacher_epochs_lm": 3
```

Every listed budget produces a separate job for GCBD methods whose behavior depends on a budget. Ordinary baselines and `GCBD-NoBudget` run once with recorded budget `0.0`, avoiding scientifically identical duplicate jobs.

### Configure walltime by experiment

SLURM walltime is a resource limit, not a runtime prediction. If a job reaches the limit, SLURM terminates it and the run will not create its `DONE` marker. The initial configuration therefore assigns the conservative Trillium maximum of `24:00:00` to every method until pilot measurements are available.

`slurm.method_time` is the fallback. Entries in `slurm.method_time_rules` can select any combination of `dataset`, `scenario`, `method`, and `budget`:

```json
"slurm": {
  "teacher_time": "24:00:00",
  "method_time": "24:00:00",
  "method_time_rules": [
    {"method": "Student", "time": "08:00:00"},
    {"dataset": "IMDb", "method": "KD", "time": "12:00:00"},
    {
      "dataset": "IMDb",
      "scenario": "Setup_4_BERT_DistilGPT2",
      "method": "GCBD-Full",
      "budget": 0.3,
      "time": "24:00:00"
    }
  ]
}
```

The most specific matching rule wins. If equally specific rules both match, the later rule wins. Thus, the exact four-field rule above overrides a general GCBD or IMDb rule. Times must use `HH:MM:SS`, be greater than zero, and not exceed Trillium's `24:00:00` limit.

After a representative pilot for each dataset–scenario–method family, inspect `Elapsed` and `State` with `sacct`, add a safety margin, and update the rules. The dry-run output shows the resolved `--time` for every job, and the submission manifest records it as `requested_time`.

## 5. Understand the H100 preset

The initial single-H100 settings are:

| Parameter | Value |
|---|---:|
| `mixed_precision` | `bf16` |
| `allow_tf32` | `true` |
| `max_length_cls` | 256 |
| `max_length_lm` | 256 |
| `train_batch_size_cls` | 32 |
| `eval_batch_size_cls` | 64 |
| `train_batch_size_lm` | 16 |
| `eval_batch_size_lm` | 32 |
| `gradient_accumulation_steps` | 2 |
| `num_workers` | 8 |

These are conservative starting values for an 80 GB H100 because teacher and student occupy the same GPU. Attention-based methods such as MiniLM and TinyBERT can consume substantially more memory than logits-only methods.

Sequence length changes the scientific input, not only memory usage. Keep it fixed across compared methods unless the research design explicitly studies sequence length.

If a job runs out of GPU memory:

1. Halve the relevant training batch size.
2. Double `gradient_accumulation_steps` to preserve the effective batch size.
3. Reduce the evaluation batch size if failure occurs during evaluation.
4. Reduce sequence length only if that change is scientifically acceptable and applied consistently.

Each successful result records its peak allocated/reserved GPU memory, per-device batch size, and effective batch size. Use those measurements before increasing a batch.

The evaluation-report defaults are also bounded for one H100 job: at most 2,000 paired examples for representation diagnostics, 1,000 paired points per model for t-SNE, 10,000 valid LM tokens for calibration/confusion diagnostics, and 1,000,000 sampled student weights. These limits affect plots only, not the full evaluation metrics. Set `generate_evaluation_plots` to `false` for a timing-only run.

## 6. Verify explicit local models and datasets

Compute jobs never resolve Hugging Face repository IDs. `model_paths` and `dataset_paths` in `trillium_experiments.json` map the canonical research identifiers to explicit directories under scratch:

```text
models/
  teachers/{bert-base-uncased,deberta-v3-base,gpt2}/
  students/{tinybert-L4-H256,distilbert-base-uncased,distilgpt2}/
  teachers_finetuned/
datasets/{SST-2,IMDb,DBPedia,WikiText-2}/
```

The downloaded datasets must be saved with `DatasetDict.save_to_disk()`. Before submission, enforce offline mode and validate every tokenizer, model configuration, checkpoint, task head, and saved dataset:

```bash
cd "$SCRATCH/gcbd-nlp/code"
source "$SCRATCH/gcbd-nlp/env/venv/bin/activate"
python verify_local_assets.py --config trillium_experiments.json
python -m json.tool local_assets_verified.json
```

`verify_local_assets.py` deserializes both the sequence-classification head and the appropriate causal- or masked-language-modeling head for every configured checkpoint, one model at a time. It also validates the required splits and columns for all four datasets. Consequently, this check can take several minutes and temporarily use host memory approximately equal to the largest model. `GCBD-NLP_RUN.py` uses `local_files_only=True` for all Transformers calls and `load_from_disk()` for all datasets. A missing mapping, corrupt checkpoint, incompatible task head, or malformed dataset is a pre-submission error; there is no remote fallback. Fine-tuned teachers are written separately under `models/teachers_finetuned/`.

## 7. Validate before submission

Validate one resolved run without loading models or data:

```bash
python GCBD-NLP_RUN.py \
  --config trillium_experiments.json \
  --mode method \
  --dataset IMDb \
  --scenario Setup_1_BERT_TinyBERT \
  --method GCBD-Full \
  --seed 18 \
  --budget 0.3 \
  --dry-run
```

Preview the entire SLURM plan:

```bash
python submit_trillium.py --config trillium_experiments.json
```

The default configuration contains five scenarios, fifteen methods, two seeds, and one budget for IMDb. It therefore plans 10 teacher jobs and 150 method jobs: non-budgeted methods run once, while budget-aware GCBD methods use the configured budgets.

## 8. Submit jobs

After reviewing the dry run:

```bash
python submit_trillium.py --config trillium_experiments.json --submit
```

The submission manifest is saved under `logs/`. The script refuses matrices above the configured `max_jobs_per_submission` safeguard.

Useful filtered submissions:

```bash
# One scenario, seed, and method
python submit_trillium.py --submit \
  --only-scenario Setup_1_BERT_TinyBERT \
  --only-seed 18 \
  --only-method GCBD-Full

# Prepare only the teachers
python submit_trillium.py --submit --teacher-only

# Submit methods after teachers already exist
python submit_trillium.py --submit --skip-teacher-jobs
```

Do not use `--skip-teacher-jobs` unless every selected teacher cache has a `DONE` marker. Method jobs fail explicitly rather than silently using an unfine-tuned teacher.

Completed runs are skipped by default. Use `--overwrite` only when you deliberately want to retrain and replace a completed result.

Teacher preparation also uses an atomic `.LOCK` directory. A duplicate teacher job fails instead of corrupting a cache. If a node failure or time limit leaves a stale lock, first confirm with `squeue` that no matching teacher job is active; then remove only that specific `.LOCK` directory and resubmit.

## 9. Monitor

```bash
squeue -u "$USER"
squeue -j JOB_ID
scontrol show job JOB_ID
jobperf JOB_ID
scancel JOB_ID
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode
```

Each job writes a log to `logs/JOB_NAME-JOB_ID.out`. The Trillium maximum walltime is 24 hours. If a run approaches that limit, reduce epochs or sample caps, or add an application-level checkpoint/resume design before launching the full matrix.

## 10. Outputs and aggregation

Method results are isolated by dataset, scenario, method/budget, and seed:

```text
outputs/
  IMDb/
    Setup_1_BERT_TinyBERT/
      GCBD-Full_B0.3/
        seed_18/
          run_config.json
          train_log.csv
          eval_metrics.json
          evaluation_report.json
          calibration_bins.csv
          confidence_hist.png
          reliability_diagram.png
          confusion_matrix.csv
          confusion_matrix.png
          tsne_embedding.csv
          tsne_teacher_student.png
          kd_feature_gaps.csv
          kd_gap_hist.png
          weight_hist.png
          training_curves.png
          query_ratio.png
          result_row.json
          student_model/
          DONE
```

After jobs finish:

```bash
source "$SCRATCH/gcbd-nlp/env/venv/bin/activate"
cd "$SCRATCH/gcbd-nlp/code"
python aggregate_results.py --config trillium_experiments.json
```

This creates:

- `method_comparison_raw.csv`;
- `method_comparison_aggregated.csv` with mean and standard deviation across seeds; and
- `teacher_query_efficiency.csv` with query cost, accuracy gain, and efficiency summaries; and
- `missing_runs.csv`, which identifies configured jobs lacking results.

Inspect `missing_runs.csv` before interpreting comparisons. A method mean computed from fewer seeds is not directly comparable to a fully completed condition.

### Query and diagnostic definitions

The NLP reports use the same training-query convention as the Vision code:

- `query_ratio` and `train_query_ratio` are aliases for total online teacher prediction/distillation queries divided by total training examples seen. `Student` is 0, dense KD baselines are 1, and GCBD is the selected active fraction.
- `final_train_query_ratio` is that ratio in the last epoch.
- `active_query_ratio` records GCBD active selection separately; `train_candidate_ratio` records the curriculum-eligible fraction.
- `teacher_geometry_queries` and `teacher_geometry_forward_ratio` report broad GCBD stage-1 representation forwards separately. They are intentionally not counted as prediction queries.
- Teacher forwards used after training solely to create teacher–student t-SNE and KD-gap plots are diagnostic and are not counted as training queries.

For classification, `aggregate_results.py` pairs each run with the `Student` result from the same dataset, scenario, and seed. It reports `accuracy_gain_over_student` and Teacher-Query Efficiency as:

`teacher_query_efficiency = accuracy_gain_over_student / train_query_ratio`

It also reports accuracy gain per 1,000 actual teacher prediction queries. A zero-query Student run has undefined efficiency and is left blank rather than divided by an epsilon. For WikiText-2, the analogous report uses relative perplexity reduction.

A single method job cannot calculate Teacher-Query Efficiency because its seed-matched Student job is independent and may not yet be complete. Per-run files therefore record `performance_per_train_query_ratio` only as a descriptive quantity and mark `teacher_query_efficiency` as deferred. The scientifically defined Teacher-Query Efficiency is calculated only by `aggregate_results.py`, after seed-matched pairing with the Student baseline.

Confidence, calibration, and confusion diagnostics are example-level for classification. For language modeling they are computed on the configured bounded sample of valid target tokens; the LM confusion matrix shows the most frequent target tokens plus an `<other>` bucket. The full-split `perplexity` remains the primary LM evaluation metric.

## 11. Direct single-job submission

The matrix submitter is recommended, but one method can be submitted manually:

```bash
export GCBD_PROJECT_ROOT="$SCRATCH/gcbd-nlp/code"
export GCBD_CONFIG="$GCBD_PROJECT_ROOT/trillium_experiments.json"
export GCBD_VENV="$SCRATCH/gcbd-nlp/env/venv"
export GCBD_HF_HOME="$SCRATCH/gcbd-nlp/cache/downloads"
mkdir -p "$SCRATCH/gcbd-nlp/logs"

sbatch --account=ACCOUNT_NAME \
  --time=24:00:00 \
  --output="$SCRATCH/gcbd-nlp/logs/%x-%j.out" \
  "$GCBD_PROJECT_ROOT/trillium_h100.sbatch" \
  --mode method \
  --dataset IMDb \
  --scenario Setup_1_BERT_TinyBERT \
  --method KD \
  --seed 18 \
  --budget 0.0
```

The required teacher must already be complete for direct method submission. For an isolated CLI run, `--epochs N` overrides student epochs for the selected task, `--teacher-epochs N` overrides teacher epochs, and repeated `--set KEY=JSON_VALUE` options override other runtime fields.
