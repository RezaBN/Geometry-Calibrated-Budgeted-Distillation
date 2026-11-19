# Geometry-Calibrated Budgeted Distillation (GCBD)

**Research code and supporting materials for _Do We Need Every Teacher Prediction? Geometry-Calibrated Budgeted Distillation_.**

GCBD is a two-stage, gray-box knowledge-distillation framework that combines reusable teacher-geometry representations, sliced-Wasserstein attention alignment (SWAT), uncertainty-guided teacher-query selection, and semantic unbalanced optimal transport (UOT). It investigates how compact students can learn with fewer **online teacher-prediction queries** across natural language processing and computer vision tasks.

**Note.** ***This repository currently contains the core framework. Full code scripts, datasets, and supplementary materials will be uploaded upon paper acceptance.***

## Repository contents

| Directory | Contents |
| --- | --- |
| [`code/nlp/`](code/nlp/) | NLP training/evaluation code, experiment configurations, and Trillium execution instructions. |
| [`code/vision/`](code/vision/) | Vision training/evaluation code, configurations, and Trillium execution instructions. |
| [`code/vision_swat_fast/`](code/vision_swat_fast/) | Vision SWAT implementation refinements, benchmarks, and verification scripts. |
| [`code/gcbd_phase2_integrated_nlp_optimized/`](code/gcbd_phase2_integrated_nlp_optimized/) | Separate Phase 2 integration workspace; consult its README for validation status. |
| [`results/original_18_settings/`](results/original_18_settings/) | Original multi-configuration NLP and Vision experimental summaries and seed-level results. |
| [`results/budget_sensitivity/`](results/budget_sensitivity/) | Additional finite-budget experiments on IMDb, SST-2, and CIFAR-100. |
| [`results/teacher_evaluations/`](results/teacher_evaluations/) | Fine-tuned teacher evaluation records (archive excludes model weights). |
| [`results/Optimized/`](results/Optimized/) | Separately reported optimized-implementation results for NLP and Vision. |
| [`supplementary/`](supplementary/) | Supplementary results/discussion PDF and high-resolution figure archive. |

## Experimental scope

The main study evaluates **18 teacher–student configurations** on **IMDb, SST-2, CIFAR-100, and ImageNet** with six random seeds per configuration. Supplementary budget-response experiments and implementation-efficiency benchmarks are **separate experimental cohorts** and should not be pooled with the original aggregate results.

The reported online prediction-query rate measures access to teacher predictions; it is **not** a measure of total teacher computation, GPU time, or financial cost. Refer to the manuscript and supplementary document for measurement definitions, comparisons, and limitations.

## Reproducibility and citation

Start with the dataset-specific instructions in [`code/nlp/README-Trillium.md`](code/nlp/README-Trillium.md) and [`code/vision/README-Vision-Trillium.md`](code/vision/README-Vision-Trillium.md), then consult the relevant configuration and result files. Execution may require separately obtained datasets, pretrained models, teacher checkpoints, and environment-specific paths. **Teacher model weights are not provided in the teacher-evaluation archive.**

If you use this repository, please cite the associated GCBD manuscript. This repository is provided as supporting research material; see the individual code directories for their implementation and validation notes.
