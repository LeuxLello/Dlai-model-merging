# Predicting Interference in Model Merging

Deep Learning & Applied AI, Sapienza University of Rome, 2025/2026.

## Overview

This project studies whether task-vector geometry predicts interference when independently
fine-tuned language models are merged in weight space. All specialists start from the same compact
BERT encoder. Only encoder parameters are merged; each task retains its own classification head.

The experiments cover six binary NLP tasks:

| Task | Dataset | Family | Primary metric |
|---|---|---|---|
| SST-2 | GLUE/SST-2 | sentence sentiment | accuracy |
| IMDb | IMDb | document sentiment | accuracy |
| MRPC | GLUE/MRPC | paraphrase detection | F1 |
| RTE | GLUE/RTE | textual entailment | accuracy |
| CoLA | GLUE/CoLA | linguistic acceptability | Matthews correlation |
| BoolQ | SuperGLUE/BoolQ | binary question answering | accuracy |

## Methods

- Mean task-vector merging
- Task Arithmetic
- Tensor-wise TIES
- Global TIES following the reference flatten-first scope
- Projection-balanced merging

The reusable implementation is in `src/dlai_merge/`. It includes specialist training, task loading,
evaluation, merging algorithms, directional diagnostics, and controlled ablations.

## Main findings

- Task-vector cosine similarity is positively associated with retained merge performance across
  training seeds.
- TIES-style coordinate conflict handling is more reliable than uniform averaging.
- Training specialists for 1,200 rather than 400 optimizer steps improves every task, but makes
  their encoders harder to merge.
- Directional diagnostics replicate on held-out seeds, confirming that merge damage can be
  asymmetric across the two tasks in a pair.
- Projection-balanced scalar weights remain close to 0.5 and do not improve merging. On the primary
  held-out comparison, projection-balanced merging is worse than tensor-wise TIES by `-0.00894`
  with bootstrap interval `[-0.01485, -0.00304]`.

## Repository layout

```text
configs/       experiment configuration
notebooks/     Kaggle experiment entry points
src/           reusable Python package
tests/         unit tests for merging and diagnostics
results/       compact CSV, JSON, figures, and per-experiment README files
references/    scientific sources and BibTeX
report/        official course template
docs/course/   original project guidelines and provenance
```

## Local setup

Python 3.11 or later is recommended.

```bash
pip install -e ".[dev]"
pytest
```

Large checkpoints are intentionally excluded from Git. The repository stores compact result tables,
figures, metadata, and the code required to reproduce them.

## Notebook order

The notebooks are numbered in execution order. Notebook 01 validates the environment; notebooks
02–04 establish the specialist and multi-seed baselines; notebooks 05–08 contain explanatory
ablations and error analysis; notebook 09 verifies global TIES; notebooks 10–11 extend the study to
six tasks, longer training, directional diagnostics, and the final held-out test.

Detailed inputs and outputs are listed in [`notebooks/README.md`](notebooks/README.md). Each result
directory contains a README with its protocol, tables, limitations, and frozen conclusion.

## Key result directories

- [`results/multiseed_confirmatory/`](results/multiseed_confirmatory/)
- [`results/global_ties_corrective/`](results/global_ties_corrective/)
- [`results/extended_tasks_budget_directionality/`](results/extended_tasks_budget_directionality/)
- [`results/projection_balanced_long_specialists/`](results/projection_balanced_long_specialists/)

Scientific sources are documented in [`references/README.md`](references/README.md). The mandatory
AI-use record is maintained separately in [`AI_USAGE.md`](AI_USAGE.md).

## Limitations

- All experiments use one compact BERT base architecture.
- The model identifies the task through its task-specific classification head.
- The 1,200-step budget comparison begins as a single-seed ablation before the final held-out run.
- CoLA fails to learn at 400 steps and is excluded from the corresponding primary sensitivity
  analysis; it becomes viable at 1,200 steps.
- The results characterize controlled model merging and are not state-of-the-art benchmark claims.
