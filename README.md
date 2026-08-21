# Predicting Interference in Model Merging

Deep Learning & Applied AI (DLAI), Sapienza University of Rome, 2025/2026.

## Research question

Can the compatibility of task vectors predict when merging independently fine-tuned models will help or hurt performance?

The project fine-tunes the same compact BERT encoder on several binary NLP tasks, merges the resulting encoder updates, and relates post-merge performance to parameter-space interference. Task-specific classification heads are **not** merged: each merged encoder is evaluated with the corresponding specialized head.

## Main hypothesis

Related tasks should produce more aligned task vectors and suffer less destructive interference. In particular, cosine similarity and sign agreement between task vectors should correlate with the performance retained after merging.

## Planned tasks

- SST-2: sentiment classification
- IMDb: sentiment classification
- MRPC: paraphrase detection
- RTE: textual entailment

SST-2 and IMDb form the expected high-compatibility pair. Cross-family pairs provide lower-compatibility controls.

## Methods

- Independent fine-tuning (specialist upper bound)
- Pretrained base model (no-task-update reference)
- Mean of task vectors
- Task Arithmetic with a tunable scaling coefficient
- TIES-Merging (trim, elect sign, merge)

## Primary measurements

- Validation score retained relative to each specialist
- Average and worst-task retained performance
- Task-vector cosine similarity
- Sign agreement and sign conflict rate
- Layer-wise update norm and alignment
- Correlation between interference indicators and merge degradation

## Repository layout

```text
configs/       experiment definitions
notebooks/     Kaggle entry points and analysis
src/dlai_merge reusable implementation
tests/         fast unit tests for merging algorithms
results/       lightweight tables and final figures
report/        official report and AI-use statement
```

## Environment

Python 3.11 is recommended.

```bash
pip install -e ".[dev]"
pytest
```

Kaggle-specific instructions will be added to `notebooks/01_kaggle_smoke_test.ipynb`. Large checkpoints and tokens must never be committed.

## Status

The multi-seed confirmation and explanatory ablations are complete. Task-vector cosine similarity
predicts retained merge performance across the three tested seeds, while simple norm equalization
and uniform early-layer attenuation do not repair fragile merges. TIES is currently the strongest
frozen baseline; further improvement work will target parameter-level conflict rather than a whole
layer group.
