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
docs/          study guide, method-to-code map, and project audit
references/    authoritative reading list and BibTeX bibliography
```

## Environment

Python 3.11 is recommended.

```bash
pip install -e ".[dev]"
pytest
```

Kaggle-specific instructions will be added to `notebooks/01_kaggle_smoke_test.ipynb`. Large checkpoints and tokens must never be committed.

## Status

The initial multi-seed confirmation and explanatory ablations are complete. Task-vector cosine similarity
is positively associated with retained merge performance across the three tested seeds, while simple norm equalization
and uniform early-layer attenuation do not repair fragile merges. TIES is currently the strongest
frozen baseline among the tested methods. The existing implementation uses tensor-wise rather than
official global trimming, so a corrective global-TIES comparison is required before final claims.
Two pre-declared improvement attempts selected their explicit no-change controls. The completed
error analysis shows that compatible sentiment merges are comparatively stable, while pairs involving
RTE produce larger and task-asymmetric losses.

## Start here

- [`docs/STUDY_GUIDE.md`](docs/STUDY_GUIDE.md): plain-language walkthrough from datasets to conclusions.
- [`references/README.md`](references/README.md): papers, exact sections, and official data/model sources.
- [`docs/METHOD_TO_CODE.md`](docs/METHOD_TO_CODE.md): formula-to-code-to-notebook traceability.
- [`docs/PROJECT_AUDIT.md`](docs/PROJECT_AUDIT.md): guideline compliance, limitations, and final roadmap.
- [`AI_USAGE.md`](AI_USAGE.md): honest working record for the mandatory disclosure.

The final corrective run is complete. With fixed data subsets, tensor-wise TIES achieved mean
retention 0.9380 versus 0.9307 for official global TIES. The paired mean difference (global minus
tensor-wise) was -0.00723 with a bootstrap interval of [-0.01489, 0.00068]. The experimental phase
is closed; see `results/global_ties_corrective/README.md` for the frozen interpretation.

An explicitly separated extension protocol is available in notebook 10. It adds CoLA and BoolQ,
expands the primary design to 45 pair-seed units, compares 400 with 1200 optimizer steps on a fixed
seed, and evaluates directional diagnostics at the constituent-task level. Its conclusions remain
pending until the Kaggle run is complete and do not alter the frozen four-task findings above.
