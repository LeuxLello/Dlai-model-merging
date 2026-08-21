# Experimental plan

## 1. Scientific scope

This is not a leaderboard comparison. The independent variable is task compatibility; the dependent variable is post-merge degradation. Merging methods are instruments used to study that relationship.

### Research question

To what extent do task-vector alignment and sign conflict predict negative transfer after weight-space model merging?

### Falsifiable hypotheses

- **H1:** SST-2 and IMDb task vectors have higher cosine similarity and sign agreement than cross-family pairs.
- **H2:** Higher alignment predicts a higher retained score after merging.
- **H3:** TIES reduces degradation most strongly for pairs with many sign conflicts.
- **H4:** Interference is concentrated in a subset of encoder layers rather than distributed uniformly.

If these relationships do not appear, that is a valid negative result and must be reported.

## 2. Controlled setup

- Base model: `prajjwal1/bert-mini` (initial choice; one shared initialization)
- Datasets: SST-2, IMDb, MRPC, RTE
- Task format: binary classification
- Seeds: 3 for the final experiment
- Selection metric: validation metric defined per dataset
- Shared controls: tokenizer, maximum length, optimizer family, update budget, evaluation cadence
- Merge scope: encoder parameters only
- Evaluation scope: merged encoder plus each task's own classifier head

The smoke test uses small subsets. Final subset sizes will be selected only after timing one complete run on Kaggle.

## 3. Experiment matrix

### Stage A - specialists

Fine-tune one model per task and seed. Save:

- encoder state;
- task head state;
- validation metrics;
- training curve;
- exact configuration.

### Stage B - pairwise merging

Evaluate all six task pairs using:

- mean task vector;
- Task Arithmetic for several scaling coefficients;
- TIES for several trim densities and scaling coefficients.

This yields both related and unrelated task combinations.

### Stage C - diagnostics

For every pair and layer compute:

- cosine similarity;
- sign agreement on non-negligible updates;
- conflict-weighted update magnitude;
- L1/L2 norm of each task vector;
- retained task performance.

### Stage D - ablations

Subject to compute budget:

- merge only early, middle, or late layers;
- remove trimming from TIES;
- remove sign election from TIES;
- compare global and layer-wise trimming.

### Frozen configuration after the seed-42 pilot

Before running seeds 7 and 123, the confirmatory comparison was frozen to Mean, Task Arithmetic
with scale 0.75, and TIES with scale 1.0 and density 0.2. No per-pair retuning is allowed for the
confirmatory runs.

### Explanatory ablations after multi-seed confirmation

The next analysis is deliberately narrow: a seed-42 layer-wise intervention on three representative
pairs, plus an equal-norm control across all six pairs. These experiments test where interference
arises and whether unequal task-vector magnitude explains fragile merges; they do not reopen the
method hyperparameter search.

### Layer-adaptive improvement protocol

Seed 42 is used as development data to choose one global early-layer update factor from
`{0.25, 0.50, 0.75, 1.00}` independently for adaptive Mean and adaptive TIES. Embedding and late-layer
factors remain fixed at 1.0. The selected factors are then frozen and evaluated on seeds 7 and 123.
The 1.0 candidate is the explicit no-adaptation control, so the selection cannot force an apparent
improvement. The primary improvement claim is based on the two held-out seeds, not on seed 42.

## 4. Success criteria

The minimum complete project contains:

1. reproducible specialists for at least three tasks;
2. all pairwise merges for Mean, Task Arithmetic, and TIES;
3. three seeds for the key comparison;
4. one plot relating alignment to retained performance;
5. one layer-wise interference plot;
6. one ablation that explains, rather than merely ranks, a method.

## 5. Compute strategy

1. CPU unit tests locally.
2. Tiny-data smoke test on Kaggle.
3. One full run and timing estimate.
4. Freeze the final budget.
5. Run seeds as separate Kaggle notebook versions.
6. Download only metrics and small summaries to GitHub.

## 6. Risks and mitigations

- **Weak specialists:** tune the shared training recipe before merging.
- **Dataset-size confound:** use a fixed update budget and report sample counts.
- **Classifier incompatibility:** never merge task heads.
- **Seed noise:** use three final seeds and confidence intervals.
- **Hyperparameter cherry-picking:** define the tuning grid before final runs.
- **Kaggle session limits:** checkpoint each specialist and make merge analysis restartable.

## 7. Deliverables checklist

- Repository link
- Reproducible code and Kaggle notebook
- Official two-page report
- References and optional appendix
- Mandatory AI-use statement
- Submission email with the exact official subject and recipients
