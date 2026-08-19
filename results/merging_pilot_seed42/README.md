# Pairwise merging pilot - seed 42

This directory contains the exact compact output of the first pairwise merging run. It includes 156
task-level evaluations: six task pairs, thirteen merge configurations per pair, and evaluation on
both constituent tasks.

## Main pilot observations

- Mean merging retention has Pearson correlation `0.520` and Spearman rank correlation `0.771`
  with task-vector cosine similarity across the six pairs.
- The raw sign-conflict fraction is a weaker predictor in this run (Pearson `-0.264`).
- SST-2 + IMDb has by far the largest cosine similarity (`0.375`), consistent with both being
  sentiment tasks.
- Pairwise best mean retention ranges from `0.879` to `0.993` of specialist performance.
- RTE pairs are generally the most fragile; this may partly reflect RTE's weaker, low-resource
  specialist and must be separated from task-compatibility effects in the final analysis.
- The best configuration is Task Arithmetic for three pairs, TIES for two, and Mean for one.

These are exploratory single-seed observations. Selecting the best hyperparameter configuration per
pair is optimistic and must not be used for a confirmatory correlation claim. Final reporting should
use pre-registered/frozen configurations and multiple seeds.

## Best configuration per pair

| Pair | Method | Scale | Density | Mean retained | Worst retained |
|---|---|---:|---:|---:|---:|
| SST-2 + IMDb | Task Arithmetic | 0.75 | - | 0.991 | 0.982 |
| SST-2 + MRPC | Task Arithmetic | 0.75 | - | 0.993 | 0.982 |
| SST-2 + RTE | Task Arithmetic | 0.75 | - | 0.899 | 0.817 |
| IMDb + MRPC | TIES | 1.00 | 0.20 | 0.977 | 0.970 |
| IMDb + RTE | Mean | 1.00 | - | 0.879 | 0.841 |
| MRPC + RTE | TIES | 1.00 | 0.20 | 0.955 | 0.911 |

## Files

- `merge_results.csv`: all 156 task-level evaluations.
- `merge_summary.csv`: 78 pair/configuration aggregates.
- `pair_diagnostics.csv`: vector alignment, sign agreement, and norms.
- `specialist_recheck.csv`: specialist reference metrics from the common evaluation path.
- `best_per_pair.csv`: exploratory best configuration for each pair.
- `alignment_vs_retention.png`: exploratory plot based on those best configurations.
- `metadata.json`: run metadata and grid.
