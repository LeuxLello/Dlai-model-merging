# Multi-seed confirmatory results

This directory contains the exact compact outputs of the confirmatory experiment on seeds 7, 42,
and 123. Configurations were frozen before running seeds 7 and 123: Mean; Task Arithmetic at scale
0.75; and TIES at scale 1.0 with density 0.2.

## Main result

For fixed Mean merging, task-vector cosine similarity is positively associated with retained
specialist performance in every seed:

| Seed | Pearson | Spearman | Sign-conflict Pearson |
|---:|---:|---:|---:|
| 7 | 0.718 | 0.714 | -0.279 |
| 42 | 0.520 | 0.771 | -0.264 |
| 123 | 0.878 | 0.600 | -0.189 |
| Mean | 0.705 | 0.695 | -0.244 |

The direction of the cosine relationship replicates across all three seeds. The raw fraction of
sign-conflicting coordinates is consistently negative but substantially weaker. With only six task
pairs per seed, these coefficients should be presented as effect-size evidence rather than a
high-powered significance test.

## Frozen method comparison

| Method | Mean retained | Between-seed SD | Worst observed task retention |
|---|---:|---:|---:|
| TIES | 0.936 | 0.003 | 0.814 |
| Task Arithmetic | 0.928 | 0.016 | 0.806 |
| Mean | 0.927 | 0.006 | 0.754 |

TIES has the best mean retention and the lowest variation across seeds. This aggregate advantage is
not universal: Mean is best for IMDb+RTE, and Task Arithmetic is best for SST-2+IMDb.

## Pair-level observations

- SST-2+IMDb, the two sentiment tasks, is consistently strong: Mean retains 0.975 and Task
  Arithmetic retains 0.983 on average.
- IMDb+MRPC performs best with TIES (0.969 mean retention).
- Pairs containing RTE are the most fragile (approximately 0.87-0.93 depending on pair/method).
- Specialist scores themselves are stable across seeds, so the merging pattern is not explained by
  a single failed training run.

## Interpretation

The confirmatory evidence supports the narrow project claim that geometric task-vector alignment is
useful for anticipating pairwise merge compatibility. It does not establish a universal causal law:
task identity, specialist quality, update norm, and dataset size remain possible confounders. A
layer-wise ablation is the next appropriate experiment to localize where interference arises.

## Files

- `all_merge_results.csv`: 108 task-level evaluations across three seeds.
- `all_pair_diagnostics.csv`: 18 pair/seed vector diagnostics.
- `all_specialist_scores.csv`: 12 specialist reference evaluations.
- `per_seed_pair.csv`: pair-level retention for each frozen method and seed.
- `final_summary.csv`: pair/method means and standard deviations across seeds.
- `method_summary.csv`: method-level results by seed.
- `correlations.csv`: confirmatory correlations by seed.
- `multiseed_confirmatory.png`: main confirmatory visualization.
- `metadata.json`: environment and frozen configuration declaration.
