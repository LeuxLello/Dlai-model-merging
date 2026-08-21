# Layer-wise and equal-norm ablations

These seed-42 explanatory experiments follow the frozen multi-seed confirmation. They do not reopen
the merge hyperparameter search.

## Layer-wise intervention

For each evaluated task, non-selected parameter groups remain exactly specialist-specific; only the
selected group is replaced by merged parameters. Therefore the score drop estimates the cost of
sharing that group while preserving the rest of the specialist.

Average retention across the three representative pairs:

| Shared scope | Mean merge | TIES |
|---|---:|---:|
| Embeddings only | 0.993 | 0.994 |
| Early blocks only | 0.971 | 0.976 |
| Late blocks only | 0.994 | 0.995 |
| All Transformer blocks | 0.944 | 0.935 |
| Full encoder | 0.934 | 0.927 |

No single late layer group is responsible for the full degradation. Sharing only late blocks or
embeddings is nearly lossless; early blocks cause the largest isolated drop. The much larger loss
when all blocks are shared indicates cumulative, distributed interference and cross-layer
interaction. This effect is strongest for IMDb+RTE.

Layer-wise task-vector alignment also distinguishes related sentiment tasks: SST-2+IMDb cosine
similarity is positive in every scope and reaches 0.461 in late blocks, whereas IMDb+RTE remains near
zero and becomes slightly negative in late blocks.

## Equal-norm control

Rescaling both task vectors to their average norm does not improve mean retention for any pair:

| Pair | Standard Mean | Equal-norm Mean | Difference |
|---|---:|---:|---:|
| IMDb+MRPC | 0.957 | 0.954 | -0.003 |
| IMDb+RTE | 0.879 | 0.860 | -0.019 |
| MRPC+RTE | 0.936 | 0.935 | -0.002 |
| SST-2+IMDb | 0.967 | 0.966 | -0.001 |
| SST-2+MRPC | 0.949 | 0.945 | -0.004 |
| SST-2+RTE | 0.889 | 0.874 | -0.015 |

Thus RTE-pair fragility is not rescued by compensating for RTE's smaller update norm. This weakens
the simple magnitude-confound explanation and is consistent with directional incompatibility, while
not proving that direction is the only cause. Rescaling can itself disturb the coupling between a
trained encoder and its task head, so this control should be interpreted conservatively.

## Files

- `layer_results.csv`: 60 task-level layer interventions.
- `layer_summary.csv`: pair/method/scope aggregates.
- `layer_diagnostics.csv`: alignment and norms for every selected scope.
- `norm_results.csv`: 24 task-level standard/equal-norm evaluations.
- `norm_summary.csv`: pair/variant aggregates.
- `layerwise_and_norm_ablation.png`: explanatory figure.
- `metadata.json`: exact scope membership and environment.
