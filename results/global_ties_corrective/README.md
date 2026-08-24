# Final corrective experiment: global versus tensor-wise TIES

## Protocol verification

- Source commit: `eeb2d72a550459f11c4afaac49701d70e0711d8a`
- Model: `prajjwal1/bert-mini`
- Tasks: SST-2, IMDb, MRPC, RTE
- Training seeds: 7, 42, 123
- Fixed data-subset seed: 2026
- Budget: 400 optimizer steps per specialist
- Frozen methods: Mean; Task Arithmetic (`scale=0.75`); tensor-wise TIES and global TIES (`density=0.2`, `scale=1.0`)
- Complete design: 144 task-level rows = 3 seeds × 6 pairs × 4 methods × 2 evaluated tasks
- Integrity checks: no missing or duplicate experimental rows

## Frozen result

| Method | Mean retention | SD across pair-seed units | Worst task retention |
|---|---:|---:|---:|
| tensor-wise TIES | 0.9380 | 0.0469 | 0.7615 |
| global TIES | 0.9307 | 0.0503 | 0.7976 |
| Mean | 0.9285 | 0.0350 | 0.7781 |
| Task Arithmetic | 0.9266 | 0.0509 | 0.8111 |

Global minus tensor-wise TIES over the 18 pair-seed units:

- mean delta: `-0.00723`;
- median delta: `-0.00674`;
- global-TIES win rate: `7/18 = 38.9%`;
- bootstrap 95% interval for the mean delta: `[-0.01489, 0.00068]`.

## Interpretation

The official flatten-first trimming scope did **not** improve average retention in this controlled setting. Tensor-wise TIES retained about 0.72 percentage points more performance on average, but the bootstrap interval narrowly includes zero; this is not strong evidence of a general advantage. Global TIES had a less severe single worst task retention than tensor-wise TIES, while Task Arithmetic had the best worst-case value. The correct conclusion is therefore a trade-off, not a universal winner.

With fixed data subsets, cosine similarity remained positively associated with Mean-merge retention in all three seeds (Pearson: 0.800, 0.557, 0.854; six pairs per seed). This replication removes the earlier confounding between training seed and subset selection, while the small number of pairs still limits generalization.

No further tuning is authorized by this result. This bundle closes the experimental phase.
