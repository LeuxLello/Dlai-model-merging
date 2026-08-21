# Results

Commit only lightweight, reproducible summaries here:

- final CSV/JSON metric tables;
- publication-ready figures;
- concise run metadata.

Raw logs and checkpoints belong in Kaggle outputs or another artifact store and are ignored by Git.

## Smoke tests

- `smoke_test_2026-08-19.json`: first end-to-end Kaggle validation. It confirms that data loading,
  specialist training, checkpoint serialization, and all three merging implementations execute. Its
  tiny-data metrics are diagnostic only and are not part of the final experiment.

## Specialist pilot

- `pilot_metrics_seed42.csv`: compact metrics for the first meaningful four-task training run.
- `pilot_summary_seed42.json`: environment, protocol, metrics, and the decision to proceed to the
  pairwise merging pilot. This remains single-seed pilot evidence, not the final multi-seed result.

## Pairwise merging pilot

- `merging_pilot_seed42/`: exact compact output, figure, and a cautious interpretation of the first
  six-pair merging grid. The directory contains all 156 task-level evaluations.

## Multi-seed confirmation

- `multiseed_confirmatory/`: frozen-method results for seeds 7, 42, and 123. This is the primary
  confirmatory evidence for the relationship between task-vector alignment and merge retention.
