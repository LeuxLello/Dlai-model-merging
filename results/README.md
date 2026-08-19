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
