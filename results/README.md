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

## Explanatory ablations

- `layerwise_norm_ablation/`: seed-42 interventions that localize cumulative cross-layer
  interference and show that equalizing task-vector norms does not rescue fragile RTE pairs.

## Layer-adaptive improvement attempt

- `layer_adaptive_improvement/`: development selection on seed 42 followed by held-out evaluation
  on seeds 7 and 123. Both adaptive methods selected the no-adaptation factor 1.0, ruling out uniform
  early-layer attenuation as an improvement under this protocol.

## Scope-density TIES improvement attempt

- `scope_density_ties/`: development selection on seed 42 and held-out evaluation on seeds 7 and
  123. Uniform density 0.20 was selected over all structured schedules, so scope-level density
  tuning did not improve the frozen TIES baseline.
