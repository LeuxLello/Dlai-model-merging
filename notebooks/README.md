# Notebooks

Planned execution order:

1. `01_kaggle_smoke_test.ipynb` - dependency, data, model, and one-batch verification.
2. `02_train_specialists_pilot.ipynb` - meaningful seed-42 specialist training and checkpoint export.
3. `03_pairwise_merging_pilot.ipynb` - seed-42 pairwise merging grid, diagnostics, and figures.
4. `04_multiseed_confirmatory.ipynb` - frozen configurations on seeds 42, 7, and 123.
5. `05_layerwise_and_norm_ablation.ipynb` - localize interference and control task-vector magnitude.
6. `06_layer_adaptive_improvement.ipynb` - select an early-layer attenuation on seed 42 and test it on held-out seeds 7 and 123.
7. `07_scope_density_ties.ipynb` - select a structured TIES density schedule and test it on held-out seeds.
8. `08_example_level_error_analysis.ipynb` - inspect prediction transitions on real examples for all six pairs.

The notebooks will call code from `src/dlai_merge`; they should not contain separate, drifting implementations of the algorithms.
