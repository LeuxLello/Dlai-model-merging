# Notebooks

Planned execution order:

1. `01_kaggle_smoke_test.ipynb` - dependency, data, model, and one-batch verification.
2. `02_train_specialists_pilot.ipynb` - meaningful seed-42 specialist training and checkpoint export.
3. `03_pairwise_merging_pilot.ipynb` - seed-42 pairwise merging grid, diagnostics, and figures.
4. `04_analysis.ipynb` - confidence intervals and final figures.

The notebooks will call code from `src/dlai_merge`; they should not contain separate, drifting implementations of the algorithms.
