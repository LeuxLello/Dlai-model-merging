# Report

The original course template is stored in `report/template/` and must retain its formatting. For one
student, the main report is limited to two pages; references may continue onto a third page. The
mandatory AI-use statement appears before the references. An optional appendix may follow the
bibliography, but the main claim and evidence must remain in the two-page report.

Use `references/references.bib` as the bibliography source and adapt `AI_USAGE.md` into a concise,
accurate disclosure. Every number in the report must be traceable to a committed result table.

## Prepared material

The report tables are stored in `report/tables/`:

- `main_results.csv` contains the held-out comparison and the separate 400/1,200-step ablation.
- `task_protocol.csv` contains the datasets, metrics, splits, and shared training protocol.
- `main_results.tex` and `task_protocol.tex` are compact versions ready for the LaTeX report.
- `report_tables.xlsx` provides the same tables in an easier format for manual inspection.

The five selected figures are stored in `report/figures/`:

1. multi-seed geometry and retention;
2. layer-wise interference and norm ablation;
3. training-budget robustness;
4. directional interference;
5. the held-out projection-balanced comparison.

The tables and figure copies can be rebuilt from the committed experiment outputs with:

```bash
python scripts/build_report_assets.py
```

The CSV, LaTeX, and figure sources are identified inside the script and in the source columns of the
tables. `report_tables.xlsx` is an optional inspection copy of the same two tables.
