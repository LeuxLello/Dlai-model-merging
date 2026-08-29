"""Build the report tables and copy the selected figures from saved results."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "report" / "tables"
FIGURE_DIR = ROOT / "report" / "figures"


def read_csv(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"


method_labels = {
    "mean": "Mean",
    "task_arithmetic": "Task Arithmetic",
    "global_ties": "Global TIES",
    "tensorwise_ties": "Tensor-wise TIES",
    "projection_balanced": "Projection-balanced",
}

heldout_summary_path = (
    "results/projection_balanced_long_specialists/heldout_no_cola_method_summary.csv"
)
heldout_raw_path = "results/projection_balanced_long_specialists/all_merge_results.csv"
paired_path = "results/projection_balanced_long_specialists/paired_comparison_summary.csv"
budget_path = "results/extended_tasks_budget_directionality/merge_budget_comparison.csv"

heldout_summary = read_csv(heldout_summary_path)
heldout_raw = read_csv(heldout_raw_path)
paired_rows = read_csv(paired_path)
budget_rows = read_csv(budget_path)
heldout_raw = [
    row
    for row in heldout_raw
    if row["seed"] in {"7", "123"} and "cola" not in row["pair"]
]
projection_comparison = next(
    row for row in paired_rows if row["baseline"] == "tensorwise_ties"
)

main_rows: list[dict[str, object]] = []
for method in [
    "tensorwise_ties",
    "global_ties",
    "projection_balanced",
    "mean",
    "task_arithmetic",
]:
    summary = next(row for row in heldout_summary if row["method"] == method)
    raw = [row for row in heldout_raw if row["method"] == method]
    budget = [row for row in budget_rows if row["method"] == method]
    budget_400 = fmean(float(row["400"]) for row in budget) if budget else None
    budget_1200 = fmean(float(row["1200"]) for row in budget) if budget else None
    primary_comparison = method == "projection_balanced"
    main_rows.append(
        {
            "method": method,
            "method_label": method_labels[method],
            "heldout_mean_score_delta": float(summary["mean_score_delta"]),
            "heldout_sd_task_delta": float(summary["sd"]),
            "heldout_worst_task_delta": float(summary["worst"]),
            "heldout_mean_retained_ratio": fmean(float(row["retained_ratio"]) for row in raw),
            "heldout_pair_seed_units": len({(row["seed"], row["pair"]) for row in raw}),
            "heldout_task_evaluations": len(raw),
            "paired_baseline": "tensorwise_ties" if primary_comparison else "",
            "paired_mean_difference": float(projection_comparison["mean_delta"]) if primary_comparison else "",
            "paired_median_difference": float(projection_comparison["median_delta"]) if primary_comparison else "",
            "paired_win_rate": float(projection_comparison["win_rate"]) if primary_comparison else "",
            "paired_ci_low": float(projection_comparison["ci_low"]) if primary_comparison else "",
            "paired_ci_high": float(projection_comparison["ci_high"]) if primary_comparison else "",
            "budget_seed42_mean_delta_400": budget_400 if budget_400 is not None else "",
            "budget_seed42_mean_delta_1200": budget_1200 if budget_1200 is not None else "",
            "budget_long_minus_short": budget_1200 - budget_400 if budget else "",
            "budget_pair_units": len(budget),
            "heldout_source": heldout_summary_path,
            "paired_source": paired_path if primary_comparison else "",
            "budget_source": budget_path if budget else "",
        }
    )

common_protocol: dict[str, object] = {
    "base_model": "prajjwal1/bert-mini",
    "merged_component": "encoder only; task-specific head retained",
    "max_length": 128,
    "train_sample_cap": 12000,
    "eval_sample_cap": 2000,
    "train_batch_size": 32,
    "eval_batch_size": 64,
    "learning_rate": 0.00002,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "primary_training_steps": 1200,
    "budget_ablation_steps": "400;1200",
    "primary_seeds": "7;42;123",
    "heldout_seeds": "7;123",
    "development_seed": 42,
    "subset_seed": 2026,
    "accelerator_recorded": "Tesla T4 (2 visible GPUs)",
    "software_recorded": "Python 3.12.13; PyTorch 2.10.0+cu128",
    "protocol_source": "results/extended_tasks_budget_directionality/metadata.json",
}
task_rows = [
    ("SST-2", "GLUE", "sst2", "sentence sentiment", "sentence", "accuracy", "train", "validation"),
    ("IMDb", "IMDb", "", "document sentiment", "text", "accuracy", "train", "test"),
    ("MRPC", "GLUE", "mrpc", "paraphrase detection", "sentence1;sentence2", "F1", "train", "validation"),
    ("RTE", "GLUE", "rte", "textual entailment", "sentence1;sentence2", "accuracy", "train", "validation"),
    ("CoLA", "GLUE", "cola", "linguistic acceptability", "sentence", "Matthews correlation", "train", "validation"),
    ("BoolQ", "SuperGLUE", "boolq", "binary question answering", "question;passage", "accuracy", "train", "validation"),
]
task_rows = [
    {
        "task": task,
        "dataset": dataset,
        "dataset_config": config,
        "family": family,
        "input_fields": inputs,
        "primary_metric": metric,
        "train_split": train_split,
        "eval_split": eval_split,
        **common_protocol,
    }
    for task, dataset, config, family, inputs, metric, train_split, eval_split in task_rows
]

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
write_csv(TABLE_DIR / "main_results.csv", main_rows)
write_csv(TABLE_DIR / "task_protocol.csv", task_rows)

main_lines = [
    rf'{latex_escape(row["method_label"])} & {fmt(float(row["heldout_mean_score_delta"]))} & '
    rf'{fmt(float(row["heldout_sd_task_delta"]))} & {fmt(float(row["heldout_worst_task_delta"]))} & '
    rf'{fmt(float(row["heldout_mean_retained_ratio"]))} \\'
    for row in main_rows
]
(TABLE_DIR / "main_results.tex").write_text(
    "\\begin{table}[t]\n\\centering\n"
    "\\caption{Held-out merging results at 1,200 training steps on 20 pair--seed units excluding CoLA. "
    "Score changes are measured relative to the corresponding specialists.}\n"
    "\\label{tab:main-results}\n\\begin{tabular}{lrrrr}\n\\hline\n"
    "Method & Mean $\\Delta$ & SD & Worst $\\Delta$ & Retained \\\\\n+\\hline\n"
    + "\n".join(main_lines)
    + "\n\\hline\n\\end{tabular}\n\\end{table}\n",
    encoding="utf-8",
)

protocol_lines = [
    rf'{latex_escape(row["task"])} & {latex_escape(row["dataset"] + ("/" + str(row["dataset_config"]) if row["dataset_config"] else ""))} & '
    rf'{latex_escape(row["family"])} & {latex_escape(row["primary_metric"])} \\'
    for row in task_rows
]
(TABLE_DIR / "task_protocol.tex").write_text(
    "\\begin{table}[t]\n\\centering\n"
    "\\caption{Tasks used in the extended protocol. Each run uses up to 12,000 training examples and "
    "2,000 evaluation examples.}\n"
    "\\label{tab:task-protocol}\n\\begin{tabular}{llll}\n\\hline\n"
    "Task & Dataset & Family & Metric \\\\\n+\\hline\n"
    + "\n".join(protocol_lines)
    + "\n\\hline\n\\end{tabular}\n\\end{table}\n",
    encoding="utf-8",
)

figures = [
    ("results/multiseed_confirmatory/multiseed_confirmatory.png", "01_multiseed_geometry_and_retention.png"),
    ("results/layerwise_norm_ablation/layerwise_and_norm_ablation.png", "02_layerwise_interference_ablation.png"),
    ("results/extended_tasks_budget_directionality/budget_robustness.png", "03_training_budget_robustness.png"),
    ("results/extended_tasks_budget_directionality/directional_interference.png", "04_directional_interference.png"),
    ("results/projection_balanced_long_specialists/projection_balanced_heldout.png", "05_projection_balanced_heldout.png"),
]
for source, destination in figures:
    shutil.copyfile(ROOT / source, FIGURE_DIR / destination)

print(f"Built {len(main_rows)} result rows, {len(task_rows)} task rows, and {len(figures)} figures.")
