"""Dataset loading and tokenization for the four controlled binary tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset_name: str
    dataset_config: str | None
    text_columns: tuple[str, ...]
    train_split: str = "train"
    validation_split: str = "validation"
    primary_metric: str = "accuracy"


TASKS: dict[str, TaskSpec] = {
    "sst2": TaskSpec("sst2", "glue", "sst2", ("sentence",)),
    "imdb": TaskSpec("imdb", "imdb", None, ("text",), validation_split="test"),
    "mrpc": TaskSpec(
        "mrpc", "glue", "mrpc", ("sentence1", "sentence2"), primary_metric="f1"
    ),
    "rte": TaskSpec("rte", "glue", "rte", ("sentence1", "sentence2")),
}


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unknown task {name!r}; choose from {sorted(TASKS)}") from error


def _limit(dataset: Dataset, maximum: int | None, seed: int) -> Dataset:
    if maximum is None or maximum >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(maximum))


def load_task_data(
    task_name: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    seed: int = 42,
) -> tuple[DatasetDict, TaskSpec]:
    """Download, subset, and tokenize a task using a common preprocessing protocol."""
    spec = get_task(task_name)
    raw = load_dataset(spec.dataset_name, spec.dataset_config)
    train = _limit(raw[spec.train_split], max_train_samples, seed)
    validation = _limit(raw[spec.validation_split], max_eval_samples, seed)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        texts = [batch[column] for column in spec.text_columns]
        return tokenizer(*texts, truncation=True, max_length=max_length)

    remove_columns = [column for column in train.column_names if column != "label"]
    encoded = DatasetDict(
        train=train.map(tokenize, batched=True, remove_columns=remove_columns),
        validation=validation.map(tokenize, batched=True, remove_columns=remove_columns),
    )
    return encoded, spec

