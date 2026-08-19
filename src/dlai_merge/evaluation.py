"""Evaluation utilities for merged encoders with task-specific classifier heads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .data import get_task, load_task_data


def classification_metrics(prediction) -> dict[str, float]:
    labels = prediction.label_ids
    predictions = np.argmax(prediction.predictions, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


class TaskEvaluator:
    """Cache one task dataset, head, and Trainer while swapping encoder states."""

    def __init__(
        self,
        task_name: str,
        head_state: dict[str, torch.Tensor],
        base_model: str = "prajjwal1/bert-mini",
        max_length: int = 128,
        max_eval_samples: int | None = 2000,
        eval_batch_size: int = 64,
        seed: int = 42,
        output_root: str = "/tmp/dlai-merge-eval",
    ) -> None:
        self.task_name = task_name
        self.task_spec = get_task(task_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        datasets, _ = load_task_data(
            task_name,
            self.tokenizer,
            max_length=max_length,
            max_train_samples=1,
            max_eval_samples=max_eval_samples,
            seed=seed,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
        self.model.load_state_dict(head_state, strict=False)
        arguments = TrainingArguments(
            output_dir=str(Path(output_root) / task_name),
            per_device_eval_batch_size=eval_batch_size,
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=seed,
            dataloader_num_workers=2,
            disable_tqdm=True,
        )
        self.trainer = Trainer(
            model=self.model,
            args=arguments,
            eval_dataset=datasets["validation"],
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=classification_metrics,
        )

    @property
    def primary_metric(self) -> str:
        return self.task_spec.primary_metric

    def evaluate(self, encoder_state: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.base_model.load_state_dict(encoder_state, strict=True)
        metrics = self.trainer.evaluate()
        return {
            "loss": float(metrics["eval_loss"]),
            "accuracy": float(metrics["eval_accuracy"]),
            "f1": float(metrics["eval_f1"]),
            "primary_score": float(metrics[f"eval_{self.primary_metric}"]),
        }
