"""Reproducible specialist training and artifact management."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
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

from .data import load_task_data


@dataclass(frozen=True)
class TrainConfig:
    task: str
    base_model: str = "prajjwal1/bert-mini"
    output_root: str = "artifacts/specialists"
    seed: int = 42
    max_length: int = 128
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    epochs: float = 3.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    train_batch_size: int = 32
    eval_batch_size: int = 64
    warmup_ratio: float = 0.06
    fp16: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(prediction) -> dict[str, float]:
    labels = prediction.label_ids
    predictions = np.argmax(prediction.predictions, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def extract_encoder_state(model) -> dict[str, torch.Tensor]:
    """Copy only the shared pretrained encoder, excluding the task classifier."""
    return {key: value.detach().cpu().clone() for key, value in model.base_model.state_dict().items()}


def extract_head_state(model) -> dict[str, torch.Tensor]:
    encoder_keys = {f"{model.base_model_prefix}.{key}" for key in model.base_model.state_dict()}
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key not in encoder_keys
    }


def train_specialist(config: TrainConfig) -> dict[str, object]:
    """Fine-tune one task specialist and save encoder/head states separately."""
    set_seed(config.seed)
    run_dir = Path(config.output_root) / config.task / f"seed-{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    datasets, task_spec = load_task_data(
        config.task,
        tokenizer,
        max_length=config.max_length,
        max_train_samples=config.max_train_samples,
        max_eval_samples=config.max_eval_samples,
        seed=config.seed,
    )
    model = AutoModelForSequenceClassification.from_pretrained(config.base_model, num_labels=2)
    use_fp16 = config.fp16 and torch.cuda.is_available()

    arguments = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.epochs,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{task_spec.primary_metric}",
        greater_is_better=True,
        save_total_limit=1,
        fp16=use_fp16,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_metrics,
    )
    train_result = trainer.train()
    metrics = trainer.evaluate()

    torch.save(extract_encoder_state(model), run_dir / "encoder.pt")
    torch.save(extract_head_state(model), run_dir / "head.pt")
    tokenizer.save_pretrained(run_dir / "tokenizer")

    summary: dict[str, object] = {
        "config": asdict(config),
        "primary_metric": task_spec.primary_metric,
        "train_samples": len(datasets["train"]),
        "eval_samples": len(datasets["validation"]),
        "train_metrics": {key: float(value) for key, value in train_result.metrics.items()},
        "eval_metrics": {key: float(value) for key, value in metrics.items()},
        "device": str(trainer.args.device),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

