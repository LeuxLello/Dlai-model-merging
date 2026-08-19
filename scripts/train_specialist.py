"""CLI entry point for a single Kaggle specialist run."""

from __future__ import annotations

import argparse
import json

from dlai_merge.training import TrainConfig, train_specialist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["sst2", "imdb", "mrpc", "rte"])
    parser.add_argument("--base-model", default="prajjwal1/bert-mini")
    parser.add_argument("--output-root", default="artifacts/specialists")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        task=args.task,
        base_model=args.base_model,
        output_root=args.output_root,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        epochs=args.epochs,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    print(json.dumps(train_specialist(config), indent=2))


if __name__ == "__main__":
    main()
