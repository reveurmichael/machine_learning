from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from . import FineTuneConfig, FineTunePipeline, EvaluationSuite

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM Fine-tune v0.03 – training & evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Train command
    t = sub.add_parser("train", help="Run fine-tuning")
    t.add_argument("--model", required=True)
    t.add_argument("--datasets", nargs="+", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch", type=int, default=2)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--block-size", type=int, default=512)
    t.add_argument("--no-lora", action="store_true")
    t.add_argument("--lora-rank", type=int, default=8)

    # Eval command
    e = sub.add_parser("evaluate", help="Quick perplexity eval of a fine-tuned model")
    e.add_argument("--model-dir", required=True)
    e.add_argument("--datasets", nargs="+", required=True)
    e.add_argument("--samples", type=int, default=128, help="Sample size for perplexity calc")

    return p


def _run_train(ns: argparse.Namespace) -> None:
    cfg = FineTuneConfig(
        model_name_or_path=ns.model,
        output_dir=Path(ns.output),
        dataset_paths=[Path(p) for p in ns.datasets],
        num_train_epochs=ns.epochs,
        per_device_batch_size=ns.batch,
        learning_rate=ns.lr,
        block_size=ns.block_size,
        use_lora=not ns.no_lora,
        lora_rank=ns.lora_rank,
    )
    FineTunePipeline(cfg).run()


def _run_eval(ns: argparse.Namespace) -> None:
    suite = EvaluationSuite(Path(ns.model_dir), [Path(p) for p in ns.datasets])
    report = suite.quick_report()
    print("📊 Evaluation report:", report)


def main() -> None:  # noqa: D401
    args = _build_parser().parse_args()
    if args.cmd == "train":
        _run_train(args)
    elif args.cmd == "evaluate":
        _run_eval(args)


if __name__ == "__main__":
    main() 