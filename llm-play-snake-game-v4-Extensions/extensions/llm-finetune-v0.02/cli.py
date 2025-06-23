from __future__ import annotations

import argparse
from pathlib import Path

from . import FineTuneConfig, FineTunePipeline

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM Fine-tune v0.02 – multi-dataset & LoRA")
    p.add_argument("--model", required=True, help="Base HF model name or local path")
    p.add_argument("--datasets", nargs="+", required=True, help="One or more JSONL dataset paths or directories")
    p.add_argument("--output", required=True, help="Output directory for checkpoints & tokenizer")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--no-lora", action="store_true", help="Disable LoRA – perform full fine-tune")
    p.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (ignored if --no-lora)")
    return p


def main() -> None:  # noqa: D401
    args = _build_parser().parse_args()

    cfg = FineTuneConfig(
        model_name_or_path=args.model,
        output_dir=Path(args.output),
        dataset_paths=[Path(p) for p in args.datasets],
        num_train_epochs=args.epochs,
        per_device_batch_size=args.batch,
        learning_rate=args.lr,
        block_size=args.block_size,
        use_lora=not args.no_lora,
        lora_rank=args.lora_rank,
    )

    pipeline = FineTunePipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main() 