from __future__ import annotations

import argparse
from pathlib import Path

from . import DistilConfig
from .distil import DistillationPipeline

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM Distillation v0.02 – LoRA-aware")
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--no-lora-student", action="store_true")
    p.add_argument("--lora-rank", type=int, default=8)
    return p


def main() -> None:  # noqa: D401
    a = _build_parser().parse_args()
    cfg = DistilConfig(
        teacher_path=a.teacher,
        student_model=a.student,
        dataset_paths=[Path(p) for p in a.datasets],
        output_dir=Path(a.output),
        num_train_epochs=a.epochs,
        per_device_batch_size=a.batch,
        learning_rate=a.lr,
        alpha=a.alpha,
        temperature=a.temperature,
        use_lora_student=not a.no_lora_student,
        lora_rank=a.lora_rank,
    )
    DistillationPipeline(cfg).run()


if __name__ == "__main__":
    main() 