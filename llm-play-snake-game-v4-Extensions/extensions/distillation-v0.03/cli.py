from __future__ import annotations

import argparse
from pathlib import Path

from . import DistilConfig, DistillationPipeline, EvaluationSuite

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM Distillation v0.03 – training & eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Run distillation")
    t.add_argument("--teacher", required=True)
    t.add_argument("--student", required=True)
    t.add_argument("--datasets", nargs="+", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--alpha", type=float, default=0.1)
    t.add_argument("--temperature", type=float, default=2.0)
    t.add_argument("--no-lora-student", action="store_true")
    t.add_argument("--lora-rank", type=int, default=8)

    e = sub.add_parser("evaluate", help="Quick perplexity evaluation")
    e.add_argument("--model-dir", required=True)
    e.add_argument("--datasets", nargs="+", required=True)
    e.add_argument("--samples", type=int, default=128)

    return p


def _run_train(ns: argparse.Namespace):
    cfg = DistilConfig(
        teacher_path=ns.teacher,
        student_model=ns.student,
        dataset_paths=[Path(p) for p in ns.datasets],
        output_dir=Path(ns.output),
        num_train_epochs=ns.epochs,
        per_device_batch_size=ns.batch,
        learning_rate=ns.lr,
        alpha=ns.alpha,
        temperature=ns.temperature,
        use_lora_student=not ns.no_lora_student,
        lora_rank=ns.lora_rank,
    )
    DistillationPipeline(cfg).run()


def _run_eval(ns: argparse.Namespace):
    suite = EvaluationSuite(Path(ns.model_dir), [Path(p) for p in ns.datasets])
    print("📊", suite.quick_report())


def main() -> None:
    ns = _build_parser().parse_args()
    if ns.cmd == "train":
        _run_train(ns)
    elif ns.cmd == "evaluate":
        _run_eval(ns)


if __name__ == "__main__":
    main() 