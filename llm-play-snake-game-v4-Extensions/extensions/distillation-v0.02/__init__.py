"""LLM Distillation v0.02 – LoRA-aware, multi-dataset.

Upgrades over v0.01:
  • Accepts multiple heuristic JSONL datasets.
  • Allows LoRA *student* (tiny Δ-weights) for memory-efficient training.
  • Splits evaluation helpers into `evaluation.py` (future work).

No legacy flags, no backward-compat code – clean and forward-looking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

__all__ = ["DistilConfig"]


@dataclass(kw_only=True, slots=True)
class DistilConfig:
    teacher_path: str
    student_model: str
    dataset_paths: List[Path]
    output_dir: Path

    # Hyper-params
    num_train_epochs: int = 3
    per_device_batch_size: int = 4
    learning_rate: float = 5e-5
    alpha: float = 0.1  # CE weight
    temperature: float = 2.0

    # Student PEFT
    use_lora_student: bool = True
    lora_rank: int = 8

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.dataset_paths:
            raise ValueError("dataset_paths required") 