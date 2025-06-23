# llm-finetune-v0.02

"""Future-proof LLM fine-tuning extension (v0.02).

Adds multi-dataset loading, optional LoRA/QLoRA adapters, evaluation helpers
and a clean CLI (see ``cli.py``).

The key philosophy for v0.02:
    • No backward-compat shims – only modern features.
    • All configuration is explicit and type-hinted via ``FineTuneConfig``.
    • Pipeline code is isolated in ``pipeline.py`` so that the CLI remains thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

__all__ = [
    "FineTuneConfig",
    "FineTunePipeline",
]


@dataclass(kw_only=True, slots=True)
class FineTuneConfig:
    """Typed configuration for one fine-tuning run."""

    model_name_or_path: str
    output_dir: Path
    dataset_paths: List[Path]

    # Training hyper-params
    num_train_epochs: int = 1
    per_device_batch_size: int = 2
    learning_rate: float = 2e-5
    block_size: int = 512

    # Advanced knobs
    use_lora: bool = True  # Switch between full fine-tune vs LoRA/QLoRA
    lora_rank: int = 8

    def __post_init__(self) -> None:  # Basic validation
        if not self.dataset_paths:
            raise ValueError("dataset_paths must not be empty")
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Importing here to avoid heavy deps during *import*
from .pipeline import FineTunePipeline  # noqa: E402 