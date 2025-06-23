"""LLM Fine-tuning v0.01 – Proof of Concept
=======================================

This second-citizen extension demonstrates how to fine-tune an open-weight
language model on the **natural-language JSONL dataset** produced by
*heuristics-v0.04*.

Version scope
-------------
* v0.01 keeps a **single** entry script (`finetune.py`) that wraps Hugging Face
  `transformers.Trainer` with sane defaults.
* No Streamlit / GUI – pure CLI at this stage (mirrors heuristics-v0.01 style).
* The package is entirely self-contained plus `extensions/common` helpers,
  therefore `llm-finetune-v0.01 + common = standalone`.

Down-stream tasks (distillation, evaluation) will build on this.
"""

from __future__ import annotations

from extensions.common.path_utils import ensure_project_root_on_path, setup_extension_paths

ensure_project_root_on_path()
setup_extension_paths()

__all__: list[str] = [] 