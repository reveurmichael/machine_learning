"""LLM Distillation v0.01 – Compress the Fine-Tuned Teacher
========================================================

This second-citizen extension (Task-5) trains a *student* language model to
imitate the fine-tuned teacher produced by **llm-finetune-v0.01**.

Dependency chain:

    heuristics → llm-finetune (teacher) → distillation (student)

v0.01 scope:
• Single CLI script `distil.py` that runs a **simple knowledge-distillation
  loss** (cross-entropy + KL divergence).  
• No GUI / Streamlit; pure head-less CLI.
• Standalone aside from `extensions/common` utilities.

Future versions can add LoRA student, curriculum distillation, dashboards, etc.
"""

from __future__ import annotations

from extensions.common.path_utils import ensure_project_root_on_path, setup_extension_paths

ensure_project_root_on_path()
setup_extension_paths()

__all__: list[str] = [] 