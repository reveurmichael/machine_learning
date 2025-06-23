"""Heuristics→LLM Fine-Tuning Integration v0.01
===========================================

Automates the two-step pipeline:
1. Generate language-rich JSONL from heuristic agents (via heuristics-v0.04 &
   common dataset generator).
2. Fine-tune an open-weight LLM on that dataset using *llm-finetune-v0.01*.

The extension is *orchestration-only*; core heuristic logic and fine-tuning
code stay in their respective packages so the conceptual ideas remain visible
there.
"""

from __future__ import annotations

from extensions.common.path_utils import ensure_project_root_on_path, setup_extension_paths
import sys as _sys

ensure_project_root_on_path()
setup_extension_paths()

# ---------------------
# Alias package name without dashes so that CLI can be invoked via
#   python -m extensions.heuristics_llm_fine_tuning_integration_v0_01.pipeline
# without breaking the *hyphenated* on-disk folder structure mandated by
# the naming-convention docs.  This keeps both FQN variants importable and
# therefore satisfies tools (ruff, pytest, IDEs) that automatically convert
# non-identifier characters to underscores when generating module paths.
# ---------------------
_pkg_tail = __name__.split('.', 1)[1] if '.' in __name__ else __name__  # drop 'extensions.' prefix
_pkg_alias = f"extensions.{_pkg_tail.replace('-', '_').replace('.', '_')}"
_sys.modules.setdefault(_pkg_alias, _sys.modules[__name__])

__all__: list[str] = [] 