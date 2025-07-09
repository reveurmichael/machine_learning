"""extensions.common – Shared building blocks for *all* extensions.

This package follows final-decision.md Guideline 3: lightweight, OOP-based common utilities 
with simple logging (print() statements) rather than complex *.log file mechanisms.

The public API is intentionally tiny:
* `config`      – ultra-lightweight configuration constants.
* `utils`       – a handful of helper functions/classes (dataset, path, csv).
* `validation`  – very simple sanity-check helpers.

If your extension needs something more advanced – make it part of *your*
package.  Do **not** add heavy machinery here.

Design Philosophy:
- Simple, object-oriented utilities that can be inherited and extended
- No tight coupling with ML/DL/RL/LLM-specific concepts
- Simple logging with print() statements (final-decision.md Guideline 3)
- Enables easy addition of new extensions without friction

Reference: docs/extensions-guideline/final-decision.md
"""

from importlib import import_module
from types import ModuleType
from typing import List

# Re-export the three main sub-packages as attributes so users can write e.g.
#   from extensions.common import utils
# or access `extensions.common.utils.dataset_utils` directly.

config: ModuleType = import_module("extensions.common.config")
utils: ModuleType = import_module("extensions.common.utils")
validation: ModuleType = import_module("extensions.common.validation")

# Import specific constants for easier access
from .config import EXTENSIONS_LOGS_DIR, HEURISTICS_LOG_PREFIX

__all__: List[str] = [
    "config",
    "utils",
    "validation",
    "EXTENSIONS_LOGS_DIR",
    "HEURISTICS_LOG_PREFIX",
] 