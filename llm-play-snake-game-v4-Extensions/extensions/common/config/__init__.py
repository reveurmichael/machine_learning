"""
Common configuration package for *all* extensions.

This package follows final-decision-10.md Guideline 3: lightweight, OOP-based common utilities 
with simple logging (print() statements) rather than complex *.log file mechanisms.

Only **generic** constants that are unlikely to change between algorithm
families live here.  Anything that is *model-specific* or *algorithm-specific*
should instead reside inside the respective extension package.

The goal of keeping this `common.config` namespace extremely lean is twofold:
1. Prevent accidental tight-coupling between unrelated extensions.
2. Make it crystal-clear where a constant is coming from when reading code.

Design Philosophy:
- Simple, object-oriented utilities that can be inherited and extended
- No tight coupling with ML/DL/RL/LLM-specific concepts
- Simple logging with print() statements (final-decision-10.md Guideline 3)
- Enables easy addition of new extensions without friction
"""

from importlib import import_module
from types import ModuleType
from typing import List

# We purposefully expose the two evergreen config modules via attribute access
# *modules* instead of wildcard ("*") import so that the public API remains
# stable even if we decide to shuffle things internally.

dataset_formats: ModuleType = import_module("extensions.common.config.dataset_formats")
path_constants: ModuleType = import_module("extensions.common.config.path_constants")

__all__: List[str] = [
    "dataset_formats",
    "path_constants",
] 