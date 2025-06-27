"""
Common configuration package for *all* extensions.

Only **generic** constants that are unlikely to change between algorithm
families live here.  Anything that is *model-specific* or *algorithm-specific*
should instead reside inside the respective extension package.

The goal of keeping this `common.config` namespace extremely lean is twofold:
1. Prevent accidental tight-coupling between unrelated extensions.
2. Make it crystal-clear where a constant is coming from when reading code.
"""

from importlib import import_module
from types import ModuleType
from typing import List

# We purposefully expose the two evergreen config modules via attribute access
# *modules* instead of wildcard ("*") import so that the public API remains
# stable even if we decide to shuffle things internally.

dataset_formats: ModuleType = import_module("extensions.common.config.dataset_formats")
path_constants: ModuleType = import_module("extensions.common.config.path_constants")
validation_rules: ModuleType = import_module("extensions.common.config.validation_rules")

__all__: List[str] = [
    "dataset_formats",
    "path_constants",
    "validation_rules",
] 