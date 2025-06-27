"""extensions.common – Shared building blocks for *all* extensions.

The public API is intentionally tiny:
* `config`      – ultra-lightweight configuration constants.
* `utils`       – a handful of helper functions/classes (dataset, path, csv).
* `validation`  – very simple sanity-check helpers.

If your extension needs something more advanced – make it part of *your*
package.  Do **not** add heavy machinery here.
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

__all__: List[str] = [
    "config",
    "utils",
    "validation",
] 