"""Dataset Directory Manager
================================

Centralised enforcement of the *grid-size‐aware* dataset-storage rule that
applies to **all** extensions (heuristics, supervised, reinforcement, …).

Rule Summary (SSoT)
-------------------
Every dataset produced by an extension **must** live under::

    logs/extensions/datasets/grid-size-{N}/

where *N* equals the side-length of the board used to generate the data.
This guarantees:

1.  Dataset discoverability – look-up by grid size is trivial.
2.  Clean experimentation – models never mix samples of different spatial
    complexity by accident.
3.  Future scalability – new grid sizes drop into their own sub-tree with
    zero changes to existing code.

Design Patterns Employed
-----------------------
* **Facade / Static Utility** – a single class exports helper methods; no
  instance state is required.
* **Single-Source-of-Truth (SSoT)** – delegates absolute paths to
  `extensions.common.config` so the rule is defined in *one* place only.
* **Validation Guard** – defensive checks prevent silent misplacement of
  files; violations raise `DatasetPathError`.

All extensions are expected to rely on this helper (or at least the
underlying `config` helpers) instead of re-implementing path logic.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from .config import DATASETS_ROOT, get_dataset_dir, SUPPORTED_GRID_SIZES, ensure_datasets_dir

__all__ = [
    "DatasetPathError",
    "DatasetDirectoryManager",
]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DatasetPathError(RuntimeError):
    """Raised when a dataset path does *not* conform to the grid-size rule."""


# ---------------------------------------------------------------------------
# Manager (static facade)
# ---------------------------------------------------------------------------

class DatasetDirectoryManager:  # pylint: disable=too-few-public-methods
    """Static helpers for grid-size-aware dataset paths.

    The class purposefully contains only **`@staticmethod`** helpers – it is a
    facade, not a service with mutable state.  Instantiation is therefore
    unnecessary and discouraged.
    """

    _GRID_DIR_PATTERN: Final[re.Pattern[str]] = re.compile(r"grid-size-(\d+)")

    # ---------------------------------------------------------------------
    # Path builders
    # ---------------------------------------------------------------------

    @staticmethod
    def dataset_root() -> Path:
        """Return `logs/extensions/datasets/` (creates it if missing)."""
        DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
        return DATASETS_ROOT

    @staticmethod
    def grid_size_dir(grid_size: int) -> Path:
        """Return the directory for *grid_size*, ensuring it exists."""
        if grid_size not in SUPPORTED_GRID_SIZES:
            raise ValueError(
                f"Unsupported grid size: {grid_size}. Update SUPPORTED_GRID_SIZES in config.py if this is intentional."
            )
        return ensure_datasets_dir(grid_size)

    # ---------------------------------------------------------------------
    # Validation helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def validate_dataset_path(path: Path) -> int:
        """Validate *path* belongs to a `grid-size-N` sub-folder.

        Returns the detected *N* (int) or raises :class:`DatasetPathError`.
        """
        match = DatasetDirectoryManager._GRID_DIR_PATTERN.search(str(path))
        if not match:
            raise DatasetPathError(
                f"{path} is not inside a 'grid-size-N' directory – rule violation."  # noqa: E501
            )

        grid_size = int(match.group(1))
        if grid_size not in SUPPORTED_GRID_SIZES:
            raise DatasetPathError(
                f"Grid size {grid_size} extracted from {path} is not in SUPPORTED_GRID_SIZES."
            )
        return grid_size

    # ---------------------------------------------------------------------
    # Convenience builders for filenames
    # ---------------------------------------------------------------------

    @staticmethod
    def make_filename(
        algorithm: str,
        grid_size: int,
        data_structure: str,
        data_format: str,
        timestamp: str | None = None,
    ) -> Path:
        """Standardise dataset filenames across extensions.

        Example output::

            tabular_bfs_20250625T101501.csv

        The extension folder should *not* prepend its name; algorithm is
        sufficient context once the grid directory rule is applied.
        """
        from datetime import datetime

        timestamp = timestamp or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        filename = f"{data_structure}_{algorithm.lower()}_{timestamp}.{data_format}"
        return DatasetDirectoryManager.grid_size_dir(grid_size) / filename 