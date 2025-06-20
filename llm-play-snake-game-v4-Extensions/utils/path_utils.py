"""Path & script-initialisation helpers (generic).

This module centralises the *boiler-plate* that appears at the top of many
`scripts/*.py` files:

    * ensure the working directory is the repository root so relative paths
      (logs/, web/static, …) work irrespective of where the script is launched
    * inject the repo root into ``sys.path`` so `import something` always
      resolves without requiring explicit `PYTHONPATH`
    * optionally enable headless PyGame mode which is needed for Flask/web
      scripts that use PyGame functionality but must not open an X11 window.

Having a single helper eliminates the R0801 "duplicate-code" warnings and makes
future maintenance (e.g. adding a new environment tweak) a one-liner.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

__all__ = [
    "ensure_repo_root",
    "enable_headless_pygame",
]

# Cache so multiple calls are cheap and idempotent
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


def ensure_repo_root() -> Path:  # noqa: D401 – simple helper
    """Change *cwd* to the repository root and prepend it to *sys.path*.

    Returns
    -------
    pathlib.Path
        The absolute path to the repository root (cached across calls).
    """

    if Path.cwd() != _REPO_ROOT:
        os.chdir(_REPO_ROOT)
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    return _REPO_ROOT


def enable_headless_pygame() -> None:  # noqa: D401 – simple helper
    """Set SDL to *dummy* so PyGame can run without an X server."""

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy") 