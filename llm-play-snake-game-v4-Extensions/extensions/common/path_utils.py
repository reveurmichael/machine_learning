"""Path utilities **specific to the *extensions* namespace*.

Historically every ``extensions/<name>/__init__.py`` duplicated a small code
snippet to ensure the repository root was on ``sys.path`` so that absolute
imports such as ``from core.game_manager import GameManager`` work when an
extension is executed *directly* (e.g. ``python extensions/heuristics-v0.02/main.py``).

The canonical implementation already lives in
``utils.path_utils.ensure_project_root`` inside the *root* package.  To avoid
code duplication (and the inevitable divergence of behaviour) this module
is now a **thin façade** that simply delegates to that canonical function.

The public API *remains unchanged* – extensions can continue to import
``ensure_project_root_on_path`` or ``setup_extension_paths`` from
``extensions.common.path_utils`` – but the actual logic is single-sourced in
the root ``utils`` package, honouring the *Single Source of Truth* rule.
"""

from __future__ import annotations

# NOTE: The heavy-lifting lives in the root utils module.  Importing at runtime
# instead of at import-time avoids cyclic dependencies if extensions are used
# by tooling *before* the root package is fully initialised (e.g. during
# editable installs or unit-test collection).

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover – only for static type checkers
    from pathlib import Path


def ensure_project_root_on_path() -> "Path":  # noqa: D401 – keep the historical name
    """Ensure repository root is in ``sys.path`` and return it.

    This is a thin alias around :pyfunc:`utils.path_utils.ensure_project_root` to
    preserve backwards-compatibility for existing extensions while enforcing the
    *single source of truth* policy.
    """

    from utils.path_utils import ensure_project_root as _ensure_root  # local import to avoid cycles

    return _ensure_root()


def setup_extension_paths() -> None:  # noqa: D401 – historical name retained
    """Legacy wrapper that now simply calls :pyfunc:`ensure_project_root_on_path`."""

    ensure_project_root_on_path()


# ---------------------
# Public re-exports – keep the historical interface intact
# ---------------------

__all__ = [
    "ensure_project_root_on_path",
    "setup_extension_paths",
]

# Ensure side-effects (cwd + sys.path) are applied as soon as the module is
# imported – this mimics the historical behaviour so that existing extensions
# continue to work without modification.

ensure_project_root_on_path()
