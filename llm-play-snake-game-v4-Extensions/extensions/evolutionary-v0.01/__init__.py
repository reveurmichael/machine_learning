"""Evolutionary-v0.01
====================

Proof-of-concept extension that demonstrates a **Genetic Algorithm (GA)**
planner for Snake.  This is the very first evolutionary package – extremely
simple on purpose.  It reuses *only* common utilities plus core base classes
and keeps every other concern inside this folder so that the pair
`evolutionary-v0.01 + extensions/common` is fully standalone.
"""

from __future__ import annotations

from extensions.common.path_utils import ensure_project_root_on_path, setup_extension_paths

# Make sure the repository root is on *sys.path* so absolute imports work
ensure_project_root_on_path()
setup_extension_paths()

__all__: list[str] = [] 