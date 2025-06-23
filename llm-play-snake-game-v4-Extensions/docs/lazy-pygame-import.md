# Lazy-loading *pygame* for Head-less Testability

*Last updated 2025-06-23*

## Why?

Historically the root (Task-0) codebase imported **pygame** at the *top-level* of
many modules. That made the project hard to test in CI environments that do not
ship SDL, and it forced even purely head-less extensions (heuristics, RL,
dataset generation…) to carry the heavyweight dependency.

## The new pattern

```python
import importlib

# inside __init__ or a setup method
self._pygame = None
if self.use_gui:
    self._pygame = importlib.import_module("pygame")
    self.clock   = self._pygame.time.Clock()
```

*   The attribute **`self._pygame`** is `None` in head-less mode, otherwise it
    holds the imported module.
*   All subsequent GUI code must check **both** `self.use_gui` *and*
    `self._pygame` before calling any SDL function.

## Affected files

* `core/game_manager.py`
* `core/game_loop.py`
* `core/game_manager_helper.py` (local import inside `process_events()`)
* `replay/replay_engine.py`

Each file's docstring now contains a brief explanation so future contributors
understand the rationale.

## Benefits

1. **CI friendliness** – Unit tests can import and run game logic on servers
   without X-server / SDL.
2. **Faster startup** for head-less scripts.
3. **Clear separation of concerns** – visualization is now an optional layer
   rather than an implicit requirement.

## Writing tests

When writing unit tests simply pass `use_gui=False` (or the CLI flag
`--no-gui`) to any manager/engine. No extra mocking is necessary – GUI paths
are skipped automatically.

```python
from core.game_manager import BaseGameManager

mgr = BaseGameManager(args)
assert mgr._pygame is None  # ensures head-less
```

## FAQ

**Q : Does this break existing GUI runs?**  > No. When `use_gui=True`
`pygame` is imported and initialised exactly as before.

**Q : Do I still need `pygame` in my environment?**  > Only if you intend to run
with a GUI. Head-less scripts/tests no longer require it. 