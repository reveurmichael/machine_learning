"""heuristics-v0.04 Configuration
--------------------

Second-citizen tasks (heuristics, supervised, RL, …) are **allowed** to import
from first-citizen modules (Task-0).  The reverse dependency is forbidden.

This `config.py` acts as a *thin facade* around the canonical `config/` package
at the repository root so that files inside `extensions/heuristics-v0.04/` can
simply write:

```python
from extensions.heuristics_v0_04 import config as cfg
cfg.DIRECTIONS
```

without leaking the absolute `config.` import path everywhere.  It also hosts
**heuristic-specific** constants that are *not* relevant for Task-0.

Design Patterns
---------------
1. *Facade* – hides the full root-level config surface behind a smaller, stable
   interface tailored to heuristics agents.
2. *Explicit Re-export* – we explicitly list first-citizen attributes we rely
   on, preventing accidental tight-coupling to the entire root config module.

Remember: if a constant becomes universally useful, promote it upstream to
`/config` and re-export it here – never the other way around.
"""

from __future__ import annotations

from utils.path_utils import ensure_project_root
ensure_project_root()

# ---------------------
# Re-export first-citizen constants (Task-0 → Task-1 direction ✓)
# ---------------------

from config.game_constants import (
    DIRECTIONS,
    VALID_MOVES,
)

from config.ui_constants import GRID_SIZE
from config.network_constants import HOST_CHOICES

# Public re-export surface
__all__: list[str] = [
    # Root constants
    "DIRECTIONS",
    "VALID_MOVES",
    "GRID_SIZE",
    "HOST_CHOICES",
    # Heuristic-specific additions (below)
    "MAX_HEURISTIC_STEPS",
    "DEFAULT_MOVE_INTERVAL_MS",
]

# ---------------------
# Heuristic-specific settings (only used by v0.04 agents / GUI)
# ---------------------

# Fallback hard cap so runaway agents never freeze the replay.
MAX_HEURISTIC_STEPS: int = 5000  # steps per game (override via CLI if needed)

# Default delay between moves in the web replay (milliseconds)
DEFAULT_MOVE_INTERVAL_MS: int = 400 