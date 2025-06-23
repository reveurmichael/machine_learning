from __future__ import annotations

from importlib import import_module as _im
from typing import Dict, Any

# Import base manager from v0.02
_BaseManager = _im("extensions.reinforcement-v0.02.game_manager").RLGameManager  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Dashboard glue (very thin) -------------------------------------------------
# ---------------------------------------------------------------------------

class RLGameManager(_BaseManager):  # type: ignore[misc]
    """v0.03 manager – identical training loop plus Streamlit hooks."""

    def __init__(self, args):
        super().__init__(args)
        self._dashboard: "MetricsBuffer | None" = None
        # Lazy import to avoid Streamlit dependency for CLI uses
        try:
            from .dashboard.training_dashboard import MetricsBuffer  # noqa: F401

            self._dashboard = MetricsBuffer(maxlen=200)
        except ModuleNotFoundError:
            # Streamlit likely not installed in headless environments
            self._dashboard = None

    # Override observer callback to feed dashboard
    def _on_agent_event(self, event_type: str, data: Dict[str, Any]) -> None:  # type: ignore[override]
        super()._on_agent_event(event_type, data)
        if self._dashboard and event_type == "episode_complete":
            self._dashboard.add_record(data["id"], data["reward"]) 