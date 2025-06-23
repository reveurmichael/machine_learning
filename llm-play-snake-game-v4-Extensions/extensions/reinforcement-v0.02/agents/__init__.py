from __future__ import annotations

"""RL v0.02 – Agent package.

Each module implements a concrete *Strategy* (RL algorithm).  They all expose a
single public class with a predictable name so the factory in the parent
package can look them up dynamically.
"""

from importlib import import_module
from typing import Any, Protocol


class RLAgentProtocol(Protocol):
    """Minimal interface every RL agent must implement."""

    def select_action(self, state: Any) -> int: ...  # noqa: D401 – stub

    def remember(self, *transition: Any) -> None: ...

    def learn(self) -> None: ...

    def add_observer(self, callback) -> None: ...


# Public helper ---------------------------------------------------------------


def get_agent_class(algo: str):
    """Return concrete agent class for ``algo`` without instantiation.

    Used mainly by Sphinx / docs generation.  Production code should call
    :pyfunc:`extensions.reinforcement-v0.02.create_rl_agent` instead.
    """
    from extensions.reinforcement-v0.02 import _AGENT_REGISTRY  # type: ignore

    if algo.upper() not in _AGENT_REGISTRY:
        raise KeyError(f"Unknown algorithm '{algo}'.")
    module_path, class_name = _AGENT_REGISTRY[algo.upper()]
    return getattr(import_module(module_path), class_name)


__all__ = [
    "RLAgentProtocol",
    "get_agent_class",
] 