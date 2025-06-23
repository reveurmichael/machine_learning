from importlib import import_module as _im

RLGameLogic = _im("extensions.reinforcement-v0.02.game_logic").RLGameLogic  # type: ignore[attr-defined]

__all__ = ["RLGameLogic"] 