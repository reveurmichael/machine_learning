from importlib import import_module as _im

RLGameData = _im("extensions.reinforcement-v0.02.game_data").RLGameData  # type: ignore[attr-defined]

__all__ = ["RLGameData"] 