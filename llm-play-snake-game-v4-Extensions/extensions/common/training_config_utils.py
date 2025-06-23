"""training_config_utils.py – Shared config helpers for ML training

Centralises JSON (de)serialization and default-value handling so that
supervised / RL / evolutionary extensions do not duplicate boilerplate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "load_config",
    "save_config",
    "get_default_config",
    "validate_config",
]


@dataclass
class ModelConfig:
    hidden_size: int = 256
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    dropout_rate: float = 0.0
    max_depth: int = 6
    n_estimators: int = 100
    validation_split: float = 0.2
    random_state: int = 42


@dataclass
class TrainingConfig:
    grid_size: int = 10
    max_games: int = 1000
    use_gui: bool = False
    verbose: bool = True
    save_frequency: int = 100
    output_dir: Optional[Path] = None
    experiment_name: Optional[str] = None


# ---------------------
# JSON helpers
# ---------------------


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)


# ---------------------
# Defaults & validation
# ---------------------


def get_default_config() -> Dict[str, Any]:
    return {"model": asdict(ModelConfig()), "training": asdict(TrainingConfig())}


def validate_config(cfg: Dict[str, Any]) -> bool:  # noqa: D401
    try:
        m = cfg.get("model", {})
        t = cfg.get("training", {})
        if not 0 < m.get("learning_rate", 0) <= 1:
            return False
        if not 0 < m.get("validation_split", 0) < 1:
            return False
        if t.get("grid_size", 0) < 5 or t.get("grid_size", 0) > 50:
            return False
        if t.get("max_games", 0) <= 0:
            return False
        return True
    except Exception:
        return False 