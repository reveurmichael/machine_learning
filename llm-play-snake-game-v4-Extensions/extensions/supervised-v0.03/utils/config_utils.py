"""
Configuration utilities for supervised learning v0.03.

Design Pattern: Configuration Object Pattern
- Centralized configuration management
- Type-safe settings with validation
- Default values for all parameters
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    hidden_size: int = 256
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    dropout_rate: float = 0.2
    max_depth: int = 6
    n_estimators: int = 100
    validation_split: float = 0.2
    random_state: int = 42
    export_onnx: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    grid_size: int = 10
    max_games: int = 1000
    use_gui: bool = False
    verbose: bool = True
    save_frequency: int = 100
    output_dir: Optional[Path] = None
    experiment_name: Optional[str] = None


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if not config_path.exists():
        return {}
    
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: Dict[str, Any], config_path: Path):
    """Save configuration to JSON file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    model_config = asdict(ModelConfig())
    training_config = asdict(TrainingConfig())
    
    return {
        "model": model_config,
        "training": training_config,
        "log_level": "INFO",
        "device": "auto"
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters."""
    try:
        # Validate model config
        model = config.get("model", {})
        if model.get("learning_rate", 0) <= 0:
            return False
        if model.get("validation_split", 0) <= 0 or model.get("validation_split", 0) >= 1:
            return False
        
        # Validate training config
        training = config.get("training", {})
        if training.get("grid_size", 0) < 5 or training.get("grid_size", 0) > 50:
            return False
        if training.get("max_games", 0) <= 0:
            return False
        
        return True
    except Exception:
        return False 