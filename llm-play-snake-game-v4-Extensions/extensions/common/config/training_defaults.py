"""
Training Default Configurations for Snake Game AI Extensions.

This module contains default training configurations, dataset splitting ratios,
and other training-related constants shared across extensions.

Design Pattern: Configuration Object Pattern
- Centralized training configuration
- Consistent defaults across extensions
- Easy to override for specific use cases

Educational Value:
Shows best practices for organizing training configurations with proper
validation and documentation of choices.
"""

from typing import Dict, List, Any, Optional

# =============================================================================
# Dataset Splitting Configuration
# =============================================================================

VALIDATION_SPLIT_RATIO: float = 0.2
"""Default validation set size as fraction of training data.
Industry standard 80/20 split for training/validation."""

TEST_SPLIT_RATIO: float = 0.1
"""Default test set size as fraction of total data.
Maintains 70/20/10 train/val/test split when combined."""

MINIMUM_TRAIN_SAMPLES: int = 100
"""Minimum number of training samples required for valid training."""

MINIMUM_VAL_SAMPLES: int = 20
"""Minimum number of validation samples required."""

# =============================================================================
# Early Stopping Configuration
# =============================================================================

EARLY_STOPPING_PATIENCE: int = 15
"""Number of epochs without improvement before early stopping.
Balanced to prevent premature stopping while avoiding overfitting."""

EARLY_STOPPING_MIN_DELTA: float = 0.001
"""Minimum change to qualify as an improvement."""

EARLY_STOPPING_METRIC: str = "val_loss"
"""Default metric to monitor for early stopping."""

RESTORE_BEST_WEIGHTS: bool = True
"""Whether to restore best weights after early stopping."""

# =============================================================================
# Learning Rate Scheduling
# =============================================================================

LR_SCHEDULER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "step": {
        "step_size": 30,
        "gamma": 0.1
    },
    "exponential": {
        "gamma": 0.95
    },
    "cosine": {
        "T_max": 50,
        "eta_min": 1e-6
    },
    "reduce_on_plateau": {
        "factor": 0.5,
        "patience": 10,
        "min_lr": 1e-6
    }
}
"""Default configurations for learning rate schedulers."""

DEFAULT_LR_SCHEDULER: str = "reduce_on_plateau"
"""Default learning rate scheduler."""

# =============================================================================
# Random Seed Configuration
# =============================================================================

RANDOM_SEED: int = 42
"""Default random seed for reproducible results.
Universal answer to reproducibility questions."""

DETERMINISTIC_TRAINING: bool = True
"""Whether to use deterministic algorithms for reproducibility."""

# =============================================================================
# Checkpoint and Saving Configuration
# =============================================================================

SAVE_CHECKPOINT_FREQUENCY: int = 10
"""Save model checkpoint every N epochs."""

SAVE_BEST_MODEL: bool = True
"""Whether to save the best model during training."""

MODEL_SAVE_FORMAT: str = "pytorch"
"""Default model save format (pytorch, onnx, sklearn)."""

CHECKPOINT_MONITOR: str = "val_loss"
"""Metric to monitor for best model saving."""

CHECKPOINT_MODE: str = "min"
"""Whether to minimize ('min') or maximize ('max') the monitored metric."""

# =============================================================================
# Training Progress and Logging
# =============================================================================

LOG_FREQUENCY: int = 1
"""Log training metrics every N epochs."""

VERBOSE_TRAINING: bool = True
"""Whether to display detailed training progress."""

PROGRESS_BAR: bool = True
"""Whether to show progress bar during training."""

METRICS_TO_LOG: List[str] = [
    "loss",
    "accuracy", 
    "val_loss",
    "val_accuracy",
    "learning_rate"
]
"""Metrics to log during training."""

# =============================================================================
# Model Validation Configuration
# =============================================================================

VALIDATION_FREQUENCY: int = 1
"""Validate model every N epochs."""

VALIDATION_BATCH_SIZE: Optional[int] = None
"""Batch size for validation (None means same as training)."""

# =============================================================================
# Hyperparameter Tuning Configuration
# =============================================================================

HYPERPARAMETER_TUNING_CONFIGS: Dict[str, Dict[str, Any]] = {
    "grid_search": {
        "cv": 5,
        "scoring": "accuracy",
        "n_jobs": -1,
        "verbose": 1
    },
    "random_search": {
        "n_iter": 50,
        "cv": 5,
        "scoring": "accuracy", 
        "n_jobs": -1,
        "random_state": RANDOM_SEED
    },
    "bayesian_optimization": {
        "n_calls": 50,
        "n_initial_points": 10,
        "random_state": RANDOM_SEED
    }
}
"""Default configurations for hyperparameter tuning methods."""

# =============================================================================
# Cross-Validation Configuration
# =============================================================================

CV_STRATEGY: str = "stratified_kfold"
"""Default cross-validation strategy."""

CV_FOLDS: int = 5
"""Number of cross-validation folds."""

CV_SHUFFLE: bool = True
"""Whether to shuffle data before splitting."""

CV_RANDOM_STATE: int = RANDOM_SEED
"""Random state for CV splitting."""

# =============================================================================
# Training Data Augmentation
# =============================================================================

DATA_AUGMENTATION_ENABLED: bool = False
"""Whether to use data augmentation by default."""

AUGMENTATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "rotation": {
        "enabled": False,
        "max_angle": 15
    },
    "noise": {
        "enabled": False,
        "noise_factor": 0.1
    },
    "dropout": {
        "enabled": True,
        "dropout_rate": 0.1
    }
}
"""Data augmentation configurations."""

# =============================================================================
# Ensemble Training Configuration
# =============================================================================

ENSEMBLE_SIZE: int = 5
"""Default number of models in ensemble."""

ENSEMBLE_VOTING: str = "soft"
"""Ensemble voting strategy ('hard' or 'soft')."""

BAGGING_ENABLED: bool = True
"""Whether to use bagging for ensemble diversity."""

BOOTSTRAP_SAMPLE_RATIO: float = 0.8
"""Fraction of data to sample for each ensemble member."""

# =============================================================================
# Training Environment Configuration
# =============================================================================

USE_GPU: bool = True
"""Whether to use GPU if available."""

MIXED_PRECISION: bool = False
"""Whether to use mixed precision training."""

DISTRIBUTED_TRAINING: bool = False
"""Whether to use distributed training."""

NUM_WORKERS: int = 4
"""Number of data loader workers.""" 