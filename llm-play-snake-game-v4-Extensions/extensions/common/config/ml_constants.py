"""
Machine Learning Constants for Snake Game AI Extensions.

This module contains ML-specific hyperparameters, thresholds, and configuration
constants shared across supervised learning, reinforcement learning, and other
ML-based extensions.

Design Pattern: Constants Module
- Centralized configuration management
- Type-safe constant definitions
- Clear documentation of parameter purposes

Educational Value:
Demonstrates how to organize ML hyperparameters in a maintainable way,
with clear explanations of why specific values are chosen as defaults.
"""

from typing import Dict, List, Tuple, Any

# =============================================================================
# Neural Network Hyperparameters
# =============================================================================

# Learning rates optimized for Snake game scenarios
DEFAULT_LEARNING_RATE: float = 0.001
"""Default learning rate for neural network training.
Chosen based on empirical testing with Snake game datasets."""

LEARNING_RATE_RANGE: Tuple[float, float] = (1e-5, 1e-1)
"""Valid range for learning rate hyperparameter tuning."""

# Batch sizes for different model types
DEFAULT_BATCH_SIZE: int = 32
"""Default batch size balancing memory usage and gradient stability."""

BATCH_SIZE_OPTIONS: List[int] = [16, 32, 64, 128, 256]
"""Common batch sizes for hyperparameter optimization."""

# Training epochs
DEFAULT_EPOCHS: int = 100
"""Default number of training epochs for supervised learning."""

MAX_EPOCHS: int = 1000
"""Maximum epochs to prevent infinite training."""

# =============================================================================
# Optimizer Configuration
# =============================================================================

SUPPORTED_OPTIMIZERS: List[str] = [
    "adam",
    "adamw", 
    "sgd",
    "rmsprop",
    "adagrad"
]
"""List of supported optimizer algorithms."""

OPTIMIZER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "adam": {
        "lr": DEFAULT_LEARNING_RATE,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01
    },
    "sgd": {
        "lr": DEFAULT_LEARNING_RATE,
        "momentum": 0.9,
        "weight_decay": 0.01
    },
    "rmsprop": {
        "lr": DEFAULT_LEARNING_RATE,
        "alpha": 0.99,
        "eps": 1e-8,
        "weight_decay": 0.01
    }
}
"""Default configurations for each optimizer."""

# =============================================================================
# Model Architecture Parameters
# =============================================================================

# MLP (Multi-Layer Perceptron) defaults
DEFAULT_HIDDEN_LAYERS: List[int] = [128, 64, 32]
"""Default hidden layer sizes for MLP models."""

DEFAULT_DROPOUT_RATE: float = 0.2
"""Default dropout rate for regularization."""

ACTIVATION_FUNCTIONS: List[str] = ["relu", "tanh", "sigmoid", "leaky_relu"]
"""Supported activation functions."""

# CNN parameters
DEFAULT_CNN_CHANNELS: List[int] = [32, 64, 128]
"""Default channel progression for CNN models."""

DEFAULT_KERNEL_SIZE: int = 3
"""Default kernel size for convolutional layers."""

# LSTM parameters  
DEFAULT_LSTM_HIDDEN_SIZE: int = 128
"""Default hidden size for LSTM layers."""

DEFAULT_LSTM_LAYERS: int = 2
"""Default number of LSTM layers."""

# =============================================================================
# Tree-Based Model Parameters
# =============================================================================

# XGBoost defaults
XGBOOST_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}
"""Default XGBoost hyperparameters optimized for Snake game data."""

# LightGBM defaults
LIGHTGBM_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1
}
"""Default LightGBM hyperparameters."""

# Random Forest defaults
RANDOM_FOREST_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42
}
"""Default Random Forest hyperparameters."""

# =============================================================================
# Performance Thresholds
# =============================================================================

MIN_ACCURACY_THRESHOLD: float = 0.6
"""Minimum acceptable model accuracy."""

CONVERGENCE_PATIENCE: int = 10
"""Number of epochs without improvement before stopping."""

GRADIENT_CLIP_VALUE: float = 1.0
"""Gradient clipping threshold to prevent exploding gradients."""

# =============================================================================
# Feature Engineering
# =============================================================================

FEATURE_SCALING_METHODS: List[str] = ["standard", "minmax", "robust", "none"]
"""Supported feature scaling methods."""

DEFAULT_SCALING_METHOD: str = "standard"
"""Default feature scaling method."""

# Categorical encoding
ENCODING_METHODS: List[str] = ["onehot", "label", "target", "none"]
"""Supported categorical encoding methods."""

# =============================================================================
# Model Evaluation
# =============================================================================

EVALUATION_METRICS: List[str] = [
    "accuracy",
    "precision", 
    "recall",
    "f1_score",
    "confusion_matrix",
    "classification_report"
]
"""Standard evaluation metrics for classification tasks."""

CROSS_VALIDATION_FOLDS: int = 5
"""Number of folds for cross-validation."""

# =============================================================================
# Hardware and Performance
# =============================================================================

DEFAULT_N_JOBS: int = -1
"""Default number of parallel jobs (-1 uses all available cores)."""

MEMORY_LIMIT_GB: int = 8
"""Memory limit for large dataset processing."""

GPU_MEMORY_FRACTION: float = 0.8
"""Fraction of GPU memory to allocate for training.""" 