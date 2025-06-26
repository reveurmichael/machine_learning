"""
Model Registry for Snake Game AI Extensions.

This module defines supported model types, their configurations, and metadata
used across different extensions for consistent model management.

Design Pattern: Registry Pattern
- Centralized model type registration
- Metadata-driven model configuration
- Type-safe model specifications

Educational Value:
Shows how to implement a registry pattern for managing different model
types with their configurations and capabilities.
"""

from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# Model Type Enumeration
# =============================================================================

class ModelType(Enum):
    """Enumeration of supported model types."""
    
    # Neural Network Models
    MLP = "mlp"
    CNN = "cnn" 
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    
    # Tree-Based Models
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "randomforest"
    GRADIENT_BOOSTING = "gradientboosting"
    
    # Traditional ML Models
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    NAIVE_BAYES = "naive_bayes"
    KNN = "knn"
    
    # Reinforcement Learning Models
    DQN = "dqn"
    PPO = "ppo"
    A3C = "a3c"
    DDPG = "ddpg"
    SAC = "sac"
    TD3 = "td3"
    
    # Evolutionary Algorithm Models
    GA = "ga"
    ES = "es"
    GP = "gp"
    NSGA2 = "nsga2"
    CMA_ES = "cma_es"
    
    # LLM Models
    GPT_FINETUNE = "gpt_finetune"
    LLAMA_FINETUNE = "llama_finetune"
    BERT_FINETUNE = "bert_finetune"
    
    # Vision-Language Models
    CLIP = "clip"
    BLIP = "blip"
    FLAMINGO = "flamingo"

# =============================================================================
# Model Metadata
# =============================================================================

@dataclass
class ModelMetadata:
    """Metadata for a model type."""
    
    name: str
    category: str
    framework: str
    input_format: str
    output_format: str
    supports_incremental: bool
    supports_gpu: bool
    supports_distributed: bool
    memory_requirements: str
    typical_training_time: str
    best_use_cases: List[str]
    hyperparameters: Dict[str, Any]
    export_formats: List[str]

# =============================================================================
# Model Configurations
# =============================================================================

SUPPORTED_MODEL_TYPES: Set[str] = {model.value for model in ModelType}
"""Set of all supported model type strings."""

MODEL_CATEGORIES: Dict[str, List[str]] = {
    "neural_networks": ["mlp", "cnn", "lstm", "gru", "transformer"],
    "tree_models": ["xgboost", "lightgbm", "randomforest", "gradientboosting"],
    "traditional_ml": ["svm", "logistic_regression", "naive_bayes", "knn"],
    "reinforcement_learning": ["dqn", "ppo", "a3c", "ddpg", "sac", "td3"],
    "evolutionary": ["ga", "es", "gp", "nsga2", "cma_es"],
    "llm": ["gpt_finetune", "llama_finetune", "bert_finetune"],
    "vision_language": ["clip", "blip", "flamingo"]
}
"""Model categories for organization and filtering."""

MODEL_FRAMEWORKS: Dict[str, List[str]] = {
    "pytorch": ["mlp", "cnn", "lstm", "gru", "transformer", "dqn", "ppo", "a3c"],
    "sklearn": ["randomforest", "svm", "logistic_regression", "naive_bayes", "knn"],
    "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm"],
    "deap": ["ga", "es", "gp"],
    "transformers": ["gpt_finetune", "llama_finetune", "bert_finetune"],
    "stable_baselines3": ["ppo", "dqn", "sac", "td3"],
    "custom": ["nsga2", "cma_es", "clip", "blip", "flamingo"]
}
"""Framework requirements for each model type."""

# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY: Dict[str, ModelMetadata] = {
    # Neural Network Models
    "mlp": ModelMetadata(
        name="Multi-Layer Perceptron",
        category="neural_networks",
        framework="pytorch",
        input_format="csv",
        output_format="classification",
        supports_incremental=True,
        supports_gpu=True,
        supports_distributed=True,
        memory_requirements="low",
        typical_training_time="minutes",
        best_use_cases=["tabular_data", "feature_based", "quick_prototyping"],
        hyperparameters={
            "hidden_layers": [128, 64, 32],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        },
        export_formats=["pytorch", "onnx"]
    ),
    
    "cnn": ModelMetadata(
        name="Convolutional Neural Network",
        category="neural_networks", 
        framework="pytorch",
        input_format="npz_spatial",
        output_format="classification",
        supports_incremental=True,
        supports_gpu=True,
        supports_distributed=True,
        memory_requirements="medium",
        typical_training_time="hours",
        best_use_cases=["spatial_data", "image_like", "pattern_recognition"],
        hyperparameters={
            "channels": [32, 64, 128],
            "kernel_size": 3,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        },
        export_formats=["pytorch", "onnx"]
    ),
    
    "lstm": ModelMetadata(
        name="Long Short-Term Memory",
        category="neural_networks",
        framework="pytorch", 
        input_format="npz_sequential",
        output_format="classification",
        supports_incremental=True,
        supports_gpu=True,
        supports_distributed=True,
        memory_requirements="medium",
        typical_training_time="hours",
        best_use_cases=["sequential_data", "temporal_patterns", "time_series"],
        hyperparameters={
            "hidden_size": 128,
            "num_layers": 2,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        },
        export_formats=["pytorch", "onnx"]
    ),
    
    # Tree-Based Models
    "xgboost": ModelMetadata(
        name="XGBoost",
        category="tree_models",
        framework="xgboost",
        input_format="csv",
        output_format="classification", 
        supports_incremental=True,
        supports_gpu=True,
        supports_distributed=True,
        memory_requirements="low",
        typical_training_time="minutes",
        best_use_cases=["tabular_data", "feature_engineering", "competitions"],
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        export_formats=["xgboost", "onnx", "sklearn"]
    ),
    
    "lightgbm": ModelMetadata(
        name="LightGBM", 
        category="tree_models",
        framework="lightgbm",
        input_format="csv",
        output_format="classification",
        supports_incremental=True,
        supports_gpu=True,
        supports_distributed=True,
        memory_requirements="low",
        typical_training_time="minutes",
        best_use_cases=["large_datasets", "fast_training", "memory_efficient"],
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        export_formats=["lightgbm", "onnx", "sklearn"]
    ),
    
    "randomforest": ModelMetadata(
        name="Random Forest",
        category="tree_models",
        framework="sklearn",
        input_format="csv",
        output_format="classification",
        supports_incremental=False,
        supports_gpu=False,
        supports_distributed=False,
        memory_requirements="medium",
        typical_training_time="minutes",
        best_use_cases=["interpretability", "feature_importance", "robust_baseline"],
        hyperparameters={
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1
        },
        export_formats=["sklearn", "onnx"]
    ),
    
    # Reinforcement Learning Models
    "dqn": ModelMetadata(
        name="Deep Q-Network",
        category="reinforcement_learning",
        framework="stable_baselines3",
        input_format="environment",
        output_format="action_values",
        supports_incremental=True,
        supports_gpu=True,
        supports_distributed=False,
        memory_requirements="medium",
        typical_training_time="hours",
        best_use_cases=["discrete_actions", "off_policy", "experience_replay"],
        hyperparameters={
            "learning_rate": 0.0001,
            "buffer_size": 50000,
            "epsilon_start": 1.0,
            "epsilon_decay": 0.995,
            "batch_size": 32
        },
        export_formats=["pytorch", "onnx"]
    ),
    
    "ppo": ModelMetadata(
        name="Proximal Policy Optimization",
        category="reinforcement_learning",
        framework="stable_baselines3",
        input_format="environment",
        output_format="policy",
        supports_incremental=True,
        supports_gpu=True,
        supports_distributed=True,
        memory_requirements="medium",
        typical_training_time="hours",
        best_use_cases=["continuous_actions", "on_policy", "stable_training"],
        hyperparameters={
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "clip_range": 0.2
        },
        export_formats=["pytorch", "onnx"]
    ),
    
    # Evolutionary Models
    "ga": ModelMetadata(
        name="Genetic Algorithm",
        category="evolutionary",
        framework="deap",
        input_format="npz_raw",
        output_format="population",
        supports_incremental=True,
        supports_gpu=False,
        supports_distributed=True,
        memory_requirements="low",
        typical_training_time="variable",
        best_use_cases=["optimization", "search_spaces", "multi_objective"],
        hyperparameters={
            "population_size": 100,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "selection_pressure": 2.0,
            "generations": 100
        },
        export_formats=["pickle", "npz"]
    )
}
"""Complete model registry with metadata."""

# =============================================================================
# Model Export Formats
# =============================================================================

MODEL_EXPORT_FORMATS: Dict[str, Set[str]] = {
    "pytorch": {".pth", ".pt"},
    "onnx": {".onnx"},
    "sklearn": {".pkl", ".joblib"},
    "xgboost": {".model", ".json"},
    "lightgbm": {".model", ".txt"},
    "tensorflow": {".h5", ".pb"},
    "pickle": {".pkl"},
    "npz": {".npz"}
}
"""Supported export formats for different model types."""

# =============================================================================
# Model Capability Matrix
# =============================================================================

MODEL_CAPABILITIES: Dict[str, Dict[str, bool]] = {
    "data_formats": {
        "csv": True,
        "npz_sequential": True,
        "npz_spatial": True,
        "npz_raw": True,
        "jsonl": False  # Models don't train on JSONL directly
    },
    "training_modes": {
        "batch": True,
        "online": True,
        "incremental": True,
        "transfer_learning": True
    },
    "inference_modes": {
        "single_prediction": True,
        "batch_prediction": True,
        "streaming": True,
        "real_time": True
    },
    "hardware_support": {
        "cpu": True,
        "gpu": True,
        "distributed": True,
        "mobile": False
    }
}
"""Capability matrix for model features."""

# =============================================================================
# Model Config Schemas
# =============================================================================

MODEL_CONFIG_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "neural_networks": {
        "required": ["architecture", "optimizer", "loss_function"],
        "optional": ["scheduler", "regularization", "callbacks"],
        "architecture": {
            "required": ["input_size", "output_size"],
            "optional": ["hidden_layers", "activation", "dropout"]
        }
    },
    "tree_models": {
        "required": ["n_estimators", "max_depth"],
        "optional": ["learning_rate", "subsample", "regularization"],
        "feature_selection": {
            "optional": ["feature_importance", "permutation_importance"]
        }
    },
    "reinforcement_learning": {
        "required": ["environment", "algorithm", "policy"],
        "optional": ["exploration", "replay_buffer", "target_network"],
        "training": {
            "required": ["episodes", "steps_per_episode"],
            "optional": ["checkpoint_frequency", "evaluation_frequency"]
        }
    }
}
"""Configuration schemas for different model categories."""

# =============================================================================
# Model Compatibility Matrix
# =============================================================================

EXTENSION_MODEL_COMPATIBILITY: Dict[str, Set[str]] = {
    "heuristics": set(),  # Heuristics don't use trainable models
    "supervised": {
        "mlp", "cnn", "lstm", "xgboost", "lightgbm", "randomforest",
        "svm", "logistic_regression", "naive_bayes", "knn"
    },
    "reinforcement": {
        "dqn", "ppo", "a3c", "ddpg", "sac", "td3"
    },
    "evolutionary": {
        "ga", "es", "gp", "nsga2", "cma_es"
    },
    "llm_finetune": {
        "gpt_finetune", "llama_finetune", "bert_finetune"
    },
    "agentic_llms": {
        "gpt_finetune", "llama_finetune", "bert_finetune"
    },
    "vision_language_model": {
        "clip", "blip", "flamingo"
    }
}
"""Model compatibility matrix for different extension types.""" 