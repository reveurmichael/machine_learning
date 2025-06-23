"""
Supervised Learning Models Package - Factory Pattern Implementation
==================================================================

This package provides a factory pattern for creating supervised learning agents.
It demonstrates software evolution through inheritance and encapsulation.

Available Models:
1. Neural Networks (PyTorch):
   - MLP - Multi-Layer Perceptron for tabular data
   - CNN - Convolutional Neural Network for board data
   - LSTM - Long Short-Term Memory for sequential data
   - GRU - Gated Recurrent Unit for sequential data

2. Tree Models:
   - XGBOOST - XGBoost gradient boosting
   - LIGHTGBM - LightGBM gradient boosting
   - RANDOMFOREST - Random Forest ensemble

3. Graph Models (PyTorch Geometric):
   - GCN - Graph Convolutional Network
   - GRAPHSAGE - GraphSAGE for large graphs
   - GAT - Graph Attention Network

Design Patterns:
- Factory Pattern: create_model() function for instantiation
- Registry Pattern: MODEL_REGISTRY for model mapping
- Inheritance: Progressive enhancement through class hierarchy
- Strategy Pattern: Interchangeable model types
"""

from __future__ import annotations

# Use standardized path setup
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))

from typing import Dict, Type, Optional, List, Any

# Import neural network models
from .neural_networks.agent_mlp import MLPAgent
from .neural_networks.agent_cnn import CNNAgent
from .neural_networks.agent_lstm import LSTMAgent
from .neural_networks.agent_gru import GRUAgent

# Import tree models
from .tree_models.agent_xgboost import XGBoostAgent
from .tree_models.agent_lightgbm import LightGBMAgent
from .tree_models.agent_randomforest import RandomForestAgent

# Import graph models
from .graph_models.agent_gcn import GCNAgent
from .graph_models.agent_graphsage import GraphSAGEAgent
from .graph_models.agent_gat import GATAgent

# Model registry mapping names to classes
MODEL_REGISTRY: Dict[str, Type] = {
    # Neural Networks
    "MLP": MLPAgent,
    "CNN": CNNAgent,
    "LSTM": LSTMAgent,
    "GRU": GRUAgent,
    
    # Tree Models
    "XGBOOST": XGBoostAgent,
    "LIGHTGBM": LightGBMAgent,
    "RANDOMFOREST": RandomForestAgent,
    
    # Graph Models
    "GCN": GCNAgent,
    "GRAPHSAGE": GraphSAGEAgent,
    "GAT": GATAgent,
}

# Aliases for convenience
MODEL_REGISTRY["XGB"] = XGBoostAgent  # XGB is an alias for XGBOOST
MODEL_REGISTRY["LGBM"] = LightGBMAgent  # LGBM is an alias for LIGHTGBM
MODEL_REGISTRY["RF"] = RandomForestAgent  # RF is an alias for RANDOMFOREST

# Default model
DEFAULT_MODEL: str = "MLP"

def create_model(model_name: str, **kwargs) -> Any:
    """
    Factory function to create a model instance.
    
    Args:
        model_name: Name of the model (case-insensitive)
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model name is not recognized
    """
    model_name = model_name.upper()
    
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available}"
        )
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)

def get_available_models() -> List[str]:
    """
    Get list of available model names.
    
    Returns:
        List of model names
    """
    return list(MODEL_REGISTRY.keys())

def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model information
        
    Raises:
        ValueError: If model name is not recognized
    """
    model_name = model_name.upper()
    
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available}"
        )
    
    model_class = MODEL_REGISTRY[model_name]
    model_instance = model_class()
    
    return {
        "name": getattr(model_instance, "name", model_name),
        "description": getattr(model_instance, "description", "No description available"),
        "model_name": getattr(model_instance, "model_name", model_name),
        "framework": _get_model_framework(model_name),
        "category": _get_model_category(model_name),
        "data_type": _get_model_data_type(model_name),
    }

def _get_model_framework(model_name: str) -> str:
    """Get framework information for the model."""
    frameworks = {
        # Neural Networks
        "MLP": "PyTorch",
        "CNN": "PyTorch", 
        "LSTM": "PyTorch",
        "GRU": "PyTorch",
        
        # Tree Models
        "XGBOOST": "XGBoost",
        "LIGHTGBM": "LightGBM",
        "RANDOMFOREST": "Scikit-learn",
        
        # Graph Models
        "GCN": "PyTorch Geometric",
        "GRAPHSAGE": "PyTorch Geometric",
        "GAT": "PyTorch Geometric",
    }
    return frameworks.get(model_name, "Unknown framework")

def _get_model_category(model_name: str) -> str:
    """Get educational category for the model."""
    categories = {
        # Neural Networks
        "MLP": "Neural Network",
        "CNN": "Neural Network",
        "LSTM": "Neural Network", 
        "GRU": "Neural Network",
        
        # Tree Models
        "XGBOOST": "Tree Model",
        "LIGHTGBM": "Tree Model",
        "RANDOMFOREST": "Tree Model",
        
        # Graph Models
        "GCN": "Graph Neural Network",
        "GRAPHSAGE": "Graph Neural Network",
        "GAT": "Graph Neural Network",
    }
    return categories.get(model_name, "Unknown category")

def _get_model_data_type(model_name: str) -> str:
    """Get preferred data type for the model."""
    data_types = {
        # Neural Networks
        "MLP": "tabular",
        "CNN": "board", 
        "LSTM": "sequential",
        "GRU": "sequential",
        
        # Tree Models
        "XGBOOST": "tabular",
        "LIGHTGBM": "tabular",
        "RANDOMFOREST": "tabular",
        
        # Graph Models
        "GCN": "graph",
        "GRAPHSAGE": "graph",
        "GAT": "graph",
    }
    return data_types.get(model_name, "unknown")

# Public API
__all__ = [
    # Neural Network models
    "MLPAgent",
    "CNNAgent", 
    "LSTMAgent",
    "GRUAgent",
    
    # Tree models
    "XGBoostAgent",
    "LightGBMAgent",
    "RandomForestAgent",
    
    # Graph models
    "GCNAgent",
    "GraphSAGEAgent",
    "GATAgent",
    
    # Factory functions
    "create_model",
    "get_available_models",
    "get_model_info",
    
    # Registry
    "MODEL_REGISTRY",
    "DEFAULT_MODEL",
] 