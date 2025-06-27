"""
Factory Utilities for Snake Game AI Extensions.

This module provides shared factory pattern implementations for creating
agents, models, data loaders, and other components across extensions.

Design Patterns:
- Abstract Factory: Create families of related objects
- Factory Method: Create objects without specifying exact classes
- Builder Pattern: Construct complex objects step by step
- Registry Pattern: Register and lookup factory implementations

Educational Value:
Demonstrates how to use multiple factory patterns together to create
a flexible and extensible object creation system that supports
different extension types while maintaining clean separation of concerns.

Key Features:
- Agent factory for different AI paradigms
- Model factory for neural networks and traditional ML
- Data processor factory for different data formats
- Configuration-driven object creation
- Extensible registry system
"""

from typing import Dict, List, Any, Optional, Type, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

# Import configuration constants (updated paths for new structure)
from ..config.model_registry import (
    ModelType, MODEL_METADATA, NEURAL_NETWORK_MODELS,
    TREE_MODELS, RL_MODELS, EVOLUTIONARY_MODELS
)
from ..config.ml_constants import DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE
from ..config.validation_rules import ALGORITHM_SUPPORT

# =============================================================================
# Factory Configuration Classes
# =============================================================================

class ComponentType(Enum):
    """Types of components that can be created by factories."""
    AGENT = "agent"
    MODEL = "model"
    DATA_PROCESSOR = "data_processor"
    OPTIMIZER = "optimizer"
    LOSS_FUNCTION = "loss_function"

@dataclass
class FactoryConfig:
    """Configuration for factory object creation."""
    component_type: ComponentType
    component_name: str
    parameters: Dict[str, Any]
    extension_type: str
    version: str
    grid_size: int

@dataclass
class ComponentSpec:
    """Specification for a component that can be created."""
    name: str
    component_type: ComponentType
    factory_class: Type
    supported_extensions: List[str]
    required_parameters: List[str]
    optional_parameters: Dict[str, Any]
    description: str

# =============================================================================
# Base Factory Classes (Abstract Factory Pattern)
# =============================================================================

class BaseFactory(ABC):
    """
    Abstract base class for component factories.
    
    Design Pattern: Abstract Factory Pattern
    Purpose: Define interface for creating families of related objects
    
    Educational Note (SUPREME_RULE NO.4):
    The Abstract Factory pattern is used here to ensure that all
    factories provide consistent interfaces while allowing for
    extension-specific implementations. This base class is designed
    to be extensible for specialized factory requirements.
    
    SUPREME_RULE NO.4 Implementation:
    - Base class provides complete factory functionality for most cases
    - Protected methods allow selective customization by subclasses
    - Virtual methods enable complete behavior replacement when needed
    - Extension-specific factories can inherit and adapt as needed
    """
    
    def __init__(self, extension_type: str):
        self.extension_type = extension_type
        self.logger = logging.getLogger(f"Factory.{self.__class__.__name__}")
        self._validate_extension_type()
        self._initialize_factory_settings()
    
    def _validate_extension_type(self) -> None:
        """Validate that extension type is supported."""
        valid_types = ["heuristics", "supervised", "reinforcement", "evolutionary", "llm"]
        if self.extension_type not in valid_types:
            raise ValueError(f"Unsupported extension type: {self.extension_type}")
    
    @abstractmethod
    def create(self, config: FactoryConfig) -> Any:
        """Create component based on configuration."""
        pass
    
    @abstractmethod
    def get_supported_components(self) -> List[str]:
        """Get list of components this factory can create."""
        pass
    
    def validate_config(self, config: FactoryConfig) -> bool:
        """Validate configuration before creation."""
        if config.component_type.value not in [comp.component_type.value 
                                               for comp in self.get_component_specs()]:
            return False
        return True
    
    @abstractmethod
    def get_component_specs(self) -> List[ComponentSpec]:
        """Get specifications for all creatable components."""
        pass
    
    def _initialize_factory_settings(self) -> None:
        """
        Initialize factory-specific settings (SUPREME_RULE NO.4 Extension Point).
        
        This method can be overridden by subclasses to set up extension-specific
        factory configurations, custom validators, or specialized creation logic.
        
        Example:
            class CustomRLFactory(BaseFactory):
                def _initialize_factory_settings(self):
                    self.custom_model_validator = RLModelValidator()
                    self.environment_integration = True
        """
        pass
