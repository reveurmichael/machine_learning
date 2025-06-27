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

# =============================================================================
# Agent Factory (Factory Method Pattern)
# =============================================================================

class AgentFactory(BaseFactory):
    """
    Factory for creating AI agents.
    
    Design Pattern: Factory Method Pattern
    Purpose: Create agents without specifying their exact classes
    
    Educational Note (SUPREME_RULE NO.4):
    This factory demonstrates how different AI paradigms can be
    unified under a common interface while maintaining their
    specific characteristics and requirements. The factory is
    designed to be extensible for specialized agent creation needs.
    
    SUPREME_RULE NO.4 Implementation:
    - Base factory provides standard agent creation
    - Registry system allows dynamic agent registration
    - Protected methods enable custom creation logic
    - Extension-specific factories can inherit and extend
    """
    
    def __init__(self, extension_type: str):
        super().__init__(extension_type)
        self._agent_registry = self._build_agent_registry()
        self._setup_extension_specific_agents()
    
    def create(self, config: FactoryConfig) -> Any:
        """Create an agent based on configuration."""
        if not self.validate_config(config):
            raise ValueError(f"Invalid configuration for agent creation: {config}")
        
        agent_name = config.component_name
        if agent_name not in self._agent_registry:
            raise ValueError(f"Unknown agent type: {agent_name}")
        
        agent_class = self._agent_registry[agent_name]["class"]
        required_params = self._agent_registry[agent_name]["required_params"]
        
        # Validate required parameters
        missing_params = set(required_params) - set(config.parameters.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        # Create agent with parameters
        try:
            return agent_class(**config.parameters)
        except Exception as e:
            raise RuntimeError(f"Failed to create agent {agent_name}: {str(e)}")
    
    def get_supported_components(self) -> List[str]:
        """Get list of agent types this factory can create."""
        return list(self._agent_registry.keys())
    
    def get_component_specs(self) -> List[ComponentSpec]:
        """Get specifications for all creatable agents."""
        specs = []
        for name, info in self._agent_registry.items():
            spec = ComponentSpec(
                name=name,
                component_type=ComponentType.AGENT,
                factory_class=type(self),
                supported_extensions=info.get("supported_extensions", [self.extension_type]),
                required_parameters=info.get("required_params", []),
                optional_parameters=info.get("optional_params", {}),
                description=info.get("description", f"{name} agent implementation")
            )
            specs.append(spec)
        return specs
    
    def _build_agent_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build registry of available agents based on extension type."""
        registry = {}
        
        if self.extension_type == "heuristics":
            # Heuristic pathfinding agents
            registry.update({
                "BFS": {
                    "class": None,  # Would be imported dynamically
                    "required_params": ["name", "grid_size"],
                    "optional_params": {"use_gui": False},
                    "description": "Breadth-First Search pathfinding agent",
                    "supported_extensions": ["heuristics"]
                },
                "ASTAR": {
                    "class": None,  # Would be imported dynamically
                    "required_params": ["name", "grid_size"],
                    "optional_params": {"heuristic": "manhattan", "use_gui": False},
                    "description": "A* pathfinding agent with configurable heuristic",
                    "supported_extensions": ["heuristics"]
                },
                "DFS": {
                    "class": None,  # Would be imported dynamically
                    "required_params": ["name", "grid_size"],
                    "optional_params": {"max_depth": 1000, "use_gui": False},
                    "description": "Depth-First Search pathfinding agent",
                    "supported_extensions": ["heuristics"]
                },
                "HAMILTONIAN": {
                    "class": None,  # Would be imported dynamically
                    "required_params": ["name", "grid_size"],
                    "optional_params": {"use_gui": False},
                    "description": "Hamiltonian cycle pathfinding agent",
                    "supported_extensions": ["heuristics"]
                }
            })
            
        elif self.extension_type == "supervised":
            # Machine learning agents
            registry.update({
                "MLP": {
                    "class": None,  # Would be imported dynamically
                    "required_params": ["input_size", "output_size"],
                    "optional_params": {
                        "hidden_layers": [64, 32],
                        "activation": "relu",
                        "learning_rate": DEFAULT_LEARNING_RATE,
                        "batch_size": DEFAULT_BATCH_SIZE
                    },
                    "description": "Multi-Layer Perceptron neural network agent",
                    "supported_extensions": ["supervised"]
                },
                "CNN": {
                    "class": None,  # Would be imported dynamically
                    "required_params": ["input_shape", "num_classes"],
                    "optional_params": {
                        "conv_layers": [(32, 3), (64, 3)],
                        "pool_size": 2,
                        "learning_rate": DEFAULT_LEARNING_RATE
                    },
                    "description": "Convolutional Neural Network agent",
                    "supported_extensions": ["supervised"]
                },
                "XGBOOST": {
                    "class": None,  # Would be imported dynamically
                    "required_params": ["n_estimators"],
                    "optional_params": {
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8
                    },
                    "description": "XGBoost gradient boosting agent",
                    "supported_extensions": ["supervised"]
                }
            })
            
        elif self.extension_type == "reinforcement":
            # Reinforcement learning agents
            registry.update({
                "DQN": {
                    "class": None,  # Would be imported dynamically
                    "required_params": ["state_size", "action_size"],
                    "optional_params": {
                        "learning_rate": 0.001,
                        "gamma": 0.95,
                        "epsilon": 0.1,
                        "memory_size": 10000
                    },
                    "description": "Deep Q-Network reinforcement learning agent",
                    "supported_extensions": ["reinforcement"]
                },
                "PPO": {
                    "class": None,  # Would be imported dynamically
                    "required_params": ["state_size", "action_size"],
                    "optional_params": {
                        "learning_rate": 0.0003,
                        "gamma": 0.99,
                        "gae_lambda": 0.95,
                        "clip_epsilon": 0.2
                    },
                    "description": "Proximal Policy Optimization agent",
                    "supported_extensions": ["reinforcement"]
                }
            })
            
        elif self.extension_type == "evolutionary":
            # Evolutionary algorithm agents
            registry.update({
                "GENETIC": {
                    "class": None,  # Would be imported dynamically
                    "required_params": ["population_size", "genome_length"],
                    "optional_params": {
                        "mutation_rate": 0.01,
                        "crossover_rate": 0.8,
                        "selection_method": "tournament"
                    },
                    "description": "Genetic Algorithm agent",
                    "supported_extensions": ["evolutionary"]
                },
                "PARTICLE_SWARM": {
                    "class": None,  # Would be imported dynamically
                    "required_params": ["population_size", "dimensions"],
                    "optional_params": {
                        "inertia": 0.9,
                        "cognitive": 1.5,
                        "social": 1.5
                    },
                    "description": "Particle Swarm Optimization agent",
                    "supported_extensions": ["evolutionary"]
                }
            })
        
        return registry
    
    def _setup_extension_specific_agents(self) -> None:
        """
        Setup extension-specific agents (SUPREME_RULE NO.4 Extension Point).
        
        This method can be overridden by subclasses to register additional
        agents specific to their extension requirements or to modify the
        standard agent configurations.
        
        Example:
            class CustomHeuristicsFactory(AgentFactory):
                def _setup_extension_specific_agents(self):
                    self._agent_registry["CUSTOM_BFS"] = {
                        "class": CustomBFSAgent,
                        "required_params": ["name", "grid_size", "custom_param"],
                        "optional_params": {"optimization": True},
                        "description": "Custom optimized BFS agent"
                    }
        """
        pass

# =============================================================================
# Model Factory (Builder Pattern)
# =============================================================================

class ModelFactory(BaseFactory):
    """
    Factory for creating ML models using Builder pattern.
    
    Design Pattern: Builder Pattern
    Purpose: Construct complex ML models step by step
    
    Educational Note:
    This factory demonstrates how the Builder pattern can be used
    to create complex objects (ML models) with many optional parameters
    while keeping the construction process clear and flexible.
    """
    
    def __init__(self, extension_type: str):
        super().__init__(extension_type)
        self._model_builders = self._build_model_builders()
    
    def create(self, config: FactoryConfig) -> Any:
        """Create a model based on configuration."""
        model_name = config.component_name
        if model_name not in self._model_builders:
            raise ValueError(f"Unknown model type: {model_name}")
        
        builder_class = self._model_builders[model_name]
        builder = builder_class()
        
        # Configure builder with parameters
        for param_name, param_value in config.parameters.items():
            method_name = f"set_{param_name}"
            if hasattr(builder, method_name):
                getattr(builder, method_name)(param_value)
        
        # Build and return model
        return builder.build()
    
    def get_supported_components(self) -> List[str]:
        """Get list of model types this factory can create."""
        return list(self._model_builders.keys())
    
    def get_component_specs(self) -> List[ComponentSpec]:
        """Get specifications for all creatable models."""
        specs = []
        for model_name in self._model_builders.keys():
            # Get model metadata from configuration
            model_info = MODEL_METADATA.get(model_name, {})
            
            spec = ComponentSpec(
                name=model_name,
                component_type=ComponentType.MODEL,
                factory_class=type(self),
                supported_extensions=model_info.get("supported_extensions", [self.extension_type]),
                required_parameters=model_info.get("required_params", []),
                optional_parameters=model_info.get("optional_params", {}),
                description=model_info.get("description", f"{model_name} model implementation")
            )
            specs.append(spec)
        return specs
    
    def _build_model_builders(self) -> Dict[str, Type]:
        """Build dictionary of model builders based on extension type."""
        builders = {}
        
        # Neural network models
        for model_name in NEURAL_NETWORK_MODELS:
            builders[model_name] = self._create_neural_builder_class(model_name)
        
        # Tree-based models
        for model_name in TREE_MODELS:
            builders[model_name] = self._create_tree_builder_class(model_name)
        
        # RL models
        if self.extension_type == "reinforcement":
            for model_name in RL_MODELS:
                builders[model_name] = self._create_rl_builder_class(model_name)
        
        # Evolutionary models
        if self.extension_type == "evolutionary":
            for model_name in EVOLUTIONARY_MODELS:
                builders[model_name] = self._create_evolutionary_builder_class(model_name)
        
        return builders
    
    def _create_neural_builder_class(self, model_name: str) -> Type:
        """Create builder class for neural network models."""
        class NeuralNetworkBuilder:
            def __init__(self):
                self.model_name = model_name
                self.input_size = None
                self.output_size = None
                self.hidden_layers = [64, 32]
                self.activation = "relu"
                self.dropout = 0.0
                self.batch_norm = False
                self.learning_rate = DEFAULT_LEARNING_RATE
            
            def set_input_size(self, size: int):
                self.input_size = size
                return self
            
            def set_output_size(self, size: int):
                self.output_size = size
                return self
            
            def set_hidden_layers(self, layers: List[int]):
                self.hidden_layers = layers
                return self
            
            def set_activation(self, activation: str):
                self.activation = activation
                return self
            
            def set_dropout(self, dropout: float):
                self.dropout = dropout
                return self
            
            def set_batch_norm(self, use_batch_norm: bool):
                self.batch_norm = use_batch_norm
                return self
            
            def build(self):
                # Would create actual neural network here
                # This is a placeholder implementation
                return {
                    "type": "neural_network",
                    "model_name": self.model_name,
                    "input_size": self.input_size,
                    "output_size": self.output_size,
                    "hidden_layers": self.hidden_layers,
                    "activation": self.activation,
                    "dropout": self.dropout,
                    "batch_norm": self.batch_norm,
                    "learning_rate": self.learning_rate
                }
        
        return NeuralNetworkBuilder
    
    def _create_tree_builder_class(self, model_name: str) -> Type:
        """Create builder class for tree-based models."""
        class TreeModelBuilder:
            def __init__(self):
                self.model_name = model_name
                self.max_depth = None
                self.min_samples_split = 2
                self.min_samples_leaf = 1
                self.criterion = "gini"
                self.n_estimators = 100
            
            def set_max_depth(self, depth: int):
                self.max_depth = depth
                return self
            
            def set_min_samples_split(self, samples: int):
                self.min_samples_split = samples
                return self
            
            def set_min_samples_leaf(self, samples: int):
                self.min_samples_leaf = samples
                return self
            
            def set_criterion(self, criterion: str):
                self.criterion = criterion
                return self
            
            def build(self):
                # Would create actual tree model here
                return {
                    "type": "tree_model",
                    "model_name": self.model_name,
                    "max_depth": self.max_depth,
                    "min_samples_split": self.min_samples_split,
                    "min_samples_leaf": self.min_samples_leaf,
                    "criterion": self.criterion,
                    "n_estimators": self.n_estimators
                }
        
        return TreeModelBuilder
    
    def _create_rl_builder_class(self, model_name: str) -> Type:
        """Create builder class for RL models."""
        class RLModelBuilder:
            def __init__(self):
                self.model_name = model_name
                self.state_size = None
                self.action_size = None
                self.learning_rate = 0.001
                self.gamma = 0.95
                self.epsilon = 0.1
                self.memory_size = 10000
            
            def set_state_size(self, size: int):
                self.state_size = size
                return self
            
            def set_action_size(self, size: int):
                self.action_size = size
                return self
            
            def set_learning_rate(self, lr: float):
                self.learning_rate = lr
                return self
            
            def set_gamma(self, gamma: float):
                self.gamma = gamma
                return self
            
            def set_epsilon(self, epsilon: float):
                self.epsilon = epsilon
                return self
            
            def build(self):
                # Would create actual RL model here
                return {
                    "type": "rl_model",
                    "model_name": self.model_name,
                    "state_size": self.state_size,
                    "action_size": self.action_size,
                    "learning_rate": self.learning_rate,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "memory_size": self.memory_size
                }
        
        return RLModelBuilder
    
    def _create_evolutionary_builder_class(self, model_name: str) -> Type:
        """Create builder class for evolutionary models."""
        class EvolutionaryModelBuilder:
            def __init__(self):
                self.model_name = model_name
                self.population_size = 50
                self.genome_length = None
                self.mutation_rate = 0.01
                self.crossover_rate = 0.8
                self.selection_method = "tournament"
            
            def set_population_size(self, size: int):
                self.population_size = size
                return self
            
            def set_genome_length(self, length: int):
                self.genome_length = length
                return self
            
            def set_mutation_rate(self, rate: float):
                self.mutation_rate = rate
                return self
            
            def set_crossover_rate(self, rate: float):
                self.crossover_rate = rate
                return self
            
            def set_selection_method(self, method: str):
                self.selection_method = method
                return self
            
            def build(self):
                # Would create actual evolutionary model here
                return {
                    "type": "evolutionary_model",
                    "model_name": self.model_name,
                    "population_size": self.population_size,
                    "genome_length": self.genome_length,
                    "mutation_rate": self.mutation_rate,
                    "crossover_rate": self.crossover_rate,
                    "selection_method": self.selection_method
                }
        
        return EvolutionaryModelBuilder

# =============================================================================
# Component Registry (Registry Pattern)
# =============================================================================

class ComponentRegistry:
    """
    Registry for managing factories and creating components.
    
    Design Pattern: Registry Pattern
    Purpose: Provide centralized lookup and creation of components
    
    Educational Note:
    The Registry pattern provides a centralized way to manage
    object creation while keeping the system flexible and extensible.
    """
    
    def __init__(self):
        self._factories: Dict[str, Dict[ComponentType, BaseFactory]] = {}
    
    def register_factory(
        self,
        extension_type: str,
        component_type: ComponentType,
        factory: BaseFactory
    ) -> None:
        """Register a factory for a specific extension and component type."""
        if extension_type not in self._factories:
            self._factories[extension_type] = {}
        
        self._factories[extension_type][component_type] = factory
    
    def get_factory(
        self,
        extension_type: str,
        component_type: ComponentType
    ) -> Optional[BaseFactory]:
        """Get factory for specific extension and component type."""
        return self._factories.get(extension_type, {}).get(component_type)
    
    def create_component(self, config: FactoryConfig) -> Any:
        """Create component using appropriate factory."""
        factory = self.get_factory(config.extension_type, config.component_type)
        if factory is None:
            raise ValueError(f"No factory registered for {config.extension_type} {config.component_type.value}")
        
        return factory.create(config)
    
    def get_available_components(
        self,
        extension_type: str,
        component_type: Optional[ComponentType] = None
    ) -> Dict[ComponentType, List[str]]:
        """Get available components for extension type."""
        result = {}
        
        if extension_type not in self._factories:
            return result
        
        factories = self._factories[extension_type]
        if component_type is not None:
            if component_type in factories:
                result[component_type] = factories[component_type].get_supported_components()
        else:
            for comp_type, factory in factories.items():
                result[comp_type] = factory.get_supported_components()
        
        return result

# =============================================================================
# Global Registry and Convenience Functions
# =============================================================================

_global_registry = ComponentRegistry()

def get_component_registry() -> ComponentRegistry:
    """Get global component registry."""
    return _global_registry

def initialize_standard_factories(extension_type: str) -> None:
    """Initialize standard factories for an extension type."""
    registry = get_component_registry()
    
    # Register agent factory
    agent_factory = AgentFactory(extension_type)
    registry.register_factory(extension_type, ComponentType.AGENT, agent_factory)
    
    # Register model factory
    model_factory = ModelFactory(extension_type)
    registry.register_factory(extension_type, ComponentType.MODEL, model_factory)

def create_agent(
    extension_type: str,
    agent_name: str,
    grid_size: int,
    **kwargs
) -> Any:
    """
    Convenience function to create an agent.
    
    Args:
        extension_type: Type of extension (heuristics, supervised, etc.)
        agent_name: Name of agent to create
        grid_size: Grid size for the agent
        **kwargs: Additional parameters for agent creation
        
    Returns:
        Created agent instance
    """
    # Ensure factories are initialized
    initialize_standard_factories(extension_type)
    
    config = FactoryConfig(
        component_type=ComponentType.AGENT,
        component_name=agent_name,
        parameters={"grid_size": grid_size, **kwargs},
        extension_type=extension_type,
        version="latest",
        grid_size=grid_size
    )
    
    registry = get_component_registry()
    return registry.create_component(config)

def create_model(
    extension_type: str,
    model_name: str,
    **kwargs
) -> Any:
    """
    Convenience function to create a model.
    
    Args:
        extension_type: Type of extension (supervised, reinforcement, etc.)
        model_name: Name of model to create
        **kwargs: Parameters for model creation
        
    Returns:
        Created model instance
    """
    # Ensure factories are initialized
    initialize_standard_factories(extension_type)
    
    config = FactoryConfig(
        component_type=ComponentType.MODEL,
        component_name=model_name,
        parameters=kwargs,
        extension_type=extension_type,
        version="latest",
        grid_size=kwargs.get("grid_size", 10)
    )
    
    registry = get_component_registry()
    return registry.create_component(config)

def list_available_components(extension_type: str) -> Dict[str, List[str]]:
    """
    List all available components for an extension type.
    
    Args:
        extension_type: Type of extension
        
    Returns:
        Dictionary mapping component types to available components
    """
    # Ensure factories are initialized
    initialize_standard_factories(extension_type)
    
    registry = get_component_registry()
    components = registry.get_available_components(extension_type)
    
    # Convert enum keys to strings for easier consumption
    return {comp_type.value: comp_list for comp_type, comp_list in components.items()}

def register_agent_type(extension_type: str, agent_name: str, agent_class: Type) -> None:
    """Register a new agent type with the factory."""
    registry = get_component_registry()
    factory = registry.get_factory(extension_type, ComponentType.AGENT)
    
    if factory is None:
        initialize_standard_factories(extension_type)
        factory = registry.get_factory(extension_type, ComponentType.AGENT)
    
    # Add to agent registry
    if hasattr(factory, '_agent_registry'):
        factory._agent_registry[agent_name] = {
            "class": agent_class,
            "required_params": ["name", "grid_size"],
            "optional_params": {},
            "description": f"Custom {agent_name} agent",
            "supported_extensions": [extension_type]
        }

def register_model_type(extension_type: str, model_name: str, builder_class: Type) -> None:
    """Register a new model type with the factory."""
    registry = get_component_registry()
    factory = registry.get_factory(extension_type, ComponentType.MODEL)
    
    if factory is None:
        initialize_standard_factories(extension_type)
        factory = registry.get_factory(extension_type, ComponentType.MODEL)
    
    # Add to model builders
    if hasattr(factory, '_model_builders'):
        factory._model_builders[model_name] = builder_class 