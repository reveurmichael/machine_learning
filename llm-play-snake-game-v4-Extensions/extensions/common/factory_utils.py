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

# Import configuration constants
from .config.model_registry import (
    ModelType, MODEL_METADATA, NEURAL_NETWORK_MODELS,
    TREE_MODELS, RL_MODELS, EVOLUTIONARY_MODELS
)
from .config.ml_constants import DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE
from .config.validation_rules import ALGORITHM_SUPPORT

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
            specs.append(ComponentSpec(
                name=name,
                component_type=ComponentType.AGENT,
                factory_class=info["class"],
                supported_extensions=info["extensions"],
                required_parameters=info["required_params"],
                optional_parameters=info.get("optional_params", {}),
                description=info.get("description", f"Agent: {name}")
            ))
        return specs
    
    def _build_agent_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build registry of available agents for this extension type."""
        registry = {}
        
        if self.extension_type == "heuristics":
            registry.update({
                "bfs": {
                    "class": None,  # Would import actual class
                    "required_params": ["grid_size"],
                    "optional_params": {"max_depth": 100},
                    "extensions": ["heuristics"],
                    "description": "Breadth-First Search agent"
                },
                "astar": {
                    "class": None,  # Would import actual class
                    "required_params": ["grid_size", "heuristic"],
                    "optional_params": {"max_iterations": 1000},
                    "extensions": ["heuristics"],
                    "description": "A* Search agent"
                },
                "hamiltonian": {
                    "class": None,  # Would import actual class
                    "required_params": ["grid_size"],
                    "optional_params": {"safety_margin": 2},
                    "extensions": ["heuristics"],
                    "description": "Hamiltonian Cycle agent"
                }
            })
        
        elif self.extension_type == "supervised":
            registry.update({
                "neural_network": {
                    "class": None,  # Would import actual class
                    "required_params": ["input_size", "output_size"],
                    "optional_params": {"hidden_layers": [64, 32], "activation": "relu"},
                    "extensions": ["supervised"],
                    "description": "Neural Network agent"
                },
                "decision_tree": {
                    "class": None,  # Would import actual class
                    "required_params": ["max_depth"],
                    "optional_params": {"criterion": "gini", "min_samples_split": 2},
                    "extensions": ["supervised"],
                    "description": "Decision Tree agent"
                }
            })
        
        elif self.extension_type == "reinforcement":
            registry.update({
                "dqn": {
                    "class": None,  # Would import actual class
                    "required_params": ["state_size", "action_size"],
                    "optional_params": {"lr": DEFAULT_LEARNING_RATE, "gamma": 0.95},
                    "extensions": ["reinforcement"],
                    "description": "Deep Q-Network agent"
                },
                "ppo": {
                    "class": None,  # Would import actual class
                    "required_params": ["state_size", "action_size"],
                    "optional_params": {"lr": DEFAULT_LEARNING_RATE, "clip_ratio": 0.2},
                    "extensions": ["reinforcement"],
                    "description": "Proximal Policy Optimization agent"
                }
            })
        
        elif self.extension_type == "evolutionary":
            registry.update({
                "genetic_algorithm": {
                    "class": None,  # Would import actual class
                    "required_params": ["population_size", "genome_length"],
                    "optional_params": {"mutation_rate": 0.01, "crossover_rate": 0.8},
                    "extensions": ["evolutionary"],
                    "description": "Genetic Algorithm agent"
                },
                "neuroevolution": {
                    "class": None,  # Would import actual class
                    "required_params": ["network_structure", "population_size"],
                    "optional_params": {"mutation_strength": 0.1, "elitism": 0.1},
                    "extensions": ["evolutionary"],
                    "description": "Neuroevolution agent"
                }
            })
        
        elif self.extension_type == "llm":
            registry.update({
                "llm_agent": {
                    "class": None,  # Would import actual class
                    "required_params": ["model_name", "api_key"],
                    "optional_params": {"temperature": 0.7, "max_tokens": 1000},
                    "extensions": ["llm"],
                    "description": "Large Language Model agent"
                }
            })
        
        return registry
    
    def _setup_extension_specific_agents(self) -> None:
        """
        Setup extension-specific agents (SUPREME_RULE NO.4 Extension Point).
        
        This method can be overridden by subclasses to register additional
        agents or modify the agent registry for specialized requirements.
        
        Example:
            class CustomEvolutionaryFactory(AgentFactory):
                def _setup_extension_specific_agents(self):
                    self._agent_registry["multi_objective_ga"] = {
                        "class": MultiObjectiveGAAgent,
                        "required_params": ["population_size", "objectives"],
                        "extensions": ["evolutionary"],
                        "description": "Multi-objective genetic algorithm"
                    }
        """
        pass

# =============================================================================
# Model Factory (Builder Pattern)
# =============================================================================

class ModelFactory(BaseFactory):
    """
    Factory for creating machine learning models.
    
    Design Pattern: Builder Pattern
    Purpose: Construct complex model objects step by step
    
    Educational Note:
    The Builder pattern is ideal for model creation because models
    often require complex configuration with many optional parameters
    that can be set in different combinations.
    """
    
    def __init__(self, extension_type: str):
        super().__init__(extension_type)
        self._model_builders = self._build_model_builders()
    
    def create(self, config: FactoryConfig) -> Any:
        """Create a model using the builder pattern."""
        model_name = config.component_name
        if model_name not in self._model_builders:
            raise ValueError(f"Unknown model type: {model_name}")
        
        builder = self._model_builders[model_name]()
        
        # Configure model using builder methods
        for param, value in config.parameters.items():
            if hasattr(builder, f"set_{param}"):
                getattr(builder, f"set_{param}")(value)
            else:
                self.logger.warning(f"Unknown parameter {param} for model {model_name}")
        
        # Build the final model
        try:
            return builder.build()
        except Exception as e:
            raise RuntimeError(f"Failed to build model {model_name}: {str(e)}")
    
    def get_supported_components(self) -> List[str]:
        """Get list of model types this factory can create."""
        return list(self._model_builders.keys())
    
    def get_component_specs(self) -> List[ComponentSpec]:
        """Get specifications for all creatable models."""
        specs = []
        for model_type in ModelType:
            if model_type.value in MODEL_METADATA:
                metadata = MODEL_METADATA[model_type.value]
                if self.extension_type in metadata.get("supported_extensions", []):
                    specs.append(ComponentSpec(
                        name=model_type.value,
                        component_type=ComponentType.MODEL,
                        factory_class=type(None),  # Would be actual builder class
                        supported_extensions=metadata["supported_extensions"],
                        required_parameters=metadata.get("required_params", []),
                        optional_parameters=metadata.get("hyperparameters", {}),
                        description=metadata.get("description", f"Model: {model_type.value}")
                    ))
        return specs
    
    def _build_model_builders(self) -> Dict[str, Type]:
        """Build registry of model builders for this extension type."""
        builders = {}
        
        # Neural network models
        if self.extension_type in ["supervised", "reinforcement"]:
            for model_name in NEURAL_NETWORK_MODELS:
                builders[model_name] = self._create_neural_builder_class(model_name)
        
        # Tree models
        if self.extension_type == "supervised":
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
        """Create a builder class for neural network models."""
        class NeuralNetworkBuilder:
            def __init__(self):
                self.config = {
                    "input_size": None,
                    "output_size": None,
                    "hidden_layers": [64, 32],
                    "activation": "relu",
                    "dropout": 0.1,
                    "batch_norm": False
                }
            
            def set_input_size(self, size: int):
                self.config["input_size"] = size
                return self
            
            def set_output_size(self, size: int):
                self.config["output_size"] = size
                return self
            
            def set_hidden_layers(self, layers: List[int]):
                self.config["hidden_layers"] = layers
                return self
            
            def set_activation(self, activation: str):
                self.config["activation"] = activation
                return self
            
            def set_dropout(self, dropout: float):
                self.config["dropout"] = dropout
                return self
            
            def set_batch_norm(self, use_batch_norm: bool):
                self.config["batch_norm"] = use_batch_norm
                return self
            
            def build(self):
                if self.config["input_size"] is None or self.config["output_size"] is None:
                    raise ValueError("Input and output sizes must be specified")
                
                # Would create actual neural network here
                # For now, return configuration
                return {
                    "type": model_name,
                    "config": self.config.copy()
                }
        
        return NeuralNetworkBuilder
    
    def _create_tree_builder_class(self, model_name: str) -> Type:
        """Create a builder class for tree-based models."""
        class TreeModelBuilder:
            def __init__(self):
                self.config = {
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "criterion": "gini"
                }
            
            def set_max_depth(self, depth: int):
                self.config["max_depth"] = depth
                return self
            
            def set_min_samples_split(self, samples: int):
                self.config["min_samples_split"] = samples
                return self
            
            def set_min_samples_leaf(self, samples: int):
                self.config["min_samples_leaf"] = samples
                return self
            
            def set_criterion(self, criterion: str):
                self.config["criterion"] = criterion
                return self
            
            def build(self):
                # Would create actual tree model here
                return {
                    "type": model_name,
                    "config": self.config.copy()
                }
        
        return TreeModelBuilder
    
    def _create_rl_builder_class(self, model_name: str) -> Type:
        """Create a builder class for RL models."""
        class RLModelBuilder:
            def __init__(self):
                self.config = {
                    "state_size": None,
                    "action_size": None,
                    "learning_rate": DEFAULT_LEARNING_RATE,
                    "gamma": 0.95,
                    "epsilon": 0.1
                }
            
            def set_state_size(self, size: int):
                self.config["state_size"] = size
                return self
            
            def set_action_size(self, size: int):
                self.config["action_size"] = size
                return self
            
            def set_learning_rate(self, lr: float):
                self.config["learning_rate"] = lr
                return self
            
            def set_gamma(self, gamma: float):
                self.config["gamma"] = gamma
                return self
            
            def set_epsilon(self, epsilon: float):
                self.config["epsilon"] = epsilon
                return self
            
            def build(self):
                if self.config["state_size"] is None or self.config["action_size"] is None:
                    raise ValueError("State and action sizes must be specified")
                
                # Would create actual RL model here
                return {
                    "type": model_name,
                    "config": self.config.copy()
                }
        
        return RLModelBuilder
    
    def _create_evolutionary_builder_class(self, model_name: str) -> Type:
        """Create a builder class for evolutionary models."""
        class EvolutionaryModelBuilder:
            def __init__(self):
                self.config = {
                    "population_size": 100,
                    "genome_length": None,
                    "mutation_rate": 0.01,
                    "crossover_rate": 0.8,
                    "selection_method": "tournament"
                }
            
            def set_population_size(self, size: int):
                self.config["population_size"] = size
                return self
            
            def set_genome_length(self, length: int):
                self.config["genome_length"] = length
                return self
            
            def set_mutation_rate(self, rate: float):
                self.config["mutation_rate"] = rate
                return self
            
            def set_crossover_rate(self, rate: float):
                self.config["crossover_rate"] = rate
                return self
            
            def set_selection_method(self, method: str):
                self.config["selection_method"] = method
                return self
            
            def build(self):
                if self.config["genome_length"] is None:
                    raise ValueError("Genome length must be specified")
                
                # Would create actual evolutionary model here
                return {
                    "type": model_name,
                    "config": self.config.copy()
                }
        
        return EvolutionaryModelBuilder

# =============================================================================
# Registry Pattern Implementation
# =============================================================================

class ComponentRegistry:
    """
    Registry for managing component factories.
    
    Design Pattern: Registry Pattern
    Purpose: Centralized registry for factory lookup and management
    
    Educational Note:
    The Registry pattern provides a centralized way to manage
    different factories and allows for dynamic registration
    of new component types.
    """
    
    def __init__(self):
        self._factories: Dict[str, Dict[ComponentType, BaseFactory]] = {}
        self.logger = logging.getLogger("ComponentRegistry")
    
    def register_factory(
        self,
        extension_type: str,
        component_type: ComponentType,
        factory: BaseFactory
    ) -> None:
        """Register a factory for an extension and component type."""
        if extension_type not in self._factories:
            self._factories[extension_type] = {}
        
        self._factories[extension_type][component_type] = factory
        self.logger.info(f"Registered factory: {extension_type}.{component_type.value}")
    
    def get_factory(
        self,
        extension_type: str,
        component_type: ComponentType
    ) -> Optional[BaseFactory]:
        """Get factory for extension and component type."""
        return self._factories.get(extension_type, {}).get(component_type)
    
    def create_component(self, config: FactoryConfig) -> Any:
        """Create component using appropriate factory."""
        factory = self.get_factory(config.extension_type, config.component_type)
        if factory is None:
            raise ValueError(
                f"No factory registered for {config.extension_type}.{config.component_type.value}"
            )
        
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
        
        for comp_type, factory in self._factories[extension_type].items():
            if component_type is None or comp_type == component_type:
                result[comp_type] = factory.get_supported_components()
        
        return result

# =============================================================================
# Global Registry Instance
# =============================================================================

# Global registry instance for shared use across extensions
_global_registry = ComponentRegistry()

def get_component_registry() -> ComponentRegistry:
    """Get the global component registry."""
    return _global_registry

def initialize_standard_factories(extension_type: str) -> None:
    """Initialize standard factories for an extension type."""
    registry = get_component_registry()
    
    # Register standard factories
    registry.register_factory(
        extension_type,
        ComponentType.AGENT,
        AgentFactory(extension_type)
    )
    
    registry.register_factory(
        extension_type,
        ComponentType.MODEL,
        ModelFactory(extension_type)
    )

# =============================================================================
# High-Level Factory Functions
# =============================================================================

def create_agent(
    extension_type: str,
    agent_name: str,
    grid_size: int,
    **kwargs
) -> Any:
    """
    High-level function to create an agent.
    
    Args:
        extension_type: Type of extension (heuristics, supervised, etc.)
        agent_name: Name of agent to create
        grid_size: Grid size for the game
        **kwargs: Additional parameters for agent creation
    
    Returns:
        Created agent instance
    """
    config = FactoryConfig(
        component_type=ComponentType.AGENT,
        component_name=agent_name,
        parameters={"grid_size": grid_size, **kwargs},
        extension_type=extension_type,
        version="v0.01",  # Default version
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
    High-level function to create a model.
    
    Args:
        extension_type: Type of extension
        model_name: Name of model to create
        **kwargs: Model configuration parameters
    
    Returns:
        Created model instance
    """
    config = FactoryConfig(
        component_type=ComponentType.MODEL,
        component_name=model_name,
        parameters=kwargs,
        extension_type=extension_type,
        version="v0.01",  # Default version
        grid_size=kwargs.get("grid_size", 10)  # Default grid size
    )
    
    registry = get_component_registry()
    return registry.create_component(config)

def list_available_components(extension_type: str) -> Dict[str, List[str]]:
    """
    List all available components for an extension type.
    
    Args:
        extension_type: Extension type to query
    
    Returns:
        Dictionary mapping component types to available component names
    """
    registry = get_component_registry()
    components = registry.get_available_components(extension_type)
    
    # Convert enum keys to strings for easier use
    return {comp_type.value: comp_list for comp_type, comp_list in components.items()} 