# Factory Design Pattern for Snake Game AI Extensions

> **Important — Authoritative Reference:** This document establishes the standard factory pattern implementation across all Snake Game AI extensions. All extensions MUST follow these patterns for consistency and maintainability.

## 🎯 **Core Philosophy: Consistent Creation Patterns**

The Factory Pattern is a cornerstone design pattern in the Snake Game AI project, enabling:
- **Flexible agent/model creation** without tight coupling to concrete classes
- **Plugin-style architecture** where new algorithms can be added seamlessly  
- **Consistent interfaces** across different extension types
- **Educational demonstration** of fundamental design patterns

### **Design Benefits**
- **Loose Coupling**: Client code doesn't depend on specific implementation classes
- **Open/Closed Principle**: Easy to add new types without modifying existing code
- **Single Responsibility**: Creation logic centralized in factory classes
- **Consistent Interface**: Uniform creation patterns across all extensions

## 🏗️ **Standard Factory Template**

### **Universal Factory Pattern**
All extensions MUST use this standardized factory implementation:

```python
class {Type}Factory:
    """
    Factory Pattern Implementation for {Type} Creation
    
    Design Pattern: Factory Pattern
    Purpose: Create {type} instances without exposing instantiation logic
    Educational Note: Demonstrates how factory patterns enable plugin architectures
    and support the Open/Closed Principle by allowing new types to be added
    without modifying existing client code.
    
    Benefits:
    - Decouples client code from concrete implementations
    - Centralizes creation logic for easy maintenance
    - Enables dynamic type selection at runtime
    - Supports plugin-style architecture
    """
    
    _registry = {
        # Map string identifiers to implementation classes
        "TYPE1": Type1Implementation,
        "TYPE2": Type2Implementation,
        "TYPE3": Type3Implementation,
    }
    
    @classmethod
    def create(cls, type_name: str, **kwargs) -> BaseType:
        """
        Create instance by type name
        
        Args:
            type_name: String identifier for the type to create
            **kwargs: Additional arguments passed to the constructor
            
        Returns:
            Instance of the requested type
            
        Raises:
            ValueError: If type_name is not registered
            
        Example:
            agent = AgentFactory.create("BFS", grid_size=10)
            model = ModelFactory.create("MLP", hidden_size=128)
        """
        type_class = cls._registry.get(type_name.upper())
        if not type_class:
            available_types = list(cls._registry.keys())
            raise ValueError(
                f"Unknown {cls.__name__.replace('Factory', '').lower()}: {type_name}. "
                f"Available types: {available_types}"
            )
        return type_class(**kwargs)
    
    @classmethod
    def register(cls, type_name: str, type_class: type) -> None:
        """
        Register a new type with the factory
        
        This method enables plugin-style registration of new implementations
        without modifying the factory code itself.
        
        Args:
            type_name: String identifier for the new type
            type_class: Class to register for this type name
        """
        cls._registry[type_name.upper()] = type_class
    
    @classmethod
    def list_available(cls) -> List[str]:
        """Return list of all available type names"""
        return list(cls._registry.keys())
    
    @classmethod
    def is_available(cls, type_name: str) -> bool:
        """Check if a type name is available"""
        return type_name.upper() in cls._registry
```

## 🧠 **Extension-Specific Implementations**

### **Heuristics Factory**
```python
# extensions/heuristics-v0.02/agents/__init__.py

from .agent_bfs import BFSAgent
from .agent_astar import AStarAgent
from .agent_dfs import DFSAgent
from .agent_hamiltonian import HamiltonianAgent

class HeuristicAgentFactory:
    """
    Factory Pattern Implementation for Heuristic Agents
    
    Design Pattern: Factory Pattern
    Purpose: Create heuristic agent instances without exposing instantiation logic
    Educational Note: Demonstrates how factory patterns enable algorithm selection
    at runtime, supporting the Strategy Pattern for different pathfinding approaches.
    """
    
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
        "HAMILTONIAN": HamiltonianAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, grid_size: int = 10, **kwargs) -> BaseAgent:
        """Create heuristic agent by algorithm name"""
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
        return agent_class(name=algorithm, grid_size=grid_size, **kwargs)
    
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """Return list of available heuristic algorithms"""
        return list(cls._registry.keys())
```

### **Supervised Learning Factory**
```python
# extensions/supervised-v0.02/models/__init__.py

from .neural_networks.agent_mlp import MLPAgent
from .neural_networks.agent_cnn import CNNAgent
from .tree_models.agent_xgboost import XGBoostAgent
from .tree_models.agent_lightgbm import LightGBMAgent

class SupervisedModelFactory:
    """
    Factory Pattern Implementation for Supervised Learning Models
    
    Design Pattern: Factory Pattern + Abstract Factory Pattern
    Purpose: Create ML model instances across different frameworks and architectures
    Educational Note: Shows how factory patterns can handle multiple model types
    (neural networks, tree models, etc.) with a unified interface.
    """
    
    _registry = {
        # Neural Network Models
        "MLP": MLPAgent,
        "CNN": CNNAgent,
        "LSTM": LSTMAgent,
        
        # Tree-Based Models  
        "XGBOOST": XGBoostAgent,
        "LIGHTGBM": LightGBMAgent,
        "RANDOMFOREST": RandomForestAgent,
        
        # Graph Neural Networks
        "GCN": GCNAgent,
        "GRAPHSAGE": GraphSAGEAgent,
    }
    
    @classmethod
    def create(cls, model_type: str, input_dim: int, **kwargs) -> BaseMLAgent:
        """Create supervised learning model by type"""
        model_class = cls._registry.get(model_type.upper())
        if not model_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model: {model_type}. Available: {available}")
        return model_class(input_dim=input_dim, **kwargs)
    
    @classmethod
    def get_model_category(cls, model_type: str) -> str:
        """Return the category of a model (neural, tree, graph)"""
        neural_models = {"MLP", "CNN", "LSTM", "GRU"}
        tree_models = {"XGBOOST", "LIGHTGBM", "RANDOMFOREST"}
        graph_models = {"GCN", "GRAPHSAGE", "GAT"}
        
        model_upper = model_type.upper()
        if model_upper in neural_models:
            return "neural"
        elif model_upper in tree_models:
            return "tree"
        elif model_upper in graph_models:
            return "graph"
        else:
            return "unknown"
```

### **Reinforcement Learning Factory**
```python
# extensions/reinforcement-v0.02/agents/__init__.py

from .agent_dqn import DQNAgent
from .agent_double_dqn import DoubleDQNAgent
from .agent_ppo import PPOAgent
from .agent_a3c import A3CAgent

class RLAgentFactory:
    """
    Factory Pattern Implementation for Reinforcement Learning Agents
    
    Design Pattern: Factory Pattern
    Purpose: Create RL agent instances with consistent interface
    Educational Note: Demonstrates factory pattern for complex RL algorithms
    with different training paradigms (value-based, policy-based, actor-critic).
    """
    
    _registry = {
        # Value-Based Methods
        "DQN": DQNAgent,
        "DOUBLE_DQN": DoubleDQNAgent,
        "DUELING_DQN": DuelingDQNAgent,
        
        # Policy-Based Methods
        "PPO": PPOAgent,
        "A2C": A2CAgent,
        
        # Actor-Critic Methods
        "A3C": A3CAgent,
        "SAC": SACAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, state_dim: int, action_dim: int, **kwargs) -> BaseRLAgent:
        """Create RL agent by algorithm name"""
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown RL algorithm: {algorithm}. Available: {available}")
        return agent_class(state_dim=state_dim, action_dim=action_dim, **kwargs)
    
    @classmethod
    def get_algorithm_type(cls, algorithm: str) -> str:
        """Return the type of RL algorithm (value, policy, actor_critic)"""
        value_based = {"DQN", "DOUBLE_DQN", "DUELING_DQN"}
        policy_based = {"PPO", "A2C"}
        actor_critic = {"A3C", "SAC", "TD3"}
        
        algo_upper = algorithm.upper()
        if algo_upper in value_based:
            return "value_based"
        elif algo_upper in policy_based:
            return "policy_based"
        elif algo_upper in actor_critic:
            return "actor_critic"
        else:
            return "unknown"
```

### **Vision-Language Model Factory**
```python
# extensions/vlm-v0.02/models/__init__.py

from .gpt4_vision import GPT4VisionProvider
from .claude_vision import ClaudeVisionProvider
from .llava import LLaVAProvider

class VLMFactory:
    """
    Factory Pattern Implementation for Vision-Language Models
    
    Design Pattern: Factory Pattern
    Purpose: Create VLM provider instances without exposing instantiation logic
    Educational Note: Demonstrates factory pattern for multimodal AI models
    with different inference backends (API-based vs local deployment).
    """
    
    _registry = {
        # API-Based Models
        "GPT4_VISION": GPT4VisionProvider,
        "CLAUDE_VISION": ClaudeVisionProvider,
        "GEMINI_VISION": GeminiVisionProvider,
        
        # Local Models
        "LLAVA": LLaVAProvider,
        "BLIP2": BLIP2Provider,
        "INSTRUCTBLIP": InstructBLIPProvider,
    }
    
    @classmethod
    def create(cls, model_type: str, grid_size: int = 10, **kwargs) -> BaseVLMProvider:
        """Create VLM provider by model type"""
        provider_class = cls._registry.get(model_type.upper())
        if not provider_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unsupported VLM: {model_type}. Available: {available}")
        return provider_class(grid_size=grid_size, **kwargs)
    
    @classmethod
    def is_local_model(cls, model_type: str) -> bool:
        """Check if model runs locally vs API-based"""
        local_models = {"LLAVA", "BLIP2", "INSTRUCTBLIP"}
        return model_type.upper() in local_models
```

## 🔧 **Advanced Factory Patterns**

### **Abstract Factory for Multi-Type Creation**
```python
class ExtensionAbstractFactory:
    """
    Abstract Factory Pattern for creating families of related objects
    
    Design Pattern: Abstract Factory Pattern
    Purpose: Create families of related agents/models without specifying concrete classes
    Educational Note: Extends simple factory to handle multiple related types
    """
    
    def __init__(self, extension_type: str):
        self.extension_type = extension_type
        self._factories = {
            "heuristics": HeuristicAgentFactory,
            "supervised": SupervisedModelFactory,
            "reinforcement": RLAgentFactory,
            "vlm": VLMFactory,
        }
    
    def get_factory(self) -> object:
        """Get appropriate factory for extension type"""
        factory_class = self._factories.get(self.extension_type.lower())
        if not factory_class:
            available = list(self._factories.keys())
            raise ValueError(f"Unknown extension type: {self.extension_type}. Available: {available}")
        return factory_class
    
    def create_agent(self, algorithm: str, **kwargs):
        """Create agent using appropriate factory"""
        factory = self.get_factory()
        return factory.create(algorithm, **kwargs)
```

### **Plugin Registration System**
```python
class PluginFactoryMixin:
    """
    Mixin to add plugin registration capabilities to factories
    
    Design Pattern: Plugin Pattern + Factory Pattern
    Purpose: Allow runtime registration of new implementations
    Educational Note: Shows how to extend factory pattern for plugin architectures
    """
    
    @classmethod
    def register_plugin(cls, plugin_name: str, plugin_class: type, replace_existing: bool = False):
        """
        Register a new plugin with the factory
        
        Args:
            plugin_name: Identifier for the plugin
            plugin_class: Implementation class
            replace_existing: Whether to replace existing plugins with same name
        """
        plugin_upper = plugin_name.upper()
        
        if plugin_upper in cls._registry and not replace_existing:
            raise ValueError(f"Plugin {plugin_name} already registered. Use replace_existing=True to override.")
        
        # Validate plugin implements required interface
        if not hasattr(plugin_class, '__bases__') or not any(
            base.__name__.startswith('Base') for base in plugin_class.__bases__
        ):
            raise ValueError(f"Plugin {plugin_name} must inherit from appropriate base class")
        
        cls._registry[plugin_upper] = plugin_class
        
    @classmethod
    def unregister_plugin(cls, plugin_name: str):
        """Remove a plugin from the factory"""
        plugin_upper = plugin_name.upper()
        if plugin_upper in cls._registry:
            del cls._registry[plugin_upper]
        else:
            raise ValueError(f"Plugin {plugin_name} not found in registry")
    
    @classmethod
    def list_plugins(cls) -> Dict[str, str]:
        """Return dictionary of plugin names and their class names"""
        return {name: cls.__name__ for name, cls in cls._registry.items()}
```

## 📚 **Educational Benefits**

### **Design Pattern Learning Progression**

1. **Simple Factory** (v0.01 extensions): Basic agent creation
2. **Factory Method** (v0.02 extensions): Multiple algorithms with inheritance
3. **Abstract Factory** (v0.03 extensions): Families of related objects
4. **Plugin Factory** (advanced): Runtime registration capabilities

### **OOP Principles Demonstrated**

- **Single Responsibility**: Each factory handles one type of object creation
- **Open/Closed**: New types can be added without modifying existing code
- **Dependency Inversion**: Client code depends on abstractions, not concrete classes
- **Interface Segregation**: Clean, focused creation interfaces

### **Real-World Applications**

- **Game Engine Architecture**: Similar patterns used in Unity, Unreal Engine
- **Machine Learning Frameworks**: PyTorch, TensorFlow use factory patterns
- **Web Frameworks**: Django, Flask use factories for component creation
- **Enterprise Software**: Spring Framework, .NET Core rely heavily on factories

## 🎯 **Implementation Guidelines**

### **Required Elements for All Factories**

1. **Class naming**: `{Type}Factory` pattern
2. **Registry attribute**: `_registry` dictionary mapping names to classes
3. **Creation method**: `create(cls, type_name: str, **kwargs)`
4. **Comprehensive docstrings**: Include design pattern explanation
5. **Error handling**: Clear error messages with available options
6. **Type hints**: Full type annotations for all methods

### **Optional Advanced Features**

1. **Registration methods**: For plugin-style architectures
2. **Validation methods**: Check type availability before creation
3. **Category methods**: Group related types (neural, tree, etc.)
4. **Configuration support**: Load factory settings from config files

### **Testing Requirements**

```python
class TestFactoryPattern:
    """Test suite for factory pattern implementations"""
    
    def test_create_valid_type(self):
        """Test creation of valid registered types"""
        agent = HeuristicAgentFactory.create("BFS", grid_size=10)
        assert isinstance(agent, BFSAgent)
        assert agent.grid_size == 10
    
    def test_create_invalid_type(self):
        """Test error handling for invalid types"""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            HeuristicAgentFactory.create("INVALID_ALGORITHM")
    
    def test_list_available(self):
        """Test listing available types"""
        available = HeuristicAgentFactory.list_algorithms()
        assert "BFS" in available
        assert "ASTAR" in available
    
    def test_plugin_registration(self):
        """Test dynamic plugin registration"""
        class CustomAgent(BaseAgent):
            pass
        
        HeuristicAgentFactory.register("CUSTOM", CustomAgent)
        agent = HeuristicAgentFactory.create("CUSTOM", grid_size=10)
        assert isinstance(agent, CustomAgent)
```

## 🔗 **Integration with Project Architecture**

### **Base Class Integration**
All factories work with the established base class hierarchy:
- `BaseAgent` for heuristic and RL agents
- `BaseMLAgent` for supervised learning models
- `BaseVLMProvider` for vision-language models

### **Configuration Integration**
```python
# extensions/common/config/factory_config.py
DEFAULT_FACTORY_SETTINGS = {
    "validation_enabled": True,
    "plugin_registration_allowed": True,
    "error_on_duplicate_registration": True,
    "case_sensitive_names": False,
}
```

### **Path Management Integration**
```python
from extensions.common.path_utils import ensure_project_root

# Factory modules use standardized path management
ensure_project_root()
```

---

**The Factory Pattern serves as a cornerstone of the Snake Game AI architecture, enabling flexible, maintainable, and educational code that demonstrates fundamental software engineering principles while supporting the project's plugin-style extension system.** 