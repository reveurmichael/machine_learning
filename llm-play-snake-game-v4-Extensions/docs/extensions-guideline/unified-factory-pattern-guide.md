# Unified Factory Pattern Guide

> **Authoritative Reference**: This document establishes the definitive factory pattern implementation across all Snake Game AI extensions.

## 🎯 **Core Philosophy: Consistent Creation Patterns**

Factory patterns enable **flexible object creation** without tight coupling to concrete classes, supporting the **plugin-style architecture** that makes extensions easily extensible.

## 🚫 **EXPLICIT DECISION: NO BaseFactory or factory_utils.py**

**CRITICAL ARCHITECTURAL DECISION**: This project **explicitly rejects**:
- ❌ **BaseFactory abstract class** in `extensions/common/utils/`
- ❌ **factory_utils.py module** in `extensions/common/utils/`
- ❌ **Any shared factory inheritance hierarchy**

**Rationale**: Simple dictionary-based factories work perfectly and follow SUPREME_RULE NO.3. Each extension creates its own simple factory following the template below.

## 🏗️ **Standard Factory Template**

### **Universal Factory Implementation**
All extensions MUST use this standardized pattern:

```python
class {Type}Factory:
    """
    Factory Pattern Implementation for {Type} Creation
    
    Design Pattern: Factory Pattern
    Purpose: Create {type} instances without exposing instantiation logic
    Educational Note: Demonstrates how factory patterns enable plugin architectures
    and support the Open/Closed Principle.
    """
    
    _registry = {
        # Map string identifiers to implementation classes
        "TYPE1": Type1Implementation,
        "TYPE2": Type2Implementation,
        "TYPE3": Type3Implementation,
    }
    
    @classmethod
    def create(cls, type_name: str, **kwargs) -> BaseType:
        """Create instance by type name"""
        type_class = cls._registry.get(type_name.upper())
        if not type_class:
            available_types = list(cls._registry.keys())
            raise ValueError(
                f"Unknown {cls.__name__.replace('Factory', '').lower()}: {type_name}. "
                f"Available types: {available_types}"
            )
        return type_class(**kwargs)
    
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

### **Heuristics Agent Factory**
```python
# extensions/heuristics-v0.02/agents/__init__.py
from .agent_bfs import BFSAgent
from .agent_astar import AStarAgent
from .agent_dfs import DFSAgent
from .agent_hamiltonian import HamiltonianAgent

class HeuristicAgentFactory:
    """Factory for creating heuristic pathfinding agents"""
    
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
```

### **Supervised Model Factory**
```python
# extensions/supervised-v0.02/models/__init__.py
from .neural_networks.agent_mlp import MLPAgent
from .neural_networks.agent_cnn import CNNAgent
from .tree_models.agent_xgboost import XGBoostAgent
from .tree_models.agent_lightgbm import LightGBMAgent

class SupervisedModelFactory:
    """Factory for creating supervised learning models"""
    
    _registry = {
        # Neural Network Models
        "MLP": MLPAgent,
        "CNN": CNNAgent,
        "LSTM": LSTMAgent,
        
        # Tree-Based Models  
        "XGBOOST": XGBoostAgent,
        "LIGHTGBM": LightGBMAgent,
        "RANDOMFOREST": RandomForestAgent,
    }
    
    @classmethod
    def create(cls, model_type: str, input_dim: int, **kwargs) -> BaseMLAgent:
        """Create supervised learning model by type"""
        model_class = cls._registry.get(model_type.upper())
        if not model_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model: {model_type}. Available: {available}")
        return model_class(input_dim=input_dim, **kwargs)
```

### **Reinforcement Learning Factory**
```python
# extensions/reinforcement-v0.02/agents/__init__.py
from .agent_dqn import DQNAgent
from .agent_ppo import PPOAgent
from .agent_a3c import A3CAgent

class RLAgentFactory:
    """Factory for creating reinforcement learning agents"""
    
    _registry = {
        "DQN": DQNAgent,
        "PPO": PPOAgent,
        "A3C": A3CAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, state_dim: int, action_dim: int, **kwargs) -> BaseRLAgent:
        """Create RL agent by algorithm name"""
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown RL algorithm: {algorithm}. Available: {available}")
        return agent_class(state_dim=state_dim, action_dim=action_dim, **kwargs)
```

## 🔧 **Advanced Factory Patterns**

### **Abstract Factory for Multi-Type Creation**
```python
class ExtensionAbstractFactory:
    """Abstract factory for creating families of related objects"""
    
    def __init__(self, extension_type: str):
        self.extension_type = extension_type
        self._factories = {
            "heuristics": HeuristicAgentFactory,
            "supervised": SupervisedModelFactory,
            "reinforcement": RLAgentFactory,
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
    """Mixin to add plugin registration capabilities to factories"""
    
    @classmethod
    def register_plugin(cls, plugin_name: str, plugin_class: type, replace_existing: bool = False):
        """Register a new plugin with the factory"""
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
```

## 📚 **Educational Benefits**

### **Design Pattern Learning Progression**
1. **Simple Factory** (v0.01): Basic agent creation
2. **Factory Method** (v0.02): Multiple algorithms with inheritance
3. **Abstract Factory** (v0.03): Families of related objects
4. **Plugin Factory** (advanced): Runtime registration capabilities

### **OOP Principles Demonstrated**
- **Single Responsibility**: Each factory handles one type of object creation
- **Open/Closed**: New types can be added without modifying existing code
- **Dependency Inversion**: Client code depends on abstractions, not concrete classes
- **Interface Segregation**: Clean, focused creation interfaces

## 🎯 **Implementation Guidelines**

### **Required Elements for All Factories**
1. **Class naming**: `{Type}Factory` pattern
2. **Registry attribute**: `_registry` dictionary mapping names to classes
3. **Creation method**: `create(cls, type_name: str, **kwargs)`
4. **Comprehensive docstrings**: Include design pattern explanation
5. **Error handling**: Clear error messages with available options
6. **Type hints**: Full type annotations for all methods

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
        available = HeuristicAgentFactory.list_available()
        assert "BFS" in available
        assert "ASTAR" in available
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

---

**The Factory Pattern serves as a cornerstone of the Snake Game AI architecture, enabling flexible, maintainable, and educational code that demonstrates fundamental software engineering principles.** 