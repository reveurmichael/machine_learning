# Factory Design Pattern Implementation

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines factory design pattern implementation with strict SUPREME_RULES compliance.

> **See also:** `final-decision-10.md`, `agents.md`, `core.md`, `standalone.md`.

## 🎯 **Core Philosophy: Canonical Factory Patterns + SUPREME_RULES**

The Factory Design Pattern in the Snake Game AI project follows **SUPREME_RULES** established in `final-decision-10.md`, ensuring:
- **Canonical `create()` method** for all factories (never `create_agent()`, `create_model()`, etc.)
- **Simple logging** (print statements only, no complex logging frameworks)
- **Lightweight, OOP-based, extensible, non-over-engineered** design
- **Educational value** through clear examples and explanations

### **Educational Value**
- **Consistent Patterns**: All factories use identical `create()` method
- **Simple Logging**: All components use print() statements only
- **OOP Principles**: Demonstrates inheritance, polymorphism, and encapsulation
- **Extensibility**: Easy to add new types without modifying existing code

## 🏗️ **SUPREME_RULES: Canonical Method is create()**

**CRITICAL REQUIREMENT**: All factory classes MUST use the canonical method name `create()` for instantiation, not `create_agent()`, `create_model()`, `make_agent()`, or any other variant. This ensures consistency and aligns with the KISS principle.

### **Reference Implementation**

A generic, educational `SimpleFactory` is provided in `extensions/common/utils/factory_utils.py`:

```python
class SimpleFactory:
    """
    Generic factory implementation following final-decision-10.md SUPREME_RULES
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for all factory types
    Educational Value: Shows how SUPREME_RULES enable consistent patterns
    across simple heuristics and complex AI systems.
    
    Reference: final-decision-10.md SUPREME_RULES for canonical method naming
    """
    
    def __init__(self):
        self._registry = {}
        print(f"[SimpleFactory] Initialized")  # Simple logging - SUPREME_RULES
    
    def register(self, name: str, cls):
        """Register a class with the factory"""
        self._registry[name] = cls
        print(f"[SimpleFactory] Registered: {name}")  # Simple logging
    
    def create(self, name: str, **kwargs):  # CANONICAL create() method - SUPREME_RULES
        """Create instance using canonical create() method following final-decision-10.md"""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown type: {name}. Available: {available}")
        print(f"[SimpleFactory] Creating: {name}")  # Simple logging - SUPREME_RULES
        return self._registry[name](**kwargs)

# ❌ FORBIDDEN: Non-canonical method names (violates SUPREME_RULES)
class SimpleFactory:
    def create_agent(self, name: str):  # FORBIDDEN - not canonical
        pass
    
    def build_model(self, name: str):  # FORBIDDEN - not canonical
        pass
    
    def make_instance(self, name: str):  # FORBIDDEN - not canonical
        pass
```

### **Usage Example**
```python
from extensions.common.utils.factory_utils import SimpleFactory

class MyAgent:
    def __init__(self, name):
        self.name = name

# Create factory and register types
factory = SimpleFactory()
factory.register("myagent", MyAgent)

# Use canonical create() method
agent = factory.create("myagent", name="TestAgent")  # CANONICAL create() method
print(f"Created agent: {agent.name}")  # Simple logging
```

## 🔧 **Factory Pattern Implementation Examples**

### **Agent Factory (Canonical Implementation)**
```python
class AgentFactory:
    """
    Factory for creating different types of agents following SUPREME_RULES.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for agent creation
    Educational Value: Shows how canonical patterns work with
    different agent types while maintaining simple logging.
    
    Reference: final-decision-10.md for canonical agent architecture
    """
    
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
        "HAMILTONIAN": HamiltonianAgent,
    }
    
    @classmethod
    def create(cls, agent_type: str, **kwargs):  # CANONICAL create() method - SUPREME_RULES
        """Create agent using canonical create() method following final-decision-10.md"""
        agent_class = cls._registry.get(agent_type.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {available}")
        print(f"[AgentFactory] Creating agent: {agent_type}")  # Simple logging - SUPREME_RULES
        return agent_class(**kwargs)

# ❌ FORBIDDEN: Non-canonical method names (violates SUPREME_RULES)
class AgentFactory:
    def create_agent(self, agent_type: str):  # FORBIDDEN - not canonical
        pass
    
    def build_agent(self, agent_type: str):  # FORBIDDEN - not canonical
        pass
    
    def make_agent(self, agent_type: str):  # FORBIDDEN - not canonical
        pass
```

### **Model Factory (Canonical Implementation)**
```python
class ModelFactory:
    """
    Factory for creating different types of models following SUPREME_RULES.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for model creation
    Educational Value: Shows how canonical patterns work with
    different model types while maintaining simple logging.
    
    Reference: final-decision-10.md for canonical model architecture
    """
    
    _registry = {
        "MLP": MLPModel,
        "CNN": CNNModel,
        "LSTM": LSTMModel,
        "XGBOOST": XGBoostModel,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs):  # CANONICAL create() method - SUPREME_RULES
        """Create model using canonical create() method following final-decision-10.md"""
        model_class = cls._registry.get(model_type.upper())
        if not model_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
        print(f"[ModelFactory] Creating model: {model_type}")  # Simple logging - SUPREME_RULES
        return model_class(**kwargs)
```

### **Configuration Factory (Canonical Implementation)**
```python
class ConfigFactory:
    """
    Factory for creating different types of configurations following SUPREME_RULES.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for configuration creation
    Educational Value: Shows how canonical patterns work with
    different configuration types while maintaining simple logging.
    
    Reference: final-decision-10.md for canonical configuration architecture
    """
    
    _registry = {
        "HEURISTIC": HeuristicConfig,
        "SUPERVISED": SupervisedConfig,
        "REINFORCEMENT": ReinforcementConfig,
    }
    
    @classmethod
    def create(cls, config_type: str, **kwargs):  # CANONICAL create() method - SUPREME_RULES
        """Create configuration using canonical create() method following final-decision-10.md"""
        config_class = cls._registry.get(config_type.upper())
        if not config_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown config type: {config_type}. Available: {available}")
        print(f"[ConfigFactory] Creating config: {config_type}")  # Simple logging - SUPREME_RULES
        return config_class(**kwargs)
```

## 📊 **Simple Logging Standards for Factory Operations**

### **Required Logging Pattern (SUPREME_RULES)**
All factory operations MUST use simple print statements as established in `final-decision-10.md`:

```python
# ✅ CORRECT: Simple logging for factory operations (SUPREME_RULES compliance)
def create_instance(factory_type: str, instance_type: str):
    print(f"[Factory] Creating {instance_type} using {factory_type}")  # Simple logging - REQUIRED
    
    # Factory creation logic
    instance = factory.create(instance_type)
    
    print(f"[Factory] Successfully created {instance_type}")  # Simple logging
    return instance

# ❌ FORBIDDEN: Complex logging frameworks (violates SUPREME_RULES)
# import logging
# logger = logging.getLogger(__name__)

# def create_instance(factory_type: str, instance_type: str):
#     logger.info(f"Creating {instance_type}")  # FORBIDDEN - complex logging
#     # This violates final-decision-10.md SUPREME_RULES
```

## 🎓 **Educational Applications with Canonical Patterns**

### **Factory Pattern Benefits**
- **Loose Coupling**: Client code doesn't need to know concrete classes
- **Easy Extension**: Add new types without modifying existing code
- **Centralized Creation**: All object creation logic in one place
- **Consistent Interface**: Same `create()` method across all factories

### **Pattern Consistency**
- **Canonical Method**: All factories use `create()` method consistently
- **Simple Logging**: Print statements provide clear operation visibility
- **Educational Value**: Canonical patterns enable predictable learning
- **SUPREME_RULES**: Advanced systems follow same standards as simple ones

## 📋 **SUPREME_RULES Implementation Checklist for Factory Patterns**

### **Mandatory Requirements**
- [ ] **Canonical Method**: All factories use `create()` method exactly (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses print() statements only for all factory operations (final-decision-10.md compliance)
- [ ] **GOOD_RULES Reference**: References `final-decision-10.md` in all factory documentation
- [ ] **Pattern Consistency**: Follows canonical patterns across all factory implementations

### **Factory-Specific Standards**
- [ ] **Registration**: Canonical factory patterns for all type registration
- [ ] **Creation**: Canonical factory patterns for all instance creation
- [ ] **Error Handling**: Canonical patterns for all error conditions
- [ ] **Validation**: Simple logging for all validation operations

### **Educational Integration**
- [ ] **Clear Examples**: Simple examples using canonical `create()` method
- [ ] **Pattern Explanation**: Clear explanation of canonical patterns in factory context
- [ ] **Best Practices**: Demonstration of SUPREME_RULES in factory systems
- [ ] **Learning Value**: Easy to understand canonical patterns regardless of factory complexity

---

**Factory Design Pattern implementation ensures consistent, educational, and maintainable object creation across all Snake Game AI extensions while maintaining strict compliance with `final-decision-10.md` SUPREME_RULES.**

## 🔗 **See Also**

- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`agents.md`**: Authoritative reference for agent implementation with canonical patterns
- **`core.md`**: Base class architecture following canonical principles
- **`standalone.md`**: Standalone principle and extension independence