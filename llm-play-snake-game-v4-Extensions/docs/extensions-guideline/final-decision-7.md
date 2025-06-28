# Final Decision 7: Factory Pattern Architecture

> **Guidelines Alignment:**
> - This document is governed by the SUPREME_RULES in `final-decision-10.md`.
> - All extension factories must use the canonical method name `create()` (never `create_agent`, `create_model`, etc.).
> - All code must use simple print logging (SUPREME_RULE NO.3).
> - Reference: `extensions/common/utils/factory_utils.py` for the canonical `SimpleFactory` implementation.
> - This file is a GOOD_RULES authoritative reference and must be cross-referenced by all related documentation.

> **See also:** `agents.md`, `core.md`, `final-decision-10.md`, `factory-design-pattern.md`, `extension-evolution-rules.md`.

## 🎯 **Core Philosophy: Factory Pattern Enables Extensible Architecture**

The Factory Pattern is the cornerstone of extensible, maintainable, and educational software architecture in the Snake Game AI project. This document establishes authoritative factory pattern standards that ensure consistency across all extensions.

### **Guidelines Alignment**
- **SUPREME_RULE NO.1**: Enforces reading all GOOD_RULES before making factory pattern changes to ensure comprehensive understanding
- **SUPREME_RULE NO.2**: Uses precise `final-decision-N.md` format consistently when referencing architectural decisions and factory patterns
- **SUPREME_RULE NO.3**: Enables lightweight common utilities with OOP extensibility while maintaining factory patterns through inheritance rather than tight coupling

## 🏗️ **Factory Pattern: Canonical Method is create()**

All extension factories must use the canonical method name `create()` for instantiation, not `create_agent()` or any other variant. This ensures consistency and aligns with SUPREME_RULES and the KISS principle. Factories should be simple, dictionary-based, and avoid over-engineering.

### Reference Implementation

A generic, educational `SimpleFactory` is provided in `extensions/common/utils/factory_utils.py`:

```python
from extensions.common.utils.factory_utils import SimpleFactory

class MyAgent:
    def __init__(self, name):
        self.name = name

factory = SimpleFactory()
factory.register("myagent", MyAgent)
agent = factory.create("myagent", name="TestAgent")
print(agent.name)  # Output: TestAgent
```

### Example Extension Factory

```python
class HeuristicAgentFactory:
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
    }
    @classmethod
    def create(cls, algorithm: str, **kwargs):
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        print(f"[HeuristicAgentFactory] Creating agent: {algorithm}")  # SUPREME_RULE NO.3
        return agent_class(**kwargs)
```

## 🏗️ **Factory Pattern Architecture**

### **Core Factory Interface**
```python
class BaseAgentFactory:
    """
    Base factory interface for all agent factories.
    
    Design Pattern: Factory Pattern
    - Defines common factory interface
    - Ensures consistent factory behavior
    - Enables factory composition and extension
    """
    
    @classmethod
    def create(cls, agent_type: str, grid_size: int) -> BaseAgent:
        """
        Create agent instance by type.
        
        Args:
            agent_type: Type of agent to create
            grid_size: Grid size for the agent
            
        Returns:
            Configured agent instance
            
        Raises:
            ValueError: If agent type is not recognized
        """
        raise NotImplementedError("Subclasses must implement create()")
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available agent types"""
        raise NotImplementedError("Subclasses must implement list_available()")
```

### **Concrete Factory Implementation**
```python
class HeuristicAgentFactory(BaseAgentFactory):
    """
    Factory for heuristic-based agents.
    
    Design Pattern: Concrete Factory
    - Implements specific agent creation logic
    - Manages heuristic algorithm registry
    - Provides consistent initialization
    """
    
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
        "HAMILTONIAN": HamiltonianAgent,
    }
    
    @classmethod
    def create(cls, agent_type: str, grid_size: int) -> BaseAgent:
        """
        Create heuristic agent instance.
        
        Args:
            agent_type: Algorithm type (case-insensitive)
            grid_size: Grid size for the agent
            
        Returns:
            Configured heuristic agent
            
        Raises:
            ValueError: If algorithm type is not recognized
        """
        agent_type_upper = agent_type.upper()
        agent_class = cls._registry.get(agent_type_upper)
        
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown algorithm: {agent_type}. Available: {available}")
        
        print(f"[HeuristicAgentFactory] Creating agent: {agent_type_upper}")  # SUPREME_RULE NO.3
        return agent_class(grid_size=grid_size)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available heuristic algorithms"""
        return list(cls._registry.keys())
    
    @classmethod
    def register(cls, name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register a new agent class.
        
        Args:
            name: Algorithm name
            agent_class: Agent class to register
        """
        cls._registry[name.upper()] = agent_class
        print(f"[HeuristicAgentFactory] Registered algorithm: {name}")  # SUPREME_RULE NO.3

## 🎯 **Extension-Specific Factory Implementations**

### **Supervised Learning Factory**
```python
class SupervisedAgentFactory(BaseAgentFactory):
    """Factory for supervised learning agents"""
    
    _registry = {
        "MLP": MLPAgent,
        "CNN": CNNAgent,
        "LSTM": LSTMAgent,
        "XGBOOST": XGBoostAgent,
        "LIGHTGBM": LightGBMAgent,
    }
    
    @classmethod
    def create(cls, agent_type: str, grid_size: int, **kwargs) -> BaseAgent:
        """Create supervised learning agent"""
        agent_type_upper = agent_type.upper()
        agent_class = cls._registry.get(agent_type_upper)
        
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model: {agent_type}. Available: {available}")
        
        print(f"[SupervisedAgentFactory] Creating agent: {agent_type_upper}")  # SUPREME_RULE NO.3
        return agent_class(grid_size=grid_size, **kwargs)

### **Reinforcement Learning Factory**
```python
class RLAgentFactory(BaseAgentFactory):
    """Factory for reinforcement learning agents"""
    
    _registry = {
        "DQN": DQNAgent,
        "PPO": PPOAgent,
        "A3C": A3CAgent,
        "DDPG": DDPGAgent,
    }
    
    @classmethod
    def create(cls, agent_type: str, grid_size: int, **kwargs) -> BaseAgent:
        """Create reinforcement learning agent"""
        agent_type_upper = agent_type.upper()
        agent_class = cls._registry.get(agent_type_upper)
        
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown RL algorithm: {agent_type}. Available: {available}")
        
        print(f"[RLAgentFactory] Creating agent: {agent_type_upper}")  # SUPREME_RULE NO.3
        return agent_class(grid_size=grid_size, **kwargs)

## 📋 **Factory Pattern Best Practices**

### **Canonical Method Requirements**
- ✅ **Always use `create()` method** - never `create_agent()`, `build()`, `make()`, etc.
- ✅ **Simple registry pattern** - dictionary-based lookup for clarity
- ✅ **Clear error messages** - list available options when type not found
- ✅ **Simple logging** - use print statements following SUPREME_RULE NO.3
- ✅ **Type validation** - validate input parameters before creation

### **Extension Integration**
All extension factories must follow the canonical pattern and integrate with the base architecture established in `core.md` and other GOOD_RULES documents.
```
