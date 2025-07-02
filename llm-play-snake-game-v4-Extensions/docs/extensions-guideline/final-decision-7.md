# Final Decision 7: Factory Pattern Architecture

> **SUPREME AUTHORITY**: This document establishes the definitive factory pattern architecture standards for all Snake Game AI extensions.

> **See also:** `factory-design-pattern.md` (Factory patterns), `agents.md` (Agent implementation), `core.md` (Base architecture), `final-decision-10.md` (SUPREME_RULES).

## 🎯 **Core Philosophy: Factory Pattern Enables Extensible Architecture**

The Factory Pattern is the cornerstone of extensible, maintainable, and educational software architecture in the Snake Game AI project. This document establishes authoritative factory pattern standards that ensure consistency across all extensions, strictly following `final-decision-10.md` SUPREME_RULES.

### **SUPREME_RULES Integration**
- **SUPREME_RULE NO.1**: Enforces reading all GOOD_RULES before making factory pattern changes to ensure comprehensive understanding
- **SUPREME_RULE NO.2**: Uses precise `final-decision-N.md` format consistently when referencing architectural decisions and factory patterns
- **SUPREME_RULE NO.3**: Enables lightweight common utilities with OOP extensibility while maintaining factory patterns through inheritance rather than tight coupling
- **SUPREME_RULE NO.4**: Ensures all markdown files are coherent and aligned through nuclear diffusion infusion process

### **GOOD_RULES Integration**
This document integrates with the **GOOD_RULES** governance system established in `final-decision-10.md`:
- **`factory-design-pattern.md`**: Authoritative reference for factory pattern implementation
- **`agents.md`**: Authoritative reference for agent implementation standards
- **`core.md`**: Authoritative reference for base architecture
- **`single-source-of-truth.md`**: Ensures factory consistency across all extensions

### **Simple Logging Examples (SUPREME_RULE NO.3)**
All code examples in this document follow **SUPREME_RULE NO.3** by using simple print() statements rather than complex logging mechanisms:

```python
# ✅ CORRECT: Simple logging as per SUPREME_RULE NO.3
class AgentFactory:
    @classmethod
    def create(cls, agent_type: str, **kwargs):
        """Create agent using canonical create() method"""
        agent_class = cls._registry.get(agent_type.upper())
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")
        print(f"[AgentFactory] Creating agent: {agent_type}")  # SUPREME_RULE NO.3
        return agent_class(**kwargs)

def register_agent(agent_type: str, agent_class):
    """Register agent with factory"""
    print(f"[Factory] Registering agent: {agent_type}")  # SUPREME_RULE NO.3
    _registry[agent_type.upper()] = agent_class
```

## 🏗️ **Factory Pattern: Canonical Method is create()**

All extension factories must use the canonical method name `create()` for instantiation, not `create_agent()` or any other variant. This ensures consistency and aligns with SUPREME_RULES and the KISS principle. Factories should be simple, dictionary-based, and avoid over-engineering.

### **Reference Implementation**

A generic, educational `SimpleFactory` is provided in `extensions/common/utils/factory_utils.py`:

```python
from utils.factory_utils import SimpleFactory

class MyAgent:
    def __init__(self, name):
        self.name = name

factory = SimpleFactory()
factory.register("myagent", MyAgent)
agent = factory.create("myagent", name="TestAgent")
print(agent.name)  # Output: TestAgent
```

### **Example Extension Factory**

```python
class HeuristicAgentFactory:
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
    }
    @classmethod
    def create(cls, algorithm: str, **kwargs):  # CANONICAL create() method
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        print(f"[HeuristicAgentFactory] Creating agent: {algorithm}")  # Simple logging
        return agent_class(**kwargs)
```

## 🏗️ **Factory Pattern Architecture**

### **Core Factory Interface**
```python
class BaseAgentFactory:
    """
    Base factory interface for all agent factories.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Define common factory interface for all agent types
    Educational Value: Shows how canonical factory patterns work with agents
    
    Reference: final-decision-10.md for canonical method naming
    """
    
    @classmethod
    def create(cls, agent_type: str, grid_size: int) -> BaseAgent:  # CANONICAL create() method
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
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create heuristic agent instances with consistent interface
    Educational Value: Shows how canonical factory patterns work with heuristic algorithms
    
    Reference: final-decision-10.md for canonical method naming
    """
    
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
        "HAMILTONIAN": HamiltonianAgent,
    }
    
    @classmethod
    def create(cls, agent_type: str, grid_size: int) -> BaseAgent:  # CANONICAL create() method
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
        
        print(f"[HeuristicAgentFactory] Creating agent: {agent_type_upper}")  # Simple logging
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
        print(f"[HeuristicAgentFactory] Registered algorithm: {name}")  # Simple logging
```

## 🎯 **Extension-Specific Factory Implementations**

### **Supervised Learning Factory**
```python
class SupervisedAgentFactory(BaseAgentFactory):
    """
    Factory for supervised learning agents.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create supervised learning agent instances
    Educational Value: Shows how canonical factory patterns work with ML models
    
    Reference: final-decision-10.md for canonical method naming
    """
    
    _registry = {
        "MLP": MLPAgent,
        "CNN": CNNAgent,
        "LSTM": LSTMAgent,
        "XGBOOST": XGBoostAgent,
        "LIGHTGBM": LightGBMAgent,
    }
    
    @classmethod
    def create(cls, agent_type: str, grid_size: int, **kwargs) -> BaseAgent:  # CANONICAL create() method
        """Create supervised learning agent"""
        agent_type_upper = agent_type.upper()
        agent_class = cls._registry.get(agent_type_upper)
        
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model: {agent_type}. Available: {available}")
        
        print(f"[SupervisedAgentFactory] Creating agent: {agent_type_upper}")  # Simple logging
        return agent_class(grid_size=grid_size, **kwargs)
```

### **Reinforcement Learning Factory**
```python
class RLAgentFactory(BaseAgentFactory):
    """
    Factory for reinforcement learning agents.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create reinforcement learning agent instances
    Educational Value: Shows how canonical factory patterns work with RL algorithms
    
    Reference: final-decision-10.md for canonical method naming
    """
    
    _registry = {
        "DQN": DQNAgent,
        "PPO": PPOAgent,
        "A3C": A3CAgent,
        "DDPG": DDPGAgent,
    }
    
    @classmethod
    def create(cls, agent_type: str, grid_size: int, **kwargs) -> BaseAgent:  # CANONICAL create() method
        """Create reinforcement learning agent"""
        agent_type_upper = agent_type.upper()
        agent_class = cls._registry.get(agent_type_upper)
        
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown RL algorithm: {agent_type}. Available: {available}")
        
        print(f"[RLAgentFactory] Creating agent: {agent_type_upper}")  # Simple logging
        return agent_class(grid_size=grid_size, **kwargs)
```

### **LLM Factory**
```python
class LLMAgentFactory(BaseAgentFactory):
    """
    Factory for LLM-based agents.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create LLM agent instances
    Educational Value: Shows how canonical factory patterns work with LLM models
    
    Reference: final-decision-10.md for canonical method naming
    """
    
    _registry = {
        "LORA": LoRAAgent,
        "DISTILLED": DistilledAgent,
        "FINETUNED": FinetunedAgent,
    }
    
    @classmethod
    def create(cls, agent_type: str, grid_size: int, **kwargs) -> BaseAgent:  # CANONICAL create() method
        """Create LLM agent"""
        agent_type_upper = agent_type.upper()
        agent_class = cls._registry.get(agent_type_upper)
        
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown LLM type: {agent_type}. Available: {available}")
        
        print(f"[LLMAgentFactory] Creating agent: {agent_type_upper}")  # Simple logging
        return agent_class(grid_size=grid_size, **kwargs)
```

## 🏭 **Universal Factory Pattern**

### **Simple Factory Implementation**
```python
# extensions/common/utils/factory_utils.py
class SimpleFactory:
    """
    Simple, generic factory implementation.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Provide simple, reusable factory functionality
    Educational Value: Shows how canonical factory patterns work in practice
    
    Reference: final-decision-10.md for canonical method naming
    """
    
    def __init__(self):
        self._registry = {}
    
    def register(self, name: str, factory_func):
        """Register a factory function"""
        self._registry[name.upper()] = factory_func
        print(f"[SimpleFactory] Registered: {name}")  # Simple logging
    
    def create(self, name: str, **kwargs):  # CANONICAL create() method
        """Create object using registered factory function"""
        factory_func = self._registry.get(name.upper())
        if not factory_func:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown type: {name}. Available: {available}")
        
        print(f"[SimpleFactory] Creating: {name}")  # Simple logging
        return factory_func(**kwargs)
    
    def list_available(self):
        """List all available types"""
        return list(self._registry.keys())
```

### **Usage Examples**
```python
# Using SimpleFactory for agents
agent_factory = SimpleFactory()
agent_factory.register("BFS", lambda **kwargs: BFSAgent(**kwargs))
agent_factory.register("MLP", lambda **kwargs: MLPAgent(**kwargs))

bfs_agent = agent_factory.create("BFS", grid_size=10)
mlp_agent = agent_factory.create("MLP", grid_size=10, hidden_size=128)

# Using SimpleFactory for validators
validator_factory = SimpleFactory()
validator_factory.register("DATASET", lambda **kwargs: DatasetValidator(**kwargs))
validator_factory.register("PATH", lambda **kwargs: PathValidator(**kwargs))

dataset_validator = validator_factory.create("DATASET", schema="csv")
path_validator = validator_factory.create("PATH", root_path="/tmp")
```

## 🎓 **Educational Value and Learning Path**

### **Learning Objectives**
- **Factory Patterns**: Understanding canonical factory pattern implementation
- **Method Naming**: Learning the importance of consistent method naming
- **Registry Pattern**: Understanding how to manage object creation registries
- **Code Reusability**: Learning to create reusable factory components

### **Implementation Examples**
- **Agent Creation**: How to create agents using factory patterns
- **Factory Registration**: How to register new types with factories
- **Error Handling**: How to handle unknown types gracefully
- **Factory Composition**: How to compose multiple factories

## 🔗 **Integration with Other Documentation**

### **GOOD_RULES Alignment**
This document aligns with:
- **`factory-design-pattern.md`**: Detailed factory pattern standards
- **`agents.md`**: Agent implementation standards
- **`core.md`**: Base architecture principles
- **`single-source-of-truth.md`**: Architectural principles

### **Extension Guidelines**
This factory pattern architecture supports:
- All extension types (heuristics, supervised, reinforcement, LLM)
- All agent types (pathfinding, ML, RL, LLM)
- Consistent object creation patterns
- Reusable factory components

---

**This factory pattern architecture ensures consistent, extensible, and educational object creation across all Snake Game AI extensions while maintaining SUPREME_RULES compliance.**

> **SUPREME_RULES COMPLIANCE**: This document strictly follows the SUPREME_RULES established in `final-decision-10.md`, ensuring coherence, educational value, and architectural integrity across the entire project.
