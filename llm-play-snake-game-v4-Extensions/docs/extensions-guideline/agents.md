> **Important — Authoritative Reference:** This guide is **supplementary** to the _Final Decision Series_ (`final-decision-0` → `final-decision-10`). **If any statement here conflicts with a Final Decision document, the latter always prevails.**

# Agent Implementation Standards

This document provides guidelines for implementing AI agents across Snake Game AI extensions, following the standardized conventions established in Final Decision 4.

## 🎯 **Agent Naming Philosophy**

The agent architecture demonstrates natural software evolution from proof-of-concept to sophisticated multi-algorithm systems, with consistent naming throughout.

### **Version Progression:**
- **v0.01**: Single agent in extension root (proof-of-concept)
- **v0.02**: Organized `agents/` package with multiple algorithms
- **v0.03**: Enhanced with dashboard integration
- **v0.04**: Advanced features with JSONL generation (heuristics only)

### **Directory Structure Rules:**

#### **v0.01 Structure (Proof-of-Concept)**
- Agents go directly in: `./extensions/[Algorithm]-v0.01/`
- Single-file implementations
- Basic game integration

#### **v0.02+ Structure (Multi-Algorithm)**
- Agents go in: `./extensions/[Algorithm]-v0.0N/agents/`
- Where N ≥ 2 (v0.02, v0.03, and for heuristics: v0.04)
- Organized package structure with multiple agents

### **Standardized Naming Convention**

Following Final Decision 4, all agents use consistent naming:

#### **File Naming Pattern**
```python
# ✅ REQUIRED PATTERN: agent_{algorithm}.py
agent_bfs.py              # Breadth-First Search
agent_astar.py            # A* pathfinding
agent_hamiltonian.py      # Hamiltonian path algorithm
agent_mlp.py              # Multi-Layer Perceptron
agent_dqn.py              # Deep Q-Network
```

#### **Class Naming Pattern**
```python
# ✅ REQUIRED PATTERN: {Algorithm}Agent
class BFSAgent(BaseAgent):              # from agent_bfs.py
class AStarAgent(BaseAgent):            # from agent_astar.py
class HamiltonianAgent(BaseAgent):      # from agent_hamiltonian.py
class MLPAgent(BaseAgent):              # from agent_mlp.py
class DQNAgent(BaseAgent):              # from agent_dqn.py
```

Note: For Task-0, the agent is named `SnakeAgent` (not `LLMSnakeAgent`) per Final Decision 10.

## 🏗️ **Directory Structure Evolution**

### **Version Progression (Final Decision 5):**
- **v0.01**: Single agent in extension root (proof-of-concept)
- **v0.02**: Organized `agents/` package with multiple algorithms
- **v0.03**: Enhanced with dashboard integration (agents/ copied exactly from v0.02)

### **Directory Structure Rules:**

#### **v0.01 Structure**
```
extensions/{algorithm}-v0.01/
├── agent_{primary}.py             # Single algorithm implementation
├── game_logic.py                  # Algorithm-specific logic
└── game_manager.py                # Algorithm-specific manager
```

#### **v0.02+ Structure**
```
extensions/{algorithm}-v0.02/
├── agents/
│   ├── __init__.py               # Agent factory
│   ├── agent_{algo1}.py
│   └── agent_{algo2}.py
├── game_logic.py
└── game_manager.py
```

## 🧠 **Agent Implementation Standards**

### **Common Agent Protocol**
All agents must extend BaseAgent from the core framework:

```python
from core.game_agents import BaseAgent

class HeuristicAgent(BaseAgent):
    """
    Base class for all heuristic agents
    
    Design Pattern: Template Method Pattern
    - Defines common structure for all heuristic agents
    - Subclasses implement specific algorithms
    - Ensures consistent interface across all heuristics
    """
    
    def __init__(self, name: str, grid_size: int):
        super().__init__(name, grid_size)
        self.algorithm_name = name
    
    @abstractmethod
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan next move based on current game state"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for new game"""
        pass
```

### **Design Philosophy**

#### **Template Method Pattern**
- Base classes define common structure
- Subclasses implement algorithm-specific logic
- Consistent error handling and logging across all agents

#### **Strategy Pattern**
- Each agent implements a specific algorithm strategy
- Pluggable into game managers via factory patterns
- Isolated algorithm logic for maintainability

#### **Factory Pattern**
- Centralized agent creation via factory classes
- Dynamic agent selection by name
- Consistent initialization across all agent types

## 🎯 **Benefits of Standardized Architecture**

### **Educational Benefits**
- **Clear Algorithm Separation**: Each agent represents one algorithm
- **Progressive Complexity**: From simple BFS to complex RL
- **Comparative Analysis**: Easy to compare different approaches
- **Modular Learning**: Students can focus on specific algorithms

### **Technical Benefits**
- **Consistent Interface**: All agents implement BaseAgent protocol
- **Easy Extension**: New algorithms can be added easily
- **Testable Components**: Each agent can be tested independently
- **Performance Monitoring**: Built-in statistics and metrics

### **Research Benefits**
- **Algorithm Isolation**: Pure algorithm implementations
- **Reproducible Results**: Consistent initialization and state management
- **Extensible Framework**: Easy to add new agent types
- **Comparative Studies**: Standardized evaluation metrics

---

**This agent architecture ensures consistent, extensible, and educational implementations across all Snake Game AI extensions while maintaining clear separation of concerns and supporting progressive learning from simple heuristics to complex machine learning approaches.**

