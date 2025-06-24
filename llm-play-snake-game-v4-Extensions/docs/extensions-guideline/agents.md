> **Important — Authoritative Reference:** This guide supplements Final Decision 4 (Agent Naming) and Final Decision 5 (Directory Structure). **For conflicts, Final Decisions prevail.**

# Agent Implementation Standards

This document establishes agent naming conventions and implementation patterns following the standardized architecture from Final Decision 4.

## 🎯 **Standardized Naming Convention**

**Philosophy**: Consistent naming enables predictable patterns across all extensions and algorithm types.

### **File and Class Naming (Final Decision 4)**
```python
# ✅ REQUIRED PATTERN: agent_{algorithm}.py → {Algorithm}Agent
agent_bfs.py              → class BFSAgent(BaseAgent)
agent_astar.py            → class AStarAgent(BaseAgent)  
agent_hamiltonian.py      → class HamiltonianAgent(BaseAgent)
agent_mlp.py              → class MLPAgent(BaseAgent)
agent_dqn.py              → class DQNAgent(BaseAgent)
agent_lora.py             → class LoRAAgent(BaseAgent)
```

**Special Case**: Task-0 agent is named `SnakeAgent` (not `LLMSnakeAgent`) per Final Decision 10.

## 🏗️ **Directory Structure Evolution**

Following Final Decision 5, agent placement evolves naturally:

### **v0.01: Proof of Concept**
```
extensions/{algorithm}-v0.01/
├── agent_{primary}.py             # Single algorithm in extension root
├── game_logic.py
└── game_manager.py
```

### **v0.02+: Multi-Algorithm**
```
extensions/{algorithm}-v0.02/
├── agents/                        # Organized package structure
│   ├── __init__.py               # Agent factory
│   ├── agent_{algo1}.py
│   ├── agent_{algo2}.py
│   └── agent_{algo3}.py
├── game_logic.py
└── game_manager.py
```

### **v0.03: Dashboard Integration**
- `agents/` directory **copied exactly** from v0.02
- Enhanced with dashboard UI components
- No changes to agent implementations

### **v0.04: Advanced Features (Heuristics Only)**
- `agents/` directory **unchanged** from v0.03
- Enhanced with JSONL generation capabilities
- Algorithm implementations remain stable

## 🧠 **Design Patterns & Philosophy**

### **Template Method Pattern**
- **BaseAgent** defines common structure for all agents
- Subclasses implement algorithm-specific logic
- Ensures consistent interface across all extensions

### **Strategy Pattern**
- Each agent encapsulates a specific algorithm strategy
- Pluggable into game managers via factory patterns
- Isolated algorithm logic for maintainability

### **Factory Pattern**
- Centralized agent creation via factory classes
- Dynamic agent selection by name/configuration
- Consistent initialization across all agent types

## 📋 **Implementation Standards**

### **Required Agent Interface**
```python
from core.game_agents import BaseAgent

class HeuristicAgent(BaseAgent):
    """Algorithm-specific agent implementation"""
    
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

### **Agent Factory Pattern**
```python
# agents/__init__.py
class AgentFactory:
    """Factory for creating algorithm-specific agents"""
    
    @staticmethod
    def create_agent(algorithm: str, grid_size: int) -> BaseAgent:
        """Create agent by algorithm name"""
        # Dynamic agent creation based on standardized naming
        pass
```

## 🎯 **Benefits of Standardized Architecture**

### **Educational Value**
- **Clear Algorithm Separation**: One algorithm per agent file
- **Progressive Complexity**: From simple BFS to complex RL/LLM
- **Comparative Analysis**: Easy comparison of different approaches

### **Technical Benefits**
- **Consistent Interface**: All agents implement BaseAgent protocol
- **Easy Extension**: New algorithms follow established patterns
- **Testable Components**: Each agent independently testable

### **Cross-Extension Compatibility**
- **Uniform Naming**: Same patterns across heuristics, ML, RL, LLM
- **Version Stability**: Agent directory structure preserved across versions
- **Factory Integration**: Consistent agent creation across all extensions

---

**This standardized architecture ensures educational value, technical consistency, and scalable agent development across all Snake Game AI extensions.**

