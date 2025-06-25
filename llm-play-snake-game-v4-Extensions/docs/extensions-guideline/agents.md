# Agent Implementation Standards

This document establishes agent naming conventions and implementation patterns for all extensions.

## 🎯 **Standardized Naming Convention**

**Philosophy**: Consistent naming enables predictable patterns across all extensions and algorithm types.

### **File and Class Naming Standards**
```python
# ✅ REQUIRED PATTERN: agent_{algorithm}.py → {Algorithm}Agent
agent_bfs.py              → class BFSAgent(BaseAgent)
agent_astar.py            → class AStarAgent(BaseAgent)  
agent_hamiltonian.py      → class HamiltonianAgent(BaseAgent)
agent_mlp.py              → class MLPAgent(BaseAgent)
agent_dqn.py              → class DQNAgent(BaseAgent)
agent_lora.py             → class LoRAAgent(BaseAgent)
```

**Critical Rule**: ALL agent files MUST use `agent_{algorithm}.py` naming pattern. No exceptions.

**Agent Placement by Version:**
- **v0.01**: Agent files in extension root directory (proof of concept simplicity)
- **v0.02+**: ALL agent files MUST be in `agents/` directory (organized structure)

**Special Case**: Task-0 agent is named `SnakeAgent` (not `LLMSnakeAgent`) following established naming conventions.

## 🏗️ **Directory Structure Evolution**

Agent placement evolves naturally across versions:

### **v0.01: Proof of Concept**
```
extensions/{algorithm}-v0.01/
├── agent_{primary}.py             # ✅ Single algorithm in extension root
├── game_logic.py
└── game_manager.py
```

### **v0.02+: Multi-Algorithm**
```
extensions/{algorithm}-v0.02/
├── agents/                        # ✅ ALL agents in organized package structure
│   ├── __init__.py               # Agent factory
│   ├── agent_{algo1}.py
│   ├── agent_{algo2}.py
│   └── agent_{algo3}.py
├── game_logic.py
└── game_manager.py
```

### **v0.03: Dashboard Integration + Allowed Enhancements**
- 🔒 **Core agents copied exactly** from v0.02 (algorithm logic unchanged)
- ➕ **Web-specific enhancements allowed** (monitoring, optimization wrappers)
- 🔒 **Factory patterns unchanged** (registration names stable)
- ➕ **UI integration utilities** for dashboard components

### **v0.04: Advanced Features (Heuristics Only)**
- 🔒 **Core agents unchanged** from v0.03 (algorithm stability maintained)
- ➕ **JSONL generation capabilities** added without modifying core algorithms
- 🔒 **All v0.03 functionality preserved** (backward compatibility)
- ➕ **Language-rich dataset generation** for LLM fine-tuning

### **Extension Evolution Rules Summary**
| Version Transition | Core Algorithms | Enhancements | Factory | Breaking Changes |
|-------------------|----------------|-------------|---------|------------------|
| **v0.01 → v0.02** | ✅ Add new | ✅ Add variants | ✅ Create factory | ✅ Allowed |
| **v0.02 → v0.03** | 🔒 **Copy exactly** | ➕ Web utilities | 🔒 **Stable** | ❌ **Forbidden** |
| **v0.03 → v0.04** | 🔒 **Copy exactly** | ➕ JSONL tools | 🔒 **Stable** | ❌ **Forbidden** |

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

