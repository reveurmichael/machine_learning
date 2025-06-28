# Agent Implementation Standards for Snake Game AI

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for agent implementation standards and supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and `extension-evolution-rules.md`.

This document defines the authoritative agent naming conventions, implementation patterns, and architectural standards that ensure consistency, educational value, and maintainability across all Snake Game AI extensions.

### **SUPREME_RULES Alignment**
- **SUPREME_RULE NO.1**: Enforces reading all GOOD_RULES before making agent architectural changes to ensure comprehensive understanding
- **SUPREME_RULE NO.2**: Uses precise `final-decision-N.md` format consistently when referencing architectural decisions and agent patterns
- **SUPREME_RULE NO.3**: Enables lightweight common utilities with OOP extensibility while maintaining agent patterns through inheritance rather than tight coupling

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

## 🏗️ **Directory Structure Evolution**

> **Authoritative Reference**: See `extension-evolution-rules.md` for complete evolution rules and stability requirements.

Agent placement follows a clear evolution pattern across versions:

### **v0.01: Proof of Concept**
```
extensions/{algorithm}-v0.01/
├── agent_{primary}.py             # ✅ Single algorithm in extension root
├── game_logic.py
└── game_manager.py
```

### **v0.02+: Multi-Algorithm (MANDATORY ORGANIZATION)**
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

**Critical Rule**: Starting from v0.02, ALL agent files MUST be in the `agents/` directory. No exceptions.

### **v0.03: Dashboard Integration + Allowed Enhancements**
- 🔒 **Core agents copied exactly** from v0.02 (algorithm logic unchanged)
- ➕ **Web-specific enhancements allowed** (monitoring, optimization wrappers following ROOT/web patterns)
- �� **Factory patterns unchanged** (registration names stable)
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
> **Authoritative Reference**: See `factory-design-pattern.md` for complete factory pattern implementation.

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

## 🔗 **See Also**

- **`extension-evolution-rules.md`**: Authoritative reference for version evolution rules
- **`final-decision-4.md`**: Naming conventions and standards
- **`final-decision-5.md`**: Directory structure standards

---

**This standardized architecture ensures educational value, technical consistency, and scalable agent development across all Snake Game AI extensions.**

