# Final Decision 4: Agent Naming Conventions & Standards

> **SUPREME AUTHORITY**: This document establishes the definitive agent naming conventions and implementation standards for the Snake Game AI project.

> **See also:** `agents.md` (Agent implementation standards), `naming-conventions.md` (Naming standards), `factory-design-pattern.md` (Factory patterns), `final-decision-10.md` (SUPREME_RULES).

## 🎯 **Executive Summary**

This document establishes the **definitive naming conventions** for agent files and classes across all Snake Game AI extensions. It standardizes the file naming pattern, class naming pattern, and directory organization to ensure consistency, clarity, and maintainability across all algorithm types and extension versions, strictly following SUPREME_RULES from `final-decision-10.md`.

### **SUPREME_RULES Integration**
- **SUPREME_RULE NO.1**: Enforces reading all GOOD_RULES before making agent naming changes to ensure comprehensive understanding
- **SUPREME_RULE NO.2**: Uses precise `final-decision-N.md` format consistently when referencing architectural decisions
- **SUPREME_RULE NO.3**: Enables lightweight common utilities with OOP extensibility while maintaining agent patterns through inheritance rather than tight coupling
- **SUPREME_RULE NO.4**: Ensures all markdown files are coherent and aligned through nuclear diffusion infusion process

### **GOOD_RULES Integration**
This document integrates with the **GOOD_RULES** governance system established in `final-decision-10.md`:
- **`agents.md`**: Authoritative reference for agent implementation standards
- **`naming-conventions.md`**: Authoritative reference for naming standards
- **`factory-design-pattern.md`**: Authoritative reference for factory pattern implementation
- **`single-source-of-truth.md`**: Ensures naming consistency across all extensions

## 📁 **DECISION: Standardized Agent Naming Pattern**

### **✅ FINALIZED NAMING CONVENTION**

#### **File Naming Pattern**
```python
# ✅ STANDARDIZED PATTERN: agent_{algorithm}.py

# File names use lowercase with underscores
agent_bfs.py              # Breadth-First Search
agent_astar.py            # A* pathfinding  
agent_hamiltonian.py      # Hamiltonian path algorithm
agent_dfs.py              # Depth-First Search
agent_mlp.py              # Multi-Layer Perceptron
agent_cnn.py              # Convolutional Neural Network
agent_lstm.py             # Long Short-Term Memory
agent_gru.py              # Gated Recurrent Unit
agent_xgboost.py          # XGBoost gradient boosting
agent_lightgbm.py         # LightGBM gradient boosting
agent_randomforest.py     # Random Forest
agent_gcn.py              # Graph Convolutional Network
agent_dqn.py              # Deep Q-Network
agent_ppo.py              # Proximal Policy Optimization
agent_a3c.py              # Asynchronous Actor-Critic
agent_lora.py             # LoRA fine-tuned LLM
agent_distilled.py        # Distilled model agent
```

#### **Class Naming Pattern**
```python
# ✅ STANDARDIZED PATTERN: {Algorithm}Agent

# Class names use PascalCase with Agent suffix
class BFSAgent(BaseAgent):              # from agent_bfs.py
class AStarAgent(BaseAgent):            # from agent_astar.py
class HamiltonianAgent(BaseAgent):      # from agent_hamiltonian.py
class DFSAgent(BaseAgent):              # from agent_dfs.py
class MLPAgent(BaseAgent):              # from agent_mlp.py
class CNNAgent(BaseAgent):              # from agent_cnn.py
class LSTMAgent(BaseAgent):             # from agent_lstm.py
class GRUAgent(BaseAgent):              # from agent_gru.py
class XGBoostAgent(BaseAgent):          # from agent_xgboost.py
class LightGBMAgent(BaseAgent):         # from agent_lightgbm.py
class RandomForestAgent(BaseAgent):     # from agent_randomforest.py
class GCNAgent(BaseAgent):              # from agent_gcn.py
class DQNAgent(BaseAgent):              # from agent_dqn.py
class PPOAgent(BaseAgent):              # from agent_ppo.py
class A3CAgent(BaseAgent):              # from agent_a3c.py
class LoRAAgent(BaseAgent):             # from agent_lora.py
class DistilledAgent(BaseAgent):        # from agent_distilled.py
```

### **Rationale for This Convention**

#### **1. File Name Benefits (`agent_*.py`)**
- **Clear Identification**: `agent_` prefix immediately identifies agent files
- **Alphabetical Grouping**: All agent files group together in directory listings
- **Namespace Clarity**: Prevents confusion with other algorithm-related files
- **Import Consistency**: Predictable import patterns across extensions
- **IDE Support**: Better autocomplete and file navigation

#### **2. Class Name Benefits (`*Agent`)**
- **Standard Convention**: Follows established Python naming conventions
- **Algorithm Prominence**: Algorithm name comes first, emphasizing the approach
- **Inheritance Clarity**: `Agent` suffix clearly indicates inheritance from BaseAgent
- **Factory Pattern Support**: Enables clean factory method implementations
- **Documentation Clarity**: Makes class purpose immediately obvious

#### **3. Consistency Benefits**
- **Cross-Extension Uniformity**: Same pattern across heuristics, ML, RL, LLM extensions
- **Version Independence**: Same naming pattern from v0.01 through v0.04
- **Maintenance Efficiency**: Predictable structure reduces cognitive load
- **Onboarding Speed**: New developers can quickly understand the codebase structure

## 🏗️ **Directory Structure Examples**

### **Heuristics Extensions**
```
extensions/heuristics-v0.02/agents/
├── __init__.py
├── agent_bfs.py                    → class BFSAgent
├── agent_bfs_safe_greedy.py        → class BFSSafeGreedyAgent
├── agent_bfs_hamiltonian.py        → class BFSHamiltonianAgent
├── agent_dfs.py                    → class DFSAgent
├── agent_astar.py                  → class AStarAgent
├── agent_astar_hamiltonian.py      → class AStarHamiltonianAgent
└── agent_hamiltonian.py            → class HamiltonianAgent

extensions/heuristics-v0.03/agents/  # ✅ Same structure as v0.02
├── __init__.py
├── agent_bfs.py                    → class BFSAgent
├── agent_bfs_safe_greedy.py        → class BFSSafeGreedyAgent
├── agent_bfs_hamiltonian.py        → class BFSHamiltonianAgent
├── agent_dfs.py                    → class DFSAgent
├── agent_astar.py                  → class AStarAgent
├── agent_astar_hamiltonian.py      → class AStarHamiltonianAgent
└── agent_hamiltonian.py            → class HamiltonianAgent

extensions/heuristics-v0.04/agents/  # ✅ Same structure as v0.03
├── __init__.py
├── agent_bfs.py                    → class BFSAgent (with JSONL generation)
├── agent_bfs_safe_greedy.py        → class BFSSafeGreedyAgent (with JSONL)
├── agent_bfs_hamiltonian.py        → class BFSHamiltonianAgent (with JSONL)
├── agent_dfs.py                    → class DFSAgent (with JSONL)
├── agent_astar.py                  → class AStarAgent (with JSONL)
├── agent_astar_hamiltonian.py      → class AStarHamiltonianAgent (with JSONL)
└── agent_hamiltonian.py            → class HamiltonianAgent (with JSONL)
```

### **Supervised Learning Extensions**
```
extensions/supervised-v0.02/models/
├── neural_networks/
│   ├── __init__.py
│   ├── agent_mlp.py                → class MLPAgent
│   ├── agent_cnn.py                → class CNNAgent
│   ├── agent_lstm.py               → class LSTMAgent
│   └── agent_gru.py                → class GRUAgent
├── tree_models/
│   ├── __init__.py
│   ├── agent_xgboost.py            → class XGBoostAgent
│   ├── agent_lightgbm.py           → class LightGBMAgent
│   └── agent_randomforest.py       → class RandomForestAgent
└── graph_models/
    ├── __init__.py
    ├── agent_gcn.py                → class GCNAgent
    ├── agent_graphsage.py          → class GraphSAGEAgent
    └── agent_gat.py                → class GATAgent

extensions/supervised-v0.03/models/  # ✅ Same structure as v0.02
├── neural_networks/
│   ├── agent_mlp.py                → class MLPAgent
│   └── (same files as v0.02)
├── tree_models/
│   ├── agent_xgboost.py            → class XGBoostAgent
│   └── (same files as v0.02)
└── graph_models/
    ├── agent_gcn.py                → class GCNAgent
    └── (same files as v0.02)
```

### **Reinforcement Learning Extensions**
```
extensions/reinforcement-v0.02/agents/
├── __init__.py
├── agent_dqn.py                    → class DQNAgent
├── agent_double_dqn.py             → class DoubleDQNAgent
├── agent_dueling_dqn.py            → class DuelingDQNAgent
├── agent_ppo.py                    → class PPOAgent
├── agent_a3c.py                    → class A3CAgent
└── agent_sac.py                    → class SACAgent

extensions/reinforcement-v0.03/agents/  # ✅ Same structure as v0.02
├── __init__.py
├── agent_dqn.py                    → class DQNAgent
├── agent_double_dqn.py             → class DoubleDQNAgent
├── agent_dueling_dqn.py            → class DuelingDQNAgent
├── agent_ppo.py                    → class PPOAgent
├── agent_a3c.py                    → class A3CAgent
└── agent_sac.py                    → class SACAgent
```

### **LLM Fine-tuning Extensions**
```
extensions/llm-finetune-v0.02/agents/
├── __init__.py
├── agent_lora.py                   → class LoRAAgent
├── agent_full_finetune.py          → class FullFinetuneAgent
├── agent_qlora.py                  → class QLoRAAgent
└── agent_prefix_tuning.py          → class PrefixTuningAgent

extensions/llm-distillation-v0.02/agents/
├── __init__.py
├── agent_distilled.py              → class DistilledAgent
├── agent_knowledge_distilled.py    → class KnowledgeDistilledAgent
└── agent_feature_distilled.py      → class FeatureDistilledAgent
```

## 💻 **Implementation Standards**

### **File Header Template**
```python
# agent_{algorithm}.py
"""
{Algorithm} agent implementation for Snake Game AI.

This module implements the {Algorithm} algorithm for Snake game playing,
following the standardized agent interface and naming conventions.

Classes:
    {Algorithm}Agent: Main agent implementation
    
Design Pattern: Strategy Pattern
Purpose: Implements {Algorithm} algorithm for Snake game decision making
Educational Value: Demonstrates {Algorithm} implementation with canonical patterns

Reference: `final-decision-10.md` for SUPREME_RULES compliance
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np

from core.agents.base_agent import BaseAgent
from extensions.common.utils.path_utils import get_dataset_path
from extensions.common.validation.dataset_validator import validate_dataset

class {Algorithm}Agent(BaseAgent):
    """
    {Algorithm} agent for Snake Game AI.
    
    Design Pattern: Strategy Pattern (Canonical Implementation)
    Purpose: Implements {Algorithm} algorithm for Snake game decision making
    Educational Value: Shows how {Algorithm} works with canonical factory patterns
    
    Reference: final-decision-10.md for canonical method naming
    """
    
    def __init__(self, name: str, grid_size: int, **kwargs):
        super().__init__(name, grid_size)
        self.algorithm_name = "{Algorithm}"
        self.config = kwargs
        print_info(f"[{Algorithm}Agent] Initialized {name} agent")  # Simple logging
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """
        Plan next move using {Algorithm} algorithm.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Move direction ('UP', 'DOWN', 'LEFT', 'RIGHT')
        """
        print_info(f"[{self.name}] Planning move using {self.algorithm_name}")  # Simple logging
        
        # {Algorithm} implementation here
        # This is where the specific algorithm logic goes
        
        move = self._implement_{algorithm_lower}(game_state)
        print_info(f"[{self.name}] {self.algorithm_name} decided: {move}")  # Simple logging
        return move
    
    def _implement_{algorithm_lower}(self, game_state: Dict[str, Any]) -> str:
        """Implement {Algorithm} algorithm logic"""
        # Algorithm-specific implementation
        # This method contains the core {Algorithm} logic
        pass
```

### **Factory Pattern Integration**
```python
# agent_factory.py
class AgentFactory:
    """
    Factory for creating agent instances
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create appropriate agent instances based on algorithm type
    Educational Value: Shows how canonical factory patterns work with agent naming
    
    Reference: final-decision-10.md for canonical method naming
    """
    
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
        "HAMILTONIAN": HamiltonianAgent,
        "MLP": MLPAgent,
        "CNN": CNNAgent,
        "LSTM": LSTMAgent,
        "XGBOOST": XGBoostAgent,
        "DQN": DQNAgent,
        "PPO": PPOAgent,
        "LORA": LoRAAgent,
        "DISTILLED": DistilledAgent,
    }
    
    @classmethod
    def create(cls, agent_type: str, **kwargs):  # CANONICAL create() method
        """Create agent using canonical create() method (SUPREME_RULES compliance)"""
        agent_class = cls._registry.get(agent_type.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {available}")
        print_info(f"[AgentFactory] Creating agent: {agent_type}")  # Simple logging
        return agent_class(**kwargs)
```

## 🎓 **Educational Value and Learning Path**

### **Learning Objectives**
- **Naming Conventions**: Understanding the importance of consistent naming
- **File Organization**: Learning to organize code with clear patterns
- **Factory Patterns**: Understanding canonical factory pattern implementation
- **Code Maintainability**: Learning to write maintainable, predictable code

### **Implementation Examples**
- **Agent Creation**: How to create agents following naming conventions
- **Factory Integration**: How to integrate agents with factory patterns
- **File Organization**: How to organize agent files consistently
- **Naming Compliance**: How to follow established naming conventions

## 🔗 **Integration with Other Documentation**

### **GOOD_RULES Alignment**
This document aligns with:
- **`agents.md`**: Detailed agent implementation standards
- **`naming-conventions.md`**: Comprehensive naming standards
- **`factory-design-pattern.md`**: Factory pattern implementation
- **`single-source-of-truth.md`**: Architectural principles

### **Extension Guidelines**
This naming convention supports:
- All extension types (heuristics, supervised, reinforcement, LLM)
- All algorithm types (pathfinding, ML, RL, LLM)
- Consistent file and class organization
- Predictable import and usage patterns

---

**This agent naming convention ensures consistent, clear, and maintainable agent organization across all Snake Game AI extensions while maintaining SUPREME_RULES compliance.**

> **SUPREME_RULES COMPLIANCE**: This document strictly follows the SUPREME_RULES established in `final-decision-10.md`, ensuring coherence, educational value, and architectural integrity across the entire project.