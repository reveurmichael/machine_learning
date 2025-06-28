# Final Decision 4: Agent Naming Conventions & Standards

> **SUPREME AUTHORITY**: This document establishes the definitive agent naming conventions and implementation standards for the Snake Game AI project.

## 🎯 **Executive Summary**

This document establishes the **definitive naming conventions** for agent files and classes across all Snake Game AI extensions. It standardizes the file naming pattern, class naming pattern, and directory organization to ensure consistency, clarity, and maintainability across all algorithm types and extension versions.

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
    
Dependencies:
    - BaseAgent from core.game_agents
    - Algorithm-specific libraries and utilities
    
Usage:
    from agents.agent_{algorithm} import {Algorithm}Agent
    
    agent = {Algorithm}Agent(grid_size=10)
    move = agent.plan_move(game_state)
"""

from typing import Dict, List, Tuple, Any, Optional
from core.game_agents import BaseAgent

class {Algorithm}Agent(BaseAgent):
    """
    {Algorithm} agent for Snake Game AI.
    
    This class implements the {Algorithm} algorithm for Snake game playing,
    providing optimal/near-optimal pathfinding and decision making.
    
    Attributes:
        algorithm_name (str): Name of the algorithm ("ALGORITHM_NAME")
        grid_size (int): Size of the game grid
        performance_stats (dict): Algorithm performance statistics
        
    Example:
        >>> agent = {Algorithm}Agent(grid_size=10)
        >>> move = agent.plan_move(game_state)
        >>> print(f"Next move: {move}")
    """
    
    def __init__(self, grid_size: int, **kwargs):
        """
        Initialize {Algorithm} agent.
        
        Args:
            grid_size: Size of the game grid (NxN)
            **kwargs: Additional algorithm-specific parameters
        """
        super().__init__(name="{ALGORITHM_NAME}", grid_size=grid_size)
        self.algorithm_name = "{ALGORITHM_NAME}"
        # Algorithm-specific initialization
        
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """
        Plan next move using {Algorithm} algorithm.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Next move direction ('UP', 'DOWN', 'LEFT', 'RIGHT')
            
        Raises:
            ValueError: If game state is invalid
            RuntimeError: If no valid move can be found
        """
        # Algorithm-specific implementation
        print(f"[{self.algorithm_name}] Planning move")  # SUPREME_RULE NO.3
        
        # Extract game state components
        head_pos = game_state.get('snake_positions', [[]])[0]
        apple_pos = game_state.get('apple_position', [])
        
        # Algorithm-specific logic here
        move = self._calculate_move(head_pos, apple_pos, game_state)
        
        print(f"[{self.algorithm_name}] Selected move: {move}")  # SUPREME_RULE NO.3
        return move
        
    def reset(self) -> None:
        """Reset agent state for new game"""
        # Algorithm-specific reset logic
        print(f"[{self.algorithm_name}] Resetting agent state")  # SUPREME_RULE NO.3
        # Reset algorithm-specific state variables

### **Agent Factory Integration**
```python
# agents/__init__.py - Factory pattern implementation
"""
Agent factory for creating algorithm instances.

This module provides a centralized factory for creating agent instances
following the standardized naming conventions.
"""

from typing import Dict, Type, Any
from .agent_bfs import BFSAgent
from .agent_astar import AStarAgent
from .agent_hamiltonian import HamiltonianAgent
# Import all other agents following the pattern

class AgentFactory:
    """
    Factory for creating agent instances using standardized naming.
    
    Design Patterns:
    - Factory Pattern: Centralized agent creation
    - Registry Pattern: Algorithm name to class mapping
    - Strategy Pattern: Pluggable algorithm implementations
    """
    
    # Registry follows naming convention: algorithm_name -> AgentClass
    _agents: Dict[str, Type[BaseAgent]] = {
        'BFS': BFSAgent,
        'ASTAR': AStarAgent,
        'A*': AStarAgent,  # Alias
        'HAMILTONIAN': HamiltonianAgent,
        'HAM': HamiltonianAgent,  # Alias
        'MLP': MLPAgent,
        'CNN': CNNAgent,
        'LSTM': LSTMAgent,
        'XGBOOST': XGBoostAgent,
        'LIGHTGBM': LightGBMAgent,
        'DQN': DQNAgent,
        'PPO': PPOAgent,
        'LORA': LoRAAgent,
        'DISTILLED': DistilledAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, grid_size: int, **kwargs) -> BaseAgent:
        """
        Create agent instance by algorithm name.
        
        Args:
            algorithm: Algorithm name (case-insensitive)
            grid_size: Game grid size
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Initialized agent instance
            
        Raises:
            ValueError: If algorithm is not supported
            
        Example:
            >>> agent = AgentFactory.create("BFS", grid_size=10)
            >>> agent = AgentFactory.create("MLP", grid_size=10, hidden_size=128)
        """
        algorithm_upper = algorithm.upper()
        
        if algorithm_upper not in cls._agents:
            available = ', '.join(sorted(cls._agents.keys()))
            raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {available}")
        
        agent_class = cls._agents[algorithm_upper]
        return agent_class(grid_size=grid_size, **kwargs)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """Get list of available algorithm names"""
        return sorted(list(cls._agents.keys()))
    
    @classmethod
    def register_agent(cls, name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register new agent class.
        
        Args:
            name: Algorithm name (will be uppercased)
            agent_class: Agent class following naming convention
        """
        cls._agents[name.upper()] = agent_class

# Convenience function following naming convention
def create_agent(algorithm: str, grid_size: int, **kwargs) -> BaseAgent:
    """Create agent instance - convenience function"""
    return AgentFactory.create(algorithm, grid_size, **kwargs)
```

### **Import Patterns**
```python
# ✅ STANDARDIZED IMPORT PATTERNS:

# From specific agent file
from agents.agent_bfs import BFSAgent
from agents.agent_astar import AStarAgent
from agents.agent_mlp import MLPAgent

# From factory (recommended for dynamic creation)
from agents import AgentFactory, create_agent

# Extension-level imports
from extensions.heuristics_v0_03.agents.agent_bfs import BFSAgent
from extensions.supervised_v0_02.models.neural_networks.agent_mlp import MLPAgent

# Factory usage
agent = AgentFactory.create("BFS", grid_size=10)
agent = create_agent("MLP", grid_size=10, hidden_size=256)
```

## 🔍 **Special Naming Cases**

### **Multi-Word Algorithms**
```

### **Algorithm Variants and Inheritance**
```python
# ✅ INHERITANCE PATTERNS following naming convention:

# Base algorithm
class BFSAgent(BaseAgent):
    """Base BFS implementation"""
    def __init__(self, grid_size: int):
        super().__init__(name="BFS", grid_size=grid_size)
        print(f"[BFSAgent] Initialized BFS agent")  # SUPREME_RULE NO.3
    
    def _calculate_move(self, head_pos, apple_pos, game_state):
        """BFS pathfinding implementation"""
        # BFS algorithm logic here
        return "UP"  # Default move

# Enhanced variants inherit from base
class BFSSafeGreedyAgent(BFSAgent):
    """BFS with safety checks and greedy optimization"""
    def __init__(self, grid_size: int):
        super().__init__(grid_size)
        self.safety_threshold = 0.8
        print(f"[BFSSafeGreedyAgent] Initialized with safety checks")  # SUPREME_RULE NO.3

class BFSHamiltonianAgent(BFSSafeGreedyAgent):
    """BFS with Hamiltonian path concepts"""
    def __init__(self, grid_size: int):
        super().__init__(grid_size)
        self.hamiltonian_cycle = self._generate_hamiltonian_cycle()
        print(f"[BFSHamiltonianAgent] Initialized with Hamiltonian cycle")  # SUPREME_RULE NO.3

# Independent but related algorithms
class AStarAgent(BaseAgent):
    """A* pathfinding algorithm"""
    def __init__(self, grid_size: int):
        super().__init__(name="A*", grid_size=grid_size)
        print(f"[AStarAgent] Initialized A* agent")  # SUPREME_RULE NO.3

class AStarHamiltonianAgent(AStarAgent):
    """A* with Hamiltonian optimization"""
    def __init__(self, grid_size: int):
        super().__init__(grid_size)
        self.hamiltonian_weight = 0.3
        print(f"[AStarHamiltonianAgent] Initialized with Hamiltonian optimization")  # SUPREME_RULE NO.3
```