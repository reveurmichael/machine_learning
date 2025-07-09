# Final Decision 4: Agent Naming Conventions & Standards

> **SUPREME AUTHORITY**: This document establishes the definitive agent naming conventions and implementation standards for the Snake Game AI project.



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
