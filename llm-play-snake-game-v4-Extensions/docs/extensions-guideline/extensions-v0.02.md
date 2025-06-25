> **Important — Authoritative Reference:** This document is **supplementary** to the _Final Decision Series_ (`final-decision-0` → `final-decision-10`). In case of discrepancy, the Final Decision documents prevail.

# Extensions v0.02: The Expansion Phase

## 🎯 **Core Philosophy: From Proof of Concept to a Multi-Algorithm System**

The `v0.02` extension represents the natural evolution from a minimalist proof of concept (`v0.01`) to a more robust and practical system. Its primary purpose is to demonstrate how a single-algorithm extension can mature to **support a diverse suite of comparable algorithms** within the same domain.

This version introduces organizational structure and dynamic selection, transforming a simple script into a versatile command-line tool. It answers the question:

> "How does our architecture scale from one solution to many?"

## 🏗️ **Architectural Upgrade: Structure and Selection**

A `v0.02` extension introduces two fundamental architectural upgrades, as defined in `final-decision-5.md`.

### **1. The `agents/` Directory: An Organized Home for Agents**

**MAJOR CHANGE from v0.01:** All agent-related code is moved from the extension root into a new, dedicated `agents/` directory. This cleans up the top-level directory and establishes a clear, organized location for all algorithm implementations.

**Version-Specific Agent Placement:**
- **v0.01**: `agent_bfs.py` in extension root (simple proof of concept)
- **v0.02+**: ALL `agent_*.py` files MUST be in `agents/` directory (organized structure)

### **2. The `--algorithm` Flag: Enabling Dynamic Selection**

The entry point, `main.py`, is enhanced to become a true command-line tool. It **must** accept an `--algorithm` argument, allowing the user to dynamically select which agent to run from the command line.

### **Updated Directory Structure**
```
extensions/{algorithm_type}-v0.02/
├── __init__.py
├── README.md
├── agents/                  # 👈 NEW: All agent code moved from root to here
│   ├── __init__.py          # Contains the Agent Factory
│   ├── agent_{algo1}.py     # ✅ v0.02+: Agents in organized directory
│   └── agent_{algo2}.py     # ✅ v0.02+: Agents in organized directory
├── game_data.py             # 👈 NEW: Often added to handle more detailed stats
├── game_logic.py
├── game_manager.py
└── main.py                  # 👈 ENHANCED: Now accepts `--algorithm` flag
```

## 🔧 **The Core Mechanism: The Agent Factory**

To power the `--algorithm` flag, a `v0.02` extension **must** implement an **Agent Factory**. This is the single most important pattern in this version.

The factory is a simple class, typically located in `agents/__init__.py`, that maps string names to agent classes.

```python
# extensions/heuristics-v0.02/agents/__init__.py

> **Important**: For complete factory pattern implementation, see `unified-factory-pattern-guide.md` - the authoritative reference for all factory implementations.

from .agent_bfs import BFSAgent
from .agent_astar import AStarAgent
from .agent_dfs import DFSAgent

class HeuristicAgentFactory(BaseAgentFactory):
    """Factory for Heuristic Pathfinding Agents - See factory pattern guide for complete implementation"""

    @classmethod
    def get_registry(cls) -> Dict[str, Type[BaseAgent]]:
        return {
            "BFS": BFSAgent,
            "ASTAR": AStarAgent,
            "DFS": DFSAgent,
        }
```

## 🚀 **Extension Evolution Rules and Stability**

### **Core Algorithm Stability Principle**
A key principle of the `v0.02` to `v0.03` evolution is **core algorithm stability**. The `agents/` folder represents the **stable algorithmic foundation**:

**✅ Stable Between Versions (Never Modified):**
- Core algorithm implementations (BFS, A*, DQN, etc.)
- Factory patterns and registry systems
- Base agent interfaces and contracts
- Fundamental algorithmic logic

**⚠️ Allowed Extensions (Can Be Added):**
- Enhanced variants of existing algorithms (BFSSafeGreedy extends BFS)
- Performance optimizations that maintain interface
- Additional utility methods for web interface support
- Monitoring and metrics collection capabilities

**❌ Forbidden Changes (Breaking Stability):**
- Modifying core algorithm behavior
- Changing factory registration names
- Breaking existing agent interfaces
- Removing or renaming existing agents

### **Version Compatibility Matrix**

| Component | v0.01 → v0.02 | v0.02 → v0.03 | v0.03 → v0.04* |
|-----------|---------------|---------------|----------------|
| **Core Algorithms** | ✅ Add new | 🔒 **Stable (copy exactly)** | 🔒 **Stable** |
| **Agent Factory** | ✅ Add registry | 🔒 **Stable** | 🔒 **Stable** |
| **Enhanced Variants** | ➕ Can add | ➕ Can add | ➕ Can add |
| **Interface Methods** | ✅ Define | 🔒 **Stable** | ➕ Can extend |
| **Web Integration** | ❌ Not present | ✅ Add dashboard | ✅ Enhance |

*v0.04 only exists for heuristics extensions

This demonstrates a powerful software engineering principle: the core algorithmic logic can be finalized and then built upon with new interfaces (like web UI in `v0.03`) without modification.

## 📋 **Compliance Checklist: The Definition of Done**

A `v0.02` extension is considered complete and successful if it meets these criteria:

- [ ] Has all agent code been moved into a dedicated `agents/` directory?
- [ ] Does the `main.py` script correctly parse an `--algorithm` command-line argument?
- [ ] Is there an `AgentFactory` class in `agents/__init__.py` that can create agents from a string name?
- [ ] Does the `GameManager` correctly use this factory to instantiate the user-selected agent?
- [ ] Is the extension still headless (no GUI)?

---

> **The `v0.02` extension marks the transition from a simple test case to a structured, reusable, and scalable software component. It lays the stable foundation upon which the user-facing features of `v0.03` will be built.**

## 🧠 **Algorithm-Specific Examples**

### **Heuristics v0.02**
```
extensions/heuristics-v0.02/
├── __init__.py
├── main.py                        # --algorithm BFS|ASTAR|DFS|HAMILTONIAN
├── game_logic.py                  # HeuristicGameLogic with algorithm switching
├── game_manager.py                # Multi-algorithm manager
├── game_data.py                   # Heuristic-specific data tracking
├── agents/
│   ├── __init__.py               # HeuristicAgentFactory
│   ├── agent_bfs.py              # BFS algorithm
│   ├── agent_bfs_safe_greedy.py  # BFS with safety heuristics
│   ├── agent_bfs_hamiltonian.py  # BFS + Hamiltonian concepts
│   ├── agent_dfs.py              # Depth-First Search
│   ├── agent_astar.py            # A* pathfinding
│   ├── agent_astar_hamiltonian.py # A* + Hamiltonian
│   └── agent_hamiltonian.py      # Pure Hamiltonian path
└── README.md
```

### **Supervised v0.02**
```
extensions/supervised-v0.02/
├── __init__.py
├── main.py                        # Model selection and evaluation
├── game_logic.py                  # ML-specific game logic
├── game_manager.py                # Multi-model manager
├── game_data.py                   # ML game data with prediction tracking
├── models/                        # ✨ Organized by data representation needs
│   ├── neural_networks/          # Sequential & spatial data models
│   │   ├── __init__.py
│   │   ├── agent_mlp.py          # 16-feature tabular input
│   │   ├── agent_cnn.py          # 2D spatial board representation
│   │   ├── agent_lstm.py         # Sequential NPZ time series
│   │   └── agent_gru.py          # Sequential NPZ time series
│   ├── tree_models/              # Tabular data specialists
│   │   ├── __init__.py
│   │   ├── agent_xgboost.py      # 16-feature CSV optimal
│   │   ├── agent_lightgbm.py     # 16-feature CSV optimal
│   │   └── agent_randomforest.py # 16-feature CSV optimal
│   └── graph_models/             # Graph structure specialists
│       ├── __init__.py
│       ├── agent_gcn.py          # Graph representation
│       ├── agent_graphsage.py    # Graph representation
│       └── agent_gat.py          # Graph representation
├── training/                      # Training scripts per model type
│   ├── train_neural.py           # Handles multiple data formats
│   ├── train_tree.py             # Optimized for 16-feature CSV
│   └── train_graph.py            # Graph-specific training
└── README.md
```

**Data Representation Strategy by Model Type:**
- **Tree Models**: Use 16-feature CSV from heuristics extensions
- **CNNs**: Convert game state to 2D spatial arrays
- **RNNs**: Use sequential NPZ format with temporal history
- **GNNs**: Create graph representations of game state relationships

### **Reinforcement v0.02**
```
extensions/reinforcement-v0.02/
├── __init__.py
├── main.py                        # --algorithm DQN|PPO|A3C
├── game_logic.py                  # RL-specific game logic
├── game_manager.py                # Multi-algorithm RL manager
├── game_data.py                   # RL game data with experience tracking
├── agents/
│   ├── __init__.py               # RLAgentFactory
│   ├── agent_dqn.py              # Deep Q-Network
│   ├── agent_double_dqn.py       # Double DQN
│   ├── agent_dueling_dqn.py      # Dueling DQN
│   ├── agent_ppo.py              # Proximal Policy Optimization
│   └── agent_a3c.py              # Asynchronous Actor-Critic
├── training/                      # RL training scripts
│   ├── train_dqn.py
│   ├── train_ppo.py
│   └── train_a3c.py
└── README.md
```

## 🏗️ **Shared Infrastructure Patterns**

### **Agent Factory Pattern:**
```python
# ✅ All extensions use factory patterns
def create_agent(algorithm: str) -> BaseAgent:
    """
    Factory Pattern Implementation
    
    Design Pattern: Factory Pattern
    Purpose: Decouple agent creation from client code
    Educational Note: This pattern makes it easy to add new algorithms
    without modifying existing code (Open/Closed Principle)
    """
    agents = {
        'BFS': BFSAgent,
        'ASTAR': AStarAgent,
        'MLP': MLPAgent,
        'DQN': DQNAgent,
    }
    return agents[algorithm]()
```

### **Inheritance Hierarchy:**
```python
# ✅ Natural algorithm evolution through inheritance
class BFSAgent(BaseAgent):
    """
    Foundation BFS implementation
    
    Design Pattern: Template Method Pattern
    Purpose: Provides base BFS algorithm that can be extended
    Educational Note: Inheritance enables algorithm specialization
    while maintaining consistent interface
    """
    pass

class BFSSafeGreedyAgent(BFSAgent):
    """
    Extends BFS with safety checks and greedy optimization
    
    Design Pattern: Decorator Pattern (via inheritance)
    Purpose: Adds safety features without modifying base BFS
    Educational Note: Shows how to enhance algorithms incrementally
    """
    pass

class AStarAgent(BaseAgent):
    """A* pathfinding with heuristics"""
    pass
```

## 🚀 **Evolution Patterns**

### **v0.01 → v0.02 Changes:**

**Heuristics:**
- ✅ **Single algorithm** → **Multi-algorithm suite**
- ✅ **No arguments** → **`--algorithm` parameter**
- ✅ **Simple structure** → **Organized agents folder**
- ✅ **Basic BFS** → **Advanced algorithmic variations**

**Supervised Learning:**
- ✅ **Neural networks only** → **All ML model types**
- ✅ **Single framework** → **Multi-framework support**
- ✅ **Basic training** → **Advanced training pipelines**
- ✅ **Limited evaluation** → **Comprehensive benchmarking**

**Reinforcement Learning:**
- ✅ **Single DQN** → **Multiple RL algorithms**
- ✅ **Basic training** → **Advanced RL training pipelines**
- ✅ **Simple evaluation** → **Comprehensive RL benchmarking**

### **v0.02 → v0.03 Preview:**
- **All**: CLI only → **Streamlit web interface**
- **All**: No replay → **PyGame + Flask web replay**
- **All**: Basic logging → **Dataset generation capabilities**

## 📋 **Implementation Guidelines**

### **Algorithm Inheritance (Heuristics):**
```python
# ✅ Natural progression through inheritance
class BFSAgent(BaseAgent):
    """Foundation BFS - simple and reliable"""
    
class BFSSafeGreedyAgent(BFSAgent):
    """Extends BFS with safety checks and greedy optimization"""
    # Inherits BFS logic, adds safety layer
    
class BFSHamiltonianAgent(BFSSafeGreedyAgent):
    """Adds Hamiltonian path concepts to safe greedy BFS"""
    # Inherits BFS + safety, adds Hamiltonian optimization
```

### **Model Framework Integration (Supervised):**
```python
# ✅ Consistent interface across different ML frameworks
class BaseMLAgent(BaseAgent):
    """Base class for all ML agents"""
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

class XGBoostAgent(BaseMLAgent):
    """XGBoost implementation with consistent interface"""
    
class PyTorchAgent(BaseMLAgent):
    """PyTorch implementation with consistent interface"""
```

## 🎯 **Shared Output Schema**

### **All Extensions Generate:**
- `game_N.json` files with game histories
- `summary.json` with experiment statistics
- **No LLM-specific fields** (removed from Task-0 schema)
- **Algorithm/model metadata** for tracking

### **JSON Schema Example:**
```json
{
  "algorithm": "BFS",  // or "MLP", "XGBOOST", "DQN", etc.
  "score": 15,
  "steps": 120,
  "duration_seconds": 2.5,
  "game_end_reason": "max_steps_reached",
  "detailed_history": {
    "moves": ["UP", "RIGHT", "DOWN", ...],
    "apple_positions": [{"x": 5, "y": 7}, ...],
    "rounds_data": {...}
  }
}
```

## 📚 **Key Success Metrics**

### **For Heuristics v0.02:**
- [ ] **Multiple algorithms** implemented and working
- [ ] **Inheritance relationships** between related algorithms
- [ ] **Performance comparison** capabilities
- [ ] **Consistent command-line interface**
- [ ] **Algorithm-specific optimizations**

### **For Supervised Learning v0.02:**
- [ ] **Multiple ML frameworks** integrated (PyTorch, XGBoost, LightGBM)
- [ ] **Different model architectures** (Neural, Tree, Graph)
- [ ] **Training pipelines** for each model type
- [ ] **Performance evaluation** and comparison
- [ ] **Model persistence** and loading

### **For Reinforcement Learning v0.02:**
- [ ] **Multiple RL algorithms** implemented (DQN, PPO, A3C)
- [ ] **Experience replay** and training mechanisms
- [ ] **Performance evaluation** and comparison
- [ ] **Model persistence** and loading

### **Shared Success Criteria:**
- [ ] **Base class reuse** from Task-0
- [ ] **No GUI components** yet (v0.03 feature)
- [ ] **Clean inheritance patterns**
- [ ] **Factory patterns** for algorithm/model creation
- [ ] **Performance comparison** frameworks

---

**Remember**: v0.02 is about **algorithmic/model diversity** and **natural evolution**. Show how systems grow from simple to sophisticated while maintaining clean architecture.







