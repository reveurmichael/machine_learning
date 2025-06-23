# Extensions v0.02 - Multi-Algorithm Expansion

This document serves as the **definitive guideline** for implementing v0.02 extensions across different algorithm types. It demonstrates the evolution from single-algorithm v0.01 to multi-algorithm systems, showing natural software progression.

## 🎯 **Core Philosophy: Algorithm Diversity & Progression**

v0.02 builds upon v0.01's foundation to demonstrate:
- **Natural software evolution**: From proof-of-concept to production-ready systems
- **Multi-algorithm support**: Multiple approaches within each domain
- **Inheritance patterns**: How algorithms can extend and improve upon each other
- **Performance comparison**: Benchmarking different approaches

## 🔧 **Heuristics v0.02 - Multi-Algorithm Suite**

### **Location:** `./extensions/heuristics-v0.02`

### **Key Evolution from v0.01:**
- **Single BFS** → **7 different algorithms**
- **No arguments** → **`--algorithm` argument with choices**
- **Simple structure** → **Organized agents folder**

### **Algorithm Portfolio:**
```
./extensions/heuristics-v0.02/agents/
├── agent_bfs.py                # Pure BFS (same as v0.01)
├── agent_bfs_safe_greedy.py    # BFS + safety heuristics
├── agent_bfs_hamiltonian.py    # BFS + Hamiltonian path concepts
├── agent_dfs.py                # Depth-First Search
├── agent_astar.py              # A* pathfinding
├── agent_astar_hamiltonian.py  # A* + Hamiltonian optimization
└── agent_hamiltonian.py        # Pure Hamiltonian path algorithm
```

### **Inheritance Hierarchy:**
```python
# ✅ Natural algorithm evolution through inheritance
class BFSAgent(BaseAgent):
    """Foundation BFS implementation"""
    pass

class BFSSafeGreedyAgent(BFSAgent):
    """Extends BFS with safety checks and greedy optimization"""
    pass

class BFSHamiltonianAgent(BFSSafeGreedyAgent):
    """Adds Hamiltonian path concepts to safe greedy BFS"""
    pass

class AStarAgent(BaseAgent):
    """A* pathfinding with heuristics"""
    pass

class AStarHamiltonianAgent(AStarAgent):
    """A* enhanced with Hamiltonian optimization"""
    pass
```

### **Command Line Interface:**
```bash
# Choose specific algorithm
python main.py --algorithm BFS --max-games 10 --verbose

# Different algorithms for comparison
python main.py --algorithm ASTAR --max-games 5
python main.py --algorithm HAMILTONIAN --max-games 3
```

### **File Structure:**
```
./extensions/heuristics-v0.02/
├── __init__.py
├── main.py              # Multi-algorithm entry point
├── game_logic.py        # Heuristic-specific game logic
├── game_manager.py      # Multi-algorithm manager
├── game_data.py         # Heuristic game data extensions
└── agents/              # Algorithm implementations
    ├── __init__.py      # Agent factory and exports
    ├── agent_bfs.py
    ├── agent_bfs_safe_greedy.py
    ├── agent_bfs_hamiltonian.py
    ├── agent_dfs.py
    ├── agent_astar.py
    ├── agent_astar_hamiltonian.py
    └── agent_hamiltonian.py
```

### **What It Still Does NOT Have:**
- **No replay mode** (pygame/web) - comes in v0.03
- **No GUI interface** - CLI only
- **No dataset generation** - comes in v0.03
- **No Streamlit app** - comes in v0.03

## 🧠 **Supervised Learning v0.02 - Multi-Model Framework**

### **Location:** `./extensions/supervised-v0.02`

### **Key Evolution from v0.01:**
- **Neural networks only** → **All supervised learning types**
- **PyTorch focus** → **Multiple ML frameworks**
- **Single training script** → **Framework-specific trainers**

### **Model Portfolio:**
```
./extensions/supervised-v0.02/models/
├── neural_networks/     # PyTorch implementations
│   ├── agent_mlp.py
│   ├── agent_cnn.py
│   ├── agent_lstm.py
│   └── agent_gru.py
├── tree_models/         # Tree-based models
│   ├── agent_xgboost.py
│   ├── agent_lightgbm.py
│   └── agent_randomforest.py
└── graph_models/        # Graph neural networks
    ├── agent_gcn.py
    ├── agent_graphsage.py
    └── agent_gat.py
```

### **Framework Integration:**
```python
# ✅ Multiple ML frameworks with consistent interface
class XGBoostAgent(BaseAgent):
    """XGBoost gradient boosting agent"""
    def __init__(self):
        import xgboost as xgb
        self.model = xgb.XGBClassifier()

class LightGBMAgent(BaseAgent):
    """LightGBM gradient boosting agent"""
    def __init__(self):
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier()

class GCNAgent(BaseAgent):
    """Graph Convolutional Network agent"""
    def __init__(self):
        import torch_geometric
        # GCN implementation
```

### **Training Scripts:**
```
./extensions/supervised-v0.02/training/
├── train_neural.py      # PyTorch neural networks
├── train_tree.py        # XGBoost, LightGBM, RandomForest
├── train_graph.py       # Graph neural networks
└── train_ensemble.py    # Ensemble methods
```

### **Command Line Interface:**
```bash
# Train different model types
python training/train_neural.py --model MLP --dataset-path ../../logs/extensions/datasets/grid-size-10/
python training/train_tree.py --model XGBOOST --dataset-path ../../logs/extensions/datasets/grid-size-10/
python training/train_graph.py --model GCN --dataset-path ../../logs/extensions/datasets/grid-size-10/

# Ensemble training
python training/train_ensemble.py --models MLP,XGBOOST,LIGHTGBM --dataset-path ../../logs/extensions/datasets/grid-size-10/
```

### **File Structure:**
```
./extensions/supervised-v0.02/
├── __init__.py
├── main.py              # Model selection and evaluation
├── game_logic.py        # ML-specific game logic
├── game_manager.py      # Multi-model manager
├── models/              # Model implementations
│   ├── neural_networks/
│   ├── tree_models/
│   └── graph_models/
├── training/            # Training scripts
├── evaluation/          # Evaluation utilities
└── utils/               # ML-specific utilities
```

### **What It Still Does NOT Have:**
- **No GUI interface** - CLI only
- **No replay visualization** - comes in v0.03
- **No Streamlit app** - comes in v0.03
- **No web interface** - comes in v0.03

## 🏗️ **Shared Infrastructure Patterns**

### **Agent Factory Pattern:**
```python
# ✅ Both heuristics and supervised learning use factory patterns
def create_heuristic_agent(algorithm: str) -> BaseAgent:
    agents = {
        "BFS": BFSAgent,
        "BFS_SAFE_GREEDY": BFSSafeGreedyAgent,
        "BFS_HAMILTONIAN": BFSHamiltonianAgent,
        "DFS": DFSAgent,
        "ASTAR": AStarAgent,
        "ASTAR_HAMILTONIAN": AStarHamiltonianAgent,
        "HAMILTONIAN": HamiltonianAgent,
    }
    return agents[algorithm]()

def create_supervised_agent(model: str) -> BaseAgent:
    agents = {
        "MLP": MLPAgent,
        "CNN": CNNAgent,
        "LSTM": LSTMAgent,
        "XGBOOST": XGBoostAgent,
        "LIGHTGBM": LightGBMAgent,
        "GCN": GCNAgent,
    }
    return agents[model]()
```

### **Performance Comparison Framework:**
```python
# ✅ Both extensions support algorithm/model comparison
class PerformanceComparator:
    def compare_algorithms(self, algorithms: List[str]) -> Dict[str, Any]:
        results = {}
        for algorithm in algorithms:
            agent = create_agent(algorithm)
            results[algorithm] = self.evaluate_agent(agent)
        return results
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

### **v0.02 → v0.03 Preview:**
- **Both**: CLI only → **Streamlit web interface**
- **Both**: No replay → **PyGame + Flask web replay**
- **Both**: Basic logging → **Dataset generation capabilities**
- **Heuristics**: No dataset output → **CSV dataset generation for ML**
- **Supervised**: Training only → **Interactive training + evaluation interface**

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

### **Both Extensions Generate:**
- `game_N.json` files with game histories
- `summary.json` with experiment statistics
- **No LLM-specific fields** (removed from Task-0 schema)
- **Algorithm/model metadata** for tracking

### **JSON Schema Example:**
```json
{
  "algorithm": "BFS",  // or "MLP", "XGBOOST", etc.
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
- [ ] **7 different algorithms** implemented and working
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

### **Shared Success Criteria:**
- [ ] **Base class reuse** from Task-0
- [ ] **No GUI components** yet (v0.03 feature)
- [ ] **Clean inheritance patterns**
- [ ] **Factory patterns** for algorithm/model creation
- [ ] **Performance comparison** frameworks

---

**Remember**: v0.02 is about **algorithmic/model diversity** and **natural evolution**. Show how systems grow from simple to sophisticated while maintaining clean architecture.







