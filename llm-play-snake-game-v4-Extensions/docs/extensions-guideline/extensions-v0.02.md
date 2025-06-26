# Extensions v0.02: Multi-Algorithm Architecture & Command Line Interface

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0` → `final-decision-10`) and builds upon `extensions-v0.01.md` foundational patterns.

## 🎯 **Core Philosophy: Algorithmic Diversity & Command Line Interface**

Extensions v0.02 represent the **algorithmic expansion phase**, transforming the single-algorithm proof of concept from v0.01 into comprehensive multi-algorithm platforms. This version introduces sophisticated command-line interfaces while maintaining the foundational architectural patterns established in v0.01.

### **Key Advancement Areas**
- **Algorithm Diversity**: Multiple algorithms per extension type
- **Command Line Interface**: Comprehensive argument parsing and configuration
- **Factory Pattern Integration**: Dynamic algorithm selection at runtime
- **Performance Benchmarking**: Comparative analysis across algorithms
- **Enhanced Logging**: Detailed algorithm-specific metrics and outputs

## 🏗️ **Universal v0.02 Characteristics**

All v0.02 extensions share these enhanced capabilities:

### **Command Line Interface**
- **Algorithm Selection**: `--algorithm` parameter for runtime choice
- **Configuration Control**: `--grid-size`, `--max-games`, `--max-steps` parameters
- **Output Management**: `--log-dir`, `--verbose` flags for detailed control
- **Performance Options**: `--benchmark`, `--compare` modes for analysis

### **Factory Pattern Implementation**
- **Dynamic Algorithm Creation**: Runtime instantiation based on user selection
- **Consistent Interfaces**: Uniform agent contracts across all algorithm types
- **Extensible Registry**: Easy addition of new algorithms without architectural changes
- **Error Handling**: Graceful management of invalid algorithm selections

### **Enhanced Logging**
- **Algorithm-Specific Metrics**: Performance data unique to each approach
- **Comparative Analysis**: Cross-algorithm benchmarking capabilities
- **Detailed Debugging**: Verbose logging for algorithm behavior analysis
- **Structured Output**: JSON schemas supporting multiple algorithm types

## 🔧 **Algorithm-Specific Implementations**

### **Heuristics v0.02: Comprehensive Pathfinding Suite**

**Location**: `./extensions/heuristics-v0.02`

**Supported Algorithms**: BFS, A*, DFS, Hamiltonian Cycle

#### **Directory Structure**
```
./extensions/heuristics-v0.02/
├── __init__.py                     # Package initialization
├── main.py                         # CLI entry point with argument parsing
├── agents/                         # 📁 Agent directory structure
│   ├── __init__.py                 # Agent package initialization
│   ├── agent_bfs.py                # Breadth-First Search implementation
│   ├── agent_astar.py              # A* pathfinding algorithm
│   ├── agent_dfs.py                # Depth-First Search implementation
│   └── agent_hamiltonian.py        # Hamiltonian cycle pathfinding
├── game_logic.py                   # Extended pathfinding game logic
├── game_manager.py                 # Enhanced heuristic game management
└── README.md                       # Comprehensive documentation
```

#### **Command Line Interface**
```bash
# BFS pathfinding with custom parameters
python main.py --algorithm BFS --grid-size 15 --max-games 5 --verbose

# A* algorithm with performance benchmarking
python main.py --algorithm ASTAR --grid-size 20 --benchmark --max-steps 1000

# Hamiltonian cycle generation with detailed logging
python main.py --algorithm HAMILTONIAN --log-dir ./custom_logs --verbose

# Compare all algorithms on standard configuration
python main.py --compare --grid-size 10 --max-games 3
```

#### **Factory Pattern Implementation**
```python
class HeuristicAgentFactory:
    """
    Factory Pattern for Dynamic Heuristic Agent Creation
    
    Design Pattern: Factory Pattern + Strategy Pattern
    Purpose: Enable runtime algorithm selection with consistent interfaces
    Educational Value: Demonstrates how factory patterns support the
    Open/Closed Principle by allowing new algorithms without modifying
    existing client code.
    """
    
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent, 
        "DFS": DFSAgent,
        "HAMILTONIAN": HamiltonianAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, grid_size: int = 10, **kwargs) -> BaseAgent:
        """
        Create heuristic agent by algorithm name
        
        Args:
            algorithm: Algorithm identifier (BFS, ASTAR, DFS, HAMILTONIAN)
            grid_size: Game board dimensions
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Configured agent instance
            
        Raises:
            ValueError: If algorithm not supported
        """
        if algorithm not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
        
        agent_class = cls._registry[algorithm]
        return agent_class(grid_size=grid_size, **kwargs)
    
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """Return list of supported algorithm names"""
        return list(cls._registry.keys())
```

#### **Algorithm Implementations**

**BFS Agent (Breadth-First Search)**
```python
class BFSAgent(BaseAgent):
    """
    Breadth-First Search pathfinding agent
    
    Algorithm: Explores all paths level by level, guaranteeing shortest path
    Time Complexity: O(V + E) where V is vertices, E is edges
    Space Complexity: O(V) for the queue storage
    
    Educational Value:
    Demonstrates classic graph traversal algorithm with guaranteed
    optimality at the cost of potentially high memory usage.
    """
    
    def find_path(self, start: Position, goal: Position, obstacles: Set[Position]) -> List[Direction]:
        """Find optimal path using BFS algorithm"""
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current_pos, path = queue.popleft()
            
            if current_pos == goal:
                return path
                
            for direction in DIRECTIONS:
                next_pos = self._apply_direction(current_pos, direction)
                
                if (next_pos not in visited and 
                    next_pos not in obstacles and 
                    self._is_valid_position(next_pos)):
                    
                    visited.add(next_pos)
                    queue.append((next_pos, path + [direction]))
        
        return []  # No path found
```

**A* Agent (A-Star Algorithm)**
```python
class AStarAgent(BaseAgent):
    """
    A* pathfinding agent with Manhattan distance heuristic
    
    Algorithm: Best-first search using f(n) = g(n) + h(n) evaluation
    Time Complexity: O(b^d) where b is branching factor, d is depth
    Space Complexity: O(b^d) for the open/closed sets
    
    Educational Value:
    Shows how heuristic functions can guide search efficiency while
    maintaining optimality guarantees (when heuristic is admissible).
    """
    
    def find_path(self, start: Position, goal: Position, obstacles: Set[Position]) -> List[Direction]:
        """Find optimal path using A* algorithm with Manhattan heuristic"""
        open_set = []
        heapq.heappush(open_set, (0, start, []))
        
        g_scores = {start: 0}
        visited = set()
        
        while open_set:
            f_score, current_pos, path = heapq.heappop(open_set)
            
            if current_pos in visited:
                continue
                
            visited.add(current_pos)
            
            if current_pos == goal:
                return path
            
            for direction in DIRECTIONS:
                next_pos = self._apply_direction(current_pos, direction)
                
                if (next_pos not in obstacles and 
                    self._is_valid_position(next_pos) and
                    next_pos not in visited):
                    
                    tentative_g = g_scores[current_pos] + 1
                    
                    if next_pos not in g_scores or tentative_g < g_scores[next_pos]:
                        g_scores[next_pos] = tentative_g
                        h_score = self._manhattan_distance(next_pos, goal)
                        f_score = tentative_g + h_score
                        
                        heapq.heappush(open_set, (f_score, next_pos, path + [direction]))
        
        return []  # No path found
    
    def _manhattan_distance(self, pos1: Position, pos2: Position) -> int:
        """Calculate Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
```

### **Supervised Learning v0.02: Comprehensive ML Suite**

**Location**: `./extensions/supervised-v0.02`

**Supported Models**: Neural Networks, Tree Models, Ensemble Methods

#### **Directory Structure**
```
./extensions/supervised-v0.02/
├── __init__.py                     # Package initialization
├── main.py                         # CLI with model selection
├── models/                         # 📁 Model directory structure  
│   ├── __init__.py                 # Model package initialization
│   ├── neural_networks/            # 📁 Neural network implementations
│   │   ├── __init__.py
│   │   ├── agent_mlp.py            # Multi-Layer Perceptron
│   │   ├── agent_cnn.py            # Convolutional Neural Network
│   │   └── agent_lstm.py           # Long Short-Term Memory
│   └── tree_models/                # 📁 Tree-based model implementations
│       ├── __init__.py
│       ├── agent_xgboost.py        # XGBoost classifier
│       ├── agent_lightgbm.py       # LightGBM classifier
│       └── agent_randomforest.py   # Random Forest classifier
├── train.py                        # Enhanced training pipeline
├── evaluate.py                     # Model evaluation utilities
├── game_logic.py                   # ML-adapted game logic
├── game_manager.py                 # Model evaluation manager
└── README.md                       # Comprehensive documentation
```

#### **Command Line Interface**
```bash
# Train MLP on tabular features
python train.py --model MLP --dataset ../heuristics-v0.02/datasets/bfs_games.csv \
    --epochs 100 --batch-size 32 --learning-rate 0.001

# Train XGBoost with hyperparameter tuning
python train.py --model XGBOOST --dataset ../heuristics-v0.02/datasets/astar_games.csv \
    --n-estimators 100 --max-depth 6 --tune-hyperparameters

# Evaluate trained CNN model
python evaluate.py --model CNN --model-path ./models/cnn_model.pth \
    --test-dataset ../heuristics-v0.02/datasets/test_games.csv

# Compare all models on benchmark dataset
python main.py --compare-models --benchmark-dataset ./datasets/benchmark.csv
```

#### **Model Factory Implementation**
```python
class SupervisedModelFactory:
    """
    Factory Pattern for ML Model Creation
    
    Design Pattern: Abstract Factory Pattern
    Purpose: Create different types of ML models with consistent interfaces
    Educational Value: Shows how factory patterns can manage complex
    object hierarchies (neural networks vs tree models) through abstraction.
    """
    
    _registry = {
        # Neural Network Models
        "MLP": MLPAgent,
        "CNN": CNNAgent,
        "LSTM": LSTMAgent,
        
        # Tree-Based Models
        "XGBOOST": XGBoostAgent,
        "LIGHTGBM": LightGBMAgent,
        "RANDOMFOREST": RandomForestAgent,
    }
    
    @classmethod
    def create(cls, model_type: str, input_dim: int, **kwargs) -> BaseMLAgent:
        """
        Create ML model by type
        
        Args:
            model_type: Model architecture identifier
            input_dim: Input feature dimensionality
            **kwargs: Model-specific hyperparameters
            
        Returns:
            Configured model instance
        """
        if model_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown model: {model_type}. Available: {available}")
        
        model_class = cls._registry[model_type]
        return model_class(input_dim=input_dim, **kwargs)
```

### **Reinforcement Learning v0.02: Multi-Algorithm RL Suite**

**Location**: `./extensions/reinforcement-v0.02`

**Supported Algorithms**: DQN, PPO, A3C, DDPG

#### **Directory Structure**
```
./extensions/reinforcement-v0.02/
├── __init__.py                     # Package initialization
├── main.py                         # CLI with algorithm selection
├── agents/                         # 📁 RL agent implementations
│   ├── __init__.py                 # Agent package initialization
│   ├── agent_dqn.py                # Deep Q-Network
│   ├── agent_ppo.py                # Proximal Policy Optimization
│   ├── agent_a3c.py                # Asynchronous Advantage Actor-Critic
│   └── agent_ddpg.py               # Deep Deterministic Policy Gradient
├── train.py                        # RL training pipeline
├── evaluate.py                     # Agent evaluation utilities
├── environments/                   # 📁 Environment wrappers
│   ├── __init__.py
│   └── snake_env.py                # Gymnasium-compatible Snake environment
├── game_logic.py                   # RL-adapted game logic
├── game_manager.py                 # RL training manager
└── README.md                       # Comprehensive documentation
```

#### **Command Line Interface**
```bash
# Train DQN with experience replay
python train.py --algorithm DQN --episodes 10000 --learning-rate 0.0001 \
    --epsilon-decay 0.995 --replay-buffer-size 100000

# Train PPO with custom hyperparameters
python train.py --algorithm PPO --episodes 5000 --batch-size 64 \
    --clip-epsilon 0.2 --entropy-coef 0.01

# Evaluate trained A3C agent
python evaluate.py --algorithm A3C --model-path ./models/a3c_model.pth \
    --num-episodes 100 --render

# Compare RL algorithms performance
python main.py --compare-algorithms --episodes 1000 --grid-size 10
```

## 🧠 **Advanced Design Pattern Integration**

### **Command Line Architecture**
All v0.02 extensions implement sophisticated argument parsing:

```python
class ExtensionArgumentParser:
    """
    Standardized argument parsing for v0.02 extensions
    
    Design Pattern: Builder Pattern
    Purpose: Construct complex command-line interfaces incrementally
    Educational Value: Shows how builder pattern can create flexible,
    maintainable configuration systems.
    """
    
    def __init__(self, extension_type: str):
        self.extension_type = extension_type
        self.parser = argparse.ArgumentParser(
            description=f"{extension_type} v0.02 - Multi-algorithm implementation"
        )
        self._setup_common_arguments()
        self._setup_extension_specific_arguments()
    
    def _setup_common_arguments(self):
        """Add arguments common to all extensions"""
        self.parser.add_argument("--grid-size", type=int, default=10,
                               help="Game board size (default: 10)")
        self.parser.add_argument("--max-games", type=int, default=1,
                               help="Maximum number of games (default: 1)")
        self.parser.add_argument("--max-steps", type=int, default=1000,
                               help="Maximum steps per game (default: 1000)")
        self.parser.add_argument("--log-dir", type=str, default="./logs",
                               help="Logging directory (default: ./logs)")
        self.parser.add_argument("--verbose", action="store_true",
                               help="Enable verbose logging")
    
    def _setup_extension_specific_arguments(self):
        """Add extension-specific arguments (implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement extension-specific arguments")
```

### **Performance Benchmarking Framework**
```python
class PerformanceBenchmark:
    """
    Comprehensive benchmarking framework for algorithm comparison
    
    Design Pattern: Observer Pattern
    Purpose: Monitor and report algorithm performance metrics
    Educational Value: Demonstrates how observer pattern enables
    decoupled monitoring and metric collection systems.
    """
    
    def __init__(self, algorithms: List[str]):
        self.algorithms = algorithms
        self.results = {}
        self.observers = []
    
    def add_observer(self, observer: BenchmarkObserver):
        """Add performance monitoring observer"""
        self.observers.append(observer)
    
    def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResults:
        """Execute comprehensive algorithm benchmark"""
        for algorithm in self.algorithms:
            print(f"Benchmarking {algorithm}...")
            
            # Run algorithm with specified configuration
            agent = self._create_agent(algorithm, config)
            metrics = self._execute_benchmark(agent, config)
            
            self.results[algorithm] = metrics
            
            # Notify observers of results
            for observer in self.observers:
                observer.on_algorithm_completed(algorithm, metrics)
        
        return BenchmarkResults(self.results)
```

## 📊 **Enhanced Logging and Analytics**

### **Algorithm-Specific Metrics**
Each algorithm type generates specialized performance data:

**Heuristics Metrics**:
- Path length optimality
- Search space exploration
- Algorithm execution time
- Memory usage patterns

**Supervised Learning Metrics**:
- Training/validation accuracy
- Loss convergence patterns
- Inference time performance
- Model complexity measures

**Reinforcement Learning Metrics**:
- Episode reward trends
- Exploration vs exploitation balance
- Q-value convergence
- Policy gradient magnitudes

### **Comparative Analysis Output**
```json
{
  "benchmark_results": {
    "configuration": {
      "grid_size": 10,
      "max_games": 5,
      "max_steps": 1000
    },
    "algorithms": {
      "BFS": {
        "average_score": 45.2,
        "average_path_length": 12.8,
        "execution_time_ms": 15.3,
        "success_rate": 0.95
      },
      "ASTAR": {
        "average_score": 47.8,
        "average_path_length": 11.2,
        "execution_time_ms": 8.7,
        "success_rate": 0.98
      }
    },
    "ranking": ["ASTAR", "BFS"],
    "summary": "A* demonstrates superior performance with 6% higher scores and 43% faster execution"
  }
}
```

## 📋 **Implementation Standards**

### **Universal v0.02 Requirements**
- [ ] **Multi-Algorithm Support**: Minimum 3 algorithms per extension
- [ ] **Factory Pattern**: Dynamic algorithm creation and selection
- [ ] **CLI Interface**: Comprehensive argument parsing with help text
- [ ] **Performance Benchmarking**: Cross-algorithm comparison capabilities
- [ ] **Enhanced Logging**: Algorithm-specific metrics and detailed output
- [ ] **Error Handling**: Graceful handling of invalid configurations
- [ ] **Documentation**: Clear usage examples and algorithm descriptions

### **Heuristics-Specific Requirements**
- [ ] **BFS Implementation**: Breadth-first search pathfinding
- [ ] **A* Implementation**: A-star with admissible heuristic
- [ ] **DFS Implementation**: Depth-first search variant
- [ ] **Hamiltonian Implementation**: Cycle-based path generation
- [ ] **Pathfinding Metrics**: Path optimality and search efficiency tracking

### **Supervised Learning Requirements**
- [ ] **Neural Networks**: MLP, CNN, LSTM implementations
- [ ] **Tree Models**: XGBoost, LightGBM, Random Forest support
- [ ] **Training Pipeline**: Complete train/validate/test workflow
- [ ] **Model Persistence**: Save/load trained model functionality
- [ ] **Hyperparameter Tuning**: Automated parameter optimization

### **Reinforcement Learning Requirements**
- [ ] **Value-Based**: DQN implementation with experience replay
- [ ] **Policy-Based**: PPO with clipped surrogate objective
- [ ] **Actor-Critic**: A3C with parallel environment workers
- [ ] **Continuous Control**: DDPG for potential continuous variants
- [ ] **Environment Integration**: Gymnasium-compatible Snake environment

## 🚀 **Evolution Preview: v0.02 → v0.03**

### **Web Interface Integration**
v0.03 will transform command-line interfaces into sophisticated Streamlit dashboards while preserving all v0.02 functionality.

### **Dataset Generation Capabilities**
v0.03 will add comprehensive dataset generation for cross-extension training and evaluation.

### **Enhanced Visualization**
v0.03 will introduce real-time algorithm visualization and interactive performance analysis.

---

**Extensions v0.02 establish the multi-algorithm foundation that demonstrates the power and flexibility of the Snake Game AI architecture. They prove that sophisticated algorithmic diversity can coexist with clean, maintainable code through proper application of design patterns and architectural principles.**







