# Heuristics as Foundation for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines heuristics as foundation patterns for extensions.

> **See also:** `agents.md`, `core.md`, `final-decision-10.md`, `factory-design-pattern.md`, `config.md`, `csv-schema-1.md`.

## 🎯 **Core Philosophy: Algorithmic Intelligence**

Heuristic algorithms provide the foundational intelligence for Snake game AI, demonstrating how systematic problem-solving approaches can achieve excellent performance. These algorithms serve as both educational tools and practical solutions.

## 🏗️ **Extension Structure**

### **Directory Layout**
```
extensions/heuristics-v0.04/
├── __init__.py
├── agents/
│   ├── __init__.py               # Agent factory
│   ├── agent_bfs.py              # Breadth-First Search
│   ├── agent_astar.py            # A* pathfinding
│   ├── agent_hamiltonian.py      # Hamiltonian cycle
│   ├── agent_dfs.py              # Depth-First Search
│   └── agent_bfs_safe_greedy.py  # Safe greedy BFS
├── algorithms/
│   ├── __init__.py
│   ├── pathfinding.py            # Pathfinding utilities
│   ├── collision_detection.py    # Collision detection
│   └── optimization.py           # Path optimization
├── game_logic.py                 # Heuristic game logic
├── game_manager.py               # Heuristic manager
└── main.py                       # CLI interface
```

## 🔧 **Implementation Patterns**

### **Heuristic Agent Factory**
```python
class HeuristicAgentFactory:
    """
    Simple factory for heuristic agents
    
    Design Pattern: Factory Pattern
    - Simple dictionary-based registry
    - Canonical create() method
    - Easy extension for new algorithms
    """
    
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "HAMILTONIAN": HamiltonianAgent,
        "DFS": DFSAgent,
        "BFS_SAFE_GREEDY": BFSSafeGreedyAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, **kwargs):
        """Create heuristic agent by type (canonical: create())"""
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
        print_info(f"[HeuristicAgentFactory] Creating: {algorithm}")  # Simple logging
        return agent_class(**kwargs)
```

### **BFS Agent Implementation**
```python
class BFSAgent(BaseAgent):
    """
    Breadth-First Search agent for Snake game
    
    Design Pattern: Strategy Pattern
    - Encapsulates BFS pathfinding logic
    - Provides shortest path to apple
    - Handles collision avoidance
    """
    
    def __init__(self, name: str, grid_size: int):
        super().__init__(name, grid_size)
        self.pathfinder = BFSPathfinder(grid_size)
        print_info(f"[BFSAgent] Initialized BFS agent: {name}")
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan move using BFS pathfinding"""
        head = game_state['snake_head']
        apple = game_state['apple_position']
        snake_body = set(game_state['snake_body'])
        
        path = self.pathfinder.find_path(head, apple, snake_body)
        
        if path:
            next_pos = path[1] if len(path) > 1 else path[0]
            move = self._get_direction(head, next_pos)
            print_info(f"[BFSAgent] Found path to apple, moving: {move}")
            return move
        else:
            print_warning("[BFSAgent] No path found, using safe move")
            return self._find_safe_move(head, snake_body)
```

## 📊 **Algorithm Comparison**

### **Performance Characteristics**
| Algorithm | Path Quality | Speed | Memory | Use Case |
|-----------|-------------|-------|--------|----------|
| **BFS** | Optimal | Fast | Medium | General purpose |
| **A*** | Optimal | Very Fast | Low | Large grids |
| **Hamiltonian** | Suboptimal | Fast | Low | Guaranteed survival |
| **DFS** | Variable | Fast | Low | Exploration |
| **BFS Safe Greedy** | Good | Fast | Medium | Safety-focused |

### **Educational Value**
- **BFS**: Demonstrates systematic search and shortest path finding
- **A***: Shows heuristic-guided search optimization
- **Hamiltonian**: Illustrates cycle-based strategies
- **DFS**: Teaches depth-first exploration concepts

## 🚀 **Advanced Features**

### **Path Optimization**
```python
class PathOptimizer:
    """
    Optimize paths for better game performance
    
    Design Pattern: Decorator Pattern
    - Adds optimization to base pathfinding
    - Maintains original algorithm interface
    - Improves path quality without changing core logic
    """
    
    def __init__(self, base_pathfinder):
        self.base_pathfinder = base_pathfinder
        print_info("[PathOptimizer] Initialized path optimizer")
    
    def find_optimized_path(self, start, goal, obstacles):
        """Find and optimize path"""
        base_path = self.base_pathfinder.find_path(start, goal, obstacles)
        if base_path:
            optimized_path = self._optimize_path(base_path, obstacles)
            print_info(f"[PathOptimizer] Optimized path length: {len(optimized_path)}")
            return optimized_path
        return None
```

### **Collision Detection**
```python
class CollisionDetector:
    """
    Advanced collision detection for heuristic agents
    
    Design Pattern: Strategy Pattern
    - Different collision detection strategies
    - Pluggable detection algorithms
    - Efficient collision prediction
    """
    
    def __init__(self, detection_strategy="standard"):
        self.strategy = detection_strategy
        print_info(f"[CollisionDetector] Using strategy: {detection_strategy}")
    
    def predict_collision(self, position, direction, snake_body, grid_size):
        """Predict if move will cause collision"""
        next_pos = self._get_next_position(position, direction)
        
        # Wall collision
        if not (0 <= next_pos[0] < grid_size and 0 <= next_pos[1] < grid_size):
            print_warning(f"[CollisionDetector] Wall collision predicted at {next_pos}")
            return True
        
        # Snake body collision
        if next_pos in snake_body:
            print_warning(f"[CollisionDetector] Body collision predicted at {next_pos}")
            return True
        
        return False
```

## 📋 **Configuration and Usage**

### **Algorithm Selection**
```bash
python main.py --algorithm BFS --grid-size 10 --max-games 5
python main.py --algorithm ASTAR --grid-size 12 --verbose
python main.py --algorithm HAMILTONIAN --grid-size 8
```

### **Educational Applications**
- **Algorithm Comparison**: Side-by-side performance analysis
- **Path Visualization**: Visual representation of search algorithms
- **Performance Metrics**: Quantitative algorithm evaluation
- **Strategy Analysis**: Understanding algorithm decision-making

## 🔗 **Integration with Other Extensions**

### **With Supervised Learning**
- Generate training datasets from heuristic gameplay
- Use heuristic performance as baseline for ML models
- Create hybrid approaches combining heuristics and ML

### **With Reinforcement Learning**
- Use heuristic policies for reward shaping
- Compare RL performance against heuristic baselines
- Create curriculum learning starting with heuristic solutions

### **With Evolutionary Algorithms**
- Use heuristics to evaluate evolved strategies
- Create hybrid evolutionary-heuristic approaches
- Generate diverse training scenarios

## 📊 **Dataset Generation**

### **CSV Dataset Format (v0.03)**
Heuristics generate structured datasets for supervised learning:
- Game state features (16 grid-size agnostic features)
- Action labels (UP, DOWN, LEFT, RIGHT)
- Performance metrics (score, survival time, efficiency)

### **JSONL Dataset Format (v0.04)**
Enhanced datasets with language explanations:
- Natural language descriptions of game states
- Reasoning explanations for actions taken
- Educational annotations for learning

## 🎯 **Best Practices**

### **Algorithm Implementation**
- **Clear Interfaces**: Consistent API across all algorithms
- **Efficient Data Structures**: Optimize for performance
- **Comprehensive Testing**: Validate algorithm correctness
- **Educational Documentation**: Explain algorithm principles

### **Performance Optimization**
- **Memory Management**: Efficient use of data structures
- **Algorithm Tuning**: Optimize parameters for different grid sizes
- **Caching Strategies**: Cache frequently computed paths
- **Parallel Processing**: Utilize multiple cores when possible

---

**Heuristic algorithms provide the foundation for understanding systematic problem-solving in game AI. They demonstrate how algorithmic intelligence can achieve excellent performance while serving as educational tools and practical solutions for the Snake game domain.**
