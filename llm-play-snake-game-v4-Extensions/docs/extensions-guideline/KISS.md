# KISS Principle for Extensions

> **Important — Authoritative Reference:** This document supplements the Final Decision Series and unified guides for extension development standards.

## 🎯 **Core Philosophy: Keep It Simple, Stupid**

The KISS principle is fundamental to the Snake Game AI project. Extensions should be **simple, clear, and focused** rather than complex, clever, and confusing.

## 🏗️ **Simple Architecture Patterns**

### **Factory Patterns**
> **Authoritative Reference**: See `unified-factory-pattern-guide.md` for complete factory implementations.

Following the unified factory pattern guide, factories are deliberately simple:

```python
class HeuristicAgentFactory:
    """Simple factory for heuristic agents"""
    
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, **kwargs) -> BaseAgent:
        """Create agent by name"""
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        return agent_class(**kwargs)
```

### **Configuration Management**
> **Authoritative Reference**: See `config.md` for complete configuration architecture.

Following the configuration standards, configuration is straightforward:

```python
# Simple, clear imports
from config.game_constants import VALID_MOVES, DIRECTIONS
from extensions.common.config.ml_constants import DEFAULT_LEARNING_RATE

# No complex configuration hierarchies
# No dynamic configuration loading
# No configuration inheritance chains
```

### **Path Management**
> **Authoritative Reference**: See `unified-path-management-guide.md` for complete path management standards.

Following the unified path management guide, path management is consistent:

```python
# Simple, standardized path utilities
from extensions.common.path_utils import ensure_project_root, get_dataset_path

# Always start with this
ensure_project_root()

# Use utilities for all paths
dataset_path = get_dataset_path(
    extension_type="heuristics",
    version="0.03",
    grid_size=grid_size,
    algorithm="bfs",
    timestamp=timestamp
)
```

## 📚 **Simple Documentation**

### **Clear, Concise Docstrings**
```python
def plan_move(self, game_state: Dict[str, Any]) -> str:
    """
    Plan next move using BFS algorithm.
    
    Args:
        game_state: Current game state
        
    Returns:
        Next move direction (UP, DOWN, LEFT, RIGHT)
    """
    # Simple, readable implementation
    path = self.find_path(game_state)
    return path[0] if path else "NONE"
```

### **No Over-Documentation**
- **Purpose**: What the function does
- **Parameters**: What it expects
- **Returns**: What it gives back
- **No**: Long explanations, design pattern discussions, educational notes

## 🔧 **Simple Implementation**

### **One Function, One Purpose**
```python
# ✅ Simple and clear
def calculate_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# ❌ Complex and confusing
def calculate_distance_with_optimization_and_caching_and_validation(
    pos1: Tuple[int, int], 
    pos2: Tuple[int, int],
    use_cache: bool = True,
    validate_inputs: bool = True,
    optimization_level: int = 1
) -> int:
    """Calculate distance with many options"""
    # Complex implementation with many branches
```

### **Simple Data Structures**
```python
# ✅ Simple and clear
snake_positions = [(x, y) for x, y in snake_body]
apple_position = (apple_x, apple_y)

# ❌ Complex and confusing
snake_positions = {
    "body": [{"x": x, "y": y, "segment_id": i} for i, (x, y) in enumerate(snake_body)],
    "metadata": {"length": len(snake_body), "head_index": 0}
}
```

## 🎯 **Simple Extension Structure**

### **Minimal File Count**
```
extensions/heuristics-v0.02/
├── agents/
│   ├── __init__.py
│   ├── agent_bfs.py
│   └── agent_astar.py
├── game_logic.py
├── game_manager.py
└── main.py
```

### **No Unnecessary Abstractions**
- **No**: Abstract base classes for simple functionality
- **No**: Complex inheritance hierarchies
- **No**: Multiple layers of indirection
- **Yes**: Direct, clear implementations

## 🚫 **What to Avoid**

### **Over-Engineering**
- **No**: Complex design patterns for simple problems
- **No**: Multiple inheritance for single-purpose classes
- **No**: Dynamic dispatch when static is sufficient
- **No**: Plugin architectures for fixed functionality

### **Premature Optimization**
- **No**: Caching for rarely-used functions
- **No**: Complex algorithms for small datasets
- **No**: Memory optimization for simple data structures
- **No**: Performance tuning before profiling

### **Over-Abstraction**
- **No**: Generic interfaces for specific use cases
- **No**: Configuration systems for fixed values
- **No**: Factory patterns for single implementations
- **No**: Strategy patterns for simple algorithms

## ✅ **What to Embrace**

### **Clear, Direct Code**
- **Yes**: Simple functions with clear names
- **Yes**: Direct data access without getters/setters
- **Yes**: Explicit error handling
- **Yes**: Readable variable names

### **Minimal Dependencies**
- **Yes**: Standard library when possible
- **Yes**: Single-purpose external libraries
- **Yes**: Clear dependency boundaries
- **Yes**: Minimal import statements

### **Straightforward Testing**
- **Yes**: Simple unit tests
- **Yes**: Clear test names
- **Yes**: Minimal test setup
- **Yes**: Direct assertions

## 🔗 **See Also**

- **`config.md`**: Simple configuration patterns
- **`unified-path-management-guide.md`**: Simple path management
- **`unified-factory-pattern-guide.md`**: Simple factory patterns
- **`final-decision-10.md`**: GOOD_RULES - Authoritative, comprehensive reference

---

**Remember: The best code is the code that's easiest to understand, maintain, and extend. Keep it simple.**
