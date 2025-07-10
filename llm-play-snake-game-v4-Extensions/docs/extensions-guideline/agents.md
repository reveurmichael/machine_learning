# Agent Implementation Standards for Snake Game AI Extensions

## 🎯 **Core Philosophy: Algorithmic Decision Making**

Agents in the Snake Game AI project represent the core decision-making components that determine how the snake moves through the game environment. Each agent implements a specific algorithm or approach to solving the pathfinding and decision-making challenges of the Snake game, strictly following SUPREME_RULES from `final-decision.md`.

### **Educational Value**
- **Algorithm Implementation**: Clear examples of different algorithmic approaches
- **Design Pattern Demonstration**: Factory patterns, strategy patterns, and inheritance
- **Comparative Analysis**: Easy comparison between different agent types
- **Extensibility**: Framework for implementing new algorithms
- **SUPREME_RULES Compliance**: Professional software engineering standards from `final-decision.md`

## 🏗️ **Factory Pattern: Canonical Method is create()**

All agent factories must use the canonical method name `create()` for instantiation, not `create_agent()` or any other variant. This ensures consistency and aligns with the KISS principle and SUPREME_RULES from `final-decision.md`.

### **Reference Implementation**

A generic, educational `SimpleFactory` is provided in `utils/factory_utils.py`:

```python
from utils.factory_utils import SimpleFactory

class MyAgent:
    def __init__(self, name):
        self.name = name

factory = SimpleFactory()
factory.register("myagent", MyAgent)
agent = factory.create("myagent", name="TestAgent")  # CANONICAL create() method per SUPREME_RULES
print_info(f"Agent name: {agent.name}")  # SUPREME_RULES compliant logging
```

### **Example Agent Factory**

```python
class AgentFactory:
    """
    Factory following SUPREME_RULES from `final-decision.md`
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create agents using canonical create() method
    Educational Value: Shows how SUPREME_RULES apply consistently across extensions
    """
    
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, **kwargs):  # CANONICAL create() method per SUPREME_RULES
        """Create agent using canonical create() method following SUPREME_RULES from `final-decision.md`"""
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
        print_info(f"[AgentFactory] Creating agent: {algorithm}")  # SUPREME_RULES compliant logging
        return agent_class(**kwargs)
```

## 🚀 **Agent Implementation Examples**

### **Heuristic Agent Implementation**

```python
class BFSAgent(BaseAgent):
    """
    Breadth-First Search agent for pathfinding.
    
    Design Pattern: Strategy Pattern
    Purpose: Implements BFS algorithm for optimal pathfinding
    Educational Value: Shows classic graph search algorithm implementation
    following SUPREME_RULES from final-decision.md
    
    Benefits: Guarantees shortest path, handles collision avoidance
    Trade-offs: Higher memory usage compared to DFS
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("BFS", config)
        self.path = []
        print_info(f"[BFSAgent] Initialized BFS agent")  # SUPREME_RULES compliant logging
    
    def get_move(self, game_state: Dict[str, Any]) -> str:
        """Get next move using BFS pathfinding"""
        snake_positions = game_state['snake_positions']
        apple_position = game_state['apple_position']
        grid_size = game_state['grid_size']
        
        print_info(f"[BFSAgent] Getting move from {snake_positions[0]} to {apple_position}")  # SUPREME_RULES compliant logging
        
        # Find path to apple using BFS
        path = self._bfs_pathfinding(snake_positions[0], apple_position, snake_positions, grid_size)
        
        if path:
            self.path = path
            next_pos = path[1] if len(path) > 1 else path[0]
            move = self._get_direction(snake_positions[0], next_pos)
            print_info(f"[BFSAgent] Selected move: {move}")  # SUPREME_RULES compliant logging
            return move
        else:
            # No path found, use fallback strategy
            fallback_move = self._fallback_move(snake_positions, grid_size)
            print_warning(f"[BFSAgent] No path found, using fallback: {fallback_move}")  # SUPREME_RULES compliant logging
            return fallback_move
```
```
