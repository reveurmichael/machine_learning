# Agent Implementation Standards for Snake Game AI Extensions

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for agent implementation standards and supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`).

> **See also:** `core.md`, `factory-design-pattern.md`, `final-decision-10.md`, `standalone.md`.

## 🎯 **Core Philosophy: Algorithmic Decision Making**

Agents in the Snake Game AI project represent the core decision-making components that determine how the snake moves through the game environment. Each agent implements a specific algorithm or approach to solving the pathfinding and decision-making challenges of the Snake game, strictly following SUPREME_RULES from `final-decision-10.md`.

### **Educational Value**
- **Algorithm Implementation**: Clear examples of different algorithmic approaches
- **Design Pattern Demonstration**: Factory patterns, strategy patterns, and inheritance
- **Comparative Analysis**: Easy comparison between different agent types
- **Extensibility**: Framework for implementing new algorithms
- **SUPREME_RULES Compliance**: Professional software engineering standards from `final-decision-10.md`

## 🏗️ **Factory Pattern: Canonical Method is create()**

All agent factories must use the canonical method name `create()` for instantiation, not `create_agent()` or any other variant. This ensures consistency and aligns with the KISS principle and SUPREME_RULES from `final-decision-10.md`.

### Reference Implementation

```python
from utils.factory_utils import SimpleFactory

class MyAgent:
    def __init__(self, name):
        self.name = name

factory = SimpleFactory()
factory.register("myagent", MyAgent)
agent = factory.create("myagent", name="TestAgent")  # CANONICAL create() method per SUPREME_RULES
print(agent.name)  # Output: TestAgent
```

### Example Agent Factory

```python
class HeuristicAgentFactory:
    """
    Factory following SUPREME_RULES from final-decision-10.md
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create heuristic agents using canonical create() method
    Educational Value: Shows how SUPREME_RULES apply to agent creation systems
    """
    
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, **kwargs):  # CANONICAL create() method per SUPREME_RULES
        """Create heuristic agent using canonical create() method following SUPREME_RULES from final-decision-10.md"""
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
        print(f"[HeuristicAgentFactory] Creating agent: {algorithm}")  # SUPREME_RULES compliant logging
        return agent_class(**kwargs)
```

## 🏗️ **Base Agent Interface**

All agents must implement a standardized interface that enables consistent integration with the game framework:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """
    Base interface for all agents across extensions.
    
    Design Pattern: Strategy Pattern
    Purpose: Defines common agent interface enabling runtime algorithm selection
    Educational Value: Shows how strategy pattern enables interchangeable algorithms
    while maintaining consistent interface following SUPREME_RULES from final-decision-10.md
    
    All subclasses must implement plan_move() method and follow SUPREME_RULES.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.stats = {}
        print(f"[BaseAgent] Initialized {name} agent")  # SUPREME_RULES compliant logging
    
    @abstractmethod
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """
        Plan the next move based on current game state.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Move direction ('UP', 'DOWN', 'LEFT', 'RIGHT')
            
        Note: All implementations must follow SUPREME_RULES from final-decision-10.md
        """
        pass
    
    def update_stats(self, game_state: Dict[str, Any], move: str, result: str):
        """Update agent statistics based on move result"""
        if 'moves_made' not in self.stats:
            self.stats['moves_made'] = 0
        self.stats['moves_made'] += 1
        
        if result == 'APPLE_EATEN':
            if 'apples_eaten' not in self.stats:
                self.stats['apples_eaten'] = 0
            self.stats['apples_eaten'] += 1
        elif result == 'GAME_OVER':
            if 'games_lost' not in self.stats:
                self.stats['games_lost'] = 0
            self.stats['games_lost'] += 1
        
        print(f"[{self.name}] Stats updated: {result}")  # SUPREME_RULES compliant logging
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current agent statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset agent statistics"""
        self.stats = {}
        print(f"[BaseAgent] Reset stats for {self.name}")  # SUPREME_RULES compliant logging
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
    following SUPREME_RULES from final-decision-10.md
    
    Benefits: Guarantees shortest path, handles collision avoidance
    Trade-offs: Higher memory usage compared to DFS
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("BFS", config)
        self.path = []
        print(f"[BFSAgent] Initialized BFS agent")  # SUPREME_RULES compliant logging
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan next move using BFS pathfinding"""
        snake_positions = game_state['snake_positions']
        apple_position = game_state['apple_position']
        grid_size = game_state['grid_size']
        
        print(f"[BFSAgent] Planning move from {snake_positions[0]} to {apple_position}")  # SUPREME_RULES compliant logging
        
        # Find path to apple using BFS
        path = self._bfs_pathfinding(snake_positions[0], apple_position, snake_positions, grid_size)
        
        if path:
            self.path = path
            next_pos = path[1] if len(path) > 1 else path[0]
            move = self._get_direction(snake_positions[0], next_pos)
            print(f"[BFSAgent] Selected move: {move}")  # SUPREME_RULES compliant logging
            return move
        else:
            # No path found, use fallback strategy
            fallback_move = self._fallback_move(snake_positions, grid_size)
            print(f"[BFSAgent] No path found, using fallback: {fallback_move}")  # SUPREME_RULES compliant logging
            return fallback_move
```

## 📊 **Agent Performance Standards**

### **Performance Metrics**
All agents should track and report these standard metrics:
- **Success Rate**: Percentage of games completed successfully
- **Average Score**: Mean score across all games
- **Average Steps**: Mean number of steps per game
- **Efficiency**: Score per step ratio
- **Algorithm-Specific Metrics**: Custom metrics for each agent type

## 📋 **Implementation Checklist**

### **Required Components (SUPREME_RULES Compliant)**
- [ ] **Base Agent Interface**: Implements `BaseAgent` abstract class
- [ ] **Factory Pattern**: Uses canonical `create()` method per `final-decision-10.md`
- [ ] **Move Planning**: Implements `plan_move()` method
- [ ] **Statistics Tracking**: Implements `update_stats()` and `get_stats()`
- [ ] **Configuration**: Supports configurable parameters
- [ ] **Error Handling**: Graceful handling of edge cases
- [ ] **Documentation**: Clear docstrings and comments
- [ ] **SUPREME_RULES Logging**: Uses print() statements for debugging

### **Quality Standards (`final-decision-10.md` Compliance)**
- [ ] **Algorithm Correctness**: Implements algorithm accurately
- [ ] **Performance**: Meets performance benchmarks
- [ ] **Robustness**: Handles edge cases gracefully
- [ ] **Extensibility**: Easy to extend and modify
- [ ] **Educational Value**: Clear and understandable implementation
- [ ] **Design Patterns**: Proper use of documented design patterns

---

**Agent implementation standards ensure consistent, high-quality algorithmic decision-making across all Snake Game AI extensions while maintaining strict `final-decision-10.md` SUPREME_RULES compliance. By following these standards, developers can create robust, educational, and performant agents that integrate seamlessly with the overall framework.**

## 🔗 **See Also**

- **`core.md`**: Base class architecture and inheritance patterns
- **`factory-design-pattern.md`**: Factory pattern implementation guide
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`standalone.md`**: Standalone principle and extension independence

