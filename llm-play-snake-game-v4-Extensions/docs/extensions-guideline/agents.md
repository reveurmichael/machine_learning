> **Important — Authoritative Reference:** This guide is **supplementary** to the _Final Decision Series_ (`final-decision-0` → `final-decision-10`). **If any statement here conflicts with a Final Decision document, the latter always prevails.**

# Agents Architecture and Implementation Guide

This document provides comprehensive guidelines for implementing and organizing AI agents across all Snake Game AI extensions, covering the progression from v0.01 to v0.04 versions.

## 🎯 **Agent Organization Overview**

The `agents/` folder structure represents the evolution from proof-of-concept implementations to sophisticated multi-algorithm systems:

### **Extension Version Progression:**
- **v0.01**: Single agent in extension root (proof-of-concept)  
- **v0.02**: Organized `agents/` package with multiple algorithms
- **v0.03**: Enhanced `agents/` with dashboard integration  
- **v0.04**: Advanced features with JSONL trajectory generation (heuristics only)

### **Directory Structure Rules:**

For extensions named `[Algorithm]-v0.0N`:

#### **v0.01 Structure (Proof-of-Concept)**
- Agents go directly in: `./extensions/[Algorithm]-v0.01/`
- Single-file agent implementations
- Basic game integration

#### **v0.02+ Structure (Multi-Algorithm)** 
- Agents go in: `./extensions/[Algorithm]-v0.0N/agents/`
- Where N ≥ 2 (v0.02, v0.03, and for heuristics: v0.04)
- Organized package structure with multiple agents
- Advanced features and integrations

### **Folder Structure Examples:**

#### **v0.01 Layout**
```
extensions/heuristics-v0.01/
├── agent_bfs.py           # BFS agent implementation
├── game_data.py          # Basic game data handling
├── game_logic.py         # Basic game logic
├── game_manager.py       # Simple game management
└── README.md            # Basic documentation
```

#### **v0.02+ Layout**
```
extensions/heuristics-v0.02/
├── __init__.py           # Extension configuration and factory
├── agents/               # Agent implementations package
│   ├── __init__.py      # Agent protocol and base classes
│   ├── agent_bfs.py     # BFS agent implementation
│   ├── agent_astar.py   # A* agent implementation
│   └── agent_hamiltonian.py # Hamiltonian agent
├── game_data.py         # Extended game data management
├── game_logic.py        # Enhanced game logic
├── game_manager.py      # Advanced game session management
├── scripts/             # CLI and automation scripts
└── README.md           # Comprehensive documentation
```

For v0.03, add the folder "dashboard"

## 🏗️ **Agent Implementation Standards**

### **Common Agent Protocol**
All agents across extensions must implement a consistent interface by extending the BaseAgent from the core framework:

```python
# All agents must extend BaseAgent from core framework
from core.game_agents import BaseAgent

class HeuristicAgent(BaseAgent):
    """
    Base class for all heuristic agents
    
    Design Pattern: Template Method Pattern
    - Defines common structure for all heuristic agents
    - Subclasses implement specific algorithms
    - Ensures consistent interface across all heuristics
    
    Philosophy:
    - All agents share common lifecycle methods
    - Algorithm-specific logic is isolated in abstract methods
    - Consistent error handling and logging
    """
    
    def __init__(self, name: str, grid_size: int):
        """
        Initialize heuristic agent
        
        Args:
            name: Unique identifier for the agent
            grid_size: Size of the game grid
        """
        super().__init__(name, grid_size)
        self.algorithm_name = name
        self.search_depth = 0
        self.nodes_explored = 0
    
    @abstractmethod
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """
        Plan next move based on current game state
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Move direction ('UP', 'DOWN', 'LEFT', 'RIGHT')
            
        Raises:
            ValueError: If game state is invalid
            RuntimeError: If no valid move can be found
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for new game"""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        return {
            'algorithm': self.algorithm_name,
            'nodes_explored': self.nodes_explored,
            'search_depth': self.search_depth,
            'grid_size': self.grid_size
        }
```

### **Heuristic Agents Implementation**

#### **BFS Agent Example**
```python
# extensions/heuristics-v0.02/agents/agent_bfs.py
"""
Breadth-First Search agent implementation

Design Pattern: Strategy Pattern
- Implements specific search strategy (BFS)
- Pluggable into game manager
- Isolated algorithm logic

Algorithm Features:
- Guarantees shortest path to food
- Explores all nodes at current depth before going deeper
- Memory intensive but optimal for short distances
"""

from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import copy

from core.game_agents import BaseAgent
from extensions.common.validation_utils import validate_game_state

class BFSAgent(BaseAgent):
    """
    Breadth-First Search agent for Snake Game
    
    Strengths:
    - Finds optimal (shortest) path to food
    - Guaranteed to find solution if one exists
    - Good for small to medium grids
    
    Weaknesses:
    - Memory intensive (stores all frontier nodes)
    - Can be slow on large grids
    - Doesn't consider snake body growth
    """
    
    def __init__(self, grid_size: int):
        super().__init__("BFS", grid_size)
        self.path_cache = {}
        self.max_search_depth = grid_size * grid_size
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """
        Plan move using BFS pathfinding
        
        Algorithm:
        1. Validate game state
        2. Check path cache for existing solution
        3. Run BFS from snake head to food
        4. Return first move in optimal path
        
        Args:
            game_state: Current game state
            
        Returns:
            Next move direction
        """
        # Validate input
        if not validate_game_state(game_state):
            raise ValueError("Invalid game state provided")
        
        # Extract game components
        snake_head = (game_state['snake'][0]['x'], game_state['snake'][0]['y'])
        food_pos = (game_state['food']['x'], game_state['food']['y'])
        snake_body = [(segment['x'], segment['y']) for segment in game_state['snake']]
        
        # Check cache first
        cache_key = (snake_head, food_pos, tuple(snake_body))
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Run BFS
        path = self._bfs_search(snake_head, food_pos, snake_body)
        
        if not path or len(path) < 2:
            # No path found, try to avoid collision
            return self._emergency_move(snake_head, snake_body)
        
        # Calculate first move
        next_pos = path[1]
        move = self._calculate_direction(snake_head, next_pos)
        
        # Cache result
        self.path_cache[cache_key] = move
        
        return move
    
    def _bfs_search(self, start: Tuple[int, int], goal: Tuple[int, int], 
                   obstacles: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Run BFS algorithm to find path from start to goal
        
        Args:
            start: Starting position (snake head)
            goal: Target position (food)
            obstacles: Positions to avoid (snake body)
            
        Returns:
            List of positions representing the path
        """
        if start == goal:
            return [start]
        
        # BFS data structures
        queue = deque([(start, [start])])
        visited = {start}
        self.nodes_explored = 0
        
        # TODO: directions should come from ROOT/config folder 
        while queue and self.nodes_explored < self.max_search_depth:
            current_pos, path = queue.popleft()
            self.nodes_explored += 1
            
            # Explore neighbors
            for dx, dy in directions:
                next_x, next_y = current_pos[0] + dx, current_pos[1] + dy
                next_pos = (next_x, next_y)
                
                # Check bounds
                if not (0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size):
                    continue
                
                # Check obstacles and visited
                if next_pos in obstacles or next_pos in visited:
                    continue
                
                # Check if goal reached
                if next_pos == goal:
                    self.search_depth = len(path)
                    return path + [next_pos]
                
                # Add to queue
                visited.add(next_pos)
                queue.append((next_pos, path + [next_pos]))
        
        return []  # No path found
    # all other code,
    # all other code
    # all other code
    # all other code
    # all other code
    # all other code
    
    def reset(self) -> None:
        """Reset agent state for new game"""
        self.path_cache.clear()
        self.nodes_explored = 0
        self.search_depth = 0
```

## 🎯 **Agent Factory Pattern**

### **Extension Agent Factory**
```python
# extensions/heuristics-v0.02/agents/__init__.py
"""
Agent factory for heuristics extension

Design Pattern: Factory Pattern
- Centralizes agent creation logic
- Supports dynamic agent selection
- Provides consistent interface
"""

from typing import Dict, Type, Any
from core.game_agents import BaseAgent
from .agent_bfs import BFSAgent
from .agent_astar import AStarAgent
from .agent_hamiltonian import HamiltonianAgent

class HeuristicAgentFactory:
    """
    Factory for creating heuristic agents
    
    Features:
    - Dynamic agent creation by name
    - Consistent initialization
    - Extension point for new algorithms
    """
    
    _agents: Dict[str, Type[BaseAgent]] = {
        'BFS': BFSAgent,
        'ASTAR': AStarAgent,
        'A*': AStarAgent,  # Alias
        'HAMILTONIAN': HamiltonianAgent,
        'HAM': HamiltonianAgent  # Alias
    }
    
    @classmethod
    def create_agent(cls, algorithm: str, grid_size: int, **kwargs) -> BaseAgent:
        """
        Create agent by algorithm name
        
        Args:
            algorithm: Algorithm name (case-insensitive)
            grid_size: Game grid size
            **kwargs: Additional agent-specific parameters
            
        Returns:
            Initialized agent instance
            
        Raises:
            ValueError: If algorithm is not supported
        """
        algorithm_upper = algorithm.upper()
        
        if algorithm_upper not in cls._agents:
            available = ', '.join(cls._agents.keys())
            raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {available}")
        
        agent_class = cls._agents[algorithm_upper]
        return agent_class(grid_size, **kwargs)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """Get list of available algorithm names"""
        return list(cls._agents.keys())
    
    @classmethod
    def register_agent(cls, name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register new agent class
        
        Args:
            name: Algorithm name
            agent_class: Agent class to register
        """
        cls._agents[name.upper()] = agent_class

# Convenience function
def create_heuristic_agent(algorithm: str, grid_size: int, **kwargs) -> BaseAgent:
    """Create heuristic agent - convenience function"""
    return HeuristicAgentFactory.create_agent(algorithm, grid_size, **kwargs)
```

## 🎯 **Benefits of Agent Architecture**

### **Educational Benefits**
- **Clear Algorithm Separation**: Each agent represents one algorithm
- **Progressive Complexity**: From simple BFS to complex RL
- **Comparative Analysis**: Easy to compare different approaches
- **Modular Learning**: Students can focus on specific algorithms

### **Technical Benefits**
- **Consistent Interface**: All agents implement BaseAgent protocol
- **Easy Extension**: New algorithms can be added easily
- **Testable Components**: Each agent can be tested independently
- **Performance Monitoring**: Built-in statistics and metrics

### **Research Benefits**
- **Algorithm Isolation**: Pure algorithm implementations
- **Reproducible Results**: Consistent initialization and state management
- **Extensible Framework**: Easy to add new agent types
- **Comparative Studies**: Standardized evaluation metrics

---

**This agent architecture ensures consistent, extensible, and educational implementations across all Snake Game AI extensions while maintaining clear separation of concerns and supporting progressive learning from simple heuristics to complex machine learning approaches.**

