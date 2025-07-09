# Extensions v0.01: Foundation Patterns

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision.md`) and defines the foundation patterns for all v0.01 extensions.

> **See also:** `core.md`, `standalone.md`, `final-decision.md`, `project-structure-plan.md`.

## 🎯 **Core Philosophy: Proof of Concept**

Extensions v0.01 represent the **foundation stage** of algorithm implementation. They focus on proving that a specific algorithmic approach can work within the Snake Game AI framework while maintaining educational clarity and technical simplicity.

### **Educational Value**
- **Algorithm Introduction**: Clear demonstration of basic algorithmic concepts
- **Framework Integration**: Understanding how algorithms integrate with the game framework
- **Implementation Patterns**: Learning standard implementation patterns
- **Debugging Skills**: Developing debugging skills with simple implementations

## 🏗️ **v0.01 Architecture Requirements**

### **Mandatory Directory Structure**
```
extensions/{algorithm}-v0.01/
├── __init__.py
├── main.py                 # Primary execution entry point
├── game_logic.py           # Extension-specific game logic
├── game_manager.py         # Extension-specific game manager
├── agent_{algorithm}.py    # Single algorithm implementation
└── README.md               # Documentation
```

### **Factory Pattern: Canonical Method is create()**
All v0.01 extensions must use the canonical method name `create()` for instantiation:

```python
class HeuristicAgentFactory:
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, **kwargs):  # CANONICAL create() method per SUPREME_RULES
        """Create agent using canonical create() method following SUPREME_RULES from final-decision.md"""
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        print_info(f"[HeuristicAgentFactory] Creating agent: {algorithm}")  # SUPREME_RULES compliant logging
        return agent_class(**kwargs)
```

## 🚀 **Implementation Examples**

### **Heuristics v0.01 Example**
```python
# agent_bfs.py
from core.game_agents import BaseAgent

class BFSAgent(BaseAgent):
    """
    Breadth-First Search agent for v0.01.
    
    Simple implementation focusing on educational clarity.
    """
    
    def __init__(self, name: str = "BFS"):
        super().__init__(name)
        print_info(f"[BFSAgent] Initialized BFS agent")  # SUPREME_RULES compliant logging
    
    def plan_move(self, game_state: dict) -> str:
        """Plan next move using BFS"""
        snake_positions = game_state['snake_positions']
        apple_position = game_state['apple_position']
        grid_size = game_state['grid_size']
        
        # Simple BFS implementation
        path = self._bfs_pathfinding(snake_positions[0], apple_position, snake_positions, grid_size)
        
        if path and len(path) > 1:
            next_pos = path[1]
            return self._get_direction(snake_positions[0], next_pos)
        else:
            return self._fallback_move(snake_positions, grid_size)
    
    def _bfs_pathfinding(self, start: tuple, goal: tuple, obstacles: list, grid_size: int) -> list:
        """Simple BFS pathfinding"""
        queue = [(start, [start])]
        visited = set()
        
        while queue:
            current, path = queue.pop(0)
            if current == goal:
                return path
            
            for neighbor in self._get_neighbors(current, grid_size):
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def _get_neighbors(self, pos: tuple, grid_size: int) -> list:
        """Get valid neighbor positions"""
        x, y = pos
        neighbors = []
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                neighbors.append((new_x, new_y))
        
        return neighbors
    
    def _get_direction(self, current: tuple, next_pos: tuple) -> str:
        """Get direction from current to next position"""
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]
        
        if dx == 1: return 'RIGHT'
        elif dx == -1: return 'LEFT'
        elif dy == 1: return 'DOWN'
        elif dy == -1: return 'UP'
        else: return 'UP'  # Default fallback
    
    def _fallback_move(self, snake_positions: list, grid_size: int) -> str:
        """Simple fallback movement"""
        head = snake_positions[0]
        for direction in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            dx, dy = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}[direction]
            new_pos = (head[0] + dx, head[1] + dy)
            if (0 <= new_pos[0] < grid_size and 0 <= new_pos[1] < grid_size and 
                new_pos not in snake_positions):
                return direction
        return 'UP'  # Last resort
```

### **Game Manager Integration**
```python
# game_manager.py
from core.game_manager import BaseGameManager

class HeuristicGameManager(BaseGameManager):
    """
    Heuristic game manager for v0.01.
    
    Simple implementation focusing on educational clarity.
    """
    
    def __init__(self, algorithm: str, grid_size: int = 10):
        super().__init__(grid_size=grid_size)
        self.algorithm = algorithm
        self.agent = self.create(algorithm)
        print_info(f"[HeuristicGameManager] Initialized with {algorithm}")  # SUPREME_RULES compliant logging
    
    def create(self, algorithm: str):
        """Create agent using simple factory"""
        from .agent_bfs import BFSAgent
        from .agent_astar import AStarAgent
        
        agents = {
            'BFS': BFSAgent,
            'ASTAR': AStarAgent,
        }
        
        if algorithm not in agents:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return agents[algorithm]()
    
    def run_game(self) -> dict:
        """Run a single game"""
        self._initialize_game()
        
        while not self._is_game_over():
            # Get agent's move
            game_state = self._get_game_state()
            move = self.agent.plan_move(game_state)
            
            # Execute move
            result = self._execute_move(move)
            
            # Update game state
            self._update_game_state(move, result)
        
        return self._get_game_result()
    
    def _initialize_game(self):
        """Initialize game state"""
        self.snake_positions = [(5, 5)]  # Start at center
        self.apple_position = self._generate_apple()
        self.score = 0
        self.steps = 0
        print_info(f"[HeuristicGameManager] Game initialized")  # SUPREME_RULES compliant logging
    
    def _get_game_state(self) -> dict:
        """Get current game state"""
        return {
            'snake_positions': self.snake_positions,
            'apple_position': self.apple_position,
            'grid_size': self.grid_size,
            'score': self.score,
            'steps': self.steps
        }
    
    def _execute_move(self, move: str) -> str:
        """Execute agent's move"""
        head = self.snake_positions[0]
        dx, dy = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}[move]
        new_head = (head[0] + dx, head[1] + dy)
        
        # Check collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake_positions):
            return 'GAME_OVER'
        
        # Check apple
        if new_head == self.apple_position:
            return 'APPLE_EATEN'
        
        return 'MOVE'
    
    def _update_game_state(self, move: str, result: str):
        """Update game state after move"""
        head = self.snake_positions[0]
        dx, dy = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}[move]
        new_head = (head[0] + dx, head[1] + dy)
        
        self.snake_positions.insert(0, new_head)
        
        if result == 'APPLE_EATEN':
            self.score += 1
            self.apple_position = self._generate_apple()
        else:
            self.snake_positions.pop()
        
        self.steps += 1
    
    def _generate_apple(self) -> tuple:
        """Generate new apple position"""
        import random
        while True:
            apple = (random.randint(0, self.grid_size-1), 
                    random.randint(0, self.grid_size-1))
            if apple not in self.snake_positions:
                return apple
    
    def _is_game_over(self) -> bool:
        """Check if game is over"""
        return len(self.snake_positions) == 0
    
    def _get_game_result(self) -> dict:
        """Get final game result"""
        return {
            'score': self.score,
            'steps': self.steps,
            'snake_length': len(self.snake_positions)
        }
```

## 📊 **v0.01 Standards**

### **Simplicity Requirements**
- **Single Algorithm**: One algorithm per extension
- **Simple Implementation**: Focus on clarity over optimization
- **Minimal Dependencies**: Use only core framework and common utilities
- **Clear Documentation**: Comprehensive comments and docstrings

### **Educational Focus**
- **Algorithm Explanation**: Clear explanation of the algorithm
- **Implementation Details**: Step-by-step implementation
- **Debugging Support**: Easy to debug and understand
- **Learning Progression**: Foundation for more complex versions

### **Quality Standards**
- **Working Implementation**: Algorithm must work correctly
- **Clear Code**: Readable and well-documented code
- **Proper Integration**: Seamless integration with game framework
- **Error Handling**: Basic error handling and validation

## 📋 **Implementation Checklist**

### **Required Components**
- [ ] **Single Agent**: One algorithm implementation
- [ ] **Game Manager**: Extension-specific game manager
- [ ] **Factory Pattern**: Canonical `create()` method
- [ ] **Main Entry Point**: Clear execution entry point
- [ ] **Documentation**: Comprehensive README

### **Code Quality**
- [ ] **Simple Implementation**: Clear, readable code
- [ ] **Proper Comments**: Comprehensive comments
- [ ] **Error Handling**: Basic error handling
- [ ] **Integration**: Proper framework integration

### **Educational Value**
- [ ] **Algorithm Explanation**: Clear algorithm description
- [ ] **Implementation Details**: Step-by-step explanation
- [ ] **Debugging Support**: Easy to debug
- [ ] **Learning Progression**: Foundation for future versions

## 🎓 **Educational Benefits**

### **Learning Objectives**
- **Algorithm Understanding**: Deep understanding of the implemented algorithm
- **Framework Integration**: How to integrate algorithms with the game framework
- **Implementation Patterns**: Standard implementation patterns
- **Debugging Skills**: Developing debugging skills

### **Best Practices**
- **Simplicity**: Keep implementations simple and clear
- **Documentation**: Comprehensive documentation and comments
- **Testing**: Basic testing and validation
- **Integration**: Proper integration with existing framework

## 🔗 **Progression to v0.02**

v0.01 extensions serve as the foundation for v0.02 extensions, which will:
- **Add Multiple Algorithms**: Implement multiple related algorithms
- **Enhance Factory Pattern**: More sophisticated agent factory
- **Improve Performance**: Optimize implementations
- **Add Features**: Additional functionality and capabilities

---

**Extensions v0.01 provide the foundation for algorithm implementation in the Snake Game AI project, focusing on educational clarity and technical simplicity while establishing the patterns for future development.**

## 🔗 **See Also**

- **`core.md`**: Base class architecture and inheritance patterns
- **`standalone.md`**: Standalone principle and extension independence
- **`final-decision.md`**: SUPREME_RULES governance system and canonical standards
- **`project-structure-plan.md`**: Project structure and organization






