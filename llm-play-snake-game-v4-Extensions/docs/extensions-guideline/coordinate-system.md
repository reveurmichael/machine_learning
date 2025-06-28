# Universal Coordinate System for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines the universal coordinate system used across all extensions.

> **See also:** `core.md`, `final-decision-10.md`, `project-structure-plan.md`.

## 🎯 **Core Philosophy: Universal Consistency**

The Snake Game AI project uses a **universal coordinate system** that ensures consistency across all extensions, algorithms, and visualizations. This system provides a single source of truth for position representation and movement calculations.

### **Educational Value**
- **Consistency**: Same coordinate system across all components
- **Clarity**: Clear understanding of position representation
- **Debugging**: Easier debugging with consistent coordinates
- **Integration**: Seamless integration between different components

## 🏗️ **Coordinate System Definition**

### **Grid Layout**
```
(0,0) (1,0) (2,0) ... (N-1,0)
(0,1) (1,1) (2,1) ... (N-1,1)
(0,2) (1,2) (2,2) ... (N-1,2)
 ...   ...   ...   ...   ...
(0,N-1) (1,N-1) (2,N-1) ... (N-1,N-1)
```

### **Position Representation**
```python
# Universal position format: (x, y) tuple
position = (x, y)  # x: column, y: row

# Examples for 10x10 grid
top_left = (0, 0)
top_right = (9, 0)
bottom_left = (0, 9)
bottom_right = (9, 9)
center = (5, 5)
```

### **Direction Mapping**
```python
# Universal direction constants
DIRECTIONS = {
    'UP': (0, -1),      # Move up (decrease y)
    'DOWN': (0, 1),     # Move down (increase y)
    'LEFT': (-1, 0),    # Move left (decrease x)
    'RIGHT': (1, 0)     # Move right (increase x)
}

# Direction to position change mapping
def get_position_change(direction: str) -> tuple:
    """Get position change for a given direction"""
    return DIRECTIONS.get(direction.upper(), (0, 0))
```

## 🚀 **Coordinate System Implementation**

### **Position Utilities**
```python
def is_valid_position(position: tuple, grid_size: int) -> bool:
    """Check if position is within grid bounds"""
    x, y = position
    return 0 <= x < grid_size and 0 <= y < grid_size

def get_neighbor_positions(position: tuple, grid_size: int) -> list:
    """Get all valid neighbor positions"""
    x, y = position
    neighbors = []
    
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_x, new_y = x + dx, y + dy
        if is_valid_position((new_x, new_y), grid_size):
            neighbors.append((new_x, new_y))
    
    return neighbors

def calculate_distance(pos1: tuple, pos2: tuple) -> int:
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_direction(from_pos: tuple, to_pos: tuple) -> str:
    """Get direction from one position to another"""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    
    if dx == 1: return 'RIGHT'
    elif dx == -1: return 'LEFT'
    elif dy == 1: return 'DOWN'
    elif dy == -1: return 'UP'
    else: return 'UP'  # Default fallback
```

### **Game State Representation**
```python
# Universal game state format
game_state = {
    'grid_size': 10,
    'snake_positions': [(5, 5), (5, 6), (5, 7)],  # Head first
    'apple_position': (3, 4),
    'score': 5,
    'steps': 25
}

# Snake representation: list of positions with head at index 0
snake_positions = [(head_x, head_y), (body_x, body_y), ...]
```

### **Movement Validation**
```python
def is_valid_move(current_pos: tuple, direction: str, snake_positions: list, grid_size: int) -> bool:
    """Check if a move is valid"""
    # Get new position
    dx, dy = get_position_change(direction)
    new_x, new_y = current_pos[0] + dx, current_pos[1] + dy
    new_pos = (new_x, new_y)
    
    # Check grid bounds
    if not is_valid_position(new_pos, grid_size):
        return False
    
    # Check collision with snake body
    if new_pos in snake_positions:
        return False
    
    return True

def get_valid_moves(current_pos: tuple, snake_positions: list, grid_size: int) -> list:
    """Get all valid moves from current position"""
    valid_moves = []
    
    for direction in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
        if is_valid_move(current_pos, direction, snake_positions, grid_size):
            valid_moves.append(direction)
    
    return valid_moves
```

## 📊 **Coordinate System in Extensions**

### **Heuristics Extensions**
```python
class BFSAgent(BaseAgent):
    def plan_move(self, game_state: dict) -> str:
        """Plan move using BFS with universal coordinates"""
        start = game_state['snake_positions'][0]  # Head position
        goal = game_state['apple_position']
        obstacles = game_state['snake_positions']
        grid_size = game_state['grid_size']
        
        # Use universal coordinate system
        path = self._bfs_pathfinding(start, goal, obstacles, grid_size)
        
        if path and len(path) > 1:
            next_pos = path[1]
            return get_direction(start, next_pos)
        else:
            return self._fallback_move(start, obstacles, grid_size)
    
    def _bfs_pathfinding(self, start: tuple, goal: tuple, obstacles: list, grid_size: int) -> list:
        """BFS using universal coordinates"""
        queue = [(start, [start])]
        visited = set()
        
        while queue:
            current, path = queue.pop(0)
            if current == goal:
                return path
            
            for neighbor in get_neighbor_positions(current, grid_size):
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
```

### **Machine Learning Extensions**
```python
class MLPAgent(BaseAgent):
    def _state_to_vector(self, game_state: dict) -> list:
        """Convert game state to neural network input using universal coordinates"""
        head_pos = game_state['snake_positions'][0]
        apple_pos = game_state['apple_position']
        grid_size = game_state['grid_size']
        
        # Normalize positions to [0, 1] range
        head_x_norm = head_pos[0] / (grid_size - 1)
        head_y_norm = head_pos[1] / (grid_size - 1)
        apple_x_norm = apple_pos[0] / (grid_size - 1)
        apple_y_norm = apple_pos[1] / (grid_size - 1)
        
        # Create feature vector
        features = [
            head_x_norm, head_y_norm,
            apple_x_norm, apple_y_norm,
            game_state['score'] / 100.0,  # Normalize score
            len(game_state['snake_positions']) / grid_size  # Normalize length
        ]
        
        return features
```

### **Reinforcement Learning Extensions**
```python
class DQNAgent(BaseAgent):
    def _get_state_representation(self, game_state: dict) -> np.ndarray:
        """Get state representation for RL using universal coordinates"""
        grid_size = game_state['grid_size']
        state_array = np.zeros((grid_size, grid_size))
        
        # Mark snake positions
        for i, pos in enumerate(game_state['snake_positions']):
            if i == 0:  # Head
                state_array[pos[1], pos[0]] = 1
            else:  # Body
                state_array[pos[1], pos[0]] = 2
        
        # Mark apple position
        apple_pos = game_state['apple_position']
        state_array[apple_pos[1], apple_pos[0]] = 3
        
        return state_array
```

## 🎯 **Visualization Integration**

### **GUI Coordinate Mapping**
```python
def gui_to_game_coordinates(gui_x: int, gui_y: int, cell_size: int) -> tuple:
    """Convert GUI coordinates to game coordinates"""
    game_x = gui_x // cell_size
    game_y = gui_y // cell_size
    return (game_x, game_y)

def game_to_gui_coordinates(game_pos: tuple, cell_size: int) -> tuple:
    """Convert game coordinates to GUI coordinates"""
    gui_x = game_pos[0] * cell_size
    gui_y = game_pos[1] * cell_size
    return (gui_x, gui_y)
```

### **Web Interface Integration**
```python
def game_state_to_json(game_state: dict) -> dict:
    """Convert game state to JSON format for web interface"""
    return {
        'grid_size': game_state['grid_size'],
        'snake_positions': game_state['snake_positions'],
        'apple_position': game_state['apple_position'],
        'score': game_state['score'],
        'steps': game_state['steps']
    }
```

## 📋 **Implementation Checklist**

### **Coordinate System Compliance**
- [ ] **Universal Format**: All positions use (x, y) tuple format
- [ ] **Grid Bounds**: Proper validation of position bounds
- [ ] **Direction Mapping**: Consistent direction constants
- [ ] **State Representation**: Standard game state format

### **Extension Integration**
- [ ] **Position Validation**: Proper position validation in all extensions
- [ ] **Movement Calculation**: Consistent movement calculations
- [ ] **State Conversion**: Proper state to vector conversion
- [ ] **Visualization**: Correct coordinate mapping for visualization

### **Quality Standards**
- [ ] **Consistency**: Same coordinate system across all components
- [ ] **Clarity**: Clear position representation
- [ ] **Efficiency**: Efficient coordinate calculations
- [ ] **Documentation**: Clear documentation of coordinate system

## 🎓 **Educational Benefits**

### **Learning Objectives**
- **Coordinate Systems**: Understanding 2D coordinate systems
- **Consistency**: Importance of consistent data representation
- **Integration**: How coordinate systems enable component integration
- **Debugging**: Using consistent coordinates for debugging

### **Best Practices**
- **Single Source of Truth**: One coordinate system for all components
- **Validation**: Proper validation of coordinate values
- **Documentation**: Clear documentation of coordinate conventions
- **Testing**: Testing coordinate calculations and conversions

---

**The universal coordinate system ensures consistency and clarity across all Snake Game AI extensions, providing a solid foundation for algorithm implementation and component integration.**

## 🔗 **See Also**

- **`core.md`**: Base class architecture and inheritance patterns
- **`final-decision-10.md`**: final-decision-10.md governance system
- **`project-structure-plan.md`**: Project structure and organization
