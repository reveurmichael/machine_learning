# Heuristics v0.01 Extension

A simple first extension for the Snake Game project that implements heuristic pathfinding algorithms using BFS (Breadth-First Search).

## Overview

This extension demonstrates how to extend the base classes from the core package to create new game-playing agents. It showcases proper use of:

- `BaseGameManager` for session management
- `BaseGameLogic` for game mechanics
- `BaseGameData` for data tracking
- `SnakeAgent` protocol for agent compatibility

## Features

- **BFS Pathfinding**: Finds shortest path to apple using breadth-first search
- **Headless Operation**: No GUI dependencies, runs entirely in terminal
- **Compatible Logging**: Generates the same JSON format as Task-0 (game_N.json, summary.json)
- **Performance Tracking**: Records pathfinding statistics and search performance
- **Extensible Design**: Easy to add more algorithms (DFS, A*, etc.)

## Architecture

The extension follows the established design patterns:

```
HeuristicGameManager (extends BaseGameManager)
├── HeuristicGameLogic (extends BaseGameLogic)
│   ├── HeuristicGameData (extends BaseGameData)
│   └── BFSAgent (implements SnakeAgent protocol)
```

### Design Patterns Used

- **Template Method**: Base classes define structure, extensions implement specifics
- **Strategy Pattern**: Pluggable pathfinding algorithms
- **Factory Pattern**: Class attributes specify data containers (`GAME_DATA_CLS`)
- **Singleton Pattern**: File manager for logging

## Usage

### Basic Usage

```bash
# Run 3 games with BFS algorithm (default)
python -m extensions.heuristics-v0.01.main

# Run 5 games with BFS
python -m extensions.heuristics-v0.01.main --max-games 5

# Run with custom step limit
python -m extensions.heuristics-v0.01.main --max-games 10 --max-steps 500
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--algorithm` | Heuristic algorithm to use | `BFS` |
| `--max-games` | Maximum number of games to play | `3` |
| `--max-steps` | Maximum steps per game | `800` |
| `--grid-size` | Size of the game grid | `10` |
| `--verbose` | Enable verbose output | `False` |

### Example Output

```
🤖 Heuristic Game Manager initialized with BFS algorithm
📂 Logs will be saved to: logs/heuristics-bfs_20250101_120000

🚀 Starting heuristic game session...
📊 Target games: 3
🧠 Algorithm: BFS

🎮 Starting Game 1
🍎 Apple eaten! Score: 1
🍎 Apple eaten! Score: 2
📊 Game 1 completed:
   Score: 2
   Steps: 45
   Rounds: 8
   Duration: 0.12s

✅ Heuristic session completed!
📊 Games played: 3
🏆 Total score: 15
```

## Output Files

The extension generates the same log format as Task-0 for compatibility:

### Game Files (`game_N.json`)

```json
{
    "score": 8,
    "steps": 55,
    "snake_length": 9,
    "game_over": true,
    "game_end_reason": "SELF",
    "round_count": 12,
    "heuristic_info": {
        "algorithm": "BFS",
        "agent_type": "BFSAgent"
    },
    "pathfinding_stats": {
        "algorithm_name": "BFS",
        "path_calculations": 12,
        "successful_paths": 12,
        "success_rate_percent": 100.0,
        "average_search_time": 0.0003
    }
}
```

### Summary File (`summary.json`)

```json
{
    "timestamp": "2025-01-01 12:00:00",
    "configuration": {
        "algorithm": "BFS",
        "max_games": 3,
        "max_steps": 800,
        "no_gui": true
    },
    "game_statistics": {
        "total_games": 3,
        "total_score": 15,
        "scores": [8, 4, 3]
    },
    "heuristic_statistics": {
        "algorithm_name": "BFS",
        "average_score": 5.0
    }
}
```

## BFS Algorithm

The BFS agent implements a simple breadth-first search pathfinding algorithm:

1. **State Representation**: Current snake head position and apple position
2. **Search Space**: All valid grid positions (not walls or snake body)
3. **Goal**: Find shortest path from head to apple
4. **Path Execution**: Return first move in optimal path

### Algorithm Characteristics

- **Optimality**: Always finds shortest path to apple
- **Completeness**: Will find solution if one exists
- **Time Complexity**: O(V + E) where V is grid cells, E is adjacency
- **Space Complexity**: O(V) for queue and visited set

## Extending the Framework

To add new algorithms:

1. **Create new agent** implementing `SnakeAgent` protocol:
```python
class DFSAgent:
    def get_move(self, game) -> str | None:
        # Implement DFS pathfinding
        pass
```

2. **Update HeuristicGameManager** to support new algorithm:
```python
def _setup_agent(self):
    if self.algorithm_name.upper() == "DFS":
        self.agent = DFSAgent()
    # ... existing code
```

3. **Add to command line choices**:
```python
parser.add_argument(
    "--algorithm",
    choices=["BFS", "DFS"],  # Add new algorithm
    # ...
)
```

## Integration with Base Classes

The extension demonstrates proper inheritance patterns:

### BaseGameManager Integration
- Inherits session management, round tracking, file logging
- Adds heuristic-specific initialization and agent management
- Maintains compatibility with existing log formats

### BaseGameLogic Integration
- Inherits core game mechanics (collision detection, apple generation)
- Adds heuristic planning methods
- Uses factory pattern for data container selection

### BaseGameData Integration
- Inherits core game state tracking
- Adds pathfinding performance metrics
- Maintains JSON serialization compatibility

## Future Enhancements

Planned improvements for future versions:

- **More Algorithms**: DFS, A*, Dijkstra, Hamiltonian cycle
- **Performance Metrics**: More detailed search statistics
- **Optimization**: Path caching, multi-step planning
- **Visualization**: Optional debug output showing search process
- **Benchmarking**: Automated performance comparison tools

## Compatibility

- **Python**: 3.10+
- **Dependencies**: Uses only core project dependencies
- **Platform**: Cross-platform (Windows, macOS, Linux)
- **Integration**: Compatible with existing project structure 