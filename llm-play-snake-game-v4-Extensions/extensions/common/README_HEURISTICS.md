# Heuristic Common Utilities

This document explains how to use the common utilities for heuristic extensions to eliminate code duplication and maintain consistency across all heuristic extensions (v0.01, v0.02, v0.03, v0.04).

## Philosophy

The goal is to extract **non-essential infrastructure code** from heuristic extensions while keeping the **core algorithmic concepts** visible and central in each extension. This follows the DRY principle while maintaining clarity.

## What's Been Moved to Common

### ✅ Infrastructure Code (Moved to Common)
- Session configuration and logging setup
- Performance tracking and metrics calculation  
- Console output formatting and colors
- Replay data extraction and navigation
- Web UI component creation (Streamlit/Flask)
- Algorithm metadata and descriptions
- Game execution error handling patterns

### ✅ Core Algorithm Code (Stays in Extensions)
- Pathfinding algorithm implementations (BFS, A*, DFS, etc.)
- Agent decision-making logic
- Heuristic-specific optimizations
- Algorithm-specific insights and analysis
- Custom game logic adaptations

## Usage Examples

### Basic Heuristic Session

```python
# In heuristics-v0.02/game_manager.py (simplified)
from extensions.common import (
    HeuristicSessionConfig,
    HeuristicLogger,
    HeuristicPerformanceTracker,
    save_heuristic_session_summary
)

class HeuristicGameManager(BaseGameManager):
    def __init__(self, args):
        super().__init__(args)
        
        # Use common configuration
        self.config = HeuristicSessionConfig(args)
        self.logger = HeuristicLogger(self.config)
        self.performance = HeuristicPerformanceTracker()
        
        # Set up logging using common utilities
        self.log_dir = setup_heuristic_logging(self.config)
    
    def run(self):
        """Run session with standardized logging."""
        try:
            self.logger.log_session_start()
            
            # Core algorithm execution (extension-specific)
            while self.game_count < self.config.max_games:
                game_result = self._run_single_game()  # Algorithm-specific
                self.performance.record_game(
                    game_result['score'],
                    game_result['steps'], 
                    game_result['rounds'],
                    game_result['duration']
                )
            
            # Common session completion
            self.logger.log_session_complete(self.game_count, self.total_score)
            save_heuristic_session_summary(self.log_dir, self.config, self.performance)
            
        except KeyboardInterrupt:
            self.logger.log_session_complete(self.game_count, self.total_score)
```

### Streamlit App Integration

```python
# In heuristics-v0.03/app.py (simplified) 
from extensions.common import (
    create_algorithm_selector,
    create_parameter_inputs,
    create_performance_display,
    build_streamlit_tabs
)

def main():
    st.title("🤖 Heuristics v0.03 - Multi-Algorithm Snake AI")
    
    # Use common UI components
    algorithm = create_algorithm_selector(
        available_algorithms=get_available_algorithms(),
        default_algorithm="BFS"
    )
    
    params = create_parameter_inputs(key_prefix="v03")
    
    if st.button("🚀 Run Session"):
        # Core algorithm execution (extension-specific)
        results = run_heuristic_session(algorithm, params)
        
        # Common performance display
        create_performance_display(results, algorithm)
```

### Replay Engine Integration

```python
# In heuristics-v0.04/replay_engine.py (simplified)
from extensions.common import (
    extract_heuristic_replay_data,
    calculate_heuristic_performance_metrics,
    format_algorithm_insights,
    validate_replay_navigation
)

class HeuristicReplayEngine:
    def load_game_data(self, game_file: str):
        """Load and process game data using common utilities."""
        with open(game_file) as f:
            raw_data = json.load(f)
        
        # Use common data extraction
        self.replay_data = extract_heuristic_replay_data(raw_data)
        self.performance_metrics = calculate_heuristic_performance_metrics(self.replay_data)
        
        # Algorithm-specific processing still in extension
        self._process_algorithm_specific_data(raw_data)
    
    def navigate(self, command: str, position: int = None):
        """Navigate replay using common validation."""
        is_valid, new_pos, error = validate_replay_navigation(
            self.current_move,
            self.total_moves,
            command,
            position
        )
        
        if is_valid:
            self.current_move = new_pos
            # Extension-specific state update
            self._update_visual_state()
        else:
            print(f"Navigation error: {error}")
```

## Available Utilities

### Session Management
- `HeuristicSessionConfig`: Configuration container
- `HeuristicLogger`: Standardized console output
- `HeuristicPerformanceTracker`: Metrics collection
- `setup_heuristic_logging()`: Directory setup
- `save_heuristic_session_summary()`: Summary generation

### Replay Utilities  
- `extract_heuristic_replay_data()`: Data extraction
- `calculate_heuristic_performance_metrics()`: Performance analysis
- `build_heuristic_state_dict()`: State building
- `validate_replay_navigation()`: Navigation validation

### Web Interface Utilities
- `create_algorithm_selector()`: Streamlit algorithm picker
- `create_parameter_inputs()`: Parameter input widgets
- `create_performance_display()`: Metrics visualization
- `format_web_state_response()`: Flask response formatting

### Algorithm Metadata
- `ALGORITHM_DISPLAY_NAMES`: User-friendly names
- `ALGORITHM_DESCRIPTIONS`: Algorithm explanations
- `format_algorithm_insights()`: Educational insights

## Benefits of This Approach

### ✅ For Extension Developers
- **Focus on algorithms**: Less boilerplate, more algorithm logic
- **Consistency**: Standardized patterns across all extensions
- **Maintainability**: Bug fixes in common code benefit all extensions

### ✅ For Users
- **Familiar interface**: All heuristic extensions work similarly  
- **Better comparisons**: Consistent metrics and visualizations
- **Educational value**: Clear separation of infrastructure vs. algorithms

### ✅ For Project Maintenance
- **DRY compliance**: Single source of truth for common patterns
- **Easy updates**: Change logging format once, applies everywhere
- **Clear architecture**: Infrastructure separated from domain logic

## Migration Path

Existing heuristic extensions can gradually adopt these utilities:

1. **Phase 1**: Use `HeuristicSessionConfig` and `HeuristicLogger`
2. **Phase 2**: Migrate to `HeuristicPerformanceTracker`  
3. **Phase 3**: Adopt replay and web utilities
4. **Phase 4**: Remove duplicated code from extensions

Each phase maintains backward compatibility while reducing duplication.

## Extension-Specific Code Examples

Even with common utilities, each extension keeps its algorithmic identity:

```python
# This stays in heuristics-v0.02/agents/agent_bfs.py
class BFSAgent(SnakeAgent):
    """Breadth-First Search pathfinding agent."""
    
    def find_path(self, start, goal, obstacles):
        """Core BFS algorithm - stays in extension."""
        queue = deque([start])
        visited = {start}
        parent = {}
        
        while queue:
            current = queue.popleft()
            if current == goal:
                return self._reconstruct_path(parent, start, goal)
            
            for neighbor in self._get_neighbors(current):
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
        
        return None  # No path found
```

The algorithm-specific logic remains in the extension, while infrastructure uses common utilities. 