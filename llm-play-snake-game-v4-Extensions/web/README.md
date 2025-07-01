# Simple Web Architecture for Snake Game AI

## Overview

The web module provides a simplified, KISS-compliant Flask architecture for Task-0 and all future extensions (Tasks 1-5). The architecture follows **no over-preparation** principles, implementing only what's needed now while remaining easily extensible.

## Design Philosophy

- **KISS**: Simple, straightforward implementations without unnecessary complexity
- **DRY**: Minimal code duplication, maximum clarity
- **No Over-Preparation**: Build only what's needed for current requirements
- **Extensible**: Easy for Tasks 1-5 to inherit and extend

## Architecture

### Base Classes

```python
# Simple base classes - no complex inheritance hierarchy
FlaskGameApp      # Basic Flask app with auto port allocation
GameFlaskApp      # Adds game-specific API routes
```

### Task-0 Applications

```python
HumanWebApp       # Human player interface using GameLogic
LLMWebApp         # Demo LLM interface (simple demo only)
ReplayWebApp      # Game replay interface using ReplayEngine
```

### Factory Functions

```python
# Simple factory functions - no complex factory classes
create_human_web_app(grid_size=10, port=None)
create_llm_web_app(provider="hunyuan", model="hunyuan-turbos-latest", grid_size=10, port=None)
create_replay_web_app(log_dir, game_number=1, port=None)
```

## Usage Examples

### Human Web Interface
```python
from web import create_human_web_app

app = create_human_web_app(grid_size=15)
app.run()  # Automatic port allocation
```

### Replay Interface
```python
from web import create_replay_web_app

app = create_replay_web_app("logs/session_dir", game_number=3)
app.run()
```

### Full LLM Interface
```bash
# Use the script for full LLM functionality (not the demo app)
python scripts/main_web.py --provider deepseek --model deepseek-chat
```

## Extension Pattern for Tasks 1-5

Extensions can easily create their own web interfaces:

```python
# Example: Heuristic extension web app
from web.base_app import GameFlaskApp
from your_heuristic_logic import HeuristicGameLogic

class HeuristicWebApp(GameFlaskApp):
    def __init__(self, algorithm: str, grid_size: int = 10, port: int = None):
        super().__init__(f"Heuristic {algorithm}", port)
        self.algorithm = algorithm
        self.game_logic = HeuristicGameLogic(algorithm, grid_size)
    
    def get_game_state(self):
        # Return current game state
        return self.game_logic.get_state()
    
    def handle_control(self, data):
        # Handle game controls
        return self.game_logic.handle_action(data)

# Simple factory function
def create_heuristic_web_app(algorithm: str, grid_size: int = 10, port: int = None):
    return HeuristicWebApp(algorithm, grid_size, port)
```

## Key Features

### Automatic Port Allocation
- All apps use `random_free_port()` by default
- No port conflicts during development
- Easy deployment across environments

### Simple API Structure
- `/` - Main page with game interface
- `/api/health` - Health check
- `/api/state` - Get current game state
- `/api/control` - Handle game controls (POST)
- `/api/move` - Handle moves (POST)
- `/api/reset` - Reset game (POST)

### Consistent Error Handling
- Simple error responses
- Clear error messages
- Graceful degradation

## Scripts

- `scripts/human_play_web.py` - Simple human web interface
- `scripts/replay_web.py` - Simple replay web interface  
- `scripts/main_web.py` - **Full LLM interface** with GameManager integration

## Benefits

### For Task-0
- Simple, working web interfaces
- No unnecessary complexity
- Easy to understand and modify

### For Extensions (Tasks 1-5)
- Clear inheritance pattern
- Minimal boilerplate code
- Consistent API structure
- Easy to add algorithm-specific features

### For Development
- Fast iteration cycles
- No over-engineering barriers
- Simple debugging
- Clear separation of concerns

## No Over-Preparation

This architecture specifically avoids:
- Complex inheritance hierarchies
- Unused abstract methods
- Speculative features
- Over-engineered factory patterns
- Complex configuration systems

Instead, it provides:
- Simple, working solutions
- Clear extension points
- Minimal viable functionality
- Easy-to-understand patterns

## Educational Value

The simplified architecture demonstrates:
- How to build web interfaces that actually work
- Clean separation between game logic and web presentation
- Simple factory patterns without over-engineering
- Extensible design without speculative complexity
- KISS principles in practice 