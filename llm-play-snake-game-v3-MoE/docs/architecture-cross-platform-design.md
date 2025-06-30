# Snake Game Architecture: Single Source of Truth and Cross-Platform Design

## Overview

This document explains the architecture of the Snake Game project, focusing on how single source of truth principles are implemented across Pygame and Web platforms, how game logic is shared, and how different modes work together.

## Table of Contents

1. [Color Management: Single Source of Truth](#color-management)
2. [Game Logic Sharing Architecture](#game-logic-sharing)
3. [Human Play Controls: Arrow Keys and WASD](#human-play-controls)
4. [Replay System Architecture](#replay-system)
5. [Web vs Pygame Architecture](#web-vs-pygame)
6. [Key Design Principles](#design-principles)

---

## Color Management: Single Source of Truth {#color-management}

### The Problem

The project needs consistent colors across multiple platforms:
- **Pygame**: Uses RGB tuples like `(255, 140, 0)`
- **Web**: Uses hex strings like `#FF8C00`
- **Multiple modes**: Human play, replay, live LLM, continue

### The Solution: Centralized Color Definition

#### 1. Single Source: `config/ui_constants.py`

```python
COLORS = {
    "SNAKE_HEAD": (255, 140, 0),  # Bright orange for snake head
    "SNAKE_BODY": (209, 204, 192),  # Light gray for snake body
    "APPLE": (192, 57, 43),  # Red for apple
    "BACKGROUND": (44, 44, 84),  # Dark blue background
    "GRID": (87, 96, 111),  # Grid lines
    "TEXT": (255, 255, 255),  # White text
    "ERROR": (231, 76, 60),  # Red for error messages
    "BLACK": (0, 0, 0),  # Black
    "WHITE": (255, 255, 255),  # White
    "GREY": (189, 195, 199),  # Light grey
    "APP_BG": (240, 240, 240),  # App background
}
```

#### 2. Pygame Usage: Direct RGB Access

```python
# gui/game_gui.py
from config.ui_constants import COLORS

def draw_snake_segment(self, x, y, is_head):
    if is_head:
        color = COLORS["SNAKE_HEAD"]  # Direct RGB tuple
    else:
        color = COLORS["SNAKE_BODY"]
    pygame.draw.rect(self.screen, color, (x * self.pixel, y * self.pixel, self.pixel, self.pixel))
```

#### 3. Web Translation: `utils/web_utils.py`

```python
def build_color_map() -> dict[str, tuple[int, int, int]]:
    """Return colour map expected by the front-end JS/HTML."""
    return {
        "snake_head": COLORS["SNAKE_HEAD"],
        "snake_body": COLORS["SNAKE_BODY"],
        "apple": COLORS["APPLE"],
        "background": COLORS["BACKGROUND"],
        "grid": COLORS["GRID"],
    }
```

#### 4. JavaScript Conversion: `web/static/js/common.js`

```javascript
// Constants - colors will be overridden by values from the server
let COLORS = {
    SNAKE_HEAD: '#3498db',    // Blue for snake head
    SNAKE_BODY: '#2980b9',    // Darker blue for snake body
    APPLE: '#e74c3c',         // Red for apple
    BACKGROUND: '#2c3e50',    // Dark background
    GRID: '#34495e',          // Grid lines
};

/**
 * Converts RGB array to hex color string
 */
function rgbArrayToHex(rgbArray) {
    if (!Array.isArray(rgbArray) || rgbArray.length < 3) {
        return null;
    }
    return `#${rgbArray[0].toString(16).padStart(2, '0')}${rgbArray[1].toString(16).padStart(2, '0')}${rgbArray[2].toString(16).padStart(2, '0')}`;
}
```

#### 5. Dynamic Color Updates in Web Mode

```javascript
// web/static/js/human_play.js
// Update colors if provided
if (gameState.colors) {
    COLORS.SNAKE_HEAD = rgbArrayToHex(gameState.colors.snake_head) || COLORS.SNAKE_HEAD;
    COLORS.SNAKE_BODY = rgbArrayToHex(gameState.colors.snake_body) || COLORS.SNAKE_BODY;
    COLORS.APPLE = rgbArrayToHex(gameState.colors.apple) || COLORS.APPLE;
    COLORS.BACKGROUND = rgbArrayToHex(gameState.colors.background) || COLORS.BACKGROUND;
    COLORS.GRID = rgbArrayToHex(gameState.colors.grid) || COLORS.GRID;
}
```

### Benefits of This Approach

1. **Single Source**: All colors defined in one place
2. **Type Safety**: Pygame gets RGB tuples, Web gets hex strings
3. **Consistency**: Same colors across all platforms
4. **Maintainability**: Change colors once, affects everywhere
5. **Fallback**: Web has default colors if server doesn't provide them

---

## Game Logic Sharing Architecture {#game-logic-sharing}

### Core Architecture: Inheritance Hierarchy

```
GameController (core/game_controller.py)
├── GameLogic (core/game_logic.py) - LLM integration
└── ReplayEngine (replay/replay_engine.py) - Replay functionality
```

### 1. Base Game Controller: `core/game_controller.py`

The foundation that all game modes inherit from:

```python
class GameController:
    """Base class for the Snake game controller."""
    
    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = True) -> None:
        # Game state variables
        self.grid_size = grid_size
        self.board = np.zeros((grid_size, grid_size))
        self.snake_positions = np.array([[grid_size//2, grid_size//2]])
        self.head_position = self.snake_positions[-1]
        
        # Game state tracker for statistics
        self.game_state = GameData()
        
        # GUI settings
        self.use_gui = use_gui
        self.gui = None
    
    def make_move(self, direction_key: str) -> Tuple[bool, bool]:
        """Execute a move in the specified direction."""
        # Core move logic - same for all modes
        # Returns (game_active, apple_eaten)
```

### 2. LLM Integration: `core/game_logic.py`

Extends GameController with LLM-specific functionality:

```python
class GameLogic(GameController):
    """Snake game with LLM agent integration."""
    
    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = True):
        super().__init__(grid_size, use_gui)
        
        # LLM interaction state
        self.planned_moves = []
        self.processed_response = ""
    
    def parse_llm_response(self, response: str):
        """Parse the LLM's response to extract multiple sequential moves."""
        try:
            return parse_llm_response(response, process_response_for_display, self)
        except Exception as e:
            # Error handling and fallback
            return None
```

### 3. Replay Engine: `replay/replay_engine.py`

Extends GameController for replaying recorded games:

```python
class ReplayEngine(GameController):
    """Engine for replaying recorded Snake games."""
    
    def __init__(self, log_dir: str, move_pause: float = 1.0, auto_advance: bool = False, use_gui: bool = True):
        super().__init__(use_gui=use_gui)
        
        # Replay-specific state
        self.log_dir = log_dir
        self.game_number = 1
        self.apple_positions = []
        self.moves = []
        self.move_index = 0
    
    def load_game_data(self, game_number: int) -> Optional[Dict[str, Any]]:
        """Load game data from JSON files."""
        # Load and parse game_N.json files
        # Extract moves, apple positions, LLM responses
```

### 4. Web-Specific Extensions

#### Human Play Web: `human_play_web.py`

```python
class WebGameController(GameController):
    """Extended game controller for web-based human play."""
    
    def __init__(self, grid_size=10):
        super().__init__(grid_size=grid_size, use_gui=False)
        self.game_over = False
        self.game_end_reason = None
    
    def make_move(self, direction_key):
        """Override to track game over state."""
        game_active, apple_eaten = super().make_move(direction_key)
        
        if not game_active:
            self.game_over = True
            if self.last_collision_type == "wall":
                self.game_end_reason = "WALL"
            elif self.last_collision_type == "self":
                self.game_end_reason = "SELF"
        
        return game_active, apple_eaten
```

### Benefits of This Architecture

1. **Code Reuse**: Core game logic shared across all modes
2. **Consistency**: Same move validation, collision detection, scoring
3. **Extensibility**: Easy to add new modes by extending base classes
4. **Separation of Concerns**: Game logic separate from UI/LLM/replay logic
5. **Testing**: Core logic can be tested independently

---

## Human Play Controls: Arrow Keys and WASD {#human-play-controls}

### Pygame Implementation: `human_play.py`

```python
def handle_input(game, gui):
    """Handle keyboard input for snake control."""
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_UP:
                game_active, _ = game.make_move("UP")
                gui.set_game_over(not game_active)
            elif event.key == K_DOWN:
                game_active, _ = game.make_move("DOWN")
                gui.set_game_over(not game_active)
            elif event.key == K_LEFT:
                game_active, _ = game.make_move("LEFT")
                gui.set_game_over(not game_active)
            elif event.key == K_RIGHT:
                game_active, _ = game.make_move("RIGHT")
                gui.set_game_over(not game_active)
            elif event.key == K_r:
                game.reset()
                gui.set_game_over(False)
```

### Web Implementation: `web/static/js/human_play.js`

```javascript
function handleKeyDown(event) {
    if (!gameState) return;
    
    // If game over, only respond to R key for reset
    if (gameState.game_over) {
        if (event.key === 'r' || event.key === 'R') {
            resetGame();
        }
        return;
    }
    
    let direction = null;
    
    // Map keys to directions - ensure all keys are captured properly
    switch (event.key) {
        case 'ArrowUp':
        case 'w':
        case 'W':
            direction = 'UP';
            break;
        case 'ArrowDown':
        case 's':
        case 'S':
            direction = 'DOWN';
            break;
        case 'ArrowLeft':
        case 'a':
        case 'A':
            direction = 'LEFT';
            break;
        case 'ArrowRight':
        case 'd':
        case 'D':
            direction = 'RIGHT';
            break;
        case 'r':
        case 'R':
            resetGame();
            return;
        default:
            return; // Ignore other keys
    }
    
    if (direction) {
        event.preventDefault(); // Prevent default behavior like scrolling
        makeMove(direction);
    }
}

async function makeMove(direction) {
    try {
        await sendApiRequest('/api/move', 'POST', { direction });
    } catch (error) {
        console.error('Error making move:', error);
    }
}
```

### Backend API: `human_play_web.py`

```python
@app.route('/api/move', methods=['POST'])
def make_move():
    """API endpoint for making a move."""
    if game_controller is None:
        return jsonify({'error': 'Game controller not initialized'})
    
    data = request.json
    direction = data.get('direction')
    
    if direction not in ["UP", "DOWN", "LEFT", "RIGHT"]:
        return jsonify({'status': 'error', 'message': 'Invalid direction'})
    
    game_active, apple_eaten = game_controller.make_move(direction)
    
    return jsonify({
        'status': 'ok',
        'game_active': game_active,
        'apple_eaten': apple_eaten,
        'score': game_controller.score,
        'steps': game_controller.steps
    })
```

### Key Features

1. **Dual Input Support**: Both arrow keys and WASD
2. **Case Insensitive**: Works with both uppercase and lowercase
3. **Game Over Handling**: Only reset key works when game is over
4. **Prevent Default**: Stops browser scrolling on arrow keys
5. **Error Handling**: Graceful handling of network errors
6. **Consistent API**: Same direction strings used across platforms

---

## Conclusion

The Snake Game project demonstrates excellent software architecture principles:

1. **Single Source of Truth**: Colors, game logic, and constants are defined once and reused everywhere
2. **Inheritance Hierarchy**: Clean separation between core logic and platform-specific features
3. **Cross-Platform Compatibility**: Same game logic works on Pygame and Web
4. **Modular Design**: Each component has a clear responsibility
5. **Error Resilience**: Graceful handling of missing data and network issues
6. **Maintainability**: Changes propagate consistently across all platforms

This architecture makes the codebase easy to understand, modify, and extend while ensuring consistent behavior across different platforms and game modes. 