# Snake Game Architecture: Single Source of Truth and Cross-Platform Design

## Overview

This document explains the architecture of the Snake Game project, focusing on how single source of truth principles are implemented across Pygame and Web platforms, how game logic is shared, and how different modes (human play, replay, continue) work together.

## Table of Contents

1. [Color Management: Single Source of Truth](#color-management-single-source-of-truth)
2. [Game Logic Sharing Architecture](#game-logic-sharing-architecture)
3. [Human Play Controls: Arrow Keys and WASD](#human-play-controls-arrow-keys-and-wasd)
4. [Replay System Architecture](#replay-system-architecture)
5. [Web vs Pygame Architecture](#web-vs-pygame-architecture)
6. [Key Design Principles](#key-design-principles)

---

## Color Management: Single Source of Truth

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

## Game Logic Sharing Architecture

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

#### Replay Web: `replay_web.py`

```python
class WebReplayEngine(ReplayEngine):
    """Extended replay engine for web-based replay."""
    
    def __init__(self, log_dir, move_pause=1.0, auto_advance=False):
        super().__init__(log_dir=log_dir, move_pause=move_pause, auto_advance=auto_advance, use_gui=False)
        self.paused = True  # Start paused until client connects
    
    def get_current_state(self):
        """Get the current state for the web interface."""
        state = self._build_state_base()
        state['snake_positions'] = to_list(state['snake_positions'])
        state['apple_position'] = to_list(state['apple_position'])
        state.update({
            'move_pause': self.pause_between_moves,
            'game_end_reason': translate_end_reason(self.game_end_reason),
            'grid_size': self.grid_size,
            'colors': build_color_map(),
        })
        return state
```

### 5. Game Manager: High-Level Orchestration

```python
class GameManager:
    """Run one or many LLM-driven Snake games and collect aggregate stats."""
    
    def __init__(self, args: "argparse.Namespace") -> None:
        self.args = args
        self.game = None
        self.game_active = True
        self.running = True
    
    def setup_game(self):
        """Set up the game logic and GUI."""
        # Initialize game logic
        self.game = GameLogic(use_gui=self.use_gui)
        
        # Set up the GUI if enabled
        if self.use_gui:
            gui = GameGUI()
            self.game.set_gui(gui)
```

### Benefits of This Architecture

1. **Code Reuse**: Core game logic shared across all modes
2. **Consistency**: Same move validation, collision detection, scoring
3. **Extensibility**: Easy to add new modes by extending base classes
4. **Separation of Concerns**: Game logic separate from UI/LLM/replay logic
5. **Testing**: Core logic can be tested independently

---

## Human Play Controls: Arrow Keys and WASD

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

## Replay System Architecture

### 1. Data Storage: Game JSON Files

Each game is saved as `game_N.json` with detailed history:

```json
{
  "score": 5,
  "steps": 25,
  "detailed_history": {
    "apple_positions": [[3, 4], [7, 2], [1, 8], [9, 1], [5, 6]],
    "moves": ["UP", "RIGHT", "UP", "LEFT", "DOWN", ...],
    "rounds_data": [
      {
        "round": 1,
        "planned_moves": ["UP", "RIGHT", "UP"],
        "llm_response": "Based on the current position...",
        "primary_llm": "ollama/deepseek-r1:14b",
        "parser_llm": "ollama/gemma3:12b-it-qat"
      }
    ]
  },
  "game_end_reason": "WALL",
  "timestamp": "2024-01-15T10:30:00"
}
```

### 2. Loading Process: `replay/replay_utils.py`

```python
def load_game_json(log_dir: str, game_number: int) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return the path and decoded JSON dict for *game_number*."""
    game_filename = get_game_json_filename(game_number)
    game_file = join_log_path(log_dir, game_filename)
    
    file_path = Path(game_file)
    if not file_path.exists():
        print(f"[replay] Game {game_number} data not found in {log_dir}")
        return str(file_path), None
    
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        return str(file_path), data
    except Exception as exc:
        print(f"[replay] Error reading {file_path}: {exc}")
        return str(file_path), None

def parse_game_data(game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a lightweight, replay-friendly view of a *game_N.json* blob."""
    detailed = game_data.get("detailed_history")
    if not isinstance(detailed, dict):
        print("[replay] detailed_history missing")
        return None
    
    # Extract apple positions and moves
    apples = detailed.get("apple_positions", [])
    moves = detailed.get("moves", [])
    
    # Validate data
    if not apples or not moves:
        print("[replay] Missing apple_positions or moves")
        return None
    
    # Extract planned moves from first round
    planned_moves = []
    rounds_data = detailed.get("rounds_data", [])
    if rounds_data:
        first_round = rounds_data[0]
        planned_moves = first_round.get("planned_moves", [])
    
    return {
        "apple_positions": apples,
        "moves": moves,
        "planned_moves": planned_moves,
        "game_end_reason": game_data.get("game_end_reason"),
        "primary_llm": game_data.get("primary_llm"),
        "secondary_llm": game_data.get("parser_llm"),
        "timestamp": game_data.get("timestamp"),
        "raw": game_data
    }
```

### 3. Replay Engine: `replay/replay_engine.py`

```python
def load_game_data(self, game_number: int) -> Optional[Dict[str, Any]]:
    """Load game data for a specific game number."""
    game_file, game_data = load_game_json(self.log_dir, game_number)
    
    if game_data is None:
        return None
    
    try:
        parsed = parse_game_data(game_data)
        if parsed is None:
            return None
        
        # Unpack parsed fields
        self.apple_positions = parsed["apple_positions"]
        self.moves = parsed["moves"]
        self.planned_moves = parsed["planned_moves"]
        self.game_end_reason = END_REASON_MAP.get(parsed["game_end_reason"], parsed["game_end_reason"])
        self.primary_llm = parsed["primary_llm"]
        self.secondary_llm = parsed["secondary_llm"]
        self.game_timestamp = parsed["timestamp"]
        
        # Reset counters
        self.move_index = 0
        self.apple_index = 0
        self.moves_made = []
        
        # Initialize game state
        self.reset()
        
        # Set initial snake position (middle of grid)
        self.snake_positions = np.array([[self.grid_size // 2, self.grid_size // 2]])
        self.head_position = self.snake_positions[-1]
        
        # Set initial apple position
        self.set_apple_position(self.apple_positions[0])
        
        # Update game board
        self._update_board()
        
        return game_data
        
    except Exception as e:
        print(f"Error loading game data: {e}")
        return None

def execute_replay_move(self, direction_key: str) -> bool:
    """Execute a move in the specified direction for replay."""
    # Handle sentinel moves (no actual movement)
    if direction_key in SENTINEL_MOVES:
        if direction_key == "INVALID_REVERSAL":
            self.game_state.record_invalid_reversal()
        elif direction_key == "EMPTY":
            self.game_state.record_empty_move()
        elif direction_key == "SOMETHING_IS_WRONG":
            self.game_state.record_something_is_wrong_move()
        elif direction_key == "NO_PATH_FOUND":
            self.game_state.record_no_path_found_move()
        return True  # Game continues, snake doesn't move
    
    # Use parent class's make_move method
    game_active, apple_eaten = super().make_move(direction_key)
    
    # If apple was eaten, advance to next apple position
    if apple_eaten and self.apple_index + 1 < len(self.apple_positions):
        self.apple_index += 1
        next_apple = self.apple_positions[self.apple_index]
        self.set_apple_position(next_apple)
    
    return game_active
```

### 4. Web Replay: `replay_web.py`

```python
def replay_thread_function(log_dir, move_pause, auto_advance):
    """Function to run the replay engine in a separate thread."""
    global replay_engine
    
    # Initialize replay engine
    replay_engine = WebReplayEngine(
        log_dir=log_dir,
        move_pause=move_pause,
        auto_advance=auto_advance
    )
    
    # Run the replay engine
    replay_engine.run_web()

@app.route('/api/state')
def get_state():
    """API endpoint to get the current game state."""
    global replay_engine
    
    if replay_engine is None:
        return jsonify({'error': 'Replay engine not initialized'})
    
    return jsonify(replay_engine.get_current_state())

@app.route('/api/control', methods=['POST'])
def control():
    """API endpoint to control the replay."""
    global replay_engine
    
    data = request.json
    command = data.get('command')
    
    if command == 'next_game':
        replay_engine.game_number += 1
        if not replay_engine.load_game_data(replay_engine.game_number):
            replay_engine.game_number -= 1
            return jsonify({'status': 'error', 'message': 'No next game'})
        return jsonify({'status': 'ok'})
    
    if command == 'prev_game':
        if replay_engine.game_number > 1:
            replay_engine.game_number -= 1
            replay_engine.load_game_data(replay_engine.game_number)
            return jsonify({'status': 'ok'})
        return jsonify({'status': 'error', 'message': 'Already at first game'})
```

### 5. JavaScript Controls: `web/static/js/replay.js`

```javascript
function handleKeyDown(event) {
    switch (event.key) {
        case ' ': // Space
            togglePlayPause();
            event.preventDefault();
            break;
        case 'ArrowLeft':
            sendCommand('prev_game');
            event.preventDefault();
            break;
        case 'ArrowRight':
            sendCommand('next_game');
            event.preventDefault();
            break;
        case 'r':
        case 'R':
            sendCommand('restart_game');
            event.preventDefault();
            break;
        case 'ArrowUp':
            sendCommand('speed_up'); // Up key - speed up (decrease pause time)
            event.preventDefault();
            break;
        case 'ArrowDown':
            sendCommand('speed_down'); // Down key - slow down (increase pause time)
            event.preventDefault();
            break;
    }
}

async function sendCommand(command) {
    try {
        const response = await fetch('/api/control', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ command })
        });
        
        const data = await response.json();
        
        if (data.error) {
            console.error(data.error);
        }
        
        // Update move pause display based on response
        if (data.move_pause) {
            movePauseValueElement.textContent = `${data.move_pause.toFixed(1)}s`;
        }
    } catch (error) {
        console.error('Error sending command:', error);
    }
}
```

### Key Features of Replay System

1. **Faithful Reproduction**: Exact replay of original games
2. **Sentinel Moves**: Handles special moves like invalid reversals
3. **Apple History**: Recreates exact apple spawn sequence
4. **Multi-Game Support**: Navigate between different games
5. **Speed Control**: Adjustable playback speed
6. **Cross-Platform**: Same replay logic for Pygame and Web
7. **Error Handling**: Graceful handling of missing or corrupted files

---

## Web vs Pygame Architecture

### Shared Components

Both platforms use the same core components:

1. **GameController**: Core game logic
2. **GameLogic**: LLM integration
3. **ReplayEngine**: Replay functionality
4. **Color Constants**: Single source of truth
5. **Game Constants**: Grid size, directions, etc.

### Pygame Architecture

```
main.py / human_play.py / replay.py
├── GameManager / GameController / ReplayEngine
├── GameGUI / HumanGameGUI / ReplayGUI
└── pygame (display, events, timing)
```

### Web Architecture

```
main_web.py / human_play_web.py / replay_web.py
├── GameManager / GameController / ReplayEngine (background thread)
├── Flask (API endpoints)
├── HTML Templates
├── JavaScript (frontend logic)
└── CSS (styling)
```

### Key Differences

#### 1. Event Handling

**Pygame**: Direct event polling
```python
for event in pygame.event.get():
    if event.type == KEYDOWN:
        if event.key == K_UP:
            game.make_move("UP")
```

**Web**: JavaScript event listeners + API calls
```javascript
document.addEventListener('keydown', handleKeyDown);

async function makeMove(direction) {
    await sendApiRequest('/api/move', 'POST', { direction });
}
```

#### 2. Rendering

**Pygame**: Direct screen drawing
```python
def draw_snake_segment(self, x, y, is_head):
    color = COLORS["SNAKE_HEAD"] if is_head else COLORS["SNAKE_BODY"]
    pygame.draw.rect(self.screen, color, (x * self.pixel, y * self.pixel, self.pixel, self.pixel))
```

**Web**: Canvas API
```javascript
function drawRect(ctx, x, y, color, pixelSize) {
    ctx.fillStyle = color;
    ctx.fillRect(
        x * pixelSize + 1,
        y * pixelSize + 1,
        pixelSize - 2,
        pixelSize - 2
    );
}
```

#### 3. State Management

**Pygame**: Direct object access
```python
score = game.score
steps = game.steps
snake_positions = game.snake_positions
```

**Web**: JSON API
```python
@app.route('/api/state')
def api_state():
    return jsonify(build_state_dict(manager))
```

```javascript
const data = await sendApiRequest('/api/state');
gameState = data;
scoreElement.textContent = gameState.score;
```

#### 4. Timing

**Pygame**: Direct clock control
```python
clock = pygame.time.Clock()
clock.tick(TIME_TICK)
```

**Web**: Polling with intervals
```javascript
updateInterval = setInterval(fetchGameState, 100);
```

### Benefits of This Architecture

1. **Code Reuse**: 90%+ of game logic shared
2. **Consistency**: Same behavior across platforms
3. **Maintainability**: Changes propagate to both platforms
4. **Performance**: Pygame for local, Web for remote access
5. **Flexibility**: Different UI paradigms for different use cases

---

## Key Design Principles

### 1. Single Source of Truth

- **Colors**: Defined once in `config/ui_constants.py`
- **Game Logic**: Core logic in `GameController`
- **Constants**: Grid size, directions, etc. in config files
- **State**: Game state managed centrally

### 2. Separation of Concerns

- **Game Logic**: Pure game mechanics
- **UI**: Platform-specific rendering
- **LLM Integration**: Separate from core game
- **Replay**: Extends game logic without modification

### 3. Inheritance Over Composition

- `GameController` → `GameLogic` → `ReplayEngine`
- `BaseGUI` → `GameGUI` → `HumanGameGUI`
- Allows code reuse while maintaining flexibility

### 4. Platform Abstraction

- Core logic works with any UI
- GUI interface standardized
- Web uses same game objects as Pygame

### 5. Error Handling

- Graceful degradation when files missing
- Fallback colors in web mode
- Network error handling
- Validation of game data

### 6. Configuration-Driven

- Colors, grid size, timing in config files
- Easy to modify without code changes
- Environment-specific defaults

### 7. Extensibility

- Easy to add new game modes
- New UI platforms can reuse core logic
- LLM providers can be swapped easily

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