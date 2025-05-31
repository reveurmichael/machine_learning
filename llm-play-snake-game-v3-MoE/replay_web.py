"""
Snake Game Web Replay Module.
Provides a web-based interface for replaying previously recorded games.
Reuses existing replay engine, constants, and game logic from the pygame implementation.
"""

import os
import sys
import json
import argparse
import threading
import time
from flask import Flask, render_template, request, jsonify, send_from_directory

from config import PAUSE_BETWEEN_MOVES_SECONDS, COLORS, GRID_SIZE
from replay.replay_engine import ReplayEngine

# Initialize Flask app
app = Flask(__name__, static_folder='web/static', template_folder='web/templates')

# Global replay engine instance
replay_engine = None
replay_thread = None
running = True

# End reason mapping - using the same mapping as in gui/replay_gui.py
END_REASON_MAP = {
    "WALL": "Hit Wall",
    "SELF": "Hit Self",
    "MAX_STEPS": "Max Steps",
    "EMPTY_MOVES": "Empty Moves",
    "ERROR": "LLM Error"
}

class WebReplayEngine(ReplayEngine):
    """Extended replay engine for web-based replay.
    Reuses the core functionality from ReplayEngine but adapts it for web display.
    """
    
    def __init__(self, log_dir, move_pause=1.0, auto_advance=False):
        """Initialize the web replay engine.
        
        Args:
            log_dir: Directory containing game logs
            move_pause: Time in seconds to pause between moves
            auto_advance: Whether to automatically advance through games
        """
        # Initialize without GUI since we're using web interface
        super().__init__(log_dir=log_dir, move_pause=move_pause, auto_advance=auto_advance, use_gui=False)
        self.paused = True  # Start paused until client connects

    def get_current_state(self):
        """Get the current state for the web interface.
        Transforms the replay engine state into a format suitable for JSON serialization.
        
        Returns:
            Dictionary with current game state
        """
        # Create state object with all needed information from replay engine
        state = {
            'snake_positions': self.snake_positions.tolist() if hasattr(self.snake_positions, 'tolist') else self.snake_positions,
            'apple_position': self.apple_position.tolist() if hasattr(self.apple_position, 'tolist') else self.apple_position,
            'game_number': self.game_number,
            'score': self.score,
            'steps': self.steps,
            'move_index': self.move_index,
            'total_moves': len(self.moves),
            'llm_response': self.llm_response,
            'primary_llm': self.primary_llm,
            'secondary_llm': self.secondary_llm,
            'paused': self.paused,
            'speed': 1.0 / self.pause_between_moves if self.pause_between_moves > 0 else 1.0,
            'timestamp': self.game_timestamp,
            'game_end_reason': self.game_end_reason,
            'grid_size': self.grid_size,
            'colors': {
                'snake_head': COLORS['SNAKE_HEAD'],
                'snake_body': COLORS['SNAKE_BODY'],
                'apple': COLORS['APPLE'],
                'background': COLORS['BACKGROUND'],
                'grid': COLORS['GRID'],
            }
        }
        
        return state
    
    def run_web(self):
        """Run the replay engine for web interface.
        Adaptation of the run() method from ReplayEngine but without GUI updates.
        """
        # Load initial game data
        self.load_game_data(self.game_number)
        
        # Main loop - similar to run() but without GUI updates
        self.running = True
        while self.running:
            # Process game updates if not paused
            if not self.paused:
                self.update()
            
            # Sleep to control update rate
            time.sleep(0.1)

def replay_thread_function(log_dir, move_pause, auto_advance):
    """Function to run the replay engine in a separate thread.
    
    Args:
        log_dir: Directory containing game logs
        move_pause: Time in seconds to pause between moves
        auto_advance: Whether to automatically advance through games
    """
    global replay_engine, running
    
    # Initialize replay engine
    replay_engine = WebReplayEngine(
        log_dir=log_dir,
        move_pause=move_pause,
        auto_advance=auto_advance
    )
    
    # Run the replay engine
    replay_engine.run_web()

# Define routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    """API endpoint to get the current game state."""
    global replay_engine
    
    if replay_engine is None:
        return jsonify({'error': 'Replay engine not initialized'})
    
    return jsonify(replay_engine.get_current_state())

@app.route('/api/control', methods=['POST'])
def control():
    """API endpoint to control the replay.
    Implements the same control functions as the keyboard handlers in the pygame version.
    """
    global replay_engine
    
    if replay_engine is None:
        return jsonify({'error': 'Replay engine not initialized'})
    
    data = request.json
    command = data.get('command')
    
    if command == 'pause':
        replay_engine.paused = True
        return jsonify({'status': 'paused'})
    
    elif command == 'play':
        replay_engine.paused = False
        return jsonify({'status': 'playing'})
    
    elif command == 'next_game':
        # Try to load next game - same logic as in replay.py
        replay_engine.game_number += 1
        if not replay_engine.load_game_data(replay_engine.game_number):
            replay_engine.game_number -= 1
            return jsonify({'status': 'error', 'message': 'No next game'})
        return jsonify({'status': 'ok'})
    
    elif command == 'prev_game':
        # Try to load previous game - same logic as in replay.py
        if replay_engine.game_number > 1:
            replay_engine.game_number -= 1
            replay_engine.load_game_data(replay_engine.game_number)
            return jsonify({'status': 'ok'})
        return jsonify({'status': 'error', 'message': 'Already at first game'})
    
    elif command == 'restart_game':
        # Restart current game - same logic as in replay.py
        replay_engine.load_game_data(replay_engine.game_number)
        return jsonify({'status': 'ok'})
    
    elif command == 'speed_up':
        # Increase replay speed - same logic as in replay.py
        replay_engine.pause_between_moves = max(0.1, replay_engine.pause_between_moves - 0.1)
        return jsonify({'status': 'ok', 'speed': 1.0 / replay_engine.pause_between_moves})
    
    elif command == 'speed_down':
        # Decrease replay speed - same logic as in replay.py
        replay_engine.pause_between_moves += 0.1
        return jsonify({'status': 'ok', 'speed': 1.0 / replay_engine.pause_between_moves})
    
    return jsonify({'status': 'error', 'message': 'Unknown command'})

def create_directories():
    """Create required directories for web interface."""
    # Create templates directory
    os.makedirs('web/templates', exist_ok=True)
    
    # Create static directory for CSS and JS
    os.makedirs('web/static/css', exist_ok=True)
    os.makedirs('web/static/js', exist_ok=True)

def main():
    """Main function to run the web replay."""
    global replay_thread
    
    # Parse command line arguments - reusing the same arguments from replay.py
    parser = argparse.ArgumentParser(description='Web-based replay for Snake game sessions.')
    parser.add_argument('log_dir', type=str, nargs='?', help='Directory containing game logs')
    parser.add_argument('--log-dir', type=str, dest='log_dir_opt', help='Directory containing game logs (alternative to positional argument)')
    parser.add_argument('--game', type=int, default=None, 
                      help='Specific game number (1-indexed) within the session to replay. If not specified, starts with game 1.')
    parser.add_argument(
        "--move-pause",
        type=float,
        default=PAUSE_BETWEEN_MOVES_SECONDS,
        help=f"Pause between moves in seconds (default: {PAUSE_BETWEEN_MOVES_SECONDS})",
    )
    parser.add_argument('--auto-advance', action='store_true', help='Automatically advance to next game')
    parser.add_argument('--start-paused', action='store_true', help='Start replay in paused state')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the web server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the web server on')
    args = parser.parse_args()

    # Use either positional argument or --log-dir option
    log_dir = args.log_dir_opt if args.log_dir_opt else args.log_dir
    
    if not log_dir:
        print("Error: Log directory must be specified either as a positional argument or using --log-dir")
        parser.print_help()
        sys.exit(1)
        
    # Create a sample log directory if 'logs/example' is requested
    if log_dir.replace('\\', '/') == 'logs/example':
        create_sample_log_directory()

    # Check if log directory exists
    if not os.path.isdir(log_dir):
        print(f"Log directory does not exist: {log_dir}")
        sys.exit(1)
    
    # Create required directories
    create_directories()
    
    # Create HTML, CSS, and JS files
    create_template_files()
    
    # Start replay engine in a separate thread
    replay_thread = threading.Thread(
        target=replay_thread_function,
        args=(log_dir, args.move_pause, args.auto_advance)
    )
    replay_thread.daemon = True
    replay_thread.start()
    
    # If specific game is provided, set it after engine starts
    if args.game is not None:
        # Wait a bit for engine to initialize
        time.sleep(1)
        if replay_engine:
            replay_engine.game_number = args.game
            replay_engine.load_game_data(args.game)
            
    # Set initial paused state if requested (matching pygame replay.py)
    if replay_engine and not args.start_paused:
        replay_engine.paused = False
    
    # Start Flask app
    print(f"\n🐍 Snake Game Web Replay starting at http://{args.host}:{args.port}")
    print("\nOpen the link in your browser to view the replay.")
    print("\nControls:")
    print("  • Play/Pause: Space bar or button")
    print("  • Navigate games: Left/Right arrow keys or buttons")
    print("  • Adjust speed: Up/Down arrow keys or +/- buttons")
    print("  • Restart game: R key or button")
    print("  • Exit: Close browser or Ctrl+C in terminal\n")
    
    # Run Flask app
    app.run(host=args.host, port=args.port, debug=False, threaded=True)

def create_template_files():
    """Create HTML, CSS, and JS template files."""
    # Create index.html
    with open('web/templates/index.html', 'w') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Snake Game Replay</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Snake Game Replay</h1>
            <div id="paused-indicator" class="paused-indicator">PAUSED</div>
        </div>
        
        <div id="loading-message" class="loading-message">
            <div class="spinner"></div>
            <p>Loading game data...</p>
        </div>
        
        <div class="game-container" id="game-container" style="display: none;">
            <div class="game-board">
                <canvas id="game-canvas"></canvas>
            </div>
            
            <div class="game-sidebar">
                <div class="stats-section">
                    <h2>Game Information</h2>
                    <div class="stats-grid">
                        <div>Game: <span id="game-number">1</span></div>
                        <div>Score: <span id="score">0</span></div>
                        <div id="end-reason-container">End Reason: <span id="end-reason"></span></div>
                    </div>
                </div>
                
                <div class="progress-section">
                    <h2>Progress</h2>
                    <div class="progress-bar-container">
                        <div id="progress-bar" class="progress-bar"></div>
                    </div>
                    <div class="progress-text">
                        <span id="progress">0/0 (0%)</span>
                    </div>
                </div>
                
                <div class="controls-section">
                    <h2>Controls</h2>
                    <div class="control-buttons">
                        <button id="prev-game">&lt;&lt; Prev</button>
                        <button id="restart">Restart</button>
                        <button id="play-pause">Play</button>
                        <button id="next-game">Next &gt;&gt;</button>
                    </div>
                    <div class="speed-controls">
                        <span>Speed: </span>
                        <button id="speed-down">-</button>
                        <span id="speed-value">1.0x</span>
                        <button id="speed-up">+</button>
                    </div>
                </div>
                
                <div class="llm-info-section">
                    <h2>LLM Information</h2>
                    <div>Primary: <span id="primary-llm">Unknown</span></div>
                    <div>Parser: <span id="secondary-llm">None</span></div>
                    <div>Timestamp: <span id="timestamp">Unknown</span></div>
                </div>
                
                <div class="llm-response-section">
                    <h2>LLM Response</h2>
                    <div id="llm-response-text" class="response-box"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="{{ url_for('static', filename='js/replay.js') }}"></script>
</body>
</html>""")
    
    # Create CSS file
    with open('web/static/css/style.css', 'w') as f:
        f.write("""/* Reset and base styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: Arial, sans-serif;
    background-color: #f5f5f5;
    color: #333;
    line-height: 1.6;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.header {
    margin-bottom: 20px;
    text-align: center;
    position: relative;
}

.header h1 {
    margin-bottom: 10px;
    color: #2c3e50;
}

.loading-message {
    text-align: center;
    margin: 50px auto;
    padding: 20px;
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    max-width: 300px;
}

.spinner {
    width: 40px;
    height: 40px;
    margin: 0 auto 20px;
    border: 4px solid rgba(0, 0, 0, 0.1);
    border-left-color: #4361ee;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.paused-indicator {
    position: absolute;
    top: 5px;
    right: 10px;
    padding: 5px 10px;
    background-color: #e63946;
    color: white;
    font-weight: bold;
    border-radius: 4px;
    display: none;
}

.game-container {
    display: flex;
    gap: 20px;
    align-items: flex-start;
}

.game-board {
    flex: 1;
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    padding: 15px;
}

#game-canvas {
    display: block;
    margin: 0 auto;
    background-color: #2c2c54;
}

.game-sidebar {
    width: 350px;
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.stats-section, .controls-section, .llm-info-section, .llm-response-section, .progress-section {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    padding: 15px;
}

h2 {
    margin-bottom: 15px;
    font-size: 18px;
    border-bottom: 1px solid #eee;
    padding-bottom: 8px;
    color: #3498db;
}

.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    font-size: 16px;
}

.control-buttons {
    display: flex;
    justify-content: space-between;
    margin-bottom: 15px;
}

button {
    padding: 8px 12px;
    background-color: #3498db;
    color: #fff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    transition: background-color 0.2s;
}

button:hover {
    background-color: #2980b9;
}

button:disabled {
    background-color: #a0a0a0;
    cursor: not-allowed;
}

.speed-controls {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-bottom: 15px;
}

.speed-controls button {
    width: 30px;
    padding: 4px;
}

.response-box {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 4px;
    padding: 10px;
    height: 200px;
    overflow-y: auto;
    font-family: monospace;
    white-space: pre-wrap;
    font-size: 12px;
}

.progress-bar-container {
    width: 100%;
    height: 20px;
    background-color: #ecf0f1;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 10px;
    border: 1px solid #bdc3c7;
}

.progress-bar {
    height: 100%;
    background-color: #3498db;
    width: 0%;
}

.progress-text {
    text-align: center;
    font-size: 14px;
    color: #7f8c8d;
}

#end-reason-container {
    grid-column: span 2;
    color: #e63946;
    font-weight: bold;
    display: none;
    margin-top: 10px;
}

/* Responsive design */
@media (max-width: 900px) {
    .game-container {
        flex-direction: column;
    }
    
    .game-sidebar {
        width: 100%;
    }
}""")
    
    # Create JavaScript file
    with open('web/static/js/replay.js', 'w') as f:
        f.write("""// Constants - colors will be overridden by values from the server
let COLORS = {
    SNAKE_HEAD: '#3498db',    // Blue for snake head
    SNAKE_BODY: '#2980b9',    // Darker blue for snake body
    APPLE: '#e74c3c',         // Red for apple
    BACKGROUND: '#2c3e50',    // Dark background
    GRID: '#34495e',          // Grid lines
};

// End reason mapping
const END_REASON_MAP = {
    "WALL": "Hit Wall",
    "SELF": "Hit Self",
    "MAX_STEPS": "Max Steps",
    "EMPTY_MOVES": "Empty Moves",
    "ERROR": "LLM Error"
};

// DOM elements
const loadingMessage = document.getElementById('loading-message');
const gameContainer = document.getElementById('game-container');
const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');
const gameNumber = document.getElementById('game-number');
const scoreElement = document.getElementById('score');
const progressElement = document.getElementById('progress');
const endReasonElement = document.getElementById('end-reason');
const endReasonContainer = document.getElementById('end-reason-container');
const primaryLlmElement = document.getElementById('primary-llm');
const secondaryLlmElement = document.getElementById('secondary-llm');
const timestampElement = document.getElementById('timestamp');
const llmResponseElement = document.getElementById('llm-response-text');
const playPauseButton = document.getElementById('play-pause');
const prevGameButton = document.getElementById('prev-game');
const nextGameButton = document.getElementById('next-game');
const restartButton = document.getElementById('restart');
const speedUpButton = document.getElementById('speed-up');
const speedDownButton = document.getElementById('speed-down');
const speedValueElement = document.getElementById('speed-value');
const progressBar = document.getElementById('progress-bar');
const pausedIndicator = document.getElementById('paused-indicator');

// Game state
let gameState = null;
let pixelSize = 0;
let updateInterval = null;
let retryCount = 0;
const MAX_RETRIES = 10;

// Initialize
function init() {
    setupEventListeners();
    startPolling();
    document.addEventListener('keydown', handleKeyDown);
}

function setupEventListeners() {
    playPauseButton.addEventListener('click', togglePlayPause);
    prevGameButton.addEventListener('click', () => sendCommand('prev_game'));
    nextGameButton.addEventListener('click', () => sendCommand('next_game'));
    restartButton.addEventListener('click', () => sendCommand('restart_game'));
    speedUpButton.addEventListener('click', () => sendCommand('speed_up'));
    speedDownButton.addEventListener('click', () => sendCommand('speed_down'));
}

function startPolling() {
    updateInterval = setInterval(fetchGameState, 100);
}

async function fetchGameState() {
    try {
        const response = await fetch('/api/state');
        const data = await response.json();
        
        if (data.error) {
            console.error(data.error);
            
            if (!gameState && retryCount < MAX_RETRIES) {
                retryCount++;
                return;
            }
            
            if (!gameState && retryCount >= MAX_RETRIES) {
                loadingMessage.innerHTML = `<p class="error-message">Error loading game data: ${data.error}</p>`;
                clearInterval(updateInterval);
                return;
            }
            
            return;
        }
        
        gameState = data;
        
        if (loadingMessage.style.display !== 'none') {
            loadingMessage.style.display = 'none';
            gameContainer.style.display = 'flex';
        }
        
        if (gameState.colors) {
            COLORS.SNAKE_HEAD = rgbArrayToHex(gameState.colors.snake_head) || COLORS.SNAKE_HEAD;
            COLORS.SNAKE_BODY = rgbArrayToHex(gameState.colors.snake_body) || COLORS.SNAKE_BODY;
            COLORS.APPLE = rgbArrayToHex(gameState.colors.apple) || COLORS.APPLE;
            COLORS.BACKGROUND = rgbArrayToHex(gameState.colors.background) || COLORS.BACKGROUND;
            COLORS.GRID = rgbArrayToHex(gameState.colors.grid) || COLORS.GRID;
        }
        
        updateUI();
        drawGame();
    } catch (error) {
        console.error('Error fetching game state:', error);
        
        if (!gameState && retryCount < MAX_RETRIES) {
            retryCount++;
            return;
        }
        
        if (!gameState && retryCount >= MAX_RETRIES) {
            loadingMessage.innerHTML = `<p class="error-message">Error connecting to server: ${error.message}</p>`;
            clearInterval(updateInterval);
            return;
        }
    }
}

function rgbArrayToHex(rgbArray) {
    if (!Array.isArray(rgbArray) || rgbArray.length < 3) {
        return null;
    }
    return `#${rgbArray[0].toString(16).padStart(2, '0')}${rgbArray[1].toString(16).padStart(2, '0')}${rgbArray[2].toString(16).padStart(2, '0')}`;
}

function updateUI() {
    if (!gameState) return;
    
    // Update game info
    gameNumber.textContent = gameState.game_number;
    scoreElement.textContent = gameState.score;
    
    // Update document title
    document.title = `Snake Game ${gameState.game_number} - Score: ${gameState.score}`;
    
    // Update progress
    progressElement.textContent = `${gameState.move_index}/${gameState.total_moves} (${Math.floor(gameState.move_index / Math.max(1, gameState.total_moves) * 100)}%)`;
    
    // Update progress bar
    if (gameState.total_moves > 0) {
        const progressPercent = (gameState.move_index / gameState.total_moves) * 100;
        progressBar.style.width = `${progressPercent}%`;
    } else {
        progressBar.style.width = '0%';
    }
    
    // Update end reason if available
    if (gameState.game_end_reason) {
        const reason = END_REASON_MAP[gameState.game_end_reason] || gameState.game_end_reason;
        endReasonElement.textContent = reason;
        endReasonContainer.style.display = 'block';
    } else {
        endReasonContainer.style.display = 'none';
    }
    
    // Update paused indicator
    pausedIndicator.style.display = gameState.paused ? 'block' : 'none';
    
    // Update LLM info
    primaryLlmElement.textContent = gameState.primary_llm || 'Unknown';
    secondaryLlmElement.textContent = gameState.secondary_llm || 'None';
    timestampElement.textContent = gameState.timestamp || 'Unknown';
    
    // Update LLM response
    if (gameState.llm_response) {
        llmResponseElement.textContent = gameState.llm_response;
    }
    
    // Update play/pause button
    playPauseButton.textContent = gameState.paused ? 'Play' : 'Pause';
    
    // Update speed display
    speedValueElement.textContent = `${gameState.speed.toFixed(1)}x`;
}

function drawGame() {
    if (!gameState) return;
    
    const gridSize = gameState.grid_size || 10;
    const maxSize = Math.min(window.innerWidth * 0.5, window.innerHeight * 0.7);
    
    canvas.width = maxSize;
    canvas.height = maxSize;
    
    pixelSize = Math.floor(maxSize / gridSize);
    
    // Clear canvas
    ctx.fillStyle = COLORS.BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid lines
    drawGrid(gridSize, pixelSize);
    
    // Draw snake
    if (gameState.snake_positions && gameState.snake_positions.length > 0) {
        // Draw body first
        for (let i = 0; i < gameState.snake_positions.length - 1; i++) {
            const [x, y] = gameState.snake_positions[i];
            drawRect(x, y, COLORS.SNAKE_BODY);
        }
        
        // Draw head
        const head = gameState.snake_positions[gameState.snake_positions.length - 1];
        drawRect(head[0], head[1], COLORS.SNAKE_HEAD);
    }
    
    // Draw apple
    if (gameState.apple_position && gameState.apple_position.length === 2) {
        const [x, y] = gameState.apple_position;
        drawRect(x, y, COLORS.APPLE);
    }
}

function drawGrid(gridSize, pixelSize) {
    ctx.strokeStyle = COLORS.GRID;
    ctx.lineWidth = 0.5;
    
    // Draw vertical lines
    for (let i = 0; i <= gridSize; i++) {
        ctx.beginPath();
        ctx.moveTo(i * pixelSize, 0);
        ctx.lineTo(i * pixelSize, gridSize * pixelSize);
        ctx.stroke();
    }
    
    // Draw horizontal lines
    for (let i = 0; i <= gridSize; i++) {
        ctx.beginPath();
        ctx.moveTo(0, i * pixelSize);
        ctx.lineTo(gridSize * pixelSize, i * pixelSize);
        ctx.stroke();
    }
}

function drawRect(x, y, color) {
    ctx.fillStyle = color;
    ctx.fillRect(
        x * pixelSize + 1,
        y * pixelSize + 1,
        pixelSize - 2,
        pixelSize - 2
    );
}

function togglePlayPause() {
    if (!gameState) return;
    const command = gameState.paused ? 'play' : 'pause';
    sendCommand(command);
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
        
        if (data.speed) {
            speedValueElement.textContent = `${data.speed.toFixed(1)}x`;
        }
    } catch (error) {
        console.error('Error sending command:', error);
    }
}

function handleKeyDown(event) {
    switch (event.key) {
        case ' ': // Space
            togglePlayPause();
            break;
        case 'ArrowLeft':
            sendCommand('prev_game');
            break;
        case 'ArrowRight':
            sendCommand('next_game');
            break;
        case 'r':
        case 'R':
            sendCommand('restart_game');
            break;
        case 'ArrowUp':
            sendCommand('speed_up');
            break;
        case 'ArrowDown':
            sendCommand('speed_down');
            break;
        case 'Escape':
            window.close();
            break;
    }
}

// Handle window resize
window.addEventListener('resize', drawGame);

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);""")

def create_sample_log_directory():
    """Create a sample log directory with game data for testing."""
    sample_dir = os.path.join('logs', 'example')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create a sample game data file
    sample_game_data = {
        "score": 3,
        "steps": 45,
        "game_end_reason": "WALL",
        "metadata": {
            "timestamp": "2023-06-15 12:34:56",
            "round_count": 3
        },
        "llm_info": {
            "primary_provider": "OpenAI",
            "primary_model": "GPT-4",
            "parser_provider": "Claude",
            "parser_model": "Claude-3"
        },
        "detailed_history": {
            "moves": ["UP", "RIGHT", "RIGHT", "UP", "LEFT", "DOWN", "RIGHT", "UP", "UP", "RIGHT"],
            "apple_positions": [
                {"x": 5, "y": 5},
                {"x": 8, "y": 3},
                {"x": 2, "y": 7}
            ],
            "llm_response": "I'll move UP first to get closer to the apple, then RIGHT twice to approach it.",
            "planned_moves": ["UP", "RIGHT", "RIGHT"]
        }
    }
    
    # Create game 1
    with open(os.path.join(sample_dir, 'game_1.json'), 'w') as f:
        json.dump(sample_game_data, f, indent=2)
    
    # Create game 2 with different end reason
    game2_data = sample_game_data.copy()
    game2_data["score"] = 5
    game2_data["steps"] = 60
    game2_data["game_end_reason"] = "SELF"
    game2_data["metadata"]["timestamp"] = "2023-06-15 13:45:22"
    with open(os.path.join(sample_dir, 'game_2.json'), 'w') as f:
        json.dump(game2_data, f, indent=2)
    
    print(f"Created sample log directory with 2 games: {sample_dir}")
    return sample_dir

if __name__ == "__main__":
    # Check if 'logs/example' is requested as an argument
    if len(sys.argv) > 1 and sys.argv[1].replace('\\', '/') == 'logs/example':
        create_sample_log_directory()
    elif '--log-dir' in sys.argv and sys.argv[sys.argv.index('--log-dir')+1].replace('\\', '/') == 'logs/example':
        create_sample_log_directory()
    
    main() 