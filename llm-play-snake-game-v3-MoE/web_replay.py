#!/usr/bin/env python3
"""
Web-based Snake Game Replay

A Flask-based web server that replays recorded Snake game sessions in a browser.
"""

import os
import json
import argparse
import time
from flask import Flask, render_template, jsonify, send_from_directory

# Initialize Flask app
app = Flask(__name__)

# Global variables to hold game data
GAME_DATA = None
LOG_DIR = None
GAME_NUMBER = None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Web-based Snake Game Replay')
    parser.add_argument('--log-dir', type=str, required=True,
                       help='Directory containing game logs')
    parser.add_argument('--game', type=int, required=True,
                       help='Game number to replay')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host to bind the server to (default: localhost)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to run the server on (default: 8000)')
    
    return parser.parse_args()

def load_game_data(log_dir, game_number):
    """Load game data from log directory.
    
    Args:
        log_dir: Path to the log directory
        game_number: Game number to load
        
    Returns:
        Dictionary with game data or None if not found
    """
    game_file = os.path.join(log_dir, f"game_{game_number}.json")
    
    if not os.path.exists(game_file):
        print(f"Error: Game file {game_file} not found.")
        return None
    
    try:
        with open(game_file, 'r') as f:
            game_data = json.load(f)
        return game_data
    except Exception as e:
        print(f"Error loading game data: {e}")
        return None

# Create templates directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)

# Create static directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)

# HTML template for the replay page
@app.route('/')
def index():
    """Render the main replay page."""
    if not GAME_DATA:
        return "Game data not loaded. Please check the command-line arguments.", 500
    
    # Extract basic game information for the template
    game_info = {
        'game_number': GAME_NUMBER,
        'score': GAME_DATA.get('score', 0),
        'steps': GAME_DATA.get('steps', 0),
        'game_end_reason': GAME_DATA.get('game_end_reason', 'Unknown'),
        'snake_length': GAME_DATA.get('snake_length', 1),
        'total_rounds': len(GAME_DATA.get('detailed_history', {}).get('rounds_data', {}))
    }
    
    return render_template('replay.html', game_info=game_info)

@app.route('/game-data')
def game_data():
    """Return the full game data as JSON for the frontend."""
    return jsonify(GAME_DATA)

# Serve the static files
@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

# Create the HTML template file
def create_template_files():
    """Create the HTML template file for the replay page."""
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'replay.html')
    
    # Only create if it doesn't exist
    if not os.path.exists(template_path):
        with open(template_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Snake Game Replay</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .game-info {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
        }
        .info-card {
            background-color: white;
            border-radius: 5px;
            padding: 10px;
            flex: 1;
            min-width: 120px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .info-card h3 {
            margin: 0;
            color: #333;
        }
        .info-card p {
            font-size: 24px;
            font-weight: bold;
            margin: 5px 0 0 0;
            color: #4CAF50;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            align-items: center;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        #speed-control {
            margin-left: auto;
        }
        #game-board {
            width: 100%;
            max-width: 600px;
            aspect-ratio: 1;
            margin: 0 auto;
            background-color: #111;
            position: relative;
            border-radius: 5px;
            overflow: hidden;
        }
        .snake-segment {
            position: absolute;
            background-color: #4CAF50;
            border-radius: 3px;
        }
        .apple {
            position: absolute;
            background-color: #FF0000;
            border-radius: 50%;
        }
        .head {
            background-color: #2E7D32;
        }
        #step-info {
            text-align: center;
            margin-top: 10px;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐍 Snake Game Replay</h1>
        </div>
        
        <div class="game-info">
            <div class="info-card">
                <h3>Game</h3>
                <p>{{ game_info.game_number }}</p>
            </div>
            <div class="info-card">
                <h3>Score</h3>
                <p>{{ game_info.score }}</p>
            </div>
            <div class="info-card">
                <h3>Steps</h3>
                <p>{{ game_info.steps }}</p>
            </div>
            <div class="info-card">
                <h3>End Reason</h3>
                <p>{{ game_info.game_end_reason }}</p>
            </div>
            <div class="info-card">
                <h3>Snake Length</h3>
                <p>{{ game_info.snake_length }}</p>
            </div>
            <div class="info-card">
                <h3>Rounds</h3>
                <p>{{ game_info.total_rounds }}</p>
            </div>
        </div>
        
        <div class="controls">
            <button id="play-btn">Play</button>
            <button id="pause-btn" disabled>Pause</button>
            <button id="step-btn">Step Forward</button>
            <button id="reset-btn">Reset</button>
            <div id="speed-control">
                <label for="speed">Speed:</label>
                <input type="range" id="speed" min="1" max="10" value="5">
            </div>
        </div>
        
        <div id="game-board"></div>
        <div id="step-info">Step: 0 / {{ game_info.steps }}</div>
    </div>
    
    <script>
        // Game state variables
        let gameData = null;
        let currentStep = 0;
        let totalSteps = {{ game_info.steps }};
        let playInterval = null;
        let speed = 5;
        let boardSize = 20; // Default board size
        let cellSize = 0;
        let snake = [];
        let apple = null;
        
        // DOM elements
        const gameBoard = document.getElementById('game-board');
        const playBtn = document.getElementById('play-btn');
        const pauseBtn = document.getElementById('pause-btn');
        const stepBtn = document.getElementById('step-btn');
        const resetBtn = document.getElementById('reset-btn');
        const speedSlider = document.getElementById('speed');
        const stepInfo = document.getElementById('step-info');
        
        // Fetch game data from the server
        async function fetchGameData() {
            try {
                const response = await fetch('/game-data');
                gameData = await response.json();
                
                // Set up the board size
                boardSize = gameData.board_size || 20;
                cellSize = gameBoard.offsetWidth / boardSize;
                
                // Initialize the game
                resetGame();
            } catch (error) {
                console.error('Error fetching game data:', error);
            }
        }
        
        // Reset the game to the beginning
        function resetGame() {
            clearInterval(playInterval);
            currentStep = 0;
            snake = [];
            apple = null;
            
            // Initialize snake at starting position (if available in data)
            if (gameData && gameData.detailed_history && gameData.detailed_history.initial_snake) {
                snake = gameData.detailed_history.initial_snake.map(pos => ({ x: pos[0], y: pos[1] }));
            } else {
                // Default starting position
                snake = [{ x: Math.floor(boardSize / 2), y: Math.floor(boardSize / 2) }];
            }
            
            // Initialize apple position (if available)
            if (gameData && gameData.detailed_history && gameData.detailed_history.apple_positions && 
                gameData.detailed_history.apple_positions.length > 0) {
                const firstApple = gameData.detailed_history.apple_positions[0];
                apple = { x: firstApple.x, y: firstApple.y };
            }
            
            updateGameBoard();
            updateStepInfo();
            
            // Update button states
            playBtn.disabled = false;
            pauseBtn.disabled = true;
            stepBtn.disabled = false;
        }
        
        // Update the visual game board
        function updateGameBoard() {
            // Clear the board
            gameBoard.innerHTML = '';
            
            // Draw snake
            snake.forEach((segment, index) => {
                const segmentElement = document.createElement('div');
                segmentElement.className = 'snake-segment';
                if (index === 0) {
                    segmentElement.classList.add('head');
                }
                
                segmentElement.style.width = `${cellSize}px`;
                segmentElement.style.height = `${cellSize}px`;
                segmentElement.style.left = `${segment.x * cellSize}px`;
                segmentElement.style.top = `${segment.y * cellSize}px`;
                
                gameBoard.appendChild(segmentElement);
            });
            
            // Draw apple
            if (apple) {
                const appleElement = document.createElement('div');
                appleElement.className = 'apple';
                appleElement.style.width = `${cellSize}px`;
                appleElement.style.height = `${cellSize}px`;
                appleElement.style.left = `${apple.x * cellSize}px`;
                appleElement.style.top = `${apple.y * cellSize}px`;
                
                gameBoard.appendChild(appleElement);
            }
        }
        
        // Update step information
        function updateStepInfo() {
            stepInfo.textContent = `Step: ${currentStep} / ${totalSteps}`;
        }
        
        // Advance one step in the game
        function step() {
            if (!gameData || currentStep >= totalSteps) {
                pausePlayback();
                return;
            }
            
            // Get move for the current step
            const move = gameData.detailed_history.moves[currentStep];
            
            if (move) {
                // Update snake position based on move
                const head = { ...snake[0] };
                
                switch (move) {
                    case 'UP':
                        head.y -= 1;
                        break;
                    case 'DOWN':
                        head.y += 1;
                        break;
                    case 'LEFT':
                        head.x -= 1;
                        break;
                    case 'RIGHT':
                        head.x += 1;
                        break;
                    default:
                        // For EMPTY moves, do nothing
                        break;
                }
                
                // Check if apple was eaten
                const appleEaten = apple && head.x === apple.x && head.y === apple.y;
                
                // Add new head to snake
                snake.unshift(head);
                
                // If apple wasn't eaten, remove tail
                if (!appleEaten) {
                    snake.pop();
                } else {
                    // Update apple position for the next round if available
                    const roundChangeAt = gameData.detailed_history.rounds_data ? 
                        Object.keys(gameData.detailed_history.rounds_data)
                            .findIndex(r => parseInt(r.split('_')[1]) > currentStep + 1) : -1;
                    
                    if (roundChangeAt >= 0) {
                        const nextRound = Object.values(gameData.detailed_history.rounds_data)[roundChangeAt];
                        if (nextRound && nextRound.apple_position) {
                            apple = { x: nextRound.apple_position[0], y: nextRound.apple_position[1] };
                        } else {
                            apple = null;
                        }
                    } else {
                        apple = null;
                    }
                }
            }
            
            currentStep++;
            updateGameBoard();
            updateStepInfo();
            
            // Stop playback if we've reached the end
            if (currentStep >= totalSteps) {
                pausePlayback();
            }
        }
        
        // Start playback
        function startPlayback() {
            if (currentStep >= totalSteps) {
                resetGame();
            }
            
            clearInterval(playInterval);
            playInterval = setInterval(step, 1000 / speed);
            
            playBtn.disabled = true;
            pauseBtn.disabled = false;
            stepBtn.disabled = true;
        }
        
        // Pause playback
        function pausePlayback() {
            clearInterval(playInterval);
            playBtn.disabled = false;
            pauseBtn.disabled = true;
            stepBtn.disabled = false;
        }
        
        // Event listeners
        playBtn.addEventListener('click', startPlayback);
        pauseBtn.addEventListener('click', pausePlayback);
        stepBtn.addEventListener('click', step);
        resetBtn.addEventListener('click', resetGame);
        
        speedSlider.addEventListener('input', function() {
            speed = parseInt(this.value);
            if (playInterval) {
                pausePlayback();
                startPlayback();
            }
        });
        
        // Handle window resize
        window.addEventListener('resize', function() {
            cellSize = gameBoard.offsetWidth / boardSize;
            updateGameBoard();
        });
        
        // Initialize
        fetchGameData();
    </script>
</body>
</html>""")
    
    # Create a simple CSS file
    css_path = os.path.join(os.path.dirname(__file__), 'static', 'style.css')
    if not os.path.exists(css_path):
        with open(css_path, 'w') as f:
            f.write("""/* Snake Game Replay Styles */
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f0f0f0;
}
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}
/* Additional styles are included inline in the HTML */
""")

def main():
    """Main entry point for the web replay."""
    global GAME_DATA, LOG_DIR, GAME_NUMBER
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Store global values
    LOG_DIR = args.log_dir
    GAME_NUMBER = args.game
    
    # Load game data
    GAME_DATA = load_game_data(LOG_DIR, GAME_NUMBER)
    
    if not GAME_DATA:
        print(f"Error: Could not load game data for game {GAME_NUMBER} from {LOG_DIR}")
        return
    
    # Create template files if they don't exist
    create_template_files()
    
    # Start Flask server
    print(f"Starting web replay server for Game {GAME_NUMBER} at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main() 