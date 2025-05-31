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
            'primary_llm': self.primary_llm,
            'secondary_llm': self.secondary_llm,
            'paused': self.paused,
            'speed': 1.0 / self.pause_between_moves if self.pause_between_moves > 0 else 1.0,
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
    print("  • Adjust move pause: Up/Down arrow keys or +/- buttons")
    print("  • Restart game: R key or button")
    print("  • Exit: Ctrl+C in terminal\n")
    
    # Run Flask app
    app.run(host=args.host, port=args.port, debug=False, threaded=True)

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