"""
Snake Game Web Human Play Mode.
Provides a web-based interface for playing the snake game.
Reuses core game logic and implements a web interface for human control.
"""

import argparse
import threading
import time
from flask import Flask, render_template, request, jsonify
import logging

from core.game_controller import GameController
from utils.network_utils import find_free_port
from utils.web_utils import build_state_dict, translate_end_reason

# Initialize Flask app
app = Flask(__name__, static_folder='web/static', template_folder='web/templates')

# Global game controller instance
game_controller = None
game_thread = None
running = True

logging.getLogger('werkzeug').setLevel(logging.WARNING)


class WebGameController(GameController):
    """Extended game controller for web-based human play.
    Reuses the core functionality from GameController but adapts it for web display.
    """

    def __init__(self, grid_size=10):
        """Initialize the web game controller.

        Args:
            grid_size: Size of the game grid (fixed at 10x10 for web version)
        """
        # Initialize controller without GUI since we're using web interface
        super().__init__(grid_size=grid_size, use_gui=False)
        self.grid_size = grid_size
        self.game_over = False
        self.game_end_reason = None

    def get_current_state(self):
        """Get the current state for the web interface.
        Transforms the game state into a format suitable for JSON serialization.

        Returns:
            Dictionary with current game state
        """
        return build_state_dict(
            self.snake_positions,
            self.apple_position,
            self.score,
            self.steps,
            self.grid_size,
            extra={
                'game_over': self.game_over,
                'game_end_reason': translate_end_reason(self.game_end_reason),
            },
        )

    def make_move(self, direction_key):
        """Override the make_move method to track game over state.

        Args:
            direction_key: String key of the direction to move in ("UP", "DOWN", etc.)

        Returns:
            Tuple of (game_active, apple_eaten) where:
                game_active: Boolean indicating if the game is still active
                apple_eaten: Boolean indicating if an apple was eaten on this move
        """
        game_active, apple_eaten = super().make_move(direction_key)

        if not game_active:
            self.game_over = True
            if self.last_collision_type == "wall":
                self.game_end_reason = "WALL"
            elif self.last_collision_type == "self":
                self.game_end_reason = "SELF"

        return game_active, apple_eaten

    def reset(self):
        """Reset the game state including game over flag."""
        super().reset()
        self.game_over = False
        self.game_end_reason = None


def game_thread_function():
    """Background no-op loop to keep daemon thread alive."""

    while running:
        time.sleep(0.1)

# Define routes


@app.route('/')
def index():
    """Render the main game page."""
    return render_template('human_play.html')


@app.route('/api/state')
def get_state():
    """API endpoint to get the current game state."""

    if game_controller is None:
        return jsonify({'error': 'Game controller not initialized'})

    return jsonify(game_controller.get_current_state())


@app.route('/api/move', methods=['POST'])
def make_move():
    """API endpoint for making a move."""

    if game_controller is None:
        return jsonify({'error': 'Game controller not initialized'})

    data = request.json if request.json is not None else {}
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


@app.route('/api/reset', methods=['POST'])
def reset_game():
    """API endpoint to reset the game."""

    if game_controller is None:
        return jsonify({'error': 'Game controller not initialized'})

    game_controller.reset()

    return jsonify({'status': 'ok'})


def main():
    """Main function to run the web human play mode."""
    global game_controller, game_thread

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Web-based human play mode for Snake game.')
    parser.add_argument('--port', type=int, default=find_free_port(8000),
                        help='Port to run the web server on')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to run the web server on')
    args = parser.parse_args()

    # Initialize game controller
    game_controller = WebGameController(grid_size=10)  # Fixed 10x10 grid for web version

    # Start game thread
    game_thread = threading.Thread(target=game_thread_function)
    game_thread.daemon = True
    game_thread.start()

    # Start Flask app
    print(f"\n🐍 Snake Game Web Human Play starting at http://{args.host}:{args.port}")
    print("\nOpen the link in your browser to play the game.")
    print("\nControls:")
    print("  • Arrow Keys or WASD: Move Snake")
    print("  • R: Reset Game")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
