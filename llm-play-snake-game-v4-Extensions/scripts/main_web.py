"""
Main Web Script - Full GameManager Integration
============================================

Script for launching LLM-controlled Snake game with full GameManager integration.
This provides complete LLM functionality with real-time web interface.

Design Philosophy:
- Full Feature Parity: Mirrors all main.py capabilities
- Real-time Integration: Background GameManager with live state updates
- No Over-Preparation: Only implements what's needed for web interface
- Extensible: Template for Tasks 1-5 LLM implementations

Educational Value:
- Shows proper Flask + GameManager integration
- Demonstrates real-time LLM game state synchronization
- Provides template for extension web interfaces

Usage:
    python scripts/main_web.py                          # Default LLM settings
    python scripts/main_web.py --provider deepseek      # Different LLM provider  
    python scripts/main_web.py --model gpt-4 --port 8080  # Custom model and port
    python scripts/main_web.py --max-games 3            # Play multiple games
    python scripts/main_web.py --continue-with-game-in-dir logs/...  # Continue session
"""

import sys
import argparse
import threading
import time
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import main.py components for full feature parity
from scripts.main import get_parser, parse_arguments
from core.game_manager import GameManager
from llm.agent_llm import SnakeAgent
from utils.web_utils import translate_end_reason, build_color_map, build_state_dict
from config.game_constants import GRID_SIZE

# Flask imports
from flask import Flask, render_template, jsonify, request

# Global state for cross-request access
manager = None
manager_thread = None


def parse_web_arguments():
    """Parse command line arguments for web mode."""
    parser = argparse.ArgumentParser(description="Snake Game - Full LLM Web Interface")
    
    # Web-specific arguments
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number (default: auto-detect)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address (default: 127.0.0.1)"
    )
    
    # Parse web-specific args first
    web_args, remaining_argv = parser.parse_known_args()
    
    # Now parse all the main.py arguments
    sys.argv_backup = sys.argv.copy()
    sys.argv = [sys.argv[0]] + remaining_argv
    
    try:
        game_args = parse_arguments()
    finally:
        sys.argv = sys.argv_backup
    
    return web_args, game_args


def manager_thread_fn(gm: GameManager, args):
    """Background worker for running the game."""
    try:
        # Handle continuation mode
        cont_dir = getattr(args, "continue_with_game_in_dir", None)
        if cont_dir:
            # Determine next game number for continuation
            from utils.file_utils import get_next_game_number
            next_game = get_next_game_number(cont_dir)
            
            # Load existing game session
            gm.continue_from_session(cont_dir, next_game)
        else:
            # Start new game session
            gm.run()
    except Exception as e:
        print(f"[main_web] GameManager thread crashed: {e}")


def build_game_state_dict(gm: GameManager):
    """Convert GameManager state to JSON-serializable dict."""
    game = gm.game
    
    # Get basic state using the utility function
    reason_code = getattr(game.game_state, "game_end_reason", None)
    end_reason_readable = translate_end_reason(reason_code)
    
    # Build extra state for LLM mode
    extra_state = {
        'game_number': gm.game_count + 1,
        'round_count': gm.round_count,
        'running': gm.running,
        'game_active': gm.game_active,
        'planned_moves': getattr(game, 'planned_moves', []),
        'llm_response': getattr(game, 'processed_response', ''),
        'move_pause': gm.get_pause_between_moves(),
        'game_end_reason': end_reason_readable,
    }
    
    # Use the utility function for basic state
    return build_state_dict(
        snake_positions=game.snake_positions,
        apple_position=game.apple_position,
        score=game.score,
        steps=game.steps,
        grid_size=GRID_SIZE,
        extra=extra_state
    )


def main():
    """Main entry point for full LLM web interface."""
    global manager, manager_thread
    
    try:
        print("[MainWebFull] Starting Snake Game - Full LLM Web Interface")
        
        # Parse arguments (web + game)
        web_args, game_args = parse_web_arguments()
        
        # Set up pygame dummy driver for headless mode
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        
        # Create Flask app
        app = Flask(__name__, static_folder='web/static', template_folder='web/templates')
        
        # Enable GUI mode for web interface to get the 3-second preview pause
        # This allows users to see the LLM's plan before the snake starts moving
        # The pause occurs in core/game_loop.py when use_gui=True and a new plan is received
        game_args.use_gui = True
        
        # -------------------------------
        # Step 1 – Create GameManager with full CLI support
        # -------------------------------
        print(f"[MainWebFull] Creating GameManager with full integration")
        manager = GameManager(game_args)
        
        # Set up LLM agent for Task-0
        manager.agent = SnakeAgent(
            manager, 
            provider=game_args.provider, 
            model=game_args.model
        )
        
        # -------------------------------
        # Step 2 – Handle continuation mode configuration
        # -------------------------------
        if game_args.continue_with_game_in_dir:
            try:
                import json
                summary_path = os.path.join(game_args.continue_with_game_in_dir, "summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    original_cfg = summary.get("configuration", {})
                    for k in (
                        "provider", "model", "parser_provider", "parser_model",
                        "move_pause", "max_steps", "max_consecutive_empty_moves_allowed",
                        "max_consecutive_something_is_wrong_allowed",
                        "max_consecutive_invalid_reversals_allowed",
                        "max_consecutive_no_path_found_allowed",
                        "sleep_after_empty_step", "no_gui",
                    ):
                        if k in original_cfg:
                            setattr(manager.args, k, original_cfg[k])
                    print(f"[MainWebFull] Loaded continuation config from {summary_path}")
            except Exception as e:
                print(f"[MainWebFull] Warning: Could not load continuation config: {e}")
        
        # -------------------------------
        # Step 3 – Start GameManager in background thread
        # -------------------------------
        manager_thread = threading.Thread(
            target=manager_thread_fn, 
            args=(manager, game_args), 
            daemon=True
        )
        manager_thread.start()
        
        # -------------------------------
        # Step 4 – Define Flask routes
        # -------------------------------
        @app.route('/')
        def index():
            """Serve the main HTML page."""
            return render_template('main.html')
        
        @app.route('/api/state')
        def api_state():
            """Return current game state for frontend polling."""
            if manager is None or manager.game is None:
                return jsonify({'error': 'game not started'})
            return jsonify(build_game_state_dict(manager))
        
        @app.route('/api/control', methods=['POST'])
        def api_control():
            """Handle control commands (pause/play)."""
            if manager is None:
                return jsonify({'status': 'error', 'msg': 'no manager'})
            
            cmd = request.json.get('command') if request.is_json else None
            if cmd == 'pause':
                manager.running = False  # stops outer while loops – this effectively pauses
                return jsonify({'status': 'paused'})
            if cmd == 'play':
                # Not trivial to resume once gm.run() has returned, so just acknowledge
                return jsonify({'status': 'unpause-not-supported'})
            
            return jsonify({'status': 'error', 'msg': 'unknown command'})
        
        # -------------------------------
        # Step 5 – Start the Flask app (blocking)
        # -------------------------------
        host = web_args.host
        port = web_args.port or 5001
        
        print(f"[MainWebFull] Server starting on http://{host}:{port}")
        print(f"[MainWebFull] LLM Provider: {game_args.provider}/{game_args.model}")
        print(f"[MainWebFull] Max Games: {game_args.max_games}")
        if game_args.continue_with_game_in_dir:
            print(f"[MainWebFull] Continuing from: {game_args.continue_with_game_in_dir}")
        print("[MainWebFull] This provides full LLM functionality with GameManager")
        print("[MainWebFull] Use the web interface to watch the game in real-time")
        print("[MainWebFull] Note: 3-second preview pause enabled for better UX")
        print("[MainWebFull] Press Ctrl+C to stop")
        
        app.run(host=host, port=port, threaded=True, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n[MainWebFull] Server stopped by user")
    except Exception as e:
        print(f"[MainWebFull] Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 