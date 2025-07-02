"""
Main Web Script - Full GameManager Integration
============================================

Script for launching LLM-controlled Snake game with full GameManager integration.
This provides complete LLM functionality with real-time web interface.

Design Philosophy:
- KISS: Use existing factory patterns instead of custom classes
- DRY: Reuse centralized web architecture 
- No Over-Preparation: Build only what's needed for main web interface
- Extensible: Template for Tasks 1-5 LLM implementations

Educational Value:
- Shows proper integration of Flask with GameManager
- Demonstrates real-time LLM game state synchronization
- Provides template for extension web interfaces

Usage:
    python scripts/main_web.py                          # Default LLM settings
    python scripts/main_web.py --provider deepseek      # Different LLM provider  
    python scripts/main_web.py --model gpt-4 --port 8080  # Custom model and port
    python scripts/main_web.py --max-games 3            # Play multiple games
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import web architecture
from utils.factory_utils import WebAppFactory
from utils.validation_utils import validate_grid_size, validate_port
from config.ui_constants import GRID_SIZE as DEFAULT_GRID_SIZE


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Snake Game - Full LLM Web Interface")
    
    parser.add_argument(
        "--provider",
        type=str,
        default="hunyuan",
        help="LLM provider (default: hunyuan)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="hunyuan-turbos-latest",
        help="LLM model (default: hunyuan-turbos-latest)"
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Grid size (default: {DEFAULT_GRID_SIZE})"
    )
    
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
    
    # GameManager related arguments (for full LLM functionality)
    parser.add_argument(
        "--max-games",
        type=int,
        default=1,
        help="Maximum number of games to play"
    )
    
    parser.add_argument(
        "--continue-with-game-in-dir",
        type=str,
        default=None,
        help="Continue from existing game directory"
    )
    
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without GUI"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for full LLM web interface."""
    try:
        print("[MainWebFull] Starting Snake Game - Full LLM Web Interface")
        
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        grid_size = validate_grid_size(args.grid_size)
        port = validate_port(args.port) if args.port else None
        
        # Create main web app using factory with full GameManager integration
        print(f"[MainWebFull] Creating main web app with GameManager integration")
        app = WebAppFactory.create(
            "main",
            provider=args.provider,
            model=args.model,
            grid_size=grid_size,
            max_games=args.max_games,
            port=port,
            continue_from_folder=args.continue_with_game_in_dir,
            no_gui=args.no_gui
        )
        
        print(f"[MainWebFull] Server starting on {app.url}")
        print(f"[MainWebFull] LLM Provider: {args.provider}/{args.model}")
        print(f"[MainWebFull] Max Games: {args.max_games}")
        if args.continue_with_game_in_dir:
            print(f"[MainWebFull] Continuing from: {args.continue_with_game_in_dir}")
        print("[MainWebFull] This provides full LLM functionality with GameManager")
        print("[MainWebFull] Use the web interface to start/stop/reset the game")
        print("[MainWebFull] Press Ctrl+C to stop")
        
        app.run(host=args.host)
        
    except KeyboardInterrupt:
        print("\n[MainWebFull] Server stopped by user")
    except Exception as e:
        print(f"[MainWebFull] Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 