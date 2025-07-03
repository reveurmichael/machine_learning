"""
Human Web Play Script - Simple Flask Interface
==============================================

Simple script to launch human-controlled Snake game web interface.
Uses simplified web architecture following KISS principles.

Design Philosophy:
- KISS: Simple, direct script without over-engineering
- DRY: Minimal code duplication
- No Over-Preparation: Only what's needed for human web play
- Extensible: Easy for Tasks 1-5 to copy and modify

Usage:
    python scripts/human_play_web.py                    # Default settings
    python scripts/human_play_web.py --grid-size 15     # Custom grid size
    python scripts/human_play_web.py --port 8080        # Specific port
"""

import sys
import argparse
import pathlib

# Ensure project root in sys.path for imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Now we can import from utils
from utils.path_utils import ensure_project_root

PROJECT_ROOT = ensure_project_root()

from utils.web_utils import get_web_dirs

TEMPLATE_DIR, STATIC_DIR = get_web_dirs()

from web import create_human_web_app
from utils.validation_utils import validate_grid_size, validate_port
from config.ui_constants import GRID_SIZE as DEFAULT_GRID_SIZE


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Snake Game - Human Web Interface")
    
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
    
    return parser.parse_args()


def main():
    """Main entry point for human web interface."""
    try:
        print("[HumanWebPlay] Starting Snake Game - Human Web Interface")
        
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        grid_size = validate_grid_size(args.grid_size)
        port = validate_port(args.port) if args.port else None
        
        # Create and run app
        app = create_human_web_app(grid_size=grid_size, port=port)
        
        print(f"[HumanWebPlay] Server starting on http://{args.host}:{app.port}")
        print("[HumanWebPlay] Use arrow keys to control snake")
        print("[HumanWebPlay] Press Ctrl+C to stop")
        
        app.run(host=args.host)
        
    except KeyboardInterrupt:
        print("\n[HumanWebPlay] Server stopped by user")
    except Exception as e:
        print(f"[HumanWebPlay] Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 
