"""
Snake Game - Task-0 LLM Web Interface (MVC Architecture)
--------------------

Flask-based web application for LLM-driven Snake gameplay using the MVC framework.
This script demonstrates how Task-0 integrates with the excellent web MVC architecture
and serves as a foundation for Task 1-5 extensions.

Features:
- Clean MVC architecture using web.factories
- LLM integration with multiple providers
- Dynamic port allocation with network utilities
- KISS principles and elegant error handling
- Extensible foundation for future tasks
- Simple logging following SUPREME_RULES

Design Patterns Used:
    - Factory Pattern: Uses web.factories for consistent component creation
    - Template Method Pattern: Leverages BaseFlaskApp lifecycle
    - Strategy Pattern: Pluggable LLM providers and game strategies
    - Observer Pattern: Real-time updates via MVC framework

Educational Goals:
    - Demonstrate clean web MVC integration for Task-0 LLM
    - Show how future extensions can reuse this pattern
    - Illustrate LLM integration in web applications
    - Provide canonical example of Task-0 LLM web interface

Extension Pattern for Future Tasks:
    Task-1 (Heuristics): Replace LLM with pathfinding algorithms
    Task-2 (RL): Replace with RL agent and training monitoring
    Task-3 (Supervised): Replace with ML model evaluation
    Task-4 (Distillation): Replace with knowledge distillation
    Task-5 (Advanced): Combine multiple AI strategies
"""

import sys
import pathlib
import argparse
import threading
import time
from typing import Optional

# Bootstrap repository root for consistent imports
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.path_utils import ensure_project_root
ensure_project_root()

# Import Task-0 LLM components
from core.game_manager import GameManager
from llm.agent_llm import SnakeAgent
from scripts.main import parse_arguments

# Import simple web framework
from web.game_flask_app import LLMGameApp, create_llm_app
from utils.network_utils import get_server_host_port

# Simple logging following SUPREME_RULES
print_log = lambda msg: print(f"[LLMWebApp] {msg}")


class LLMWebApp(LLMGameApp):
    """
    Task-0 LLM Game Web Application.
    
    Extends LLMGameApp with specialized LLM game configuration.
    Demonstrates how to integrate Task-0 GameManager with simple web architecture.
    
    Design Pattern: Template Method Pattern (Flask Application Lifecycle)
    Educational Value: Shows how to extend simple applications for LLM gameplay
    Extension Pattern: Future tasks can extend this for their AI-specific needs
    """
    
    def __init__(self, game_args, **config):
        """
        Initialize LLM game web application.
        
        Args:
            game_args: Parsed arguments from scripts.main
            **config: Additional configuration options
        """
        super().__init__(
            llm_provider=game_args.provider,
            grid_size=getattr(game_args, 'grid_size', 10),
            **config
        )
        self.game_args = game_args
        self.game_manager = None
        print_log(f"Initialized for LLM play with {game_args.provider}/{game_args.model}")
    
    def get_application_info(self) -> dict:
        """Get LLM-specific application information."""
        return {
            "name": "Task-0 LLM Player",
            "task_name": "task0",
            "game_mode": "llm",
            "llm_provider": self.game_args.provider,
            "llm_model": self.game_args.model,
            "grid_size": getattr(self.game_args, 'grid_size', 10),
            "url": f"http://127.0.0.1:{getattr(self, 'port', 5000)}",
            "features": [
                "LLM-driven gameplay",
                "Real-time state updates",
                "Game statistics tracking",
                "Pause/resume controls",
                "Performance monitoring",
                "Multiple LLM providers"
            ]
        }


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for LLM web interface.
    
    Educational Value: Shows how to combine existing argument parsing with web-specific options
    Extension Pattern: Future tasks can extend this pattern for their specific arguments
    """
    parser = argparse.ArgumentParser(
        description="Snake Game - LLM Web Interface (Task-0 Foundation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/main_web.py                              # Default LLM settings
  python scripts/main_web.py --provider hunyuan          # Specific LLM provider
  python scripts/main_web.py --model gpt-4 --port 8080   # Custom model and port
  python scripts/main_web.py --debug                     # Debug mode

Extension Pattern:
  Future tasks can copy this script and modify:
  - Replace GameManager with their algorithm/model manager
  - Add task-specific LLM integration
  - Customize web interface for their needs
  - Maintain same elegant MVC structure

LLM Providers:
  - hunyuan: Tencent Hunyuan models
  - openai: OpenAI GPT models  
  - anthropic: Anthropic Claude models
  - local: Local/self-hosted models
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the web server (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number for the web server (default: auto-detect free port)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode for development"
    )
    
    return parser


def main() -> int:
    """
    Main entry point for Task-0 LLM web interface.
    
    Educational Value: Shows elegant integration of existing CLI args with web interface
    Extension Pattern: Future tasks can copy this exact pattern
    
    Returns:
        Exit code: 0 for success, 1 for failure
    """
    try:
        # Parse web-specific arguments first
        web_parser = create_argument_parser()
        web_args, remaining_argv = web_parser.parse_known_args()
        
        # Parse game arguments using existing main.py parser
        argv_backup = sys.argv.copy()
        sys.argv = [sys.argv[0]] + remaining_argv
        try:
            game_args = parse_arguments()
        finally:
            sys.argv = argv_backup
        
        # Get host and port using network utilities
        host, port = get_server_host_port(default_host=web_args.host, default_port=web_args.port)
        # Network utilities handle environment variables and port conflicts automatically
        
        print_log("🐍 Starting Snake Game - LLM Web Interface")
        print_log(f"📊 Architecture: Task-0 MVC Framework")
        print_log(f"🤖 LLM: {game_args.provider}/{game_args.model}")
        print_log(f"📐 Grid: {getattr(game_args, 'grid_size', 10)}x{getattr(game_args, 'grid_size', 10)}")
        print_log(f"🌐 Server: http://{host}:{port}")
        print()
        
        # Create LLM game application using elegant architecture
        app = LLMWebApp(
            provider=game_args.provider,
            model=game_args.model,
            grid_size=getattr(game_args, 'grid_size', 10)
        )
        
        # Show application info
        app_info = app.get_application_info()
        print_log("🎯 Application Information:")
        print_log(f"   Name: {app_info['name']}")
        print_log(f"   Task: {app_info['task_name']}")
        print_log(f"   Mode: {app_info.get('game_mode', 'unknown')}")
        print_log(f"   Provider: {app_info.get('llm_provider', 'unknown')}")
        print_log(f"   Model: {app_info.get('llm_model', 'unknown')}")
        print_log(f"   URL: {app_info['url']}")
        print()
        
        print_log("🎯 MVC Components:")
        mvc_info = app_info.get('mvc_components', {})
        print_log(f"   Controller: {mvc_info.get('controller', 'Unknown')}")
        print_log(f"   Model: {mvc_info.get('model', 'Unknown')}")
        print_log(f"   View: {mvc_info.get('view_renderer', 'Unknown')}")
        print()
        
        print_log("📡 API Endpoints:")
        print_log("   GET  /                 - Main game interface")
        print_log("   GET  /api/state        - Current game state")
        print_log("   POST /api/control      - Game commands (pause/resume)")
        print_log("   POST /api/reset        - Reset game")
        print_log("   GET  /api/health       - System health check")
        print()
        
        print_log("🎮 Controls:")
        print_log("   Web Interface: Real-time LLM gameplay")
        print_log("   Pause/Resume: Available via web controls")
        print_log("   Ctrl+C: Stop server")
        print()
        
        print_log("🚀 Extension Pattern:")
        print_log("   Future tasks can copy this script structure")
        print_log("   Replace GameManager with task-specific managers")
        print_log("   Integrate different AI approaches seamlessly")
        print_log("   Maintain same elegant MVC architecture")
        print()
        
        # Start the web application server
        print_log("✅ Starting web server...")
        app.run(host=host, debug=web_args.debug)
        
        return 0
        
    except KeyboardInterrupt:
        print_log("🛑 Server stopped by user")
        return 0
    except Exception as e:
        print_log(f"❌ Failed to start LLM web interface: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 