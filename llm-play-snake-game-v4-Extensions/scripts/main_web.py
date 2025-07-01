"""
Snake Game - LLM Web Interface Script (Enhanced Architecture)
============================================================

Flask-based web application for LLM-controlled Snake gameplay using the enhanced
layered web infrastructure. This script demonstrates Task-0's modern LLM web
architecture and serves as the canonical template for all future extension LLM
web interfaces.

Enhanced Features:
- Layered Web Infrastructure: BaseWebApp → SimpleFlaskApp → LLMWebGameApp
- Enhanced Naming: Clear, explicit naming throughout the LLM application stack
- Universal Factory Utilities: Uses factory_utils from ROOT/utils/ following SSOT
- Dynamic Port Allocation: Network utilities with conflict resolution
- LLM Integration: Multiple providers with enhanced configuration management
- Future-Proof Design: Template for Task 1-5 extension LLM interfaces

Design Patterns (Enhanced):
    - Template Method Pattern: Layered LLM application lifecycle
    - Factory Pattern: Universal factory utilities with canonical create() method
    - Strategy Pattern: Pluggable LLM providers with enhanced configuration
    - Facade Pattern: Simplified LLM application launcher interface

Educational Goals:
    - Demonstrate enhanced LLM web architecture for Task-0
    - Show layered inheritance patterns for LLM functionality
    - Illustrate LLM provider integration with web infrastructure
    - Provide canonical template for extension LLM interfaces

Extension Template for Future Tasks:
    Task-1 (Heuristics): Copy this structure, replace with HeuristicLLMWebGameApp
    Task-2 (Supervised): Copy this structure, replace with SupervisedLLMWebGameApp
    Task-3 (RL): Copy this structure, replace with RLLLMWebGameApp
    Task-4 (LLM Fine-tuning): Copy this structure, replace with LLMFinetuningWebGameApp
    Task-5 (Distillation): Copy this structure, replace with DistillationLLMWebGameApp

Reference: docs/extensions-guideline/mvc-flask-factory.md for comprehensive patterns
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Ensure project root for consistent imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import enhanced web infrastructure with layered architecture
from web import LLMWebGameApp, LLMWebAppLauncher
from web.factories import create_llm_web_game_app

# Import universal utilities following SSOT principles
from utils.validation_utils import validate_llm_web_arguments
from utils.print_utils import create_logger
from config.ui_constants import GRID_SIZE as DEFAULT_GRID_SIZE

# Enhanced logging with consistent naming
print_log = create_logger("LLMWebScript")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for LLM web game interface.
    
    Educational Value: Shows consistent argument handling with enhanced naming
    Extension Template: Copy this exact pattern for all extension LLM scripts
    
    Returns:
        Configured argument parser with LLM-specific options
    """
    parser = argparse.ArgumentParser(
        description="Snake Game - LLM Web Game Interface (Enhanced Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Examples:
  python scripts/main_web.py                             # Default enhanced LLM settings
  python scripts/main_web.py --provider hunyuan         # Specific LLM provider with validation
  python scripts/main_web.py --model gpt-4 --port 8080  # Custom model and port with conflict detection
  python scripts/main_web.py --grid-size 15 --debug     # Larger grid with debug mode

Extension Template Pattern:
  Future extensions should copy this LLM script structure:
  1. Import enhanced LLM infrastructure: {ExtensionType}LLMWebGameApp
  2. Use universal factory functions: create_{extension}_llm_web_game_app()
  3. Apply enhanced naming conventions throughout
  4. Leverage universal validation utilities
  5. Maintain same elegant LLM launcher pattern

LLM Architecture Benefits:
  - Layered inheritance: BaseWebApp → SimpleFlaskApp → LLMWebGameApp
  - Enhanced naming: Clear domain indication (LLM + Web + Game + App)
  - Multi-provider support: Hunyuan, OpenAI, Anthropic, local models
  - Educational clarity: Self-documenting enhanced LLM architecture

LLM Providers:
  - hunyuan: Tencent Hunyuan models (default)
  - openai: OpenAI GPT models
  - anthropic: Anthropic Claude models
  - deepseek: DeepSeek models
  - local: Local/self-hosted models
        """
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="hunyuan",
        help="LLM provider with enhanced configuration (default: hunyuan)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="hunyuan-turbos-latest",
        help="LLM model name with provider validation (default: hunyuan-turbos-latest)"
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Size of the game grid with validation (default: {DEFAULT_GRID_SIZE})"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for the web server (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number with conflict detection (default: auto-detect free port)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode with enhanced logging"
    )
    
    return parser


def create_llm_web_application(args) -> LLMWebGameApp:
    """
    Create LLM web game application using enhanced factory function.
    
    Args:
        args: Validated command line arguments
        
    Returns:
        Configured LLM web game application
        
    Educational Value: Shows enhanced factory pattern usage for LLM applications
    Extension Template: Copy this creation pattern for all extension LLM applications
    """
    print_log("Creating LLM web game application using enhanced factory...")
    
    # Use enhanced factory function with universal utilities
    app = create_llm_web_game_app(
        provider=args.provider,
        model=args.model,
        grid_size=args.grid_size,
        port=args.port
    )
    
    print_log(f"LLM web game app created: {app.name} on port {app.port}")
    return app


def display_application_info(app: LLMWebGameApp, host: str) -> None:
    """
    Display enhanced application information with clear formatting.
    
    Args:
        app: LLM web game application
        host: Server host address
        
    Educational Value: Shows enhanced naming and information display for LLM
    Extension Template: Copy this display pattern for all extension LLM applications
    """
    print_log("🎯 Enhanced LLM Application Information:")
    print_log(f"   Application: {app.name}")
    print_log(f"   Architecture: Enhanced Layered Web Infrastructure")
    print_log(f"   Type: LLM Web Game App")
    print_log(f"   LLM Provider: {app.provider}")
    print_log(f"   LLM Model: {app.model}")
    print_log(f"   Grid Size: {app.grid_size}x{app.grid_size}")
    print_log(f"   Port: {app.port} (with conflict detection)")
    print_log(f"   URL: http://{host}:{app.port}")
    print_log("")
    
    print_log("🤖 Enhanced LLM Features:")
    print_log("   AI-driven gameplay with advanced reasoning")
    print_log("   Real-time LLM decision visualization")
    print_log("   Multi-provider support with seamless switching")
    print_log("   Game state analysis and strategy display")
    print_log("   Performance monitoring and statistics")
    print_log("   Ctrl+C: Stop server")
    print_log("")
    
    print_log("🏗️ LLM Architecture Layers:")
    print_log("   BaseWebApp → SimpleFlaskApp → LLMWebGameApp")
    print_log("   LLM provider integration with enhanced configuration")
    print_log("   Universal utilities from ROOT/utils/ (SSOT compliance)")
    print_log("   Enhanced naming for educational clarity")
    print_log("")


def main() -> int:
    """
    Main entry point for LLM web game interface script.
    
    Educational Value: Shows elegant LLM application launcher lifecycle
    Extension Template: Copy this exact main() pattern for all extension LLM scripts
    
    Returns:
        Exit code: 0 for success, 1 for failure
    """
    try:
        print_log("🤖 Starting Snake Game - LLM Web Interface (Enhanced)")
        print_log("📊 Architecture: Enhanced Layered Web Infrastructure")
        print_log("🎯 Mode: LLM Player with Enhanced Naming")
        print_log("")
        
        # Parse and validate command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        validate_llm_web_arguments(args)
        
        # Create enhanced LLM web application
        app = create_llm_web_application(args)
        
        # Display enhanced application information
        display_application_info(app, args.host)
        
        print_log("🚀 Extension Template Information:")
        print_log("   This script demonstrates enhanced LLM web architecture")
        print_log("   Copy this structure for all extension LLM interfaces")
        print_log("   Replace LLMWebGameApp with {Extension}LLMWebGameApp")
        print_log("   Maintain enhanced naming and layered LLM architecture")
        print_log("")
        
        # Start the enhanced LLM web application
        print_log("✅ Starting enhanced LLM web server...")
        app.run(host=args.host, port=app.port, debug=args.debug)
        
        return 0
        
    except KeyboardInterrupt:
        print_log("🛑 LLM server stopped by user")
        return 0
    except Exception as e:
        print_log(f"❌ Failed to start LLM web interface: {e}")
        print_log("   Check enhanced error handling and LLM validation")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 