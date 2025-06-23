#!/usr/bin/env python3
"""
Supervised Learning v0.02 - Main Entry Point
===========================================

Multi-model entry point for supervised learning v0.02, supporting all ML model types.
Follows the same pattern as heuristics v0.02 - multiple algorithms with --model argument.

Design Pattern: Factory Pattern + Strategy Pattern
- Factory pattern for model creation based on --model argument
- Strategy pattern for different model types (neural, tree, graph)
- Organized structure with models folder
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))

from extensions.common.path_utils import setup_extension_paths
from extensions.supervised_v0_02.game_manager import SupervisedGameManager
setup_extension_paths()


def create_arguments(model: str, max_games: int = 10, grid_size: int = 10, 
                    verbose: bool = True, log_dir: str = None) -> argparse.Namespace:
    """
    Create arguments object for supervised learning v0.02.
    
    Factory pattern for argument creation based on model type.
    
    Args:
        model: Model type to use
        max_games: Maximum number of games to run
        grid_size: Size of the game grid
        verbose: Whether to enable verbose output
        log_dir: Log directory path
        
    Returns:
        Arguments object
    """
    args = argparse.Namespace()
    args.model = model
    args.max_games = max_games
    args.grid_size = grid_size
    args.verbose = verbose
    args.log_dir = log_dir
    args.max_steps = 1000
    args.no_gui = True  # v0.02 has no GUI
    return args


def validate_model_type(model: str) -> bool:
    """
    Validate that the model type is supported.
    
    Args:
        model: Model type to validate
        
    Returns:
        True if valid, False otherwise
    """
    valid_models = [
        # Neural networks
        "MLP", "CNN", "LSTM", "GRU",
        # Tree models
        "XGBOOST", "LIGHTGBM", "RANDOMFOREST",
        # Graph models
        "GCN", "GRAPHSAGE", "GAT"
    ]
    
    return model.upper() in valid_models


def get_model_categories() -> Dict[str, List[str]]:
    """
    Get model categories and their supported models.
    
    Returns:
        Dictionary mapping categories to model lists
    """
    return {
        "Neural Networks": ["MLP", "CNN", "LSTM", "GRU"],
        "Tree Models": ["XGBOOST", "LIGHTGBM", "RANDOMFOREST"],
        "Graph Models": ["GCN", "GRAPHSAGE", "GAT"]
    }


def print_model_info():
    """Print information about available models."""
    print("Available Models:")
    print("=" * 50)
    
    categories = get_model_categories()
    for category, models in categories.items():
        print(f"\n{category}:")
        for model in models:
            print(f"  - {model}")
    
    print("\n" + "=" * 50)


def main():
    """
    Main entry point for supervised learning v0.02.
    
    Multi-model CLI with --model argument, supporting all ML model types.
    Demonstrates evolution from v0.01 to v0.02 with multiple model support.
    """
    parser = argparse.ArgumentParser(
        description="Supervised Learning v0.02 - Multi-Model Snake Game Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run MLP neural network
  python main.py --model MLP --max-games 10

  # Run XGBoost tree model
  python main.py --model XGBOOST --max-games 5

  # Run GCN graph model
  python main.py --model GCN --max-games 3

  # List available models
  python main.py --list-models
        """
    )
    
    # Model selection
    parser.add_argument("--model", type=str, default="MLP",
                       help="Model type to use (default: MLP)")
    parser.add_argument("--list-models", action="store_true",
                       help="List all available models and exit")
    
    # Game parameters
    parser.add_argument("--max-games", type=int, default=10,
                       help="Maximum number of games to run (default: 10)")
    parser.add_argument("--grid-size", type=int, default=10,
                       help="Size of the game grid (default: 10)")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum steps per game (default: 1000)")
    
    # Output options
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose output (default: True)")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Log directory (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Handle list models request
    if args.list_models:
        print_model_info()
        return
    
    # Validate model type
    if not validate_model_type(args.model):
        print(f"Error: Unknown model type '{args.model}'")
        print("\nAvailable models:")
        print_model_info()
        sys.exit(1)
    
    # Print header
    print("=" * 60)
    print("Supervised Learning v0.02 - Multi-Model")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Max games: {args.max_games}")
    print(f"Max steps per game: {args.max_steps}")
    print("=" * 60)
    
    try:
        # Create and run game manager
        manager = SupervisedGameManager(args)
        manager.run()
        
        print("\n" + "=" * 60)
        print("Supervised Learning v0.02 completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 