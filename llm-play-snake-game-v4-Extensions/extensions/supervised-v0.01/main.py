#!/usr/bin/env python3
"""
Supervised Learning v0.01 - Main Entry Point
===========================================

Simple entry point for supervised learning v0.01, focusing on neural networks only.
Follows the same pattern as heuristics v0.01 - minimal complexity, proof of concept.

Design Pattern: Template Method
- Simple CLI interface with basic arguments
- Focused on single model type (neural networks)
- Extends base classes from Task-0
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from extensions.common.path_utils import setup_extension_paths
from game_manager import SupervisedGameManager
setup_extension_paths()


def main():
    """
    Main entry point for supervised learning v0.01.
    
    Simple CLI with basic arguments, focused on neural networks only.
    Demonstrates base class reuse and extension capabilities.
    """
    parser = argparse.ArgumentParser(description="Supervised Learning v0.01 - Neural Networks")
    parser.add_argument("--grid-size", type=int, default=10, 
                       help="Grid size for the game (default: 10)")
    parser.add_argument("--max-games", type=int, default=10,
                       help="Maximum number of games to run (default: 10)")
    parser.add_argument("--model-type", choices=["MLP", "CNN", "LSTM"], default="MLP",
                       help="Model type to use (default: MLP)")
    parser.add_argument("--training-mode", action="store_true",
                       help="Enable training mode")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum steps per game (default: 1000)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Supervised Learning v0.01 - Neural Networks")
    print("=" * 60)
    print(f"Grid Size: {args.grid_size}")
    print(f"Model Type: {args.model_type}")
    print(f"Max Games: {args.max_games}")
    print(f"Training Mode: {args.training_mode}")
    print("=" * 60)
    
    try:
        # Create and run game manager
        manager = SupervisedGameManager(args)
        manager.run()
        
        print("\n" + "=" * 60)
        print("Supervised Learning v0.01 completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 