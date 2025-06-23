#!/usr/bin/env python3
"""
Reinforcement Learning v0.01 - Training Script
--------------------

Training script for DQN reinforcement learning agent.
Provides CLI interface for training with configurable parameters.

Design Pattern: Template Method
- CLI interface with comprehensive arguments
- Focused on DQN training pipeline
- Extends base classes from Task-0
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from extensions.common.path_utils import setup_extension_paths
from game_manager import RLGameManager
setup_extension_paths()


def main():
    """
    Main training entry point for reinforcement learning v0.01.
    
    CLI interface for DQN training with comprehensive parameter control.
    Demonstrates base class reuse and extension capabilities.
    """
    parser = argparse.ArgumentParser(description="Reinforcement Learning v0.01 - DQN Training")
    
    # Game parameters
    parser.add_argument("--grid-size", type=int, default=10, 
                       help="Grid size for the game (default: 10)")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum steps per episode (default: 1000)")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes (default: 1000)")
    parser.add_argument("--epsilon", type=float, default=0.1,
                       help="Epsilon for epsilon-greedy exploration (default: 0.1)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate for DQN (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training (default: 32)")
    parser.add_argument("--memory-size", type=int, default=10000,
                       help="Replay memory size (default: 10000)")
    
    # Model parameters
    parser.add_argument("--hidden-size", type=int, default=256,
                       help="Hidden layer size for DQN (default: 256)")
    parser.add_argument("--save-interval", type=int, default=100,
                       help="Save model every N episodes (default: 100)")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for models (default: auto-generated)")
    parser.add_argument("--model-name", type=str, default="dqn_model",
                       help="Model name prefix (default: dqn_model)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Reinforcement Learning v0.01 - DQN Training")
    print("=" * 60)
    print(f"Grid Size: {args.grid_size}")
    print(f"Episodes: {args.episodes}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Hidden Size: {args.hidden_size}")
    print("=" * 60)
    
    try:
        # Create and run game manager
        manager = RLGameManager(args)
        manager.run()
        
        print("\n" + "=" * 60)
        print("Reinforcement Learning v0.01 training completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTraining cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 