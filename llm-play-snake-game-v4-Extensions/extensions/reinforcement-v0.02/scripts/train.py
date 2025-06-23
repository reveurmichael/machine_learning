#!/usr/bin/env python3
"""
Reinforcement Learning v0.02 - Training Script
=============================================

Modern CLI training script for reinforcement learning agents.
Supports multiple RL algorithms with standardized training and evaluation.

Design Pattern: Template Method
- CLI interface with comprehensive arguments
- Standardized agent training pipeline
- Grid size flexibility
- Hyperparameter optimization support
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from extensions.common.path_utils import setup_extension_paths
from extensions.common.model_utils import save_model_standardized, get_model_directory
setup_extension_paths()


def main():
    """Main training entry point for reinforcement learning v0.02."""
    parser = argparse.ArgumentParser(description="Reinforcement Learning v0.02 - Training")
    
    # Agent selection
    parser.add_argument("--agent", type=str, required=True,
                       choices=["DQN", "PPO", "A3C", "SAC"],
                       help="RL agent type to train")
    
    # Game parameters
    parser.add_argument("--grid-size", type=int, default=10,
                       help="Grid size for the game (default: 10)")
    parser.add_argument("--max-episodes", type=int, default=1000,
                       help="Maximum number of training episodes (default: 1000)")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum steps per episode (default: 1000)")
    
    # Training parameters
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training (default: 32)")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor (default: 0.99)")
    parser.add_argument("--epsilon", type=float, default=0.1,
                       help="Exploration rate (default: 0.1)")
    
    # Agent-specific parameters
    parser.add_argument("--hidden-size", type=int, default=256,
                       help="Hidden layer size for neural networks (default: 256)")
    parser.add_argument("--memory-size", type=int, default=10000,
                       help="Replay buffer size (default: 10000)")
    parser.add_argument("--target-update", type=int, default=100,
                       help="Target network update frequency (default: 100)")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for models (default: auto-generated)")
    parser.add_argument("--agent-name", type=str, default=None,
                       help="Agent name prefix (default: auto-generated)")
    parser.add_argument("--save-frequency", type=int, default=100,
                       help="Save model every N episodes (default: 100)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose output (default: True)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Reinforcement Learning v0.02 - Training")
    print("=" * 60)
    print(f"Agent: {args.agent}")
    print(f"Grid Size: {args.grid_size}")
    print(f"Max Episodes: {args.max_episodes}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Gamma: {args.gamma}")
    print(f"Epsilon: {args.epsilon}")
    print("=" * 60)
    
    try:
        # Import and train based on agent type
        if args.agent == "DQN":
            train_dqn_agent(args)
        elif args.agent == "PPO":
            train_ppo_agent(args)
        elif args.agent == "A3C":
            train_a3c_agent(args)
        elif args.agent == "SAC":
            train_sac_agent(args)
        else:
            raise ValueError(f"Unknown agent type: {args.agent}")
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTraining cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def train_dqn_agent(args):
    """Train DQN agent."""
    print(f"Training DQN agent...")
    
    # Import DQN components
    from agents.dqn_agent import DQNAgent
    
    # Create agent
    agent = DQNAgent(
        grid_size=args.grid_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
        memory_size=args.memory_size,
        target_update=args.target_update
    )
    
    # Train the agent
    print(f"Training DQN for {args.max_episodes} episodes...")
    training_result = agent.train(
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        save_frequency=args.save_frequency
    )
    
    # Save the final model
    agent_name = args.agent_name or f"{args.agent.lower()}_agent"
    agent.save_model(agent_name)
    
    print(f"Training metrics: {training_result}")


def train_ppo_agent(args):
    """Train PPO agent."""
    print(f"Training PPO agent...")
    
    # Import PPO components
    from agents.ppo_agent import PPOAgent
    
    # Create agent
    agent = PPOAgent(
        grid_size=args.grid_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma
    )
    
    # Train the agent
    print(f"Training PPO for {args.max_episodes} episodes...")
    training_result = agent.train(
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        save_frequency=args.save_frequency
    )
    
    # Save the final model
    agent_name = args.agent_name or f"{args.agent.lower()}_agent"
    agent.save_model(agent_name)
    
    print(f"Training metrics: {training_result}")


def train_a3c_agent(args):
    """Train A3C agent."""
    print(f"Training A3C agent...")
    
    # Import A3C components
    from agents.a3c_agent import A3CAgent
    
    # Create agent
    agent = A3CAgent(
        grid_size=args.grid_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma
    )
    
    # Train the agent
    print(f"Training A3C for {args.max_episodes} episodes...")
    training_result = agent.train(
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        save_frequency=args.save_frequency
    )
    
    # Save the final model
    agent_name = args.agent_name or f"{args.agent.lower()}_agent"
    agent.save_model(agent_name)
    
    print(f"Training metrics: {training_result}")


def train_sac_agent(args):
    """Train SAC agent."""
    print(f"Training SAC agent...")
    
    # Import SAC components
    from agents.sac_agent import SACAgent
    
    # Create agent
    agent = SACAgent(
        grid_size=args.grid_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma
    )
    
    # Train the agent
    print(f"Training SAC for {args.max_episodes} episodes...")
    training_result = agent.train(
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        save_frequency=args.save_frequency
    )
    
    # Save the final model
    agent_name = args.agent_name or f"{args.agent.lower()}_agent"
    agent.save_model(agent_name)
    
    print(f"Training metrics: {training_result}")


if __name__ == "__main__":
    main() 