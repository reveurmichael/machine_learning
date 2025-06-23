"""
Reinforcement Learning v0.02 - Multi-Agent Framework
--------------------

This extension implements a comprehensive reinforcement learning framework
supporting multiple RL algorithms and agents.

Design Philosophy:
- Multi-agent support (DQN, PPO, A3C, SAC)
- Standardized training and evaluation
- Grid size flexibility and proper directory structure
- No backward compatibility - only modern, future-proof code

Key Components:
- Multiple RL agents (DQN, PPO, A3C, SAC)
- Training scripts with hyperparameter tuning
- Evaluation and comparison tools
- Standardized model utilities with metadata
- Replay buffer and experience management

Usage:
    python scripts/train.py --agent DQN --grid-size 15
    python scripts/evaluate.py --agent-path logs/extensions/models/grid-size-N/dqn/
    python scripts/compare.py --agents DQN,PPO --grid-size 10
"""

__version__ = "0.02"
__author__ = "Snake Game Extensions"
__description__ = "Reinforcement Learning Framework for Snake Game" 