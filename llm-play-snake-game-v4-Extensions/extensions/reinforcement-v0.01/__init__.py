"""
Reinforcement Learning v0.01 - DQN Focus
========================================

This extension implements Deep Q-Network (DQN) reinforcement learning for the Snake game.
It demonstrates the foundational RL concepts with a single, well-implemented algorithm.

Design Philosophy:
- Single algorithm focus (DQN) for proof of concept
- Clean inheritance from base classes
- Comprehensive training pipeline
- No GUI by default (headless training)

Key Components:
- DQNAgent: Deep Q-Network implementation
- RLGameLogic: Reinforcement learning game logic
- RLGameManager: Training session management
- train.py: Training script with CLI interface

Usage:
    python train.py --episodes 1000 --epsilon 0.1 --output-dir ./models
"""

__version__ = "0.01"
__author__ = "Snake Game Extensions"
__description__ = "Deep Q-Network Reinforcement Learning for Snake Game" 