"""
Hamiltonian Cycle Snake Game Extension (v0.03)
==============================================

This extension implements the Hamiltonian Cycle algorithm for playing Snake.
The Hamiltonian cycle is the most advanced Snake algorithm that guarantees
infinite survival by following a pre-computed cycle that visits every cell.

Key Features:
- Hamiltonian cycle generation for any grid size
- Guaranteed infinite survival (never gets stuck)
- Pre-computed optimal cycle path
- Same architecture as previous heuristics but with cycle-based navigation
- Standalone implementation with no external dependencies

Design Philosophy:
- Extends BaseGameManager, BaseGameData, BaseGameLogic, BaseGameController
- Uses Factory pattern for pluggable components
- Compatible with Task-0 logging format
- Demonstrates perfect inheritance from core base classes

Algorithm Details:
- Generates a Hamiltonian cycle that visits every grid cell exactly once
- Snake follows this cycle, guaranteeing it never collides with itself
- When apple is encountered, snake can "shortcut" for efficiency
- Falls back to cycle following for safety when no shortcuts available
- Perfect for infinite survival scenarios

Performance Characteristics:
- 100% survival rate (theoretically infinite game length)
- May not optimize for score (doesn't always take shortest path to apple)
- Excellent for safety-critical applications
- Demonstrates advanced algorithmic techniques
"""

__version__ = "0.03"
__author__ = "Snake-GTP Extensions"
__description__ = "Hamiltonian Cycle Snake Game Algorithm"

# Make key classes available at package level
from .hamiltonian_agent import HamiltonianAgent
from .game_data import HeuristicGameData
from .game_logic import HeuristicGameLogic
from .game_manager import HeuristicGameManager

__all__ = [
    "HamiltonianAgent",
    "HeuristicGameData", 
    "HeuristicGameLogic",
    "HeuristicGameManager"
] 