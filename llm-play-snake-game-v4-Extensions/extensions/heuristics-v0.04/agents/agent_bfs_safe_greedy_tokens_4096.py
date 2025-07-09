from __future__ import annotations
from typing import List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS Safe Greedy 4096 Token Agent - Full detailed BFS with Safety Validation for Snake Game v0.04
----------------

This module implements a full inheritance BFS-SAFE-GREEDY agent (4096 tokens) that inherits
completely from the standard BFS-SAFE-GREEDY agent with no modifications.

Design Patterns:
- Full Inheritance: Complete inheritance from BFSSafeGreedyAgent with no overrides
- Strategy Pattern: Identical safe-greedy pathfinding and explanation generation
- SSOT: Uses all parent methods without any modifications
"""

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import extension-specific components using relative imports
from .agent_bfs_safe_greedy import BFSSafeGreedyAgent


class BFSSafeGreedy4096TokenAgent(BFSSafeGreedyAgent):
    """
    BFS Safe Greedy Agent with full 4096-token explanations (identical to original BFS-SAFE-GREEDY).
    
    Full Inheritance Pattern:
    - Complete inheritance from BFSSafeGreedyAgent with no overrides
    - Maintains identical algorithm behavior and explanation generation
    - Only changes algorithm_name for identification purposes
    
    Token Limit: ~4096 tokens (full detailed explanations, identical to BFS-SAFE-GREEDY)
    """

    def __init__(self) -> None:
        """Initialize BFS Safe Greedy 4096-token agent, exactly like base BFS-SAFE-GREEDY."""
        super().__init__()  # Initialize parent BFS Safe Greedy agent
        self.algorithm_name = "BFS-SAFE-GREEDY-4096"
        
    # No method overrides - this agent is exactly identical to BFSSafeGreedyAgent
    # except for the algorithm_name for identification purposes
