from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS 4096 Token Agent - Full detailed BFS pathfinding for Snake Game v0.04
----------------

This module implements a full inheritance BFS agent (4096 tokens) that inherits
completely from the standard BFS agent with no modifications.

Design Patterns:
- Full Inheritance: Complete inheritance from BFSAgent with no overrides
- Strategy Pattern: Identical BFS pathfinding and explanation generation
- SSOT: Uses all parent methods without any modifications
"""

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import extension-specific components using relative imports
from .agent_bfs import BFSAgent

class BFS4096TokenAgent(BFSAgent):
    """
    BFS Agent with full 4096-token explanations (identical to original BFS).
    
    Full Inheritance Pattern:
    - Complete inheritance from BFSAgent with no overrides
    - Maintains identical algorithm behavior and explanation generation
    - Only changes algorithm_name for identification purposes
    
    Token Limit: ~4096 tokens (full detailed explanations, identical to BFS)
    """

    def __init__(self):
        """Initialize BFS 4096-token agent, exactly like base BFS."""
        super().__init__()  # Initialize parent BFS agent
        self.algorithm_name = "BFS-4096"
        
    # No method overrides - this agent is exactly identical to BFSAgent
    # except for the algorithm_name for identification purposes


