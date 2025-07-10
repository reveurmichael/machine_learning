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
        # Control whether to include ASCII board representation in prompts (saves tokens)
        self.include_board_representation = True
        
    # No method overrides - this agent is exactly identical to BFSAgent
    # except for the algorithm_name for identification purposes

    def format_metrics_for_completion(self, metrics: dict, additional_metrics: dict = None) -> str:
        """
        Format metrics for completion text. Users can override this method to customize
        which metrics to include and how to format them.
        
        Args:
            metrics: Agent's own metrics dictionary
            additional_metrics: Additional metrics from dataset generator (optional)
        
        Returns:
            Formatted metrics string for completion
        """
        # Default implementation - users can override for custom formatting
        formatted_metrics = []
        
        # Include basic metrics
        if 'valid_moves' in metrics:
            formatted_metrics.append(f"- Valid moves: {metrics['valid_moves']}")
        
        if 'manhattan_distance' in metrics:
            formatted_metrics.append(f"- Manhattan distance to apple: {metrics['manhattan_distance']}")
        
        # Include additional metrics if provided
        if additional_metrics:
            if 'apple_direction' in additional_metrics:
                formatted_metrics.append(f"- Apple direction: {additional_metrics['apple_direction']}")
            
            if 'danger_assessment' in additional_metrics:
                formatted_metrics.append(f"- Danger assessment: {additional_metrics['danger_assessment']}")
            
            if 'free_space' in additional_metrics:
                formatted_metrics.append(f"- Free space: {additional_metrics['free_space']}")
        
        return "Metrics:\n" + "\n".join(formatted_metrics) if formatted_metrics else ""


