"""
Heuristics v0.02 - Agent Package
===============================

Collection of heuristic algorithms for snake game pathfinding.
Demonstrates evolution from v0.01 single-algorithm to multi-algorithm system.

Available Algorithms:
- BFS: Breadth-First Search (shortest path)
- BFS-Safe-Greedy: BFS with safety checks and greedy improvements
- BFS-Hamiltonian: BFS with Hamiltonian cycle considerations
- DFS: Depth-First Search (exploration-based)
- A*: A-Star with Manhattan distance heuristic
- A*-Hamiltonian: A-Star with Hamiltonian cycle considerations
- Hamiltonian: Pure Hamiltonian cycle pathfinding

Design Philosophy:
- Each algorithm builds upon simpler ones (progressive enhancement)
- Factory pattern for easy algorithm selection
- Consistent interface across all agents
- Educational progression from basic to advanced algorithms
"""

from typing import Dict, Type, Optional, Any
import sys
from pathlib import Path

# Add root directory to Python path for base classes
root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

# Import all agent classes
from .agent_bfs import BFSAgent
from .agent_bfs_safe_greedy import BFSSafeGreedyAgent
from .agent_bfs_hamiltonian import BFSHamiltonianAgent
from .agent_dfs import DFSAgent
from .agent_astar import AStarAgent
from .agent_astar_hamiltonian import AStarHamiltonianAgent
from .agent_hamiltonian import HamiltonianAgent


# Algorithm Registry - Factory Pattern
ALGORITHM_REGISTRY: Dict[str, Type[Any]] = {
    "BFS": BFSAgent,
    "BFS-SAFE-GREEDY": BFSSafeGreedyAgent,
    "BFS-HAMILTONIAN": BFSHamiltonianAgent,
    "DFS": DFSAgent,
    "ASTAR": AStarAgent,
    "A*": AStarAgent,  # Alias for A*
    "ASTAR-HAMILTONIAN": AStarHamiltonianAgent,
    "A*-HAMILTONIAN": AStarHamiltonianAgent,  # Alias for A*-Hamiltonian
    "HAMILTONIAN": HamiltonianAgent,
}

# Default algorithm
DEFAULT_ALGORITHM = "BFS"

# Algorithm categories for educational purposes
ALGORITHM_CATEGORIES = {
    "Basic Search": ["BFS", "DFS"],
    "Enhanced BFS": ["BFS-SAFE-GREEDY", "BFS-HAMILTONIAN"],
    "Heuristic Search": ["ASTAR", "ASTAR-HAMILTONIAN"],
    "Advanced": ["HAMILTONIAN"]
}

# Algorithm complexity information
ALGORITHM_COMPLEXITY = {
    "BFS": {"time": "O(V+E)", "space": "O(V)", "optimal": True},
    "BFS-SAFE-GREEDY": {"time": "O(V+E)", "space": "O(V)", "optimal": False},
    "BFS-HAMILTONIAN": {"time": "O(V+E)", "space": "O(V)", "optimal": False},
    "DFS": {"time": "O(V+E)", "space": "O(V)", "optimal": False},
    "ASTAR": {"time": "O(b^d)", "space": "O(b^d)", "optimal": True},
    "ASTAR-HAMILTONIAN": {"time": "O(b^d)", "space": "O(b^d)", "optimal": False},
    "HAMILTONIAN": {"time": "O(n!)", "space": "O(n)", "optimal": False}
}


def create_agent(algorithm_name: str) -> Optional[Any]:
    """
    Factory function to create agent instances.
    
    Design Pattern: Factory Method
    - Encapsulates agent creation logic
    - Allows easy addition of new algorithms
    - Provides consistent interface for agent instantiation
    
    Args:
        algorithm_name: Name of the algorithm (case-insensitive)
        
    Returns:
        Agent instance or None if algorithm not found
        
    Examples:
        >>> agent = create_agent("BFS")
        >>> agent = create_agent("astar-hamiltonian")
        >>> agent = create_agent("A*")  # Alias support
    """
    algorithm_key = algorithm_name.upper().replace("-", "-")
    
    if algorithm_key in ALGORITHM_REGISTRY:
        agent_class = ALGORITHM_REGISTRY[algorithm_key]
        return agent_class()
    
    return None


def get_available_algorithms() -> list[str]:
    """
    Get list of all available algorithm names.
    
    Returns:
        List of algorithm names (primary names only, no aliases)
    """
    # Return primary names only (exclude aliases)
    primary_algorithms = []
    for alg in ALGORITHM_REGISTRY.keys():
        if alg not in ["A*", "A*-HAMILTONIAN"]:  # Skip aliases
            primary_algorithms.append(alg)
    
    return sorted(primary_algorithms)


def get_algorithm_info(algorithm_name: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about an algorithm.
    
    Args:
        algorithm_name: Name of the algorithm
        
    Returns:
        Dictionary with algorithm information or None if not found
    """
    algorithm_key = algorithm_name.upper().replace("-", "-")
    
    if algorithm_key not in ALGORITHM_REGISTRY:
        return None
    
    # Find category
    category = "Unknown"
    for cat, algorithms in ALGORITHM_CATEGORIES.items():
        if algorithm_key in algorithms:
            category = cat
            break
    
    info = {
        "name": algorithm_key,
        "category": category,
        "class": ALGORITHM_REGISTRY[algorithm_key].__name__,
        "complexity": ALGORITHM_COMPLEXITY.get(algorithm_key, {}),
        "description": _get_algorithm_description(algorithm_key)
    }
    
    return info


def _get_algorithm_description(algorithm_name: str) -> str:
    """Get human-readable description of algorithm."""
    descriptions = {
        "BFS": "Breadth-First Search - finds shortest path by exploring level by level",
        "BFS-SAFE-GREEDY": "Enhanced BFS with safety checks and greedy optimizations",
        "BFS-HAMILTONIAN": "BFS with Hamiltonian cycle considerations for safer paths",
        "DFS": "Depth-First Search - explores as far as possible before backtracking",
        "ASTAR": "A* algorithm with Manhattan distance heuristic for optimal pathfinding",
        "ASTAR-HAMILTONIAN": "A* enhanced with Hamiltonian cycle considerations",
        "HAMILTONIAN": "Pure Hamiltonian cycle pathfinding for complete board coverage"
    }
    
    return descriptions.get(algorithm_name, "Advanced heuristic pathfinding algorithm")


# Export public interface
__all__ = [
    # Factory functions
    'create_agent',
    'get_available_algorithms', 
    'get_algorithm_info',
    
    # Constants
    'DEFAULT_ALGORITHM',
    'ALGORITHM_REGISTRY',
    'ALGORITHM_CATEGORIES',
    'ALGORITHM_COMPLEXITY',
    
    # Agent classes (for direct import if needed)
    'BFSAgent',
    'BFSSafeGreedyAgent', 
    'BFSHamiltonianAgent',
    'DFSAgent',
    'AStarAgent',
    'AStarHamiltonianAgent',
    'HamiltonianAgent'
] 