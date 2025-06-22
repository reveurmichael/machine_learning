"""
Heuristic Agents Package
========================

Collection of all 7 heuristic algorithm agents for Snake game.

This package contains implementations of various pathfinding and space-filling
algorithms adapted for the Snake game environment. All agents follow the same
interface and can be used interchangeably through the factory pattern.

Available Algorithms:
- BFS: Breadth-First Search (optimal shortest path)
- BFS-Safe-Greedy: BFS with safety validation (best performer)
- BFS-Hamiltonian: BFS with Hamiltonian fallback (hybrid)
- DFS: Depth-First Search (educational/experimental)
- A*: A* Algorithm (optimal with heuristics)
- A*-Hamiltonian: A* with Hamiltonian fallback (advanced hybrid)
- Hamiltonian: Hamiltonian Cycle (space-filling)
"""

from .agent_bfs import BFSAgent
from .agent_bfs_safe_greedy import BFSSafeGreedyAgent
from .agent_bfs_hamiltonian import BFSHamiltonianAgent
from .agent_dfs import DFSAgent
from .agent_astar import AStarAgent
from .agent_astar_hamiltonian import AStarHamiltonianAgent
from .agent_hamiltonian import HamiltonianAgent

__all__ = [
    'BFSAgent',
    'BFSSafeGreedyAgent', 
    'BFSHamiltonianAgent',
    'DFSAgent',
    'AStarAgent',
    'AStarHamiltonianAgent',
    'HamiltonianAgent'
]

# Agent registry for factory pattern
AGENT_REGISTRY = {
    'bfs': BFSAgent,
    'bfs-safe-greedy': BFSSafeGreedyAgent,
    'bfs-hamiltonian': BFSHamiltonianAgent,
    'dfs': DFSAgent,
    'astar': AStarAgent,
    'astar-hamiltonian': AStarHamiltonianAgent,
    'hamiltonian': HamiltonianAgent
}

def create_agent(algorithm_name: str):
    """
    Factory function to create agent instances.
    
    Args:
        algorithm_name: Name of the algorithm ('bfs', 'astar', etc.)
        
    Returns:
        Agent instance
        
    Raises:
        ValueError: If algorithm_name is not recognized
    """
    if algorithm_name not in AGENT_REGISTRY:
        available = ', '.join(AGENT_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm '{algorithm_name}'. Available: {available}")
    
    return AGENT_REGISTRY[algorithm_name]()

def get_available_algorithms():
    """Get list of available algorithm names."""
    return list(AGENT_REGISTRY.keys()) 