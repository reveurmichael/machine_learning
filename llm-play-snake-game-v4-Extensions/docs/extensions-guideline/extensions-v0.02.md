# Extensions v0.02: Multi-Algorithm Patterns

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision.md`) and defines the multi-algorithm patterns for all v0.02 extensions.

> **See also:** `core.md`, `standalone.md`, `final-decision.md`, `factory-design-pattern.md`.

## 🎯 **Core Philosophy: Algorithm Comparison**

Extensions v0.02 extend v0.01 by implementing **multiple related algorithms** within the same extension. This enables comparative analysis and demonstrates how different algorithmic approaches solve the same problem. GUI components are optional per SUPREME_RULES from `final-decision.md`.

### **Educational Value**
- **Algorithm Comparison**: Comparing different algorithmic approaches
- **Factory Patterns**: Learning factory patterns for algorithm selection
- **Performance Analysis**: Understanding algorithm performance differences
- **Code Organization**: Managing multiple algorithms in a single extension
- **GUI Philosophy**: Understanding optional GUI components per SUPREME_RULES

## 🏗️ **v0.02 Architecture Requirements**

### **Mandatory Directory Structure**
```
extensions/{algorithm}-v0.02/
├── __init__.py
├── main.py                 # Primary execution entry point
├── game_logic.py           # Extension-specific game logic
├── game_manager.py         # Extension-specific game manager
├── agents/                 # 🎯 MANDATORY: Multiple algorithm implementations
│   ├── __init__.py
│   ├── agent_bfs.py
│   ├── agent_astar.py
│   ├── agent_dfs.py
│   └── agent_hamiltonian.py
└── README.md               # Documentation
```

### **Factory Pattern: Canonical Method is create()**
All v0.02 extensions must use the canonical method name `create()` for instantiation:

```python
from utils.print_utils import print_info

class HeuristicAgentFactory:
    """Simple factory for current needs"""
    
    _agents = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent
    }
    
    @classmethod
    def create(cls, algorithm: str, grid_size: int) -> BaseAgent:
        """Create agent directly without over-engineering"""
        agent_class = cls._agents.get(algorithm.upper())
        if not agent_class:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        print_info(f"[HeuristicAgentFactory] Creating agent: {algorithm}")
        return agent_class(algorithm, grid_size)
```

## 🚀 **Implementation Examples**

### **Multiple Agent Implementations**
```python
# agents/agent_bfs.py
from core.game_agents import BaseAgent
from utils.print_utils import print_info

class BFSAgent(BaseAgent):
    """Breadth-First Search agent for v0.02."""
    
    def __init__(self, name: str = "BFS"):
        super().__init__(name)
        print_info(f"[BFSAgent] Initialized BFS agent")
    
    def plan_move(self, game_state: dict) -> str:
        """Plan next move using BFS"""
        snake_positions = game_state['snake_positions']
        apple_position = game_state['apple_position']
        grid_size = game_state['grid_size']
        
        path = self._bfs_pathfinding(snake_positions[0], apple_position, snake_positions, grid_size)
        
        if path and len(path) > 1:
            next_pos = path[1]
            return self._get_direction(snake_positions[0], next_pos)
        else:
            return self._fallback_move(snake_positions, grid_size)
    
    def _bfs_pathfinding(self, start: tuple, goal: tuple, obstacles: list, grid_size: int) -> list:
        """BFS pathfinding implementation"""
        queue = [(start, [start])]
        visited = set()
        
        while queue:
            current, path = queue.pop(0)
            if current == goal:
                return path
            
            for neighbor in self._get_neighbors(current, grid_size):
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None

# agents/agent_astar.py
class AStarAgent(BaseAgent):
    """A* Search agent for v0.02."""
    
    def __init__(self, name: str = "ASTAR"):
        super().__init__(name)
        from utils.print_utils import print_info
        print_info(f"[AStarAgent] Initialized A* agent")
    
    def plan_move(self, game_state: dict) -> str:
        """Plan next move using A*"""
        snake_positions = game_state['snake_positions']
        apple_position = game_state['apple_position']
        grid_size = game_state['grid_size']
        
        path = self._astar_pathfinding(snake_positions[0], apple_position, snake_positions, grid_size)
        
        if path and len(path) > 1:
            next_pos = path[1]
            return self._get_direction(snake_positions[0], next_pos)
        else:
            return self._fallback_move(snake_positions, grid_size)
    
    def _astar_pathfinding(self, start: tuple, goal: tuple, obstacles: list, grid_size: int) -> list:
        """A* pathfinding implementation"""
        import heapq
        
        open_set = [(0, start, [start])]
        closed_set = set()
        
        while open_set:
            f_score, current, path = heapq.heappop(open_set)
            
            if current == goal:
                return path
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            for neighbor in self._get_neighbors(current, grid_size):
                if neighbor not in closed_set and neighbor not in obstacles:
                    g_score = len(path)
                    h_score = self._manhattan_distance(neighbor, goal)
                    f_score = g_score + h_score
                    
                    heapq.heappush(open_set, (f_score, neighbor, path + [neighbor]))
        
        return None
    
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Calculate Manhattan distance"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# agents/agent_dfs.py
class DFSAgent(BaseAgent):
    """Depth-First Search agent for v0.02."""
    
    def __init__(self, name: str = "DFS"):
        super().__init__(name)
        from utils.print_utils import print_info
        print_info(f"[DFSAgent] Initialized DFS agent")
    
    def plan_move(self, game_state: dict) -> str:
        """Plan next move using DFS"""
        snake_positions = game_state['snake_positions']
        apple_position = game_state['apple_position']
        grid_size = game_state['grid_size']
        
        path = self._dfs_pathfinding(snake_positions[0], apple_position, snake_positions, grid_size)
        
        if path and len(path) > 1:
            next_pos = path[1]
            return self._get_direction(snake_positions[0], next_pos)
        else:
            return self._fallback_move(snake_positions, grid_size)
    
    def _dfs_pathfinding(self, start: tuple, goal: tuple, obstacles: list, grid_size: int) -> list:
        """DFS pathfinding implementation"""
        stack = [(start, [start])]
        visited = set()
        
        while stack:
            current, path = stack.pop()
            
            if current == goal:
                return path
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor in self._get_neighbors(current, grid_size):
                if neighbor not in visited and neighbor not in obstacles:
                    stack.append((neighbor, path + [neighbor]))
        
        return None
```

### **Enhanced Game Manager with Factory**
```python
# game_manager.py
from core.game_manager import BaseGameManager
from .agents.agent_bfs import BFSAgent
from .agents.agent_astar import AStarAgent
from .agents.agent_dfs import DFSAgent
from .agents.agent_hamiltonian import HamiltonianAgent
from utils.print_utils import print_info

class HeuristicGameManager(BaseGameManager):
    """
    Enhanced game manager for v0.02 with multiple algorithms.
    
    Uses factory pattern for algorithm selection and comparison.
    """
    
    def __init__(self, algorithm: str, grid_size: int = 10, max_games: int = 1):
        super().__init__(grid_size=grid_size)
        self.algorithm = algorithm
        self.max_games = max_games
        self.agent = self.create(algorithm)
        print_info(f"[HeuristicGameManager] Initialized with {algorithm}")
    
    def create(self, algorithm: str):
        """Create agent using factory pattern"""
        agents = {
            'BFS': BFSAgent,
            'ASTAR': AStarAgent,
            'DFS': DFSAgent,
            'HAMILTONIAN': HamiltonianAgent,
        }
        
        if algorithm not in agents:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return agents[algorithm]()
    
    def run_comparison(self, algorithms: list) -> dict:
        """Run comparison between multiple algorithms"""
        print_info(f"[HeuristicGameManager] Running comparison for {algorithms}")
        
        comparison_results = {}
        
        for algorithm in algorithms:
            print_info(f"[HeuristicGameManager] Testing {algorithm}")
            
            # Create agent for this algorithm
            self.agent = self.create(algorithm)
            
            # Run games
            algorithm_results = self.run_multiple_games()
            comparison_results[algorithm] = algorithm_results
        
        return comparison_results
    
    def run_multiple_games(self) -> dict:
        """Run multiple games with current agent"""
        from utils.print_utils import print_info, print_warning
        results = {
            'algorithm': self.agent.name,
            'games_played': 0,
            'total_score': 0,
            'successful_games': 0,
            'average_score': 0.0,
            'success_rate': 0.0
        }
        
        for game_id in range(self.max_games):
            print_info(f"[HeuristicGameManager] Running game {game_id + 1} with {self.agent.name}")
            
            try:
                game_result = self.run_single_game()
                results['games_played'] += 1
                results['total_score'] += game_result['score']
                
                if game_result['success']:
                    results['successful_games'] += 1
                
                print_info(f"[HeuristicGameManager] Game {game_id + 1} completed, score: {game_result['score']}")
                
            except Exception as e:
                print_warning(f"[HeuristicGameManager] ERROR in game {game_id + 1}: {e}")
                continue
        
        # Calculate averages
        if results['games_played'] > 0:
            results['average_score'] = results['total_score'] / results['games_played']
            results['success_rate'] = results['successful_games'] / results['games_played']
        
        return results
```

### **Enhanced Main Entry Point**
```python
# main.py
#!/usr/bin/env python3
"""
Main entry point for heuristics v0.02 extension.

Supports single algorithm execution and algorithm comparison.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from extensions.common.utils.path_utils import ensure_project_root
from game_manager import HeuristicGameManager
from utils.print_utils import print_info

def main():
    """Main execution function with algorithm comparison"""
    # Ensure proper working directory
    ensure_project_root()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run heuristic algorithms with comparison")
    parser.add_argument("--algorithm", help="Single algorithm to run")
    parser.add_argument("--algorithms", nargs="+", help="Multiple algorithms for comparison")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size")
    parser.add_argument("--max-games", type=int, default=10, help="Number of games per algorithm")
    parser.add_argument("--comparison", action="store_true", help="Run algorithm comparison")
    
    args = parser.parse_args()
    
    # Create game manager
    manager = HeuristicGameManager(
        algorithm=args.algorithm or "BFS",
        grid_size=args.grid_size,
        max_games=args.max_games
    )
    
    if args.comparison or args.algorithms:
        # Run comparison
        algorithms_to_test = args.algorithms or ["BFS", "ASTAR", "DFS"]
        results = manager.run_comparison(algorithms_to_test)
        
        # Print comparison results
        print_info(f"\nAlgorithm Comparison Results:")
        print_info(f"  Grid Size: {args.grid_size}x{args.grid_size}")
        print_info(f"  Games per Algorithm: {args.max_games}")
        print_info(f"  {'Algorithm':<12} {'Avg Score':<10} {'Success Rate':<12}")
        print_info(f"  {'-'*12} {'-'*10} {'-'*12}")
        
        for algorithm, result in results.items():
            print_info(f"  {algorithm:<12} {result['average_score']:<10.2f} {result['success_rate']:<12.2%}")
        
    else:
        # Run single algorithm
        results = manager.run_multiple_games()
        
        # Print single algorithm results
        print_info(f"\nSingle Algorithm Results:")
        print_info(f"  Algorithm: {results['algorithm']}")
        print_info(f"  Grid Size: {args.grid_size}x{args.grid_size}")
        print_info(f"  Games Played: {results['games_played']}")
        print_info(f"  Average Score: {results['average_score']:.2f}")
        print_info(f"  Success Rate: {results['success_rate']:.2%}")
    
    return results

if __name__ == "__main__":
    main()
```

## 📊 **v0.02 Standards**

### **Multi-Algorithm Requirements**
- **Multiple Algorithms**: At least 3 related algorithms
- **Factory Pattern**: Canonical `create()` method for algorithm selection
- **Comparison Capability**: Ability to compare algorithm performance
- **Consistent Interface**: All algorithms implement the same interface

### **Educational Focus**
- **Algorithm Comparison**: Clear comparison between different approaches
- **Performance Analysis**: Understanding algorithm performance differences
- **Factory Patterns**: Learning factory patterns for algorithm selection
- **Code Organization**: Managing multiple algorithms in a single extension

### **Quality Standards**
- **Working Algorithms**: All algorithms must work correctly
- **Performance Metrics**: Comprehensive performance comparison
- **Code Reuse**: Shared utilities between algorithms
- **Clear Documentation**: Documentation for each algorithm

## 📋 **Implementation Checklist**

### **Required Components**
- [ ] **Multiple Agents**: At least 3 algorithm implementations
- [ ] **Factory Pattern**: Canonical `create()` method
- [ ] **Comparison Logic**: Algorithm comparison functionality
- [ ] **Performance Metrics**: Comprehensive performance analysis
- [ ] **Shared Utilities**: Common utilities for all algorithms

### **Code Quality**
- [ ] **Algorithm Correctness**: All algorithms work correctly
- [ ] **Code Reuse**: Shared code between algorithms
- [ ] **Performance Analysis**: Meaningful performance comparison
- [ ] **Error Handling**: Proper error handling for all algorithms

### **Educational Value**
- [ ] **Algorithm Comparison**: Clear comparison between approaches
- [ ] **Factory Patterns**: Understanding factory pattern usage
- [ ] **Performance Analysis**: Learning to analyze algorithm performance
- [ ] **Code Organization**: Managing multiple algorithms effectively

## 🎓 **Educational Benefits**

### **Learning Objectives**
- **Algorithm Comparison**: Understanding differences between algorithms
- **Factory Patterns**: Learning factory patterns for object creation
- **Performance Analysis**: Analyzing and comparing algorithm performance
- **Code Organization**: Managing multiple related implementations

### **Best Practices**
- **Modular Design**: Clear separation between algorithms
- **Factory Patterns**: Using factories for object creation
- **Performance Metrics**: Meaningful performance comparison
- **Code Reuse**: Sharing common utilities between algorithms

## 🔗 **Progression to v0.03**

v0.02 extensions serve as the foundation for v0.03 extensions, which will:
- **Add Data Generation**: Generate training datasets from algorithm execution
- **Add Visualization**: Enhanced visualization and analysis capabilities
- **Add Scripts**: Backend scripts for batch processing
- **Add Dashboard**: Streamlit script launcher interface for algorithm comparison (SUPREME_RULES)

---

**Extensions v0.02 demonstrate multi-algorithm patterns and factory usage, providing educational value through algorithm comparison and performance analysis while establishing patterns for more sophisticated extensions.**

## 🔗 **See Also**

- **`core.md`**: Base class architecture and inheritance patterns
- **`standalone.md`**: Standalone principle and extension independence
- **`final-decision.md`**: SUPREME_RULES governance system and canonical standards
- **`factory-design-pattern.md`**: Factory pattern implementation guide







