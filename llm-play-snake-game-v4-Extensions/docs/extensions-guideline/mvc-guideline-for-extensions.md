# MVC Architecture & Game Runner Guidelines for Extensions

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and `flask.md`. It provides comprehensive guidance on implementing both web interfaces and game runners for extensions (Tasks 1-5).

> **See also:** `final-decision-10.md`, `flask.md`, `scripts.md`, `core.md`.

## 🎯 **Core Philosophy: Minimal Web + Simple Game Runners**

Extensions must implement **two complementary interfaces**:
1. **Web Interface**: Flask-based web applications using the minimal KISS architecture
2. **Game Runner**: Simple programmatic interface for testing and automation

Both follow the same principles: **KISS, DRY, extensible, and educational**.

### **Design Philosophy**
- **Minimal Flask Apps**: Simple, direct integration without complex MVC patterns
- **Copy-Paste Templates**: Extensions copy Task-0 patterns and modify core components
- **Dual Interfaces**: Same functionality available via web and programmatic APIs
- **Educational Value**: Clear examples demonstrating different AI approaches

## 🌐 **Web Interface Architecture**

### **Task-0 Foundation: Minimal Flask Apps**

Task-0 provides **three perfect web application templates** in `ROOT/web/game_flask_app.py`:

```python
# ROOT/web/game_flask_app.py - Foundation Templates

class SimpleFlaskApp:
    """Minimal Flask application foundation for all tasks"""
    # Template Method Pattern: configure_app() → setup_routes() → run()

class HumanGameApp(SimpleFlaskApp):
    """Human player web application template"""
    # Override: get_game_data(), get_api_state(), handle_control()

class LLMGameApp(SimpleFlaskApp):
    """LLM player web application template"""
    # Override: get_game_data(), get_api_state(), handle_control()

class ReplayGameApp(SimpleFlaskApp):
    """Replay web application template"""
    # Override: get_game_data(), get_api_state(), handle_control()
```

### **Extension Web Implementation Pattern**

**Step 1: Copy Template**
```bash
# Copy appropriate Task-0 web script
cp scripts/human_play_web.py extensions/heuristics-v0.03/scripts/heuristic_web.py
```

**Step 2: Create Extension Flask App**
```python
# extensions/heuristics-v0.03/web/heuristic_flask_app.py
from web.game_flask_app import SimpleFlaskApp

class HeuristicGameApp(SimpleFlaskApp):
    """
    Heuristic algorithms web application.
    
    Extension Pattern: Copy SimpleFlaskApp → Override 3 methods
    Educational Value: Shows pathfinding algorithm integration
    """
    
    def __init__(self, algorithm: str = "BFS", grid_size: int = 10, **config):
        super().__init__(f"Heuristic Snake - {algorithm}")
        self.algorithm = algorithm
        self.grid_size = grid_size
        
        # Initialize algorithm-specific components
        from agents import AgentFactory
        self.pathfinder = AgentFactory.create(algorithm, grid_size)
        print_log(f"Heuristic mode: {algorithm} on {grid_size}x{grid_size} grid")
    
    def get_game_data(self) -> Dict[str, Any]:
        """Get heuristic-specific game data for template rendering."""
        return {
            'name': self.name,
            'mode': 'heuristic',
            'algorithm': self.algorithm,
            'grid_size': self.grid_size,
            'features': [
                f"{self.algorithm} pathfinding",
                "Real-time visualization",
                "Step-by-step analysis",
                "Performance metrics"
            ],
            'status': 'ready'
        }
    
    def get_api_state(self) -> Dict[str, Any]:
        """Get current algorithm state via API."""
        return {
            'mode': 'heuristic',
            'algorithm': self.algorithm,
            'grid_size': self.grid_size,
            'path_length': len(self.pathfinder.current_path) if hasattr(self.pathfinder, 'current_path') else 0,
            'nodes_explored': getattr(self.pathfinder, 'nodes_explored', 0),
            'status': 'running' if self.pathfinder.is_active() else 'ready'
        }
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle algorithm-specific controls."""
        action = data.get('action', '')
        
        if action == 'step':
            move = self.pathfinder.get_next_move()
            return {'action': 'move', 'direction': move, 'status': 'processed'}
        elif action == 'reset':
            self.pathfinder.reset()
            return {'action': 'reset', 'status': 'processed'}
        elif action == 'change_algorithm':
            new_algorithm = data.get('algorithm', self.algorithm)
            self.algorithm = new_algorithm
            self.pathfinder = AgentFactory.create(new_algorithm, self.grid_size)
            return {'action': 'algorithm_changed', 'algorithm': new_algorithm, 'status': 'processed'}
        
        return {'error': 'Unknown action'}

# Factory function following SUPREME_RULES canonical create() pattern
def create_heuristic_app(algorithm: str = "BFS", grid_size: int = 10, **config) -> HeuristicGameApp:
    """Canonical create() method for heuristic web applications."""
    return HeuristicGameApp(algorithm=algorithm, grid_size=grid_size, **config)
```

**Step 3: Create Web Script**
```python
# extensions/heuristics-v0.03/scripts/heuristic_web.py
from web.heuristic_flask_app import HeuristicGameApp
from utils.network_utils import get_server_host_port

class HeuristicWebApp(HeuristicGameApp):
    """Web script wrapper for heuristic algorithms."""
    
    def __init__(self, algorithm: str = "BFS", grid_size: int = 10, **config):
        super().__init__(algorithm=algorithm, grid_size=grid_size, **config)

def main():
    """Main entry point for heuristic web interface."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    host, port = get_server_host_port(default_host=args.host, default_port=args.port)
    
    app = HeuristicWebApp(
        algorithm=args.algorithm,
        grid_size=args.grid_size
    )
    
    app.run(host=host, debug=args.debug)
```

## 🎮 **Game Runner Architecture**

### **Task-0 Foundation: core/game_runner.py**

Task-0 provides the foundational game runner pattern in `core/game_runner.py`:

```python
# core/game_runner.py - Foundation Pattern
def play(agent: BaseAgent, max_steps: int = 1_000, render: bool = False, 
         *, seed: Optional[int] = None) -> List[dict]:
    """Execute a game and return trajectory as state dictionaries."""
    # Template implementation for all extensions
```

### **Extension Game Runner Implementation**

**Each extension MUST implement `game_runner.py`** following this pattern:

```python
# extensions/heuristics-v0.03/game_runner.py
"""
Heuristic algorithms game runner - Simple programmatic interface.

This module provides simple functions for running heuristic algorithms
programmatically, useful for testing, automation, and integration.

Design Philosophy:
- Simple function interface (no classes unless necessary)
- Direct algorithm access without web overhead
- Consistent with core/game_runner.py patterns
- Educational value through clear examples
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
import sys

# Ensure project root access
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.game_runner import play as core_play
from agents import AgentFactory

# Simple logging following SUPREME_RULES
print_log = lambda msg: print(f"[HeuristicRunner] {msg}")


def play_bfs(grid_size: int = 10, max_steps: int = 1000, render: bool = False, 
             seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Run BFS pathfinding algorithm.
    
    Args:
        grid_size: Size of the game grid
        max_steps: Maximum steps before timeout
        render: Whether to show visual interface
        seed: Random seed for reproducibility
        
    Returns:
        List of game state dictionaries (trajectory)
    """
    print_log(f"Running BFS on {grid_size}x{grid_size} grid")
    
    agent = AgentFactory.create("BFS", grid_size)
    trajectory = core_play(agent, max_steps=max_steps, render=render, seed=seed)
    
    print_log(f"BFS completed: {len(trajectory)} steps, score: {trajectory[-1].get('score', 0)}")
    return trajectory


def play_astar(grid_size: int = 10, max_steps: int = 1000, render: bool = False,
               seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Run A* pathfinding algorithm."""
    print_log(f"Running A* on {grid_size}x{grid_size} grid")
    
    agent = AgentFactory.create("ASTAR", grid_size)
    trajectory = core_play(agent, max_steps=max_steps, render=render, seed=seed)
    
    print_log(f"A* completed: {len(trajectory)} steps, score: {trajectory[-1].get('score', 0)}")
    return trajectory


def play_hamiltonian(grid_size: int = 10, max_steps: int = 1000, render: bool = False,
                     seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Run Hamiltonian cycle algorithm."""
    print_log(f"Running Hamiltonian on {grid_size}x{grid_size} grid")
    
    agent = AgentFactory.create("HAMILTONIAN", grid_size)
    trajectory = core_play(agent, max_steps=max_steps, render=render, seed=seed)
    
    print_log(f"Hamiltonian completed: {len(trajectory)} steps, score: {trajectory[-1].get('score', 0)}")
    return trajectory


def play_heuristic(algorithm: str, grid_size: int = 10, max_steps: int = 1000, 
                   render: bool = False, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Run any heuristic algorithm by name.
    
    Args:
        algorithm: Algorithm name ("BFS", "ASTAR", "DFS", "HAMILTONIAN")
        grid_size: Size of the game grid
        max_steps: Maximum steps before timeout
        render: Whether to show visual interface
        seed: Random seed for reproducibility
        
    Returns:
        List of game state dictionaries (trajectory)
    """
    print_log(f"Running {algorithm} on {grid_size}x{grid_size} grid")
    
    agent = AgentFactory.create(algorithm.upper(), grid_size)
    trajectory = core_play(agent, max_steps=max_steps, render=render, seed=seed)
    
    final_score = trajectory[-1].get('score', 0) if trajectory else 0
    print_log(f"{algorithm} completed: {len(trajectory)} steps, score: {final_score}")
    return trajectory


def compare_algorithms(algorithms: List[str], grid_size: int = 10, max_steps: int = 1000,
                      seed: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple heuristic algorithms.
    
    Args:
        algorithms: List of algorithm names to compare
        grid_size: Size of the game grid
        max_steps: Maximum steps before timeout
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping algorithm names to performance metrics
    """
    print_log(f"Comparing algorithms: {algorithms}")
    
    results = {}
    
    for algorithm in algorithms:
        print_log(f"Running {algorithm}...")
        trajectory = play_heuristic(algorithm, grid_size, max_steps, render=False, seed=seed)
        
        if trajectory:
            final_state = trajectory[-1]
            results[algorithm] = {
                'steps': len(trajectory),
                'score': final_state.get('score', 0),
                'success': final_state.get('game_active', False) == False and final_state.get('score', 0) > 0,
                'trajectory_length': len(trajectory)
            }
        else:
            results[algorithm] = {
                'steps': 0,
                'score': 0,
                'success': False,
                'trajectory_length': 0
            }
    
    print_log(f"Comparison complete: {results}")
    return results


# Convenience aliases for interactive use
play = play_heuristic  # Default function
quick_bfs = lambda: play_bfs(render=True)
quick_astar = lambda: play_astar(render=True)
quick_hamiltonian = lambda: play_hamiltonian(render=True)


if __name__ == "__main__":
    """Demo usage when run directly."""
    print_log("Heuristic Game Runner Demo")
    print_log("Running BFS with visualization...")
    
    trajectory = play_bfs(grid_size=10, render=True, seed=42)
    print_log(f"Demo complete: {len(trajectory)} steps")
```

## 🔗 **Integration Patterns**

### **Web + Game Runner Integration**

Extensions should integrate both interfaces seamlessly:

```python
# extensions/heuristics-v0.03/scripts/main.py
"""Main CLI interface that uses game_runner.py internally."""

import argparse
from game_runner import play_heuristic, compare_algorithms

def main():
    parser = argparse.ArgumentParser(description="Heuristic algorithms CLI")
    parser.add_argument("--algorithm", default="BFS", choices=["BFS", "ASTAR", "DFS", "HAMILTONIAN"])
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--compare", nargs="+", help="Compare multiple algorithms")
    
    args = parser.parse_args()
    
    if args.compare:
        results = compare_algorithms(args.compare, args.grid_size)
        print("Comparison Results:")
        for algo, metrics in results.items():
            print(f"  {algo}: {metrics['score']} score, {metrics['steps']} steps")
    else:
        trajectory = play_heuristic(args.algorithm, args.grid_size, render=args.render)
        print(f"Final score: {trajectory[-1].get('score', 0)}")

if __name__ == "__main__":
    main()
```

### **Streamlit Integration Pattern**

```python
# extensions/heuristics-v0.03/dashboard/tab_main.py
"""Streamlit tab that launches both web interface and game runner."""

import streamlit as st
import subprocess
from game_runner import play_heuristic

class MainTab:
    def render(self, session_state):
        st.header("🎮 Heuristic Algorithms")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Quick Play")
            algorithm = st.selectbox("Algorithm", ["BFS", "ASTAR", "DFS", "HAMILTONIAN"])
            
            if st.button("🚀 Run Algorithm"):
                with st.spinner("Running algorithm..."):
                    trajectory = play_heuristic(algorithm, render=False)
                    st.success(f"Completed! Score: {trajectory[-1].get('score', 0)}")
        
        with col2:
            st.subheader("Web Interface")
            if st.button("🌐 Launch Web Interface"):
                subprocess.Popen(["python", "scripts/heuristic_web.py", "--algorithm", algorithm])
                st.info("Web interface launched in background")
```

## 📋 **Extension Implementation Checklist**

### **Web Interface Requirements**
- [ ] **Copy Task-0 template** from appropriate web script
- [ ] **Create extension Flask app** inheriting from `SimpleFlaskApp`
- [ ] **Override 3 methods**: `get_game_data()`, `get_api_state()`, `handle_control()`
- [ ] **Add factory function** with canonical `create()` method name
- [ ] **Create web script** with argument parsing and network utilities
- [ ] **Test web interface** with `python scripts/extension_web.py --help`

### **Game Runner Requirements**
- [ ] **Create game_runner.py** in extension root directory
- [ ] **Implement algorithm-specific functions** (e.g., `play_bfs()`, `play_dqn()`)
- [ ] **Add generic play function** accepting algorithm parameter
- [ ] **Include comparison function** for multiple algorithms/models
- [ ] **Add convenience aliases** for interactive use
- [ ] **Test game runner** with `python -c "from game_runner import play; play('BFS', render=True)"`

### **Integration Requirements**
- [ ] **CLI scripts use game_runner** internally
- [ ] **Streamlit tabs integrate both** web and programmatic interfaces
- [ ] **Documentation includes examples** for both interfaces
- [ ] **Factory patterns consistent** across web and game runner
- [ ] **Error handling graceful** in both interfaces

## 🎯 **Extension Examples by Task**

### **Task-1 (Heuristics)**
```python
# Web: HeuristicGameApp(algorithm="BFS")
# Runner: play_bfs(), play_astar(), play_hamiltonian()
# Integration: CLI → game_runner → web interface launch
```

### **Task-2 (Reinforcement Learning)**
```python
# Web: RLGameApp(agent_type="DQN")
# Runner: play_dqn(), play_ppo(), train_agent()
# Integration: Training monitoring via web + programmatic evaluation
```

### **Task-3 (Supervised Learning)**
```python
# Web: MLGameApp(model_type="XGBoost")
# Runner: play_xgboost(), play_neural_net(), evaluate_model()
# Integration: Model comparison via web + batch evaluation
```

### **Task-4 (LLM Fine-tuning)**
```python
# Web: FineTuningApp(model_name="snake-gpt")
# Runner: play_finetuned(), compare_models(), evaluate_reasoning()
# Integration: Fine-tuning monitoring + model comparison
```

### **Task-5 (Advanced)**
```python
# Web: MultiStrategyApp(strategies=["heuristic", "rl", "supervised"])
# Runner: play_ensemble(), compare_strategies(), analyze_performance()
# Integration: Strategy comparison dashboard + automated benchmarking
```

---

**This architecture ensures that every extension provides both user-friendly web interfaces and powerful programmatic APIs, following KISS principles while maintaining educational value and extensibility.** 