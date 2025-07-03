# Core Architecture Documentation

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for core architecture standards and supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`).

## ✅ **Current Core Architecture Assessment**

Based on comprehensive analysis of the `core` folder, the architecture is excellently designed and perfectly prepared for future extensions. The core demonstrates exemplary SOLID principles and requires no refactoring, strictly following SUPREME_RULES from `final-decision-10.md`.

## 🎯 **Game Runner Pattern for Extensions**

### **Why Extensions Need Their Own game_runner.py**

The current `core/game_runner.py` is Task-0 specific (LLM-focused) and should be complemented by extension-specific runners that provide:

1. **Agent-Specific Entry Points**: Direct execution of heuristic/RL/ML agents
2. **Simplified Testing**: Quick validation of algorithms without full session overhead  
3. **Research Workflows**: Rapid prototyping and experimentation
4. **Educational Clarity**: Clear demonstration of how each agent type works

### **Extension game_runner.py Pattern**

```python
# extensions/heuristics-v0.03/game_runner.py
"""Quick-play helper for heuristic algorithms.

Provides direct execution of pathfinding agents without LLM overhead.
Ideal for rapid testing, benchmarking, and educational demonstrations.

Example
-------
>>> from agents import BFSAgent
>>> from game_runner import play_heuristic
>>> trajectory = play_heuristic(BFSAgent(), max_steps=500, render=True)
"""

from __future__ import annotations
from typing import List, Optional

from agents import BaseHeuristicAgent  # Extension-specific base
from game_logic import HeuristicGameLogic
from extensions.common.utils.path_utils import setup_extension_paths

setup_extension_paths()

__all__ = ["play_heuristic"]

def play_heuristic(
    agent: BaseHeuristicAgent,
    max_steps: int = 1_000,
    render: bool = False,
    *,
    seed: Optional[int] = None,
) -> List[dict]:
    """Execute heuristic agent and return trajectory.
    
    Parameters
    ----------
    agent
        Heuristic agent implementing BaseHeuristicAgent protocol
    max_steps
        Safety cap for simulation
    render
        Enable PyGame visualization
    seed
        RNG seed for reproducibility
    """
    if seed is not None:
        from utils.seed_utils import seed_everything
        seed_everything(seed)
    
    game = HeuristicGameLogic(use_gui=render)
    trajectory: List[dict] = []
    
    for _ in range(max_steps):
        trajectory.append(game.get_state_snapshot())
        
        # Use heuristic-specific planning
        move = agent.plan_move(game.get_state_snapshot()) or "EMPTY"
        active, _ = game.make_move(move)
        if not active:
            break
        if render:
            game.draw()
    
    return trajectory

# Convenience functions for common algorithms
def play_bfs(**kwargs) -> List[dict]:
    """Quick BFS execution."""
    from agents.agent_bfs import BFSAgent
    return play_heuristic(BFSAgent(), **kwargs)

def play_astar(**kwargs) -> List[dict]:
    """Quick A* execution."""  
    from agents.agent_astar import AStarAgent
    return play_heuristic(AStarAgent(), **kwargs)
```

### **Reinforcement Learning game_runner.py**

```python
# extensions/reinforcement-v0.02/game_runner.py
"""Quick-play helper for RL agents.

Provides direct execution of trained RL models without session overhead.
Supports both training and inference modes.
"""

from __future__ import annotations
from typing import List, Optional, Union
import torch

from agents import BaseRLAgent
from game_logic import RLGameLogic
from extensions.common.utils.path_utils import setup_extension_paths

setup_extension_paths()

__all__ = ["play_rl", "train_rl"]

def play_rl(
    agent: Union[BaseRLAgent, str],  # Agent instance or model path
    max_steps: int = 1_000,
    render: bool = False,
    *,
    seed: Optional[int] = None,
) -> List[dict]:
    """Execute trained RL agent and return trajectory."""
    
    if isinstance(agent, str):
        # Load from checkpoint
        agent = load_rl_agent(agent)
    
    if seed is not None:
        from utils.seed_utils import seed_everything
        seed_everything(seed)
    
    game = RLGameLogic(use_gui=render)
    trajectory: List[dict] = []
    
    agent.eval()  # Set to inference mode
    
    for _ in range(max_steps):
        trajectory.append(game.get_state_snapshot())
        
        state_tensor = agent.preprocess_state(game.get_state_snapshot())
        with torch.no_grad():
            action = agent.select_action(state_tensor, explore=False)
        
        move = agent.action_to_move(action)
        active, _ = game.make_move(move)
        if not active:
            break
        if render:
            game.draw()
    
    return trajectory

def train_rl(
    agent: BaseRLAgent,
    episodes: int = 1000,
    render: bool = False,
    save_path: Optional[str] = None
) -> dict:
    """Train RL agent and return training statistics."""
    # Training implementation specific to RL
    pass
```

## 🏗️ **Factory Pattern: Canonical Method is create()**

All extension factories must use the canonical method name `create()` for instantiation, not `create_agent()` or any other variant. This ensures consistency and aligns with the KISS principle and SUPREME_RULES from `final-decision-10.md`. Factories should be simple, dictionary-based, and avoid over-engineering.

### Reference Implementation

A generic, educational `SimpleFactory` is provided in `utils/factory_utils.py`:

```python
from utils.factory_utils import SimpleFactory

class MyAgent:
    def __init__(self, name):
        self.name = name

factory = SimpleFactory()
factory.register("myagent", MyAgent)
agent = factory.create("myagent", name="TestAgent")  # CANONICAL create() method per SUPREME_RULES
print(agent.name)  # Output: TestAgent
```

### Example Extension Factory

```python
class HeuristicAgentFactory:
    """
    Factory following SUPREME_RULES from `final-decision-10.md`
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create heuristic agents using canonical create() method
    Educational Value: Shows how SUPREME_RULES apply consistently across extensions
    """
    
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, **kwargs):  # CANONICAL create() method per SUPREME_RULES
        """Create heuristic agent using canonical create() method following SUPREME_RULES from `final-decision-10.md`"""
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
        print_info(f"[HeuristicAgentFactory] Creating agent: {algorithm}")  # SUPREME_RULES compliant logging
        return agent_class(**kwargs)
```

---

## 🎯 **Existing Base Classes Architecture**

### **Perfect Foundation Classes**

#### **1. `BaseGameManager` - Generic Session Management**
- Contains **only** generic attributes: `game_count`, `total_score`, `round_count`
- **Zero LLM-specific code**: no `llm_response`, `awaiting_plan`, `token_stats`
- Factory pattern with `GAME_LOGIC_CLS` for pluggable game logic
- Ready for all task types without modification

#### **2. `BaseGameData` - Universal Game State Tracking**
- Contains **only** universal attributes: `score`, `steps`, `snake_positions`, `apple_position`
- Uses `BaseGameStatistics()` (not `GameStatistics`)
- **Zero LLM-specific counters**: no `consecutive_empty_steps`, `consecutive_something_is_wrong`
- Clean separation of concerns for different task types

#### **3. `BaseGameController` - Pure Game Logic Controller**
- Contains **only** core game mechanics: `board`, `snake_positions`, `apple_position`
- Factory pattern with `GAME_DATA_CLS` for pluggable data containers
- **Zero LLM dependencies** - completely generic
- Universal coordinate system and collision detection

#### **4. `BaseGameLogic` - Generic Planning Layer**
- Contains **only** universal planning: `planned_moves`, `get_next_planned_move()`
- **Zero LLM-specific processing** - pure algorithmic interface
- Perfect abstraction for any decision-making system

## 🏗️ **Perfect Inheritance Hierarchy**

The inheritance structure demonstrates ideal software architecture following SUPREME_RULES from `final-decision-10.md`:

```
BaseGameManager → GameManager (Task-0 adds LLM features)
BaseGameData → GameData (Task-0 adds LLM statistics)  
BaseGameController → GameController (Task-0 adds LLM data tracking)
BaseGameLogic → GameLogic (Task-0 adds LLM response parsing)
```

## 🚀 **Extension Integration Examples**

### **Task 1 (Heuristics) Integration**

```python
class HeuristicGameManager(BaseGameManager):
    """
    Inherits all session management, adds pathfinding algorithms
    
    Design Pattern: Template Method Pattern
    Purpose: Extends base functionality with heuristic-specific features
    Educational Value: Shows clean inheritance following SUPREME_RULES from `final-decision-10.md`
    """
    GAME_LOGIC_CLS = HeuristicGameLogic  # Factory pattern
    
    def initialize(self):
        # Set up pathfinding algorithms using canonical patterns
        self.pathfinder = PathfindingFactory.create("ASTAR")  # CANONICAL create() method per SUPREME_RULES
        print_info("[HeuristicGameManager] Initialized pathfinder and ready to run")  # SUPREME_RULES compliant logging
    
    def run(self):
        # Inherits all generic game loop logic from BaseGameManager
        # Only implements heuristic-specific planning
        for game in range(self.args.max_games):
            self.setup_game()  # Inherited method
            while self.game_active:  # Inherited attribute
                path = self.pathfinder.find_path(self.game.get_state_snapshot())
                self.game.planned_moves = path  # Inherited attribute

class HeuristicGameData(BaseGameData):
    """
    Inherits universal game state, adds heuristic-specific data
    
    Design Pattern: Decorator Pattern
    Purpose: Extends base data with heuristic-specific tracking
    Educational Value: Shows how to extend base classes following SUPREME_RULES from `final-decision-10.md`
    """
    # Inherits: consecutive_invalid_reversals, consecutive_no_path_found
    # Does NOT inherit: consecutive_empty_steps, consecutive_something_is_wrong
    # Uses: BaseGameStatistics() - perfect for heuristics
    
    def __init__(self):
        super().__init__()
        # Add heuristic-specific extensions
        self.algorithm_name = "A*"
        self.path_calculations = 0
        print_info("[HeuristicGameData] Initialized heuristic data tracking")  # SUPREME_RULES compliant logging

class HeuristicGameLogic(BaseGameLogic):
    """
    Inherits planning interface, implements pathfinding logic
    
    Design Pattern: Strategy Pattern  
    Purpose: Implements heuristic decision-making strategies
    Educational Value: Shows algorithmic strategy implementation following SUPREME_RULES from `final-decision-10.md`
    """
    GAME_DATA_CLS = HeuristicGameData  # Factory pattern
    
    def __init__(self, grid_size=10, use_gui=True):  # GUI is optional per SUPREME_RULE NO.5
        super().__init__(grid_size, use_gui)
        # Inherits: planned_moves, get_next_planned_move()
        self.pathfinder = PathfindingFactory.create("ASTAR")  # CANONICAL create() method per SUPREME_RULES
        print_info(f"[HeuristicGameLogic] Initialized with {grid_size}x{grid_size} grid")  # SUPREME_RULES compliant logging
    
    def plan_next_moves(self):
        # Use inherited get_state_snapshot()
        current_state = self.get_state_snapshot()
        path = self.pathfinder.find_path(current_state)
        self.planned_moves = path  # Inherited attribute
        print_info(f"[HeuristicGameLogic] Planned {len(path)} moves")  # SUPREME_RULES compliant logging
```

### **Task 2 (Reinforcement Learning) Integration**

```python
class RLGameManager(BaseGameManager):
    """
    Inherits session management, adds RL training capabilities
    
    Design Pattern: Observer Pattern
    Purpose: Manages RL training with episode tracking
    Educational Value: Shows RL integration following SUPREME_RULES from `final-decision-10.md`
    """
    GAME_LOGIC_CLS = RLGameLogic
    
    def initialize(self):
        self.agent = AgentFactory.create("DQN")  # CANONICAL create() method per SUPREME_RULES
        print_info("[RLGameManager] DQN agent initialized")  # SUPREME_RULES compliant logging
    
    def run(self):
        # Inherits all session management
        # Only implements RL training loop
        for episode in range(self.args.max_episodes):
            self.setup_game()  # Inherited
            while self.game_active:  # Inherited
                state = self.game.get_state_snapshot()  # Inherited
                action = self.agent.select_action(state)
                self.game.planned_moves = [action]  # Inherited

class RLGameData(BaseGameData):
    """
    Inherits universal game state, adds RL-specific extensions
    
    Design Pattern: Composite Pattern
    Purpose: Composes base data with RL-specific metrics
    Educational Value: Shows data composition following SUPREME_RULES
    """
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.action_history = []
        print_info("[RLGameData] Initialized RL data tracking")  # SUPREME_RULES logging

class RLGameLogic(BaseGameLogic):
    """
    Inherits planning interface, implements RL decision logic
    
    Design Pattern: Command Pattern
    Purpose: Encapsulates RL actions as commands
    Educational Value: Shows command pattern with SUPREME_RULES compliance
    """
    GAME_DATA_CLS = RLGameData
    
    def get_observation(self):
        # Use inherited state snapshot
        state = self.get_state_snapshot()
        print_info(f"[RLGameLogic] Generated observation")  # SUPREME_RULES logging
        return state
```

## 🎯 **Key Architectural Strengths**

### **1. Factory Pattern Excellence (SUPREME_RULES Compliant)**
- `GAME_LOGIC_CLS` and `GAME_DATA_CLS` enable pluggable components
- Clean dependency injection without tight coupling
- Easy to extend for new task types
- All factories use canonical `create()` method per `final-decision-10.md`

### **2. Clean Separation of Concerns**
- Base classes contain **zero** LLM-specific code
- Perfect abstraction boundaries between different responsibilities  
- No over-preparation or unused functionality
- Follows `final-decision-10.md` lightweight principles

### **3. SOLID Principles Implementation**
- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Perfect inheritance relationships
- **Interface Segregation**: Clean, focused interfaces
- **Dependency Inversion**: Depends on abstractions via factory patterns

### **4. Future-Ready Design (SUPREME_RULES)**
- Tasks 1-5 can inherit directly from base classes
- No modification needed for new extension types
- Perfect balance of functionality without over-engineering
- Strict compliance with `final-decision-10.md` principles

## 📊 **Attribute Distribution Analysis**

### **Universal Attributes (In Base Classes)**
**Perfect for All Tasks:**
- `score`, `steps`, `snake_positions`, `apple_position`
- `consecutive_invalid_reversals`, `consecutive_no_path_found`
- `planned_moves`, `need_new_plan`
- `game_count`, `total_score`, `round_count`
- `use_gui`, `gui`, `game_active`

### **Task-0 Specific Attributes (Properly Isolated)**
**LLM-Only Features:**
- `llm_response`, `awaiting_plan`, `token_stats`
- `consecutive_empty_steps`, `consecutive_something_is_wrong`
- `continue_from_session`, `continue_from_directory`

## 🎨 **File Naming Excellence**

All files follow the consistent `game_*.py` pattern aligned with `final-decision-10.md` standards:
- `game_manager.py` - Session management
- `game_data.py` - State tracking
- `game_controller.py` - Game mechanics
- `game_logic.py` - Decision making
- `game_loop.py` - Execution flow
- `game_stats.py` - Performance metrics

## 🏆 **Conclusion: Perfect Architecture**

The `core` folder demonstrates **exceptional software architecture** and requires **zero refactoring** while fully complying with `final-decision-10.md` SUPREME_RULES:

- ✅ **Perfect Base Classes**: Generic, reusable, no pollution
- ✅ **Factory Patterns**: Pluggable components via class attributes with canonical `create()` methods
- ✅ **Clean Inheritance**: Each task inherits exactly what it needs
- ✅ **Future-Ready**: Tasks 1-5 can inherit directly from base classes
- ✅ **No Over-preparation**: Contains only code actually used by Task-0
- ✅ **SUPREME_RULES Compliance**: Full adherence to `final-decision-10.md` standards

This architecture serves as a **perfect reference implementation** for how the entire codebase should be structured, demonstrating world-class software engineering principles and `final-decision-10.md` compliance in practice.

## 🔗 **See Also**

- **`agents.md`**: Authoritative reference for agent implementation standards
- **`factory-design-pattern.md`**: Factory pattern implementation guide  
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards

---

**This core architecture ensures educational value, technical consistency, and scalable development across all Snake Game AI extensions while maintaining strict `final-decision-10.md` SUPREME_RULES compliance.**

from utils.factory_utils import SimpleFactory








