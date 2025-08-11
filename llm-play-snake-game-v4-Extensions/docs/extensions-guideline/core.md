# Core Architecture Documentation

> This document aligns with `final-decision.md`. Use exact filenames when citing docs. Keep examples minimal and link to related docs explicitly.

## ✅ **Current Core Architecture Assessment**

Based on comprehensive analysis of the `core` folder, the architecture is excellently designed and perfectly prepared for future extensions. The core demonstrates exemplary SOLID principles and requires no refactoring.

## 🎯 **Game Runner Pattern for Extensions**

### **Why Extensions Need Their Own game_runner.py**

The current `core/game_runner.py` is Task-0 specific (LLM-focused) and should be complemented by extension-specific runners that provide:

1. **Agent-Specific Entry Points**: Direct execution of heuristic/RL/ML agents
2. **Simplified Testing**: Quick validation of algorithms without full session overhead  
3. **Research Workflows**: Rapid prototyping and experimentation
4. **Educational Clarity**: Clear demonstration of how each agent type works

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

The inheritance structure demonstrates ideal software architecture following SUPREME_RULES from `final-decision.md`:

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
    Educational Value: Shows clean inheritance following SUPREME_RULES from `final-decision.md`
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
    Educational Value: Shows how to extend base classes following SUPREME_RULES from `final-decision.md`
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
    Educational Value: Shows algorithmic strategy implementation following SUPREME_RULES from `final-decision.md`
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
    Educational Value: Shows RL integration following SUPREME_RULES from `final-decision.md`
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
- All factories use canonical `create()` method per `final-decision.md`

### **2. Clean Separation of Concerns**
- Base classes contain **zero** LLM-specific code
- Perfect abstraction boundaries between different responsibilities  
- No over-preparation or unused functionality
- Follows `final-decision.md` lightweight principles

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
- Strict compliance with `final-decision.md` principles

### **5. Generic Game Data Management**
- `generate_game_data()`: Template method for game data generation
- `save_game_data()`: Standardized game data saving
- `display_game_results()`: Consistent result display
- `save_session_summary()`: Template for session summaries
- `determine_game_end_reason()`: Canonical end reason determination
- `update_session_stats()`: Standardized statistics tracking
- `finalize_game()`: Complete game finalization workflow
- `run_single_game()`: Template method for game execution

### **6. Elegant Limits Management Integration**
- Automatic integration with `core/game_limits_manager.py`
- Consistent limit tracking across all extensions
- Configurable limit enforcement strategies
- Thread-safe operations with singleton pattern

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

All files follow the consistent `game_*.py` pattern aligned with `final-decision.md` standards:
- `game_manager.py` - Session management
- `game_data.py` - State tracking
- `game_controller.py` - Game mechanics
- `game_logic.py` - Decision making
- `game_loop.py` - Execution flow
- `game_stats.py` - Performance metrics

## 🏆 **Conclusion: Perfect Architecture**

The `core` folder demonstrates **exceptional software architecture** and requires **zero refactoring** while fully complying with `final-decision.md` SUPREME_RULES:

- ✅ **Perfect Base Classes**: Generic, reusable, no pollution
- ✅ **Factory Patterns**: Pluggable components via class attributes with canonical `create()` methods
- ✅ **Clean Inheritance**: Each task inherits exactly what it needs
- ✅ **Future-Ready**: Tasks 1-5 can inherit directly from base classes
- ✅ **No Over-preparation**: Contains only code actually used by Task-0
- ✅ **SUPREME_RULES Compliance**: Full adherence to `final-decision.md` standards

This architecture serves as a **perfect reference implementation** for how the entire codebase should be structured, demonstrating world-class software engineering principles and `final-decision.md` compliance in practice.

## 🔗 **See Also**

- **`agents.md`**: Authoritative reference for agent implementation standards
- **`factory-design-pattern.md`**: Factory pattern implementation guide  
- **`final-decision.md`**: SUPREME_RULES governance system and canonical standards

---

**This core architecture ensures educational value, technical consistency, and scalable development across all Snake Game AI extensions while maintaining strict `final-decision.md` SUPREME_RULES compliance.**

from utils.factory_utils import SimpleFactory

## 🚀 **Extension Development Experience**

### **Simplified Extension Creation**
The refactored architecture makes creating new extensions significantly easier:

```python
# Minimal extension game manager (example)
class MyExtensionGameManager(BaseGameManager):
    GAME_LOGIC_CLS = MyExtensionGameLogic
    
    def __init__(self, args):
        super().__init__(args)
        self.my_extension_config = getattr(args, "my_config", "default")
    
    def initialize(self):
        # Setup extension-specific components
        self._setup_logging()
        self.setup_game()
    
    def run(self):
        # Inherits all generic game management
        for game_id in range(1, self.args.max_games + 1):
            game_duration = self.run_single_game()  # Inherited method
            self.finalize_game(game_duration)  # Inherited method
            self.display_game_results(game_duration)  # Inherited method
        self.save_session_summary()  # Inherited method
```

### **Universal Summary Generator**
All extensions use the same summary generation system:

```python
# In any extension
from core.game_summary_generator import BaseGameSummaryGenerator

class MyExtensionSummaryGenerator(BaseGameSummaryGenerator):
    def _add_task_specific_game_fields(self, summary, game_data):
        # Add extension-specific fields
        summary["my_algorithm"] = self.algorithm_name
        summary["my_metrics"] = self.calculate_my_metrics()
    
    def _add_task_specific_session_fields(self, summary, session_data):
        # Add extension-specific session fields
        summary["my_session_stats"] = self.aggregate_my_stats()
```

### **Unified Statistics Collection**
All extensions use the same statistics collector:

```python
# In any extension
from core.game_stats_manager import GameStatisticsCollector

class MyExtensionStatsCollector(GameStatisticsCollector):
    def _add_task_specific_game_metrics(self, game_data):
        # Record extension-specific metrics
        self.custom_metrics["my_metric"] = self.calculate_my_metric()
    
    def _add_task_specific_session_metrics(self, session_data):
        # Add extension-specific session metrics
        session_data["my_aggregated_metrics"] = self.aggregate_my_metrics()
```

### **Centralized File/Session Management**
All extensions use the same file management system:

```python
# In any extension
from core.game_file_manager import BaseFileManager

# Automatic game and session summary saving
file_manager = BaseFileManager()
file_manager.save_game_summary(game_data, duration, game_number, log_dir)
file_manager.save_session_summary(session_data, session_duration, log_dir)
```

## 🎯 **Key Architectural Strengths**

### **1. Factory Pattern Excellence (SUPREME_RULES Compliant)**
- `GAME_LOGIC_CLS` and `GAME_DATA_CLS` enable pluggable components
- Clean dependency injection without tight coupling
- Easy to extend for new task types
- All factories use canonical `create()` method per `final-decision.md`

### **2. Clean Separation of Concerns**
- Base classes contain **zero** LLM-specific code
- Perfect abstraction boundaries between different responsibilities  
- No over-preparation or unused functionality
- Follows `final-decision.md` lightweight principles

### **3. SOLID Principles Implementation**
- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: All subclasses work with base interfaces
- **Interface Segregation**: Clean, focused interfaces
- **Dependency Inversion**: High-level modules don't depend on low-level details

### **4. Template Method Pattern Excellence**
- Base classes define workflow, subclasses fill details
- Consistent behavior across all extensions
- Easy to understand and maintain
- Perfect for educational purposes

### **5. Singleton Pattern for Shared Resources**
- File managers, statistics collectors, and summary generators use singleton pattern
- Thread-safe operations across all extensions
- Memory efficiency and consistent behavior
- Single source of truth for shared resources

## 📊 **Unified Data Flow Architecture**

### **Game Execution Flow**
```
Extension GameManager
├── Inherits BaseGameManager (session management)
├── Uses BaseGameSummaryGenerator (summary creation)
├── Uses GameStatisticsCollector (stats aggregation)
├── Uses BaseFileManager (file operations)
└── Extension-specific hooks for customization
```

### **Data Generation Flow**
```
Game State Changes
├── GameStatisticsCollector records metrics
├── BaseGameSummaryGenerator creates summaries
├── BaseFileManager saves files
└── Extension hooks add custom data
```

### **Extension Customization Points**
- `_add_task_specific_game_data()` - Add custom game fields
- `_add_task_specific_summary_data()` - Add custom session fields
- `_add_task_specific_stats()` - Add custom metrics
- `_display_task_specific_results()` - Add custom display
- `_update_task_specific_stats()` - Add custom stat updates

## 🎓 **Educational Benefits**

### **Learning Objectives**
- **Design Patterns**: Factory, Template Method, Singleton, Strategy
- **SOLID Principles**: Clean inheritance and composition
- **Single Source of Truth**: Centralized data management
- **Extension Development**: Easy to create new algorithms
- **Code Quality**: Elegant, maintainable architecture

### **Best Practices Demonstrated**
- **KISS Principle**: Simple, clear implementations
- **DRY Principle**: No code duplication
- **Fail-Fast**: Robust error handling
- **Documentation**: Comprehensive docstrings and examples
- **Testing**: Easy to test individual components

## 🔧 **Implementation Guidelines**

### **Creating New Extensions**
1. **Inherit from Base Classes**: Use `BaseGameManager`, `BaseGameLogic`, etc.
2. **Override Extension Hooks**: Implement task-specific methods
3. **Use Unified Components**: Leverage summary generator, stats collector, file manager
4. **Follow Naming Conventions**: Use established patterns from `naming-conventions.md`
5. **Document Design Patterns**: Explain why patterns were chosen

### **Extension-Specific Customization**
```python
class MyExtensionGameManager(BaseGameManager):
    def _add_task_specific_game_data(self, game_data, game_duration):
        # Add my extension's game-specific data
        game_data["my_algorithm"] = self.algorithm_name
        game_data["my_metrics"] = self.calculate_my_metrics()
    
    def _add_task_specific_summary_data(self, summary):
        # Add my extension's session-specific data
        summary["my_aggregated_stats"] = self.aggregate_my_stats()
    
    def _display_task_specific_results(self, game_duration):
        # Display my extension's specific results
        print_info(f"My Algorithm: {self.algorithm_name}")
        print_info(f"My Performance: {self.calculate_performance()}")
```

## 📈 **Performance and Scalability**

### **Memory Efficiency**
- Singleton pattern reduces memory usage
- Lazy loading of heavy components
- Efficient data structures and algorithms

### **Thread Safety**
- All shared resources use thread-safe patterns
- Consistent behavior across concurrent operations
- Robust error handling and recovery

### **Extensibility**
- Easy to add new algorithms and approaches
- Clean separation of concerns
- Minimal coupling between components

---

**The core architecture provides a solid foundation for all Snake Game AI extensions, ensuring consistency, maintainability, and educational value while supporting rapid development and experimentation.**








