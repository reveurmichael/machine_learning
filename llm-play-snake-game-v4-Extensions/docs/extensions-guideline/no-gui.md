# No GUI Architecture for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`` → `final-decision.md`) and defines no GUI architecture patterns for extensions.

> **See also:** `core.md`, `final-decision.md`, `project-structure-plan.md`.


### **1. ✅ Universal Headless Controller (Perfect for Tasks 0-5)**

**Location:** `core/game_controller.py`

**✅ BaseGameController - Completely Generic:**
```python
class BaseGameController:
    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = True):
        # ✅ Universal attributes (used by ALL tasks)
        self.grid_size = grid_size
        self.board = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.snake_positions = np.array([[grid_size//2, grid_size//2]])
        self.head_position = self.snake_positions[-1]
        self.apple_position = self._generate_apple()
        self.current_direction = None
        self.last_collision_type = None
        
        # ✅ Generic GUI management - works headless or with GUI
        self.use_gui = use_gui
        self.gui = None
        
        # ✅ Universal game state tracking
        self.game_state = self.GAME_DATA_CLS()  # Polymorphic data container
        
    def set_gui(self, gui_instance: "BaseGUI") -> None:
        """✅ Dependency injection - works with ANY GUI implementation"""
        self.gui = gui_instance
        self.use_gui = gui_instance is not None
        
    def draw(self) -> None:
        """✅ Safe GUI drawing - no-op in headless mode"""
        if self.use_gui and self.gui:
            pass  # Delegated to GUI implementation
```

**🎯 How Tasks 1-5 Use Headless Mode:**
```python
# Task-1 (Heuristics) - Already working
class HeuristicWebController(BaseGameController):
    def __init__(self, grid_size: int = 15):
        super().__init__(grid_size=grid_size, use_gui=False)  # ✅ Headless

# Task-2 (RL) - Will work seamlessly  
class RLController(BaseGameController):
    def __init__(self, grid_size: int = 20):
        super().__init__(grid_size=grid_size, use_gui=False)  # ✅ Headless

# Task-3 (Genetic) - Will work seamlessly
class GeneticController(BaseGameController):
    def __init__(self, grid_size: int = 15):
        super().__init__(grid_size=grid_size, use_gui=False)  # ✅ Headless
```

---

### **2. ✅ Universal Manager Headless Support (Perfect)**

**Location:** `core/game_manager.py`

**✅ BaseGameManager - Generic Headless Logic:**
```python
class BaseGameManager:
    def __init__(self, args: "argparse.Namespace"):
        # ✅ Universal GUI detection
        self.use_gui: bool = not getattr(args, "no_gui", False)
        
        # ✅ Conditional pygame initialization (only when needed)
        if self.use_gui:
            self.clock = pygame.time.Clock()
            self.time_delay = TIME_DELAY
            self.time_tick = TIME_TICK
        else:
            self.clock = None      # ✅ No pygame dependency in headless
            self.time_delay = 0    # ✅ No artificial delays
            self.time_tick = 0     # ✅ Maximum performance
```

**🎯 How Tasks 1-5 Use Manager Headless Mode:**
```python
# Task-1 (Heuristics) - Already working
class HeuristicGameManager(BaseGameManager):
    def __init__(self, args):
        super().__init__(args)  # ✅ Inherits headless logic
        # Runs at maximum speed when use_gui=False

# Task-2 (RL) - Will work seamlessly
class RLGameManager(BaseGameManager):
    def __init__(self, args):
        super().__init__(args)  # ✅ Same headless inheritance
        # Perfect for training loops (no GUI overhead)
```

---

### **3. ✅ Universal Replay Engine Headless Support (Perfect)**

**Location:** `replay/replay_engine.py`

**✅ BaseReplayEngine - Generic Headless Replay:**
```python
class BaseReplayEngine(BaseGameController):
    def __init__(self, log_dir: str, pause_between_moves: float = 1.0, 
                 auto_advance: bool = False, use_gui: bool = True):
        super().__init__(use_gui=use_gui)  # ✅ Inherits headless capability
        
        # ✅ Universal replay attributes (work headless or with GUI)
        self.log_dir = log_dir
        self.pause_between_moves = pause_between_moves
        self.running = True
        self.paused = False
        
    def load_game_data(self, game_number: int):
        # ✅ GUI-agnostic data loading
        if self.use_gui and self.gui and hasattr(self.gui, "move_history"):
            self.gui.move_history = []  # ✅ Safe GUI update
```

**🎯 How Tasks 1-5 Use Headless Replay:**
```python
# Task-1 (Heuristics) - Can replay BFS/A* sessions headlessly
class HeuristicReplayEngine(BaseReplayEngine):
    def __init__(self, log_dir: str, use_gui: bool = False):
        super().__init__(log_dir, use_gui=use_gui)  # ✅ Headless replay

# Task-2 (RL) - Can analyze training episodes headlessly  
class RLReplayEngine(BaseReplayEngine):
    def __init__(self, log_dir: str, use_gui: bool = False):
        super().__init__(log_dir, use_gui=use_gui)  # ✅ Headless analysis
```

---

### **4. ✅ Web Interface Perfect Headless Integration**

**Current Implementation:**
```python
# scripts/human_play_web.py - Task-0 web interface
class WebGameController(GameController):
    def __init__(self, grid_size: int = DEFAULT_GRID_SIZE):
        super().__init__(grid_size=grid_size, use_gui=False)  # ✅ Headless web

# extensions/heuristics/web/routes.py - Task-1 web interface  
class HeuristicWebController(BaseGameController):
    def __init__(self, grid_size: int = 15):
        super().__init__(grid_size=grid_size, use_gui=False)  # ✅ Headless web
```

**🎯 Future Tasks Follow Same Pattern:**
```python
# extensions/rl/web/routes.py - Future Task-2 web interface
class RLWebController(BaseGameController):
    def __init__(self, grid_size: int = 20):
        super().__init__(grid_size=grid_size, use_gui=False)  # ✅ Headless web

# extensions/genetic/web/routes.py - Future Task-3 web interface
class GeneticWebController(BaseGameController):
    def __init__(self, grid_size: int = 15):
        super().__init__(grid_size=grid_size, use_gui=False)  # ✅ Headless web
```

---

## **🎯 Perfect Inter-Class Dependencies - Zero Coupling Issues**

### **✅ Conditional Import Pattern**

**Current Architecture:**
```python
# core/game_controller.py - Safe GUI imports
if TYPE_CHECKING:
    from gui.base_gui import BaseGUI  # ✅ Only for type hints

# gui/base_gui.py - Deferred pygame import
def init_display(self, title: str = "Snake Game"):
    import pygame  # ✅ Only imported when GUI actually needed
```

### **✅ Safe GUI Method Calls**

**Universal Pattern:**
```python
# All controllers use this safe pattern
def draw(self) -> None:
    if self.use_gui and self.gui:  # ✅ Double-check prevents errors
        self.gui.draw_board(...)   # ✅ Only called when GUI exists

def reset(self) -> None:
    # ... game logic ...
    if self.use_gui and self.gui:  # ✅ Safe GUI update
        self.draw()
```

---

## **🚀 How Tasks 1-5 Leverage Perfect Headless Architecture**

### **Task-1 (Heuristics) - Already Working Perfectly:**
```python
# ✅ Headless training/evaluation
python -m extensions.heuristics.main --algorithm BFS --no-gui --max-games 1000

# ✅ Headless web interface
class HeuristicWebController(BaseGameController):
    def __init__(self, grid_size: int = 15):
        super().__init__(grid_size=grid_size, use_gui=False)  # ✅ Perfect
```


