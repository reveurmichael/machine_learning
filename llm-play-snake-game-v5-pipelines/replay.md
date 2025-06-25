# Replay System Architecture for Extensions

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ and extension guidelines. Replay components follow the same architectural patterns established in the GOODFILES.

## 🎯 **Core Philosophy: Universal Replay Foundation**

The replay system demonstrates perfect base class architecture where generic replay engines provide foundation functionality while extension-specific implementations add specialized visualization and data handling.

### **Design Philosophy**
- **Universal Base Classes**: Generic replay foundation for all extensions
- **Data Abstraction**: Flexible data structures supporting all algorithm types
- **Consistent Interface**: Uniform replay patterns across extensions
- **Educational Value**: Clear demonstration of Template Method pattern

## 🏗️ **Perfect Base Class Architecture**

### **BaseReplayEngine: Universal Foundation**
Following Final Decision patterns for extensible architecture:

```python
class BaseReplayEngine:
    """Generic replay engine for all extensions"""
    
    def __init__(self, log_dir: str, pause_between_moves: float = 1.0, 
                 auto_advance: bool = False, use_gui: bool = True):
        # Universal replay attributes (used by ALL tasks)
        self.pause_between_moves = pause_between_moves
        self.auto_advance = auto_advance
        self.running = True
        self.game_number = 1
        self.move_index = 0
        self.planned_moves = []
        
    # Universal methods for all extensions
    def set_gui(self, gui_instance): pass
    def load_next_game(self) -> None: pass
    def execute_replay_move(self, direction_key: str) -> bool: pass
    def handle_events(self): pass  # Abstract
    def _build_state_base(self) -> Dict[str, Any]: pass
```

### **BaseReplayData: Universal Data Structure**
Following SOLID principles with minimal data contracts:

```python
@dataclass(slots=True)
class BaseReplayData:
    """Minimal subset required for vanilla playback"""
    # Universal attributes (used by ALL tasks)
    apple_positions: List[List[int]]
    moves: List[str]
    game_end_reason: Optional[str]
```

### **Task-0 Extensions: LLM-Specific Only**
```python
@dataclass(slots=True)
class ReplayData(BaseReplayData):
    """Extended replay data for Task-0 LLM GUI overlay"""
    # LLM-specific additions only
    planned_moves: List[str]
    primary_llm: str
    secondary_llm: str
    timestamp: Optional[str]
    llm_response: Optional[str]
    full_json: Dict[str, Any]
```

## 🔧 **Extension Integration Patterns**

### **Heuristics Replay Implementation**
```python
class HeuristicReplayEngine(BaseReplayEngine):
    """Replay engine for heuristic algorithm sessions"""
    
    def __init__(self, log_dir: str, pause_between_moves: float = 1.0, 
                 auto_advance: bool = False, use_gui: bool = True):
        super().__init__(log_dir, pause_between_moves, auto_advance, use_gui)
        
        # Heuristic-specific extensions only
        self.algorithm_name: Optional[str] = None
        self.search_stats: Dict[str, Any] = {}
        self.path_efficiency: float = 0.0
        
    def load_game_data(self, game_number: int) -> Optional[Dict[str, Any]]:
        """Uses generic load_game_json utility"""
        game_file, game_data = load_game_json(self.log_dir, game_number)
        if game_data is None:
            return None
            
        # Parse heuristic-specific data
        self.algorithm_name = game_data.get("algorithm", "Unknown")
        self.search_stats = game_data.get("search_stats", {})
        
        # Use inherited generic parsing for common fields
        self.apple_positions = game_data["detailed_history"]["apple_positions"]
        self.moves = game_data["detailed_history"]["moves"]
        self.game_end_reason = game_data.get("game_end_reason")
        
        # Use inherited game state setup
        self.reset()
        self.snake_positions = np.array([[self.grid_size // 2, self.grid_size // 2]])
        self.set_apple_position(self.apple_positions[0])
        return game_data
        
    def _build_state_base(self) -> Dict[str, Any]:
        """Extend generic state with heuristic-specific data"""
        base_state = super()._build_state_base()
        base_state.update({
            "algorithm_name": self.algorithm_name,
            "search_stats": self.search_stats,
            "path_efficiency": self.path_efficiency,
        })
        return base_state
```

### **Supervised Learning Replay Implementation**
```python
class SupervisedReplayEngine(BaseReplayEngine):
    """Replay engine for supervised learning sessions"""
    
    def __init__(self, log_dir: str, pause_between_moves: float = 1.0, 
                 auto_advance: bool = False, use_gui: bool = True):
        super().__init__(log_dir, pause_between_moves, auto_advance, use_gui)
        
        # Supervised learning-specific extensions only
        self.model_name: Optional[str] = None
        self.prediction_confidence: List[float] = []
        self.training_metrics: Dict[str, Any] = {}
        
    def load_game_data(self, game_number: int) -> Optional[Dict[str, Any]]:
        """Uses generic load_game_json utility"""
        game_file, game_data = load_game_json(self.log_dir, game_number)
        if game_data is None:
            return None
            
        # Parse supervised learning-specific data
        self.model_name = game_data.get("model_name", "Unknown")
        self.prediction_confidence = game_data.get("prediction_confidence", [])
        
        # Use inherited generic parsing for common fields
        self.apple_positions = game_data["detailed_history"]["apple_positions"]
        self.moves = game_data["detailed_history"]["moves"]
        self.game_end_reason = game_data.get("game_end_reason")
        
        # Use inherited game state setup
        self.reset()
        self.snake_positions = np.array([[self.grid_size // 2, self.grid_size // 2]])
        self.set_apple_position(self.apple_positions[0])
        return game_data
```

### **Reinforcement Learning Replay Implementation**
```python
class RLReplayEngine(BaseReplayEngine):
    """Replay engine for reinforcement learning sessions"""
    
    def __init__(self, log_dir: str, pause_between_moves: float = 1.0, 
                 auto_advance: bool = False, use_gui: bool = True):
        super().__init__(log_dir, pause_between_moves, auto_advance, use_gui)
        
        # RL-specific extensions only
        self.agent_type: Optional[str] = None
        self.q_values: List[List[float]] = []
        self.rewards: List[float] = []
        self.episode_stats: Dict[str, Any] = {}
        
    def load_game_data(self, game_number: int) -> Optional[Dict[str, Any]]:
        """Uses generic load_game_json utility"""
        game_file, game_data = load_game_json(self.log_dir, game_number)
        if game_data is None:
            return None
            
        # Parse RL-specific data
        self.agent_type = game_data.get("agent_type", "DQN")
        self.q_values = game_data.get("q_values", [])
        self.rewards = game_data.get("rewards", [])
        
        # Use inherited generic parsing for common fields
        self.apple_positions = game_data["detailed_history"]["apple_positions"]
        self.moves = game_data["detailed_history"]["moves"]
        self.game_end_reason = game_data.get("game_end_reason")
        
        return game_data
```

## 🔧 **Generic Replay Utilities**

### **Universal File Loading**
Task-agnostic utilities for all extensions:

```python
def load_game_json(log_dir: str, game_number: int) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return the path and decoded JSON dict for game_number.
    
    This function is NOT Task0 specific - works for any extension.
    """
    # Uses FileManager singleton (generic)
    game_filename = _file_manager.get_game_json_filename(game_number)
    game_file = _file_manager.join_log_path(log_dir, game_filename)
    # Pure file I/O - works for any task
```

## 🎯 **Architectural Benefits**

### **Perfect Separation of Concerns**
- **BaseReplayEngine**: Generic replay workflow (NO LLM dependencies)
- **BaseReplayData**: Minimal universal data contract
- **Extension Engines**: Algorithm-specific visualization and data handling
- **Utility Functions**: Task-agnostic file operations

### **Extension Benefits**
- **Code Reuse**: All extensions inherit core replay functionality
- **Consistent Interface**: Same replay controls across all algorithm types
- **Easy Extension**: New algorithms add specialized data without changing base
- **Scalable Architecture**: Template Method pattern enables specialized behavior

### **Educational Value**
- **Design Patterns**: Demonstrates Template Method, Strategy, and Factory patterns
- **Clean Architecture**: Clear separation between generic and specific functionality
- **Progressive Enhancement**: From simple data playback to rich visualizations
- **SOLID Principles**: Open for extension, closed for modification

---

**The replay system architecture exemplifies the extension principles established in the Final Decision series, providing a robust foundation that scales from simple Task-0 playback to sophisticated multi-algorithm analysis while maintaining consistent interface patterns and demonstrating professional software architecture principles.**



