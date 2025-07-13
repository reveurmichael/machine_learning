# Single Source of Truth (SSOT) Principles

## 🎯 **Core Philosophy: Eliminate Redundancy**

The Single Source of Truth (SSOT) principle ensures that every piece of information has exactly one authoritative location. This eliminates contradictions, reduces maintenance burden, and ensures consistency across the entire codebase.

### **Educational Value**
- **Consistency**: Understanding how to maintain consistent information
- **Maintenance**: Learning to reduce maintenance burden through SSOT
- **Reliability**: Ensuring data accuracy and reliability
- **Architecture**: Understanding SSOT in system design

## 🏗️ **SSOT Implementation Patterns**

### **Configuration Management**
```python
# ✅ CORRECT: Single source for configuration
from config.game_constants import VALID_MOVES, GRID_SIZES

class GameController:
    def __init__(self):
        self.valid_moves = VALID_MOVES  # Single source
        self.grid_sizes = GRID_SIZES    # Single source
        print_info("[GameController] Using SSOT configuration")  # SUPREME_RULES compliant logging

# ❌ WRONG: Multiple sources for same information
class GameController:
    def __init__(self):
        self.valid_moves = ["UP", "DOWN", "LEFT", "RIGHT"]  # Duplicate definition
        self.grid_sizes = [8, 10, 12, 16, 20]              # Duplicate definition
```

### **Path Management**
```python
# ✅ CORRECT: Single source for path management
from utils.path_utils import ensure_project_root

def setup_environment():
    project_root = ensure_project_root()  # Single source
    print_info(f"[Setup] Using project root: {project_root}")  # SUPREME_RULES compliant logging

# ❌ WRONG: Multiple path management implementations
def setup_environment():
    # Custom path finding logic - violates SSOT
    current = Path(__file__).resolve()
    for _ in range(10):
        if (current / "config").is_dir():
            os.chdir(str(current))
            break
        current = current.parent
```

### **Data Generation**
```python
# ✅ CORRECT: Single source for data generation
class DatasetGenerator:
    def _create_jsonl_record(self, game_state: dict, move: str, explanation: dict) -> dict:
        """Single method to generate JSONL record"""
        return {
            "prompt": self._create_prompt(game_state),
            "completion": self._create_completion(move, explanation),
            "metadata": self._create_metadata(game_state)
        }
    
    def _create_csv_record(self, game_state: dict, move: str) -> dict:
        """Single method to generate CSV record"""
        return {
            "head_x": game_state["head_position"][0],
            "head_y": game_state["head_position"][1],
            "apple_x": game_state["apple_position"][0],
            "apple_y": game_state["apple_position"][1],
            "target_move": move
        }

# ❌ WRONG: Multiple data generation methods
class DatasetGenerator:
    def create_jsonl_record_v1(self, game_state: dict) -> dict:
        # Duplicate implementation
        pass
    
    def create_jsonl_record_v2(self, game_state: dict) -> dict:
        # Another duplicate implementation
        pass
```

## 📊 **SSOT Standards**

### **Configuration SSOT**
- **Game Constants**: All game-related constants in `config/game_constants.py`
- **LLM Constants**: All LLM-related constants in `config/llm_constants.py`
- **Network Constants**: All network-related constants in `config/network_constants.py`
- **UI Constants**: All UI-related constants in `config/ui_constants.py`

### **Path Management SSOT**
- **Project Root**: Single `ensure_project_root()` function in `utils/path_utils.py`
- **Dataset Paths**: Single path generation utilities in `extensions/common/utils/path_utils.py`
- **Log Paths**: Single log path management in `core/game_file_manager.py`

### **Data Generation SSOT**
- **CSV Generation**: Single CSV generation logic in `extensions/common/utils/csv_utils.py`
- **JSONL Generation**: Single JSONL generation logic in agent classes
- **Game Data**: Single game data generation in `core/game_data.py`

### **Logging SSOT**
- **Print Functions**: Single logging functions in `utils/print_utils.py`
- **No .log Files**: No complex logging frameworks (SUPREME_RULE NO.3)
- **Simple Logging**: Use only `print_info`, `print_warning`, `print_error`, `print_success`

## 🔧 **SSOT Implementation Guidelines**

### **1. Identify Duplicate Information**
```python
# Look for duplicate definitions
VALID_MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]  # In multiple files
GRID_SIZES = [8, 10, 12, 16, 20]              # In multiple files

# Consolidate to single source
# config/game_constants.py
VALID_MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]
GRID_SIZES = [8, 10, 12, 16, 20]
```

### **2. Create Single Source**
```python
# ✅ CORRECT: Single source for all imports
from config.game_constants import VALID_MOVES, GRID_SIZES

class GameController:
    def __init__(self):
        self.valid_moves = VALID_MOVES  # Single source
        self.grid_sizes = GRID_SIZES    # Single source
```

### **3. Update All References**
```python
# Update all files to use single source
# game_logic.py
from config.game_constants import VALID_MOVES

# agent_bfs.py
from config.game_constants import VALID_MOVES

# game_manager.py
from config.game_constants import VALID_MOVES
```

## 📋 **SSOT Compliance Checklist**

### **Configuration Management**
- [ ] **Game Constants**: All game constants in `config/game_constants.py`
- [ ] **LLM Constants**: All LLM constants in `config/llm_constants.py`
- [ ] **Network Constants**: All network constants in `config/network_constants.py`
- [ ] **UI Constants**: All UI constants in `config/ui_constants.py`

### **Path Management**
- [ ] **Project Root**: Use `utils/path_utils.ensure_project_root()` only
- [ ] **Dataset Paths**: Use `extensions/common/utils/path_utils.py` only
- [ ] **Log Paths**: Use `core/game_file_manager.py` only

### **Data Generation**
- [ ] **CSV Generation**: Use `extensions/common/utils/csv_utils.py` only
- [ ] **JSONL Generation**: Use agent-specific methods only
- [ ] **Game Data**: Use `core/game_data.py` only

### **Logging**
- [ ] **Print Functions**: Use `utils/print_utils.py` only
- [ ] **No .log Files**: No complex logging frameworks
- [ ] **Simple Logging**: Use only simple print functions

## 🎓 **Educational Benefits**

### **Learning Objectives**
- **Consistency Management**: Understanding how to maintain consistent information
- **Maintenance Reduction**: Learning to reduce maintenance burden
- **Reliability**: Ensuring data accuracy and reliability
- **Architecture Design**: Understanding SSOT in system design

### **Best Practices**
- **Identify Duplicates**: Find and eliminate duplicate information
- **Create Single Source**: Establish authoritative sources for information
- **Update References**: Ensure all code uses single sources
- **Maintain Consistency**: Keep single sources up-to-date

## ✅ **Success Indicators**

### **Working Implementation Examples**
- **Heuristics v0.04**: Successfully uses SSOT for all data generation
- **Configuration Management**: All constants properly centralized
- **Path Management**: Single path management implementation
- **Data Generation**: Single data generation methods per format
- **Logging**: Consistent logging across all components

### **Quality Standards**
- **No Duplicates**: No duplicate definitions of same information
- **Single Sources**: All information has single authoritative source
- **Consistent References**: All code references single sources
- **Maintainable**: Easy to update and maintain

---

**Single Source of Truth principles ensure consistency, reliability, and maintainability across all Snake Game AI extensions while reducing complexity and eliminating contradictions.**

## 🔗 **See Also**

- **`final-decision.md`**: SUPREME_RULES governance system and canonical standards
- **`config.md`**: Configuration management standards
- **`cwd-and-logs.md`**: Path management and logging standards
- **`data-format-decision-guide.md`**: Data format selection guidelines









