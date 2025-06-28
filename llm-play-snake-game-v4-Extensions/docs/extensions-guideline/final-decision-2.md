# Final Decision 2: Configuration, Validation, and Architectural Standards

## 🎯 **Executive Summary**

This document establishes the **definitive architectural standards** for configuration organization, validation systems, singleton patterns, file naming conventions, and other critical structural decisions across all Snake Game AI extensions. These decisions resolve all major TODOs and provide concrete implementation guidelines.

### **GOOD_RULES Integration**
This document integrates with the **GOOD_RULES** governance system established in `final-decision-10.md`:
- **`config.md`**: Authoritative reference for configuration access rules and LLM constants whitelist
- **`validation.md`**: Authoritative reference for validation system standards
- **`naming-conventions.md`**: Authoritative reference for naming standards
- **`single-source-of-truth.md`**: Ensures configuration consistency across all extensions

### **Simple Logging Examples (SUPREME_RULE NO.3)**
All code examples in this document follow **SUPREME_RULE NO.3** by using simple print() statements rather than complex logging mechanisms:

```python
# ✅ CORRECT: Simple logging as per SUPREME_RULE NO.3
def validate_config_access(extension_type: str, imported_modules: List[str]):
    """Validate extension configuration access compliance"""
    print(f"[ConfigValidator] Validating {extension_type} extension")  # SUPREME_RULE NO.3
    
    for module in imported_modules:
        if module.startswith('config.llm_constants'):
            if not extension_type.startswith(('agentic-llms', 'llm', 'vision-language-model')):
                print(f"[ConfigValidator] WARNING: {extension_type} accessing forbidden LLM constants")  # SUPREME_RULE NO.3
                raise ImportError(f"LLM constants forbidden for {extension_type}")
    
    print(f"[ConfigValidator] Configuration access validated for {extension_type}")  # SUPREME_RULE NO.3

def get_universal_config(module: str, key: str):
    """Simple universal configuration access"""
    print(f"[Config] Accessing universal config: {module}.{key}")  # SUPREME_RULE NO.3
    
    if module == "game":
        from config.game_constants import VALID_MOVES, DIRECTIONS
        config_map = {"VALID_MOVES": VALID_MOVES, "DIRECTIONS": DIRECTIONS}
    elif module == "ui":
        from config.ui_constants import COLORS, GRID_SIZE
        config_map = {"COLORS": COLORS, "GRID_SIZE": GRID_SIZE}
    else:
        config_map = {}
    
    value = config_map.get(key)
    print(f"[Config] Retrieved {module}.{key} = {value}")  # SUPREME_RULE NO.3
    return value

## 🔧 **DECISION 1: Configuration Organization**

### **Finalized Structure**

```
ROOT/config/               # Task-0 specific (LLM-related configs)
├── game_constants.py      # ✅ Universal game rules (used by all tasks)
├── ui_constants.py        # ✅ Universal UI settings (used by all tasks) 
├── llm_constants.py       # 🚫 General extensions must not import; ✅ LLM-focused extensions (agentic-llms, vision-language-model, llm-finetune, etc.) may use
├── prompt_templates.py    # ❌ Task-0 only (LLM prompts); ✅ LLM-focused extensions (agentic-llms, vision-language-model, llm-finetune, etc.) may use
├── network_constants.py   # ✅ Universal HTTP/WebSocket settings (used by all tasks)
└── web_constants.py       # ✅ Universal Flask Web settings (used by all tasks)

extensions/common/config/  # Extension-specific configurations
├── __init__.py
├── dataset_formats.py     # Data format specifications
├── path_constants.py      # Directory path templates
└── validation_rules.py    # Validation thresholds and rules

# Note: Following SUPREME_RULE NO.3, we avoid patterns like:
# ml_constants.py, training_defaults.py, model_registry.py
# Instead, define extension-specific constants locally in each extension
```

### **Usage Patterns**

```python
# ✅ Universal constants (used by all tasks)
from config.game_constants import VALID_MOVES, DIRECTIONS, MAX_STEPS_ALLOWED
from config.ui_constants import COLORS, GRID_SIZE, WINDOW_WIDTH

# ✅ Extension-specific constants (SUPREME_RULE NO.3: define locally in extensions)
# Local constants in each extension instead of importing from common config
DEFAULT_LEARNING_RATE = 0.001
BATCH_SIZES = [16, 32, 64, 128]
EARLY_STOPPING_PATIENCE = 10

# ✅ Common utilities (lightweight, generic)
from extensions.common.config.dataset_formats import CSV_SCHEMA_VERSION

# ❌ Task-0 only (extensions should NOT import these)
# from config.llm_constants import AVAILABLE_PROVIDERS  # 🚫 Forbidden for non-LLM extensions
# from config.prompt_templates import SYSTEM_PROMPT     # 🚫 Forbidden for non-LLM extensions
```

### **Rationale**
- **Clear Separation**: Universal vs task-specific vs extension-specific
- **Single Source of Truth**: Each constant has one authoritative location
- **Import Safety**: Extensions cannot accidentally depend on LLM-specific configs
- **Scalability**: Easy to add new extension-specific configurations

## 🔍 **DECISION 2: Validation System Organization**

### **Finalized Structure**

```
extensions/common/validation/
├── __init__.py                    # Export main validation functions
├── game_state_validation.py       # Game state schema validation
├── dataset_validation.py          # Dataset format/quality validation
├── model_validation.py            # Model artifact validation
├── directory_validation.py        # File structure compliance
├── coordinate_validation.py       # Coordinate system compliance
├── cross_extension_validation.py  # Inter-extension compatibility
├── config_validation.py           # Configuration consistency validation
└── schema_validation.py           # JSON/CSV schema validation
```

### **Implementation Example**

```python
# extensions/common/validation/__init__.py
"""
Comprehensive validation system for Snake Game AI extensions.

Design Patterns:
- Strategy Pattern: Different validation strategies for different data types
- Template Method Pattern: Common validation workflow with specific implementations
- Factory Pattern: Create appropriate validators based on data type
"""

from .game_state_validation import validate_game_state, GameStateValidator
from .dataset_validation import validate_dataset, DatasetValidator
from .model_validation import validate_model_artifacts, ModelValidator
from .directory_validation import validate_directory_structure, DirectoryValidator
from .coordinate_validation import validate_coordinates, CoordinateValidator

# SUPREME_RULE NO.3: Simple validation functions instead of complex classes
def validate_game_state(state):
    """Simple game state validation"""
    if not state or 'snake_positions' not in state:
        raise ValueError("Invalid game state")
    print("[Validator] Game state is valid")
    return True

def validate_dataset_format(dataset_path):
    """Simple dataset format validation"""
    if not dataset_path.endswith('.csv'):
        raise ValueError("Expected CSV format")
    print(f"[Validator] Dataset format valid: {dataset_path}")
    return True

def validate_model_artifacts(model_path):
    """Simple model validation"""
    if not any(model_path.endswith(ext) for ext in ['.pth', '.pkl', '.onnx']):
        raise ValueError("Expected model file")
    print(f"[Validator] Model artifacts valid: {model_path}")
    return True

# Usage in extensions - simple function calls
def validate_extension_data(extension_path: str, data: dict):
    """Simple validation for extension data"""
    print(f"[Validator] Validating extension: {extension_path}")
    
    validate_game_state(data.get('game_state'))
    validate_dataset_format(data.get('dataset_path', ''))
    validate_model_artifacts(data.get('model_path', ''))
    
    print("[Validator] All validations passed")
    return True
```

### **Rationale**
- **Comprehensive Coverage**: All validation types organized systematically
- **Extensible**: Easy to add new validation types
- **Reusable**: Shared validation logic across all extensions
- **Educational**: Demonstrates multiple design patterns

## 🔄 **DECISION 3: Simple Utility Functions (SUPREME_RULE NO.3)**

### **Using Simple Functions Instead of Complex Singletons**

Following **SUPREME_RULE NO.3**, complex singleton managers have been simplified to lightweight utility functions that encourage experimentation and flexibility.

### **Simplified Utility Functions**

```python
# ✅ SIMPLIFIED UTILITY FUNCTIONS (SUPREME_RULE NO.3):
    """
    Manages all directory structure and path operations.
    
    Singleton Justification:
    - Global file system state
    - Consistent path resolution across entire application
    - Expensive initialization (directory scanning, validation)
    """
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._path_cache = {}
            self._validate_project_structure()
    
    @abstractmethod
    def get_dataset_path(self, extension_type: str, version: str, 
                        grid_size: int, algorithm: str) -> Path:
        """Get standardized dataset path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(f"logs/extensions/datasets/grid-size-{grid_size}/{extension_type}_v{version}_{timestamp}/{algorithm}")
        print(f"[PathManager] Generated dataset path: {path}")  # SUPREME_RULE NO.3
        return path

# SUPREME_RULE NO.3: Simple configuration access instead of complex managers
def get_config_value(config_type: str, key: str) -> Any:
    """Simple configuration value retrieval"""
    print(f"[Config] Getting {config_type}.{key}")
    
    # Simple config access without complex class hierarchies
    if config_type == "game":
        from config.game_constants import VALID_MOVES, DIRECTIONS
        return {"VALID_MOVES": VALID_MOVES, "DIRECTIONS": DIRECTIONS}.get(key)
    elif config_type == "ui":
        from config.ui_constants import COLORS, GRID_SIZE
        return {"COLORS": COLORS, "GRID_SIZE": GRID_SIZE}.get(key)
    
    return None

# SUPREME_RULE NO.3: Simple validation registry instead of complex singletons
_validators = {}  # Simple module-level registry

def register_validator(data_type: str, validator_func):
    """Simple validator registration"""
    print(f"[Registry] Registering validator for {data_type}")
    _validators[data_type] = validator_func

def get_validator(data_type: str):
    """Simple validator retrieval"""
    return _validators.get(data_type, lambda x: True)  # Default: always valid

# SUPREME_RULE NO.3: Simple schema functions instead of complex managers
def get_csv_schema(grid_size: int) -> list:
    """Simple CSV schema retrieval"""
    print(f"[Schema] Getting CSV schema for grid {grid_size}x{grid_size}")
    
    # Standard 16-feature schema works for any grid size
    return [
        'head_x', 'head_y', 'apple_x', 'apple_y', 'snake_length',
        'apple_dir_up', 'apple_dir_down', 'apple_dir_left', 'apple_dir_right',
        'danger_straight', 'danger_left', 'danger_right',
        'free_space_up', 'free_space_down', 'free_space_left', 'free_space_right',
        'game_id', 'step_in_game', 'target_move'
    ]

# SUPREME_RULE NO.3: Simple model registry instead of complex managers
_model_types = {}  # Simple module-level registry

def register_model_type(model_name: str, model_class):
    """Simple model registration"""
    print(f"[ModelRegistry] Registering model: {model_name}")
    _model_types[model_name] = model_class

def get_model_class(model_name: str):
    """Simple model class retrieval"""
    return _model_types.get(model_name)

def list_available_models():
    """List all registered models"""
    return list(_model_types.keys())
```

## 🚫 **EXPLICIT ARCHITECTURAL REJECTIONS**

### **Factory Pattern Rejections**
- ❌ **BaseFactory abstract class** in `extensions/common/utils/`
- ❌ **factory_utils.py module** in `extensions/common/utils/`
- ❌ **Shared factory inheritance hierarchy**
- ✅ **Instead**: Simple dictionary-based factories in each extension (SUPREME_RULE NO.3)

### **Singleton Pattern Rejections**  
- ❌ **singleton_utils.py in extensions/common/utils/**
- ❌ **Any wrapper around ROOT/utils/singleton_utils.py**
- ❌ **Duplicating singleton functionality in extensions/common/**
- ✅ **Instead**: Use ROOT/utils/singleton_utils.py when truly needed, prefer simple functions (SUPREME_RULE NO.3)

### **NOT Singleton Classes**

```python
# ❌ NOT RECOMMENDED as Singletons:

# Game-specific classes - need multiple instances
class GameManager:        # Different games need separate instances
class GameData:          # Each game has its own data
class GameController:    # Multiple games can run simultaneously

# Agent classes - need multiple instances for comparison
class BFSAgent:          # Need multiple agents for A/B testing
class MLPAgent:          # Need separate instances for different models

# Training classes - need separate instances for different experiments
class TrainingManager:   # Multiple training sessions
class ModelTrainer:      # Different models training simultaneously
```

### **Rationale**
- **True Global State**: Only classes with genuinely global, stateless responsibilities
- **Expensive Initialization**: Classes with costly setup that benefits from sharing
- **Consistency Requirements**: Classes that must maintain consistency across application
- **Avoid Over-Singletonization**: Preserve flexibility for multi-instance scenarios

## 📁 **DECISION 4: File and Class Naming Conventions**

### **Agent Naming Convention**

```python
# ✅ STANDARDIZED PATTERN:

# File names: agent_{algorithm}.py
agents/
├── agent_bfs.py              # Breadth-First Search
├── agent_astar.py            # A* pathfinding
├── agent_hamiltonian.py      # Hamiltonian path
├── agent_mlp.py              # Multi-Layer Perceptron
├── agent_cnn.py              # Convolutional Neural Network
├── agent_xgboost.py          # XGBoost gradient boosting
├── agent_dqn.py              # Deep Q-Network
├── agent_ppo.py              # Proximal Policy Optimization
└── agent_lora.py             # LoRA fine-tuned LLM

# Class names: {Algorithm}Agent
class BFSAgent(BaseAgent):               # from agent_bfs.py
class AStarAgent(BaseAgent):             # from agent_astar.py
class HamiltonianAgent(BaseAgent):       # from agent_hamiltonian.py
class MLPAgent(BaseAgent):               # from agent_mlp.py
class CNNAgent(BaseAgent):               # from agent_cnn.py
class XGBoostAgent(BaseAgent):           # from agent_xgboost.py
class DQNAgent(BaseAgent):               # from agent_dqn.py
class PPOAgent(BaseAgent):               # from agent_ppo.py
class LoRAAgent(BaseAgent):              # from agent_lora.py
```

### **Configuration File Naming**

```python
# ✅ STANDARDIZED PATTERN:

# Universal constants (ROOT/config/)
game_constants.py       # Universal game rules
ui_constants.py         # Universal UI settings
coordinate_constants.py # Universal coordinate system

# Task-0 specific (ROOT/config/)
llm_constants.py        # LLM providers, models
prompt_templates.py     # LLM prompt templates
network_constants.py    # HTTP, WebSocket settings

dataset_formats.py      # Data format specifications
validation_rules.py     # Validation thresholds and rules
```

### **Extension Structure Naming**

```python
# ✅ STANDARDIZED PATTERN:

extensions/
├── {algorithm}-v0.01/           # Proof of concept
├── {algorithm}-v0.02/           # Multi-algorithm expansion
├── {algorithm}-v0.03/           # Web interface + dataset generation
└── heuristics-v0.04/            # Advanced features (heuristics only)

# Where {algorithm} is:
heuristics-v0.0N/        # Pathfinding algorithms (v0.01-v0.04)
supervised-v0.0N/        # Machine learning models (v0.01-v0.03)
reinforcement-v0.0N/     # Reinforcement learning (v0.01-v0.03)
llm-finetune-v0.0N/      # LLM fine-tuning (v0.01-v0.03)
llm-distillation-v0.0N/  # Model distillation (v0.01-v0.03)
evolutionary-v0.0N/      # Genetic algorithms (v0.01-v0.03)
```

## 📊 **DECISION 5: CSV Schema Grid-Size Handling**

### **Grid-Size Agnostic Feature Engineering**

```python
# ✅ FINALIZED: 16 normalized features work for any grid size

GRID_SIZE_AGNOSTIC_FEATURES = {
    # Normalized absolute positions (0.0 to 1.0)
    'head_x_normalized': 'head_x / grid_size',
    'head_y_normalized': 'head_y / grid_size', 
    'apple_x_normalized': 'apple_x / grid_size',
    'apple_y_normalized': 'apple_y / grid_size',
    
    # Relative directional features (binary 0/1)
    'apple_dir_up': '1 if apple_y > head_y else 0',
    'apple_dir_down': '1 if apple_y < head_y else 0',
    'apple_dir_left': '1 if apple_x < head_x else 0', 
    'apple_dir_right': '1 if apple_x > head_x else 0',
    
    # Immediate danger detection (binary 0/1)
    'danger_straight': '1 if collision ahead else 0',
    'danger_left': '1 if collision to left else 0',
    'danger_right': '1 if collision to right else 0',
    
    # Proportional free space (0.0 to 1.0)
    'free_space_up_ratio': 'free_space_up / grid_size',
    'free_space_down_ratio': 'free_space_down / grid_size',
    'free_space_left_ratio': 'free_space_left / grid_size', 
    'free_space_right_ratio': 'free_space_right / grid_size',
    
    # Game state (absolute value)
    'snake_length': 'len(snake_positions)'
}

# Additional metadata columns (not features)
METADATA_COLUMNS = ['game_id', 'step_in_game', 'grid_size']
TARGET_COLUMN = ['target_move']

# Total: 16 features + 3 metadata + 1 target = 20 columns
```

### **Implementation**

```python
# extensions/common/csv_schema.py
class GridSizeAgnosticCSVSchema:
    """
    CSV schema that works consistently across all grid sizes.
    
    Design Principles:
    - Normalized features (0.0 to 1.0 range)
    - Grid-size independent feature count
    - Consistent model training across different grid sizes
    """
    
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.feature_count = 16  # Always 16 features regardless of grid_size
    
    def extract_features(self, game_state: dict) -> dict:
        """Extract normalized features from game state"""
        head_x, head_y = game_state['snake'][0]['x'], game_state['snake'][0]['y']
        apple_x, apple_y = game_state['food']['x'], game_state['food']['y']
        
        features = {
            # Normalized positions
            'head_x_normalized': head_x / self.grid_size,
            'head_y_normalized': head_y / self.grid_size,
            'apple_x_normalized': apple_x / self.grid_size,
            'apple_y_normalized': apple_y / self.grid_size,
            
            # Relative directions (binary)
            'apple_dir_up': 1 if apple_y > head_y else 0,
            'apple_dir_down': 1 if apple_y < head_y else 0,
            'apple_dir_left': 1 if apple_x < head_x else 0,
            'apple_dir_right': 1 if apple_x > head_x else 0,
            
            # Danger detection (binary)
            'danger_straight': self._check_danger_straight(game_state),
            'danger_left': self._check_danger_left(game_state), 
            'danger_right': self._check_danger_right(game_state),
            
            # Proportional free space
            'free_space_up_ratio': self._get_free_space_up(game_state) / self.grid_size,
            'free_space_down_ratio': self._get_free_space_down(game_state) / self.grid_size,
            'free_space_left_ratio': self._get_free_space_left(game_state) / self.grid_size,
            'free_space_right_ratio': self._get_free_space_right(game_state) / self.grid_size,
            
            # Game state
            'snake_length': len(game_state['snake'])
        }
        
        return features
```

## 🛠️ **DECISION 6: Path Management Standardization**

### **Mandatory Usage Pattern**

```python
# ✅ ALL extensions must use this pattern:
from extensions.common.path_utils import ensure_project_root, get_extension_path

def setup_extension_environment():
    """Standard setup for all extensions"""
    project_root = ensure_project_root()
    extension_path = get_extension_path(__file__)
    return project_root, extension_path

def main():
    """Main entry point with proper path management"""
    project_root, extension_path = setup_extension_environment()
    
    # Initialize extension components
    print(f"[Main] Starting extension execution")  # SUPREME_RULE NO.3
    
    # Load configuration
    config = load_extension_config(extension_path)
    
    # Run extension logic
    results = execute_extension_logic(config)
    
    # Save results
    save_extension_results(results, project_root)
    
    print(f"[Main] Extension execution completed")  # SUPREME_RULE NO.3
```

### **Forbidden Patterns**

```python
# ❌ FORBIDDEN: Manual path finding
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
os.chdir(project_root)

# ❌ FORBIDDEN: Hardcoded paths
sys.path.insert(0, "/Users/someone/project/root")
os.chdir("/Users/someone/project/root")

# ❌ FORBIDDEN: Relative path assumptions
sys.path.insert(0, "../../../")
os.chdir("../../../")
```
