# Final Decision 3: Simple Utility Functions Architecture

> **SUPREME AUTHORITY**: This document establishes the definitive standards for simple utility functions following SUPREME_RULE NO.3.

> **See also:** `utils.md` (Utility standards), `elegance.md` (Code quality), `no-over-preparation.md` (Simplicity principles), `final-decision-10.md` (SUPREME_RULES).

## 🎯 **Core Philosophy: Lightweight, OOP-Based Common Utilities**

The `extensions/common/` folder should serve as a lightweight, reusable foundation for all extensions, supporting experimentation and flexibility. Its code must be simple, preferably object-oriented (OOP) but never over-engineered, strictly following SUPREME_RULES from `final-decision-10.md`.

### **SUPREME_RULES Integration**
- **SUPREME_RULE NO.1**: Enforces reading all GOOD_RULES before making utility architectural changes to ensure comprehensive understanding
- **SUPREME_RULE NO.2**: Uses precise `final-decision-N.md` format consistently when referencing architectural decisions
- **SUPREME_RULE NO.3**: Enables lightweight common utilities with OOP extensibility while maintaining utility patterns through inheritance rather than tight coupling
- **SUPREME_RULE NO.4**: Ensures all markdown files are coherent and aligned through nuclear diffusion infusion process

### **GOOD_RULES Integration**
This document integrates with the **GOOD_RULES** governance system established in `final-decision-10.md`:
- **`utils.md`**: Authoritative reference for utility function standards
- **`elegance.md`**: Authoritative reference for code quality and simplicity standards
- **`single-source-of-truth.md`**: Ensures utility consistency across all extensions
- **`no-over-preparation.md`**: Prevents over-engineering of utility functions

### **Simple Logging Examples (SUPREME_RULE NO.3)**
All code examples in this document follow **SUPREME_RULE NO.3** by using simple print() statements rather than complex logging mechanisms:

```python
# ✅ CORRECT: Simple logging as per SUPREME_RULE NO.3
def get_dataset_path(extension_type: str, version: str, grid_size: int, algorithm: str) -> str:
    """Simple dataset path generation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/extensions/datasets/grid-size-{grid_size}/{extension_type}_v{version}_{timestamp}/{algorithm}"
    print(f"[Path] Generated dataset path: {path}")  # SUPREME_RULE NO.3
    return path

def validate_grid_size(grid_size: int):
    """Simple grid size validation"""
    if grid_size < 5 or grid_size > 50:
        print(f"[Validator] Invalid grid size: {grid_size} (must be 5-50)")  # SUPREME_RULE NO.3
        raise ValueError(f"Grid size should be reasonable (5-50), got {grid_size}")
    print(f"[Validator] Grid size {grid_size} is valid")  # SUPREME_RULE NO.3

def register_validator(data_type: str, validator_func):
    """Simple validator registration"""
    print(f"[Registry] Registering validator for {data_type}")  # SUPREME_RULE NO.3
    _validators[data_type] = validator_func
```

## 🎯 **Executive Summary**

This document establishes **lightweight utility functions** for the Snake Game AI project following **SUPREME_RULE NO.3**: "The extensions/common/ folder should stay lightweight and generic." Complex singleton patterns have been simplified to simple, easy-to-understand functions.

## 🛠️ **Existing Infrastructure (Available When Needed)**

The project includes a robust singleton implementation in `ROOT/utils/singleton_utils.py`:
- **`SingletonABCMeta`**: Generic, thread-safe metaclass for all tasks
- **Double-checked locking**: High-performance singleton implementation
- **Testing utilities**: Available for any extension that truly needs singleton behavior

## 🚫 **EXPLICIT DECISION: NO singleton_utils.py in extensions/common/**

**CRITICAL ARCHITECTURAL DECISION**: This project **explicitly rejects**:
- ❌ **singleton_utils.py in extensions/common/utils/**
- ❌ **Any wrapper around ROOT/utils/singleton_utils.py**
- ❌ **Duplicating singleton functionality in extensions/common/**

**Rationale**: 
- **ROOT/utils/singleton_utils.py is already generic** and works for all tasks (0-5)
- **SUPREME_RULE NO.3**: Avoid unnecessary duplication and complexity
- **Most use cases should use simple functions** instead of singletons

## 🔄 **DECISION: Simple Functions Over Complex Singletons**

### **✅ SIMPLIFIED UTILITY FUNCTIONS**

#### **1. Simple Path Management Functions**
```python
from datetime import datetime
from pathlib import Path

def get_dataset_path(extension_type: str, version: str, grid_size: int, algorithm: str) -> str:
    """Simple dataset path generation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/extensions/datasets/grid-size-{grid_size}/{extension_type}_v{version}_{timestamp}/{algorithm}"
    print(f"[Path] Generated dataset path: {path}")
    return path

def get_model_path(extension_type: str, version: str, grid_size: int, model_name: str) -> str:
    """Simple model path generation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/extensions/models/grid-size-{grid_size}/{extension_type}_v{version}_{timestamp}/{model_name}"
    print(f"[Path] Generated model path: {path}")
    return path

def ensure_directory_exists(path: str):
    """Simple directory creation"""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"[Path] Ensured directory exists: {path}")

def validate_grid_size(grid_size: int):
    """Simple grid size validation"""
    if grid_size < 5 or grid_size > 50:
        raise ValueError(f"Grid size should be reasonable (5-50), got {grid_size}")
    print(f"[Path] Grid size {grid_size} is valid")
```

#### **2. Simple Configuration Access**
```python
# SUPREME_RULE NO.3: Simple configuration access instead of complex singletons
def get_universal_config(module: str, key: str):
    """Simple universal configuration access"""
    print(f"[Config] Accessing universal config: {module}.{key}")
    
    # Direct imports - simple and clear
    if module == "game":
        from config.game_constants import VALID_MOVES, DIRECTIONS
        config_map = {"VALID_MOVES": VALID_MOVES, "DIRECTIONS": DIRECTIONS}
    elif module == "ui":
        from config.ui_constants import COLORS, GRID_SIZE
        config_map = {"COLORS": COLORS, "GRID_SIZE": GRID_SIZE}
    else:
        config_map = {}
    
    return config_map.get(key)

def get_extension_config(module: str, key: str, default=None):
    """Simple extension configuration access"""
    print(f"[Config] Accessing extension config: {module}.{key}")
    
    # Extension-specific constants defined locally (SUPREME_RULE NO.3)
    if module == "dataset":
        local_config = {"CSV_SCHEMA_VERSION": "1.0", "FEATURE_COUNT": 16}
    else:
        local_config = {}
    
    return local_config.get(key, default)
```

#### **3. Simple Validation Functions**
```python
# SUPREME_RULE NO.3: Simple validation functions instead of complex registries
_validators = {}  # Simple module-level registry

def register_validator(data_type: str, validator_func):
    """Simple validator registration"""
    print(f"[Validation] Registering validator for {data_type}")
    _validators[data_type] = validator_func

def validate_data(data_type: str, data):
    """Simple data validation"""
    validator = _validators.get(data_type, lambda x: True)  # Default: always valid
    try:
        result = validator(data)
        print(f"[Validation] {data_type} validation: {'PASS' if result else 'FAIL'}")
        return result
    except Exception as e:
        print(f"[Validation] {data_type} validation error: {e}")
        return False

def get_schema(schema_type: str, version: str = "latest"):
    """Simple schema retrieval"""
    print(f"[Schema] Getting {schema_type} schema v{version}")
    
    # Simple schema definitions without complex caching
    if schema_type == "csv":
        return ["head_x", "head_y", "apple_x", "apple_y", "snake_length", "target_move"]
    
    return []
```

#### **4. Simple Schema Management**
```python
# SUPREME_RULE NO.3: Simple schema functions instead of complex managers
def get_csv_schema(grid_size: int, version: str = "v1"):
    """Simple CSV schema retrieval - grid-size agnostic"""
    print(f"[Schema] Getting CSV schema v{version} for grid {grid_size}x{grid_size}")
    
    # Standard 16-feature schema works for any grid size
    return [
        'head_x', 'head_y', 'apple_x', 'apple_y', 'snake_length',
        'apple_dir_up', 'apple_dir_down', 'apple_dir_left', 'apple_dir_right',
        'danger_straight', 'danger_left', 'danger_right',
        'free_space_up', 'free_space_down', 'free_space_left', 'free_space_right',
        'game_id', 'step_in_game', 'target_move'
    ]

def extract_features(game_state, grid_size: int):
    """Simple feature extraction function"""
    print(f"[Features] Extracting features for grid {grid_size}x{grid_size}")
    
    # Simple feature extraction without complex classes
    features = {
        'head_x': game_state.get('head_position', [0, 0])[0],
        'head_y': game_state.get('head_position', [0, 0])[1],
        'snake_length': len(game_state.get('snake_positions', []))
    }
    return features

def validate_dataset_compatibility(dataset_path: str, expected_schema: str):
    """Simple dataset compatibility validation"""
    print(f"[Validation] Validating dataset compatibility: {dataset_path}")
    
    # Simple validation without complex classes
    if not dataset_path.endswith('.csv'):
        print(f"[Validation] Dataset must be CSV format")
        return False
    
    print(f"[Validation] Dataset compatibility validated")
    return True
```

## 🏭 **Factory Pattern Integration**

### **Simple Factory Functions**
```python
# SUPREME_RULE NO.3: Simple factory functions instead of complex classes
def create_agent_factory():
    """Create agent factory using canonical factory pattern"""
    return AgentFactory.create("default")  # CANONICAL create() method

def create_validator_factory():
    """Simple validator factory creation"""
    print(f"[Factory] Creating validator factory")
    
    registry = {
        "DATASET": DatasetValidator,
        "PATH": PathValidator,
        "SCHEMA": SchemaValidator,
    }
    
    def create(validator_type: str, **kwargs):  # CANONICAL create() method
        """Create validator using canonical create() method (SUPREME_RULES compliance)"""
        validator_class = registry.get(validator_type.upper())
        if not validator_class:
            available = list(registry.keys())
            raise ValueError(f"Unknown validator type: {validator_type}. Available: {available}")
        print(f"[Factory] Creating validator: {validator_type}")  # Simple logging
        return validator_class(**kwargs)
    
    return create
```

## 🎓 **Educational Value and Learning Path**

### **Learning Objectives**
- **Simple Functions**: Understanding when to use simple functions vs complex classes
- **Utility Design**: Learning to design lightweight, reusable utilities
- **Factory Patterns**: Understanding canonical factory pattern implementation
- **Code Simplicity**: Learning to avoid over-engineering

### **Implementation Examples**
- **Path Management**: How to create simple path utilities
- **Configuration Access**: How to safely access configuration constants
- **Validation**: How to build simple validation systems
- **Schema Management**: How to handle data schemas efficiently

## 🔗 **Integration with Other Documentation**

### **GOOD_RULES Alignment**
This document aligns with:
- **`utils.md`**: Detailed utility function standards
- **`elegance.md`**: Code quality and simplicity principles
- **`no-over-preparation.md`**: Avoiding over-engineering
- **`single-source-of-truth.md`**: Architectural principles

### **Extension Guidelines**
This utility architecture supports:
- All extension types (heuristics, supervised, reinforcement, LLM)
- All utility needs (paths, configuration, validation, schemas)
- Simple, lightweight implementations
- Consistent patterns across all extensions

---

**This simple utility functions architecture ensures lightweight, reusable, and educational utilities across all Snake Game AI extensions while maintaining SUPREME_RULES compliance.**

> **SUPREME_RULES COMPLIANCE**: This document strictly follows the SUPREME_RULES established in `final-decision-10.md`, ensuring coherence, educational value, and architectural integrity across the entire project.