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
├── dataset_validator.py           # Dataset format validation
├── path_validator.py              # Path structure compliance
└── schema_validator.py            # JSON/CSV schema validation
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

from .dataset_validator import validate_dataset, DatasetValidator
from .path_validator import validate_directory_structure, PathValidator
from .schema_validator import validate_schema, SchemaValidator

# SUPREME_RULE NO.3: Simple validation functions instead of complex classes
def validate_dataset_format(dataset_path):
    """Simple dataset format validation"""
    if not dataset_path.endswith('.csv'):
        raise ValueError("Expected CSV format")
    print(f"[Validator] Dataset format valid: {dataset_path}")
    return True

def validate_path_structure(path):
    """Simple path structure validation"""
    if not path or not path.exists():
        raise ValueError("Invalid path structure")
    print(f"[Validator] Path structure valid: {path}")
    return True

def validate_schema_compliance(data, schema):
    """Simple schema compliance validation"""
    if not data or not schema:
        raise ValueError("Invalid data or schema")
    print(f"[Validator] Schema compliance valid")
    return True

# Usage in extensions - simple function calls
def validate_extension_data(extension_path: str, data: dict):
    """Simple validation for extension data"""
    print(f"[Validator] Validating extension: {extension_path}")
    
    validate_dataset_format(data.get('dataset_path', ''))
    validate_path_structure(data.get('model_path', ''))
    validate_schema_compliance(data.get('schema_data'), data.get('schema'))
    
    print("[Validator] All validations passed")
    return True
```

### **Rationale**
- **Essential Coverage**: Focus on core validation needs
- **Extensible**: Easy to add new validation types
- **Reusable**: Shared validation logic across all extensions
- **Educational**: Demonstrates multiple design patterns

## 🔄 **DECISION 3: Simple Utility Functions (SUPREME_RULE NO.3)**

### **Using Simple Functions Instead of Complex Singletons**

Following **SUPREME_RULE NO.3**, complex singleton managers have been simplified to lightweight utility functions that encourage experimentation and flexibility.

### **Simplified Utility Functions**

```python
# ✅ SIMPLIFIED UTILITY FUNCTIONS (SUPREME_RULE NO.3):
def get_project_root():
    """Get project root directory"""
    print("[PathUtils] Getting project root")  # Simple logging
    # Implementation here
    return project_root

def get_extension_path(extension_type: str, version: str):
    """Get extension directory path"""
    print(f"[PathUtils] Getting path for {extension_type}-{version}")  # Simple logging
    # Implementation here
    return extension_path

def validate_extension_structure(extension_path):
    """Validate extension directory structure"""
    print(f"[Validator] Validating structure: {extension_path}")  # Simple logging
    # Implementation here
    return True

# ❌ FORBIDDEN: Complex singleton managers (violates SUPREME_RULE NO.3)
# class PathManager:
#     _instance = None
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance
```

### **Rationale**
- **Simplicity**: Lightweight functions instead of complex classes
- **Flexibility**: Easy to modify and extend
- **Educational**: Clear, understandable code
- **Performance**: No singleton overhead

## 🏗️ **DECISION 4: File Naming Conventions**

### **Finalized Standards**

```
# Core files (ROOT/)
game_*.py                  # Game-related functionality
llm_*.py                   # LLM-specific functionality
web_*.py                   # Web interface functionality
app.py                     # Main application entry point

# Extension files (extensions/)
{extension_type}-v{version}/  # Extension directories
├── agents/                   # Agent implementations
├── config.py                 # Extension-specific configuration
├── main.py                   # Extension entry point
└── app.py                    # Streamlit application (v0.03+)

# Common utilities (extensions/common/)
utils/                        # Utility functions
├── factory_utils.py          # Factory pattern utilities
├── path_utils.py             # Path management utilities
├── dataset_utils.py          # Dataset handling utilities
└── csv_schema_utils.py       # CSV schema utilities

validation/                   # Validation functions
├── dataset_validator.py      # Dataset validation
├── path_validator.py         # Path validation
└── schema_validator.py       # Schema validation

config/                       # Configuration constants
├── dataset_formats.py        # Data format specifications
├── path_constants.py         # Path templates
└── validation_rules.py       # Validation rules
```

### **Rationale**
- **Consistency**: Clear, predictable naming patterns
- **Discoverability**: Easy to find relevant files
- **Scalability**: Works for any number of extensions
- **Educational**: Demonstrates good file organization practices

## 📋 **Implementation Checklist**

### **Configuration Standards**
- [ ] **Universal Constants**: All extensions use `config/game_constants.py` and `config/ui_constants.py`
- [ ] **Extension-Specific**: Each extension defines its own constants locally
- [ ] **LLM Constants**: Only LLM-focused extensions import `config/llm_constants.py`
- [ ] **Import Safety**: No accidental imports of task-specific constants

### **Validation Standards**
- [ ] **Simple Functions**: Use lightweight validation functions (SUPREME_RULE NO.3)
- [ ] **Essential Coverage**: Focus on core validation needs
- [ ] **Reusable**: Shared validation logic across extensions
- [ ] **Educational**: Demonstrate design patterns clearly

### **Utility Standards**
- [ ] **Simple Functions**: Lightweight utility functions instead of complex classes
- [ ] **Flexibility**: Easy to modify and extend
- [ ] **Performance**: No unnecessary overhead
- [ ] **Clarity**: Clear, understandable code

### **Naming Standards**
- [ ] **Consistency**: Follow established naming conventions
- [ ] **Predictability**: Clear, logical file organization
- [ ] **Scalability**: Works for any number of extensions
- [ ] **Educational**: Demonstrate good practices

---

**This document establishes the definitive architectural standards for the Snake Game AI project, ensuring consistency, clarity, and educational value across all extensions while following SUPREME_RULES compliance.**

## 🔗 **See Also**

- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`config.md`**: Configuration management standards
- **`validation.md`**: Validation system standards
- **`naming-conventions.md`**: Naming standards
- **`single-source-of-truth.md`**: Architectural principles
