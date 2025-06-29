# Forbidden Import Patterns for Extensions

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines forbidden import patterns that violate the standalone principle.

> **See also:** `final-decision-10.md`, `standalone.md`, `extensions-move-guidelines.md`.

## 🚫 **Absolutely Forbidden Import Patterns**

The following import patterns are **completely forbidden** in any extension and will cause immediate build failures:

### **Cross-Extension Imports**
```python
# ❌ FORBIDDEN: Importing from other extension types
from extensions.heuristics_v0_03 import BFSAgent
from extensions.supervised_v0_02 import MLPAgent
from extensions.reinforcement_v0_01 import DQNAgent

# ❌ FORBIDDEN: Importing from different versions of same extension
from extensions.heuristics_v0_02 import AStarAgent
from extensions.supervised_v0_01 import OldMLPAgent
```

### **Version-Specific Imports**
```python
# ❌ FORBIDDEN: Direct version-specific imports
from extensions.heuristics_v0_02.agents import agent_bfs
from extensions.supervised_v0_01.models import neural_agent
from extensions.reinforcement_v0_01.agents import dqn_agent
```

### **Extension-to-Extension Communication**
```python
# ❌ FORBIDDEN: Direct extension communication
heuristic_result = heuristics_v0_02.run_algorithm()
ml_model = supervised_v0_01.train_model(heuristic_result)
```

## ✅ **Allowed Import Patterns**

### **Core Framework Imports**
```python
# ✅ ALLOWED: Core framework components
from core.game_manager import BaseGameManager
from core.game_logic import BaseGameLogic
from core.game_data import BaseGameData
from core.game_controller import BaseGameController
```

### **Common Utilities Imports**
```python
# ✅ ALLOWED: Shared utilities from common folder
from extensions.common.utils.path_utils import ensure_project_root
from extensions.common.utils.dataset_utils import load_csv_dataset
from extensions.common.utils.csv_schema_utils import generate_csv_schema
from extensions.common.validation.dataset_validator import validate_game_state
```

### **Extension-Specific Imports**
```python
# ✅ ALLOWED: Within the same extension
from .agents.agent_bfs import BFSAgent
from .agents.agent_astar import AStarAgent
from .game_logic import HeuristicGameLogic
```

## 🎯 **Standalone Principle Enforcement**

### **Extension Independence**
Each extension must be completely independent:
- **No dependencies** on other extensions
- **No shared code** between extensions
- **No cross-extension communication**
- **No version-specific dependencies**

### **Common Folder Role**
The `extensions/common/` folder provides:
- **Utility functions** for common tasks
- **Path management** utilities
- **Data validation** functions
- **Configuration** helpers
- **No algorithmic knowledge** or extension-specific logic

## 📋 **Validation Checklist**

### **Import Analysis**
- [ ] **No cross-extension imports** found
- [ ] **No version-specific imports** found
- [ ] **No extension-to-extension communication** found
- [ ] **Only core framework imports** used
- [ ] **Only common utilities imports** used
- [ ] **Only extension-specific imports** used

### **Dependency Analysis**
- [ ] **Extension is standalone** with common folder
- [ ] **No external extension dependencies**
- [ ] **No shared code between extensions**
- [ ] **Clean separation of concerns**

## 🔍 **Detection Methods**

### **Static Analysis**
```python
def check_forbidden_imports(file_path: str) -> List[str]:
    """Check for forbidden import patterns in a file"""
    forbidden_patterns = [
        r'from extensions\.heuristics_v0_',
        r'from extensions\.supervised_v0_',
        r'from extensions\.reinforcement_v0_',
        r'from extensions\.evolutionary_v0_',
        r'from extensions\.distillation_v0_',
        r'from heuristics_v0_',
        r'from supervised_v0_',
        r'from reinforcement_v0_',
        r'from evolutionary_v0_',
        r'from distillation_v0_'
    ]
    
    violations = []
    with open(file_path, 'r') as f:
        content = f.read()
        for pattern in forbidden_patterns:
            if re.search(pattern, content):
                violations.append(f"Found forbidden pattern: {pattern}")
    
    return violations
```

### **Build-Time Validation**
```python
def validate_extension_imports(extension_path: str) -> bool:
    """Validate that extension has no forbidden imports"""
    python_files = Path(extension_path).rglob("*.py")
    
    for file_path in python_files:
        violations = check_forbidden_imports(str(file_path))
        if violations:
            print(f"Import violations in {file_path}:")  # Simple logging - SUPREME_RULES
            for violation in violations:
                print(f"  - {violation}")  # Simple logging - SUPREME_RULES
            return False
    
    return True
```

## 🚨 **Consequences of Violations**

### **Build Failures**
- **Immediate rejection** of extension
- **Build system** will detect violations
- **CI/CD pipeline** will fail
- **No deployment** allowed

### **Architectural Issues**
- **Violation of standalone principle**
- **Tight coupling** between extensions
- **Maintenance complexity** increase
- **Educational value** reduction

## 🎓 **Educational Value**

### **Learning Objectives**
- **Modular Design**: Understanding independent modules
- **Clean Architecture**: Separation of concerns
- **Dependency Management**: Proper dependency isolation
- **Best Practices**: Avoiding tight coupling

### **Design Principles**
- **Single Responsibility**: Each extension has one purpose
- **Open/Closed**: Open for extension, closed for modification
- **Dependency Inversion**: Depend on abstractions, not concretions
- **Interface Segregation**: Clean, focused interfaces

## 🔗 **Cross-References and Integration**

### **Related Documents**
- **`final-decision-10.md`**: SUPREME_RULES for import patterns
- **`standalone.md`**: Standalone principle and extension independence
- **`extensions-move-guidelines.md`**: Extension development workflow

### **Implementation Files**
- **`extensions/common/utils/factory_utils.py`**: Canonical factory utilities
- **`extensions/common/utils/path_utils.py`**: Path management with factory patterns
- **`extensions/common/utils/csv_schema_utils.py`**: Schema utilities with factory patterns

### **Educational Resources**
- **Design Patterns**: Import patterns as foundation for modular design
- **SUPREME_RULES**: Canonical patterns ensure consistency across all extensions
- **Simple Logging**: Print statements provide clear operation visibility
- **OOP Principles**: Import patterns demonstrate effective module separation

---

**Forbidden import patterns ensure that each extension remains independent and maintainable, following the standalone principle and promoting clean architecture across the Snake Game AI project.**

## 🔗 **See Also**

- **`standalone.md`**: Standalone principle and extension independence
- **`final-decision-10.md`**: final-decision-10.md governance system
- **`extensions-move-guidelines.md`**: Extension development workflow

