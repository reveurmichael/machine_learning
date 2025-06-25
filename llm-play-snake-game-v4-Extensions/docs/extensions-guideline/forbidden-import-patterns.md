# Forbidden Import Patterns

No from {algorithm}_v0.0N import patterns
No from extensions.{algorithm}_v0.0N import patterns

## 🚫 **Strictly Forbidden Import Patterns**

These import patterns are **never allowed** in the Snake Game AI project:

```python
# ❌ NEVER DO THIS - Direct extension-to-extension imports
from heuristics_v0.03 import some_module
from supervised_v0.02 import some_module  
from reinforcement_v0.01 import some_module
from extensions.heuristics_v0.03 import some_module
from extensions.supervised_v0.02 import some_module
```

## ✅ **Allowed Import Patterns**

```python
# ✅ CORRECT - Import from common utilities
from extensions.common.path_utils import get_dataset_path
from extensions.common.csv_schema import create_csv_row
from extensions.common.dataset_loader import load_dataset_for_training

# ✅ CORRECT - Import from core/base classes
from core.game_manager import BaseGameManager
from core.game_logic import BaseGameLogic

# ✅ CORRECT - Import from config (universal constants)
from config.game_constants import VALID_MOVES, DIRECTIONS
from config.ui_constants import COLORS, GRID_SIZE
```

## 🎯 **Why These Patterns Are Forbidden**

### **Standalone Principle**
Each extension `{algorithm}-v0.0N` plus the common folder must form a completely standalone unit. Direct imports between extensions violate this principle.

### **Conceptual Clarity**
Extensions represent distinct AI approaches (heuristics, supervised learning, reinforcement learning, etc.). Cross-extension imports blur these conceptual boundaries.

### **Maintainability**
Direct dependencies between extensions create tight coupling, making the system harder to maintain and evolve independently.

## 🔧 **How to Share Code Between Extensions**

### **Use the Common Folder**
Place shared utilities in `extensions/common/`:
```python
# ✅ CORRECT - Shared via common folder
from extensions.common.validation import validate_dataset_format
from extensions.common.path_utils import get_extension_path
```

### **Use Universal Constants**
Import from `config/` for universal game rules:
```python
# ✅ CORRECT - Universal constants
from config.game_constants import VALID_MOVES, DIRECTIONS, MAX_STEPS_ALLOWED
from config.ui_constants import COLORS, GRID_SIZE, WINDOW_WIDTH
```

## 🚨 **Validation Requirements**

All extensions must validate import compliance:
```python
from extensions.common.validation import validate_import_patterns

def validate_extension_imports(imported_modules: List[str]):
    """Validate extension import pattern compliance"""
    validate_import_patterns(imported_modules)
```

## 🔗 **See Also**

- **`standalone.md`**: Complete standalone architecture documentation
- **`extensions/common/`**: Shared utilities for all extensions
- **`config/`**: Universal configuration constants

