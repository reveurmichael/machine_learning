# Single Source of Truth Principle

## 🎯 **Core Philosophy: One Truth, One Place**

### **Guidelines Alignment**
- **final-decision.md Guideline 1**: Enforces reading all GOOD_RULES before making architectural changes to ensure SSOT compliance
- **final-decision.md Guideline 2**: Uses precise `final-decision-N.md` format consistently throughout architectural references
- **simple logging**: Promotes lightweight common utilities with simple logging using only the print functions from `ROOT/utils/print_utils.py` (such as `print_info`, `print_warning`, `print_success`, `print_error`, `print_important`) to maintain SSOT for logging patterns. Never use raw print().

The Single Source of Truth (SSOT) principle ensures that every piece of information has exactly one authoritative location. This eliminates contradictions, reduces maintenance burden, and ensures consistency across the entire Snake Game AI ecosystem.

## 🏗️ **SSOT Architecture**

### **Configuration Hierarchy**
> For an expanded rationale and hierarchy diagram see **`config.md`** and **``**.

| Level | Location | Purpose | Example |
|-------|----------|---------|---------|
| **Universal** | `config/` | Core game rules, UI, coordinate system | `VALID_MOVES`, `DIRECTIONS` |
| **Shared Extension** | `extensions/common/config/` | Cross-extension settings |  ???? |
| **Experiment** | Local to script | Runtime parameters | CLI flags, YAML configs |

### **Path Management SSOT**
All non-trivial path logic is consolidated in **`unified-path-management-guide.md`** and its implementation `extensions/common/path_utils.py`.  *Do not* copy `ensure_project_root()` or sibling helpers into extension folders – just import them.

### **Data Format SSOT**
All data format decisions are centralized in **`data-format-decision-guide.md`**. This includes:
- When to use CSV vs NPZ vs JSONL
- Feature engineering standards
- Storage structure requirements
- Format validation rules

## 🔧 **SSOT Implementation**

### **Configuration Access**
```python
# ✅ CORRECT: Single source for each type
from config.game_constants import VALID_MOVES          # Universal

# ❌ WRONG: Duplicate definitions
VALID_MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]  # Don't redefine
```

### **Path Management**
```python
# ✅ CORRECT: Use centralized utilities
from extensions.common.path_utils import ensure_project_root, get_dataset_path

# ❌ WRONG: Manual path construction
import os
dataset_path = os.path.join(os.getcwd(), "logs", "datasets")  # Don't construct manually
```

### **Data Format Decisions**
```python
# ✅ CORRECT: Follow authoritative guide
from extensions.common.csv_schema import create_csv_row  # For tree models
from extensions.common.npz_utils import create_sequential_dataset  # For RNNs

# ❌ WRONG: Ad-hoc format decisions
if model_type == "xgboost":
    format = "csv"  # Don't decide locally
```

## 🚫 **SSOT Violations to Avoid**

### **Configuration Duplication**
```python
# ❌ WRONG: Multiple definitions
# file1.py
MAX_STEPS = 1000

# file2.py  
MAX_STEPS = 1000  # Duplicate!

# ✅ CORRECT: Single definition
# config/game_constants.py
MAX_STEPS = 1000

# file1.py and file2.py
from config.game_constants import MAX_STEPS
```

### **Path Logic Scattering**
```python
# ❌ WRONG: Path logic everywhere
# file1.py
def get_log_path():
    return os.path.join(os.getcwd(), "logs")

# file2.py
def get_log_path():
    return Path.cwd() / "logs"  # Different implementation!

# ✅ CORRECT: Centralized path utilities
from extensions.common.path_utils import get_log_path
```

### **Format Decision Fragmentation**
```python
# ❌ WRONG: Format decisions scattered
# file1.py
if algorithm == "bfs":
    save_as_csv(data)

# file2.py
if model_type == "xgboost":
    save_as_csv(data)  # Duplicate logic!

# ✅ CORRECT: Centralized format decisions
from extensions.common.data_formats import save_dataset
save_dataset(data, algorithm, model_type)  # Centralized decision
```

## 🎯 **SSOT Benefits**

### **Consistency**
- **Uniform Behavior**: Same configuration, same behavior across extensions
- **Predictable Paths**: Standardized path resolution everywhere
- **Consistent Formats**: Same data format decisions across all extensions

### **Maintainability**
- **Single Update Point**: Change once, affects everywhere
- **Reduced Bugs**: No synchronization issues between copies
- **Clear Ownership**: Each piece of information has one owner

### **Educational Value**
- **Clear Learning Path**: One place to learn each concept
- **Reduced Confusion**: No contradictory information
- **Focused Documentation**: Each document has one clear purpose

### **Manual Review Checklist**
- [ ] No duplicate constant definitions
- [ ] All paths use centralized utilities
- [ ] All format decisions follow authoritative guide
- [ ] No local redefinitions of standard patterns
- [ ] Clear documentation of SSOT locations

---

**The Single Source of Truth principle ensures that the Snake Game AI project remains consistent, maintainable, and educational by eliminating contradictions and centralizing authoritative information.**









