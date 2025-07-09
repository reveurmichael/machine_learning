# Forbidden Import Patterns for Extensions

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`` → `final-decision.md`) and defines forbidden import patterns that violate the standalone principle.

> **See also:** `standalone.md`, `extensions-move-guidelines.md`, `final-decision.md`.

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
