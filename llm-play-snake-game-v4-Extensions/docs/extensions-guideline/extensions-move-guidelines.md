# Extensions Move Guidelines

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines extension move guidelines.

> **See also:** `final-decision-10.md`, `standalone.md`, `conceptual-clarity.md`.

## 🎯 **Core Philosophy: Standalone Independence**

Extensions must maintain **standalone independence** when moved or reorganized. Each extension should be self-contained with minimal dependencies, ensuring it can operate independently while following `final-decision-10.md` SUPREME_RULES.

### **Educational Value**
- **Modularity**: Understanding how to create independent modules
- **Dependency Management**: Learning to minimize external dependencies
- **Portability**: Creating code that can be easily moved and reused
- **Self-Containment**: Building systems that work independently

## 📦 **Extension Structure Requirements**

### **Standalone Directory Structure**
```
extensions/{algorithm}-v0.0N/
├── __init__.py                 # Extension initialization
├── agents/                     # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py          # Base agent class
│   ├── bfs_agent.py           # BFS implementation
│   ├── astar_agent.py         # A* implementation
│   └── dfs_agent.py           # DFS implementation
├── app.py                      # Streamlit application (v0.03+)
├── dashboard/                  # UI components (v0.03+)
│   ├── __init__.py
│   ├── tab_main.py            # Main algorithm interface
│   ├── tab_evaluation.py      # Performance analysis
│   └── tab_visualization.py   # Results display
├── scripts/                    # CLI entry points
│   ├── main.py                # Core functionality
│   ├── generate_dataset.py    # Dataset generation
│   └── replay.py              # Replay functionality
├── config.py                   # Extension-specific configuration
└── requirements.txt            # Extension dependencies
```

### **Common Dependencies Only**
```python
# ✅ CORRECT: Only import from common utilities
from extensions.common.utils.factory_utils import SimpleFactory
from extensions.common.utils.path_utils import get_dataset_path
from extensions.common.utils.csv_schema_utils import create_csv_row

# ✅ CORRECT: Extension-specific imports
from .agents.bfs_agent import BFSAgent
from .config import DEFAULT_CONFIG

# ❌ INCORRECT: Cross-extension imports
from extensions.heuristics_v0.03.agents import BFSAgent  # Wrong
from extensions.supervised_v0.02.models import MLPModel  # Wrong
```

## 🔄 **Move Process Guidelines**

### **1. Pre-Move Checklist**
```python
# ✅ CORRECT: Validate extension before moving
def validate_extension_for_move(extension_path: Path) -> bool:
    """Validate extension is ready for move."""
    
    # Check for cross-extension imports
    cross_imports = find_cross_extension_imports(extension_path)
    if cross_imports:
        print(f"[MoveValidation] ERROR: Found cross-extension imports: {cross_imports}")  # Simple logging
        return False
    
    # Check for missing common dependencies
    missing_deps = check_common_dependencies(extension_path)
    if missing_deps:
        print(f"[MoveValidation] ERROR: Missing common dependencies: {missing_deps}")  # Simple logging
        return False
    
    # Check for canonical patterns
    if not validate_canonical_patterns(extension_path):
        print(f"[MoveValidation] ERROR: Missing canonical patterns")  # Simple logging
        return False
    
    print(f"[MoveValidation] Extension ready for move: {extension_path}")  # Simple logging
    return True
```

### **2. Dependency Analysis**
```python
# ✅ CORRECT: Analyze dependencies before moving
def analyze_extension_dependencies(extension_path: Path) -> dict:
    """Analyze extension dependencies."""
    
    dependencies = {
        'common_utils': [],
        'extension_specific': [],
        'external_packages': [],
        'cross_extension': []  # Should be empty
    }
    
    # Scan all Python files in extension
    for py_file in extension_path.rglob("*.py"):
        imports = extract_imports(py_file)
        
        for import_stmt in imports:
            if import_stmt.startswith('extensions.common'):
                dependencies['common_utils'].append(import_stmt)
            elif import_stmt.startswith('extensions.'):
                dependencies['cross_extension'].append(import_stmt)
            elif import_stmt.startswith('.'):
                dependencies['extension_specific'].append(import_stmt)
            else:
                dependencies['external_packages'].append(import_stmt)
    
    print(f"[DependencyAnalysis] Found {len(dependencies['cross_extension'])} cross-extension imports")  # Simple logging
    return dependencies
```

### **3. Move Execution**
```python
# ✅ CORRECT: Execute extension move
def move_extension(source_path: Path, target_path: Path) -> bool:
    """Move extension to new location."""
    
    print(f"[ExtensionMove] Moving extension from {source_path} to {target_path}")  # Simple logging
    
    # Validate source extension
    if not validate_extension_for_move(source_path):
        return False
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Copy extension files
    copy_extension_files(source_path, target_path)
    
    # Update internal paths if needed
    update_internal_paths(target_path)
    
    # Validate moved extension
    if not validate_extension_for_move(target_path):
        print(f"[ExtensionMove] ERROR: Moved extension validation failed")  # Simple logging
        return False
    
    print(f"[ExtensionMove] Extension moved successfully to {target_path}")  # Simple logging
    return True
```

## 🏗️ **Canonical Pattern Requirements**

### **Factory Pattern Compliance**
```python
# ✅ CORRECT: Canonical factory pattern in moved extension
class AgentFactory:
    """Factory for creating agents in moved extension."""
    
    _registry = {
        'BFS': BFSAgent,
        'ASTAR': AStarAgent,
        'DFS': DFSAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, **kwargs):  # CANONICAL create() method
        """Create agent using canonical factory pattern."""
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        print(f"[AgentFactory] Creating {algorithm} agent")  # Simple logging
        return agent_class(**kwargs)

# ❌ INCORRECT: Non-canonical factory pattern
class AgentFactory:
    def create_agent(self, algorithm: str, **kwargs):  # Wrong method name
        pass
```

### **Simple Logging Compliance**
```python
# ✅ CORRECT: Simple logging in moved extension
class GameManager:
    def __init__(self, config: dict):
        self.config = config
        print(f"[GameManager] Initialized with config: {config}")  # Simple logging
    
    def start_game(self):
        print(f"[GameManager] Starting game")  # Simple logging
        # Game logic here
        print(f"[GameManager] Game completed")  # Simple logging

# ❌ INCORRECT: Complex logging in moved extension
import logging
logger = logging.getLogger(__name__)
logger.info("Starting game")  # Violates SUPREME_RULES
```

## 📋 **Move Validation Checklist**

### **Pre-Move Validation**
- [ ] **Cross-Extension Imports**: No imports from other extensions
- [ ] **Common Dependencies**: Only imports from extensions/common
- [ ] **Canonical Patterns**: Factory methods use `create()` name
- [ ] **Simple Logging**: Uses print() statements only
- [ ] **Self-Contained**: All required files present

### **Post-Move Validation**
- [ ] **Path Updates**: Internal paths updated correctly
- [ ] **Import Resolution**: All imports resolve correctly
- [ ] **Functionality**: Extension works in new location
- [ ] **Pattern Compliance**: Canonical patterns maintained
- [ ] **Logging Compliance**: Simple logging maintained

### **Dependency Management**
- [ ] **Common Utils**: Only depends on extensions/common
- [ ] **External Packages**: Minimal external dependencies
- [ ] **Extension Specific**: Self-contained extension code
- [ ] **No Cross-Dependencies**: No dependencies on other extensions

## 🎯 **Best Practices**

### **1. Self-Contained Design**
```python
# ✅ CORRECT: Self-contained extension design
class ExtensionConfig:
    """Extension-specific configuration."""
    
    def __init__(self):
        # Extension-specific settings
        self.algorithm = "BFS"
        self.grid_size = 10
        self.max_games = 100
        
        print(f"[ExtensionConfig] Initialized with algorithm: {self.algorithm}")  # Simple logging
    
    def validate(self) -> bool:
        """Validate configuration."""
        if self.grid_size < 5:
            print(f"[ExtensionConfig] ERROR: Grid size too small: {self.grid_size}")  # Simple logging
            return False
        return True
```

### **2. Minimal Dependencies**
```python
# ✅ CORRECT: Minimal, focused dependencies
# Only import what's needed from common utilities
from extensions.common.utils.factory_utils import SimpleFactory
from extensions.common.utils.path_utils import get_dataset_path

# Extension-specific imports
from .agents.bfs_agent import BFSAgent
from .config import ExtensionConfig

# ❌ INCORRECT: Excessive dependencies
from extensions.common.utils import *  # Import everything
from extensions.heuristics_v0.03 import *  # Cross-extension import
```

### **3. Clear Boundaries**
```python
# ✅ CORRECT: Clear extension boundaries
class ExtensionManager:
    """Manages extension-specific operations."""
    
    def __init__(self, extension_type: str, version: str):
        self.extension_type = extension_type
        self.version = version
        self.factory = SimpleFactory()
        
        print(f"[ExtensionManager] Initialized {extension_type} v{version}")  # Simple logging
    
    def create_agent(self, algorithm: str) -> BaseAgent:
        """Create agent using canonical factory pattern."""
        return self.factory.create(algorithm)  # CANONICAL create() method
```

## 🎓 **Educational Benefits**

### **Learning Objectives**
- **Modularity**: Understanding how to create independent modules
- **Dependency Management**: Learning to minimize external dependencies
- **Portability**: Creating code that can be easily moved and reused
- **Self-Containment**: Building systems that work independently

### **Best Practices**
- **Minimal Dependencies**: Only depend on what's absolutely necessary
- **Clear Boundaries**: Define clear interfaces and boundaries
- **Self-Containment**: Make extensions work independently
- **Canonical Patterns**: Maintain consistent patterns across moves

---

**Extension move guidelines ensure that extensions remain standalone, portable, and maintainable while preserving canonical patterns and educational value.**

## 🔗 **See Also**

- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`standalone.md`**: Standalone principle and extension independence
- **`conceptual-clarity.md`**: Conceptual clarity guidelines for extensions
