# Extensions Move Guidelines

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines extension move guidelines.

> **See also:** `final-decision-10.md`, `standalone.md`, `conceptual-clarity.md`.

## �� **Core Philosophy: Clean Extension Transitions**

Extension moves involve transferring code between different extension versions while maintaining **clean architecture, educational value, and SUPREME_RULES compliance**. This process ensures that extensions evolve systematically without breaking existing functionality or creating inconsistencies.

### **Educational Value**
- **Clean Transitions**: Learn how to move code between versions systematically
- **Architecture Preservation**: Maintain clean architecture during moves
- **Dependency Management**: Understand proper dependency handling
- **Version Control**: Learn systematic version management

## 🏗️ **Factory Pattern: Canonical Method is create()**

All factories must use the canonical method name `create()` for instantiation, not `create_agent()` or any other variant. This ensures consistency and aligns with the KISS principle.

### **Extension Move Factory**
```python
class ExtensionMoveFactory:
    """
    Factory for managing extension moves following SUPREME_RULES.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Manages systematic extension transitions
    Educational Value: Shows how canonical patterns work in extension management
    """
    
    _registry = {
        "HEURISTICS": HeuristicsMoveManager,
        "SUPERVISED": SupervisedMoveManager,
        "REINFORCEMENT": ReinforcementMoveManager,
    }
    
    @classmethod
    def create(cls, extension_type: str, **kwargs):  # CANONICAL create() method
        """Create move manager using canonical create() method (SUPREME_RULES compliance)"""
        manager_class = cls._registry.get(extension_type.upper())
        if not manager_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown extension type: {extension_type}. Available: {available}")
        print(f"[ExtensionMoveFactory] Creating move manager: {extension_type}")  # Simple logging
        return manager_class(**kwargs)
```

## 📁 **Extension Move Process**

### **Standard Directory Structure**
```
extensions/{algorithm}-v0.0N/
├── app.py                         # Main application entry point
├── dashboard/                     # Dashboard components (v0.03+)
│   ├── __init__.py
│   ├── components/                # Reusable UI components
│   ├── pages/                     # Multi-page application structure
│   └── utils/                     # Dashboard-specific utilities
├── scripts/                       # Backend execution scripts
│   ├── main.py                    # Core algorithm execution
│   ├── generate_dataset.py        # Dataset generation
│   └── replay.py                  # Replay functionality
└── agents/                        # Algorithm implementations
    └── [agent files]
```

### **Move Process Steps**
```python
class ExtensionMoveManager:
    """
    Manages systematic extension moves following SUPREME_RULES.
    
    Design Pattern: Template Method Pattern
    Purpose: Provides consistent move process across all extensions
    Educational Value: Shows systematic approach to code transitions
    """
    
    def __init__(self, source_version: str, target_version: str):
        self.source_version = source_version
        self.target_version = target_version
        print(f"[ExtensionMoveManager] Initialized move: {source_version} → {target_version}")  # Simple logging
    
    def execute_move(self):
        """Execute systematic extension move"""
        print(f"[ExtensionMoveManager] Starting move process")  # Simple logging
        
        # Step 1: Validate source extension
        self._validate_source()
        
        # Step 2: Create target directory structure
        self._create_target_structure()
        
        # Step 3: Copy and adapt code
        self._copy_and_adapt_code()
        
        # Step 4: Update dependencies
        self._update_dependencies()
        
        # Step 5: Validate target extension
        self._validate_target()
        
        print(f"[ExtensionMoveManager] Move completed successfully")  # Simple logging
    
    def _validate_source(self):
        """Validate source extension structure"""
        # Validation logic here
        print(f"[ExtensionMoveManager] Source validation completed")  # Simple logging
    
    def _create_target_structure(self):
        """Create target directory structure"""
        # Structure creation logic here
        print(f"[ExtensionMoveManager] Target structure created")  # Simple logging
    
    def _copy_and_adapt_code(self):
        """Copy and adapt code for target version"""
        # Code adaptation logic here
        print(f"[ExtensionMoveManager] Code copied and adapted")  # Simple logging
    
    def _update_dependencies(self):
        """Update dependencies for target version"""
        # Dependency update logic here
        print(f"[ExtensionMoveManager] Dependencies updated")  # Simple logging
    
    def _validate_target(self):
        """Validate target extension"""
        # Target validation logic here
        print(f"[ExtensionMoveManager] Target validation completed")  # Simple logging
```

## 🔧 **Code Adaptation Patterns**

### **Import Statement Updates**
```python
# ✅ CORRECT: Use common utilities (SUPREME_RULES compliance)
from extensions.common.utils.csv_schema_utils import create_csv_row

# ❌ FORBIDDEN: Cross-extension imports (violates SUPREME_RULES)
# from extensions.heuristics_v0.03.agents import BFSAgent  # Wrong
# from extensions.heuristics_v0.03 import *  # Cross-extension import
```

### **Configuration Updates**
```python
class MoveConfiguration:
    """Configuration for extension moves"""
    
    def __init__(self, source_version: str, target_version: str):
        self.source_version = source_version
        self.target_version = target_version
        self.adaptations = self._get_adaptations()
        print(f"[MoveConfiguration] Configured for {source_version} → {target_version}")  # Simple logging
    
    def _get_adaptations(self) -> dict:
        """Get required adaptations for move"""
        adaptations = {
            'imports': self._get_import_adaptations(),
            'dependencies': self._get_dependency_adaptations(),
            'structure': self._get_structure_adaptations()
        }
        return adaptations
    
    def _get_import_adaptations(self) -> list:
        """Get import statement adaptations"""
        return [
            # Update import paths to use common utilities
            ('from extensions.heuristics_v0.03.agents import BFSAgent', 
             'from extensions.common.utils.factory_utils import SimpleFactory'),
            # Add new imports for target version
            ('', 'from extensions.common.utils.csv_schema_utils import create_csv_row')
        ]
    
    def _get_dependency_adaptations(self) -> list:
        """Get dependency adaptations"""
        return [
            # Update dependencies for target version
            ('streamlit==1.28.0', 'streamlit==1.29.0'),
            # Add new dependencies
            ('', 'pandas>=2.0.0')
        ]
    
    def _get_structure_adaptations(self) -> list:
        """Get directory structure adaptations"""
        return [
            # Add new directories for target version
            ('', 'dashboard/'),
            ('', 'dashboard/components/'),
            ('', 'dashboard/pages/'),
            ('', 'dashboard/utils/')
        ]
```

## 📊 **Move Validation Standards**

### **Source Validation**
```python
def validate_source_extension(extension_path: Path) -> bool:
    """Validate source extension before move"""
    required_files = [
        'app.py',
        'scripts/main.py',
        'agents/__init__.py'
    ]
    
    for file_path in required_files:
        if not (extension_path / file_path).exists():
            print(f"[Validation] Missing required file: {file_path}")  # Simple logging
            return False
    
    print(f"[Validation] Source extension validated")  # Simple logging
    return True
```

### **Target Validation**
```python
def validate_target_extension(extension_path: Path) -> bool:
    """Validate target extension after move"""
    # Check directory structure
    required_dirs = [
        'dashboard',
        'scripts',
        'agents'
    ]
    
    for dir_path in required_dirs:
        if not (extension_path / dir_path).exists():
            print(f"[Validation] Missing required directory: {dir_path}")  # Simple logging
            return False
    
    # Check for forbidden patterns
    forbidden_patterns = [
        'from extensions.heuristics_v0.03',
        'import logging',
        'logger = logging.getLogger'
    ]
    
    for pattern in forbidden_patterns:
        if contains_pattern(extension_path, pattern):
            print(f"[Validation] Found forbidden pattern: {pattern}")  # Simple logging
            return False
    
    print(f"[Validation] Target extension validated")  # Simple logging
    return True
```

## 🎯 **Move Automation**

### **Automated Move Script**
```python
def automate_extension_move(source_version: str, target_version: str):
    """Automate extension move process"""
    print(f"[Automation] Starting automated move: {source_version} → {target_version}")  # Simple logging
    
    # Create move manager
    move_manager = ExtensionMoveFactory.create("HEURISTICS", 
                                              source_version=source_version,
                                              target_version=target_version)
    
    # Execute move
    move_manager.execute_move()
    
    # Validate result
    if validate_target_extension(Path(f"extensions/heuristics-{target_version}")):
        print(f"[Automation] Move completed successfully")  # Simple logging
    else:
        print(f"[Automation] Move validation failed")  # Simple logging
```

## 📋 **Implementation Checklist**

### **Pre-Move Requirements**
- [ ] **Source Validation**: Validate source extension structure
- [ ] **Dependency Analysis**: Identify required dependencies
- [ ] **Import Analysis**: Identify import statement changes
- [ ] **Structure Planning**: Plan target directory structure

### **Move Execution**
- [ ] **Directory Creation**: Create target directory structure
- [ ] **Code Copying**: Copy code with adaptations
- [ ] **Import Updates**: Update import statements
- [ ] **Dependency Updates**: Update dependencies
- [ ] **Configuration Updates**: Update configuration files

### **Post-Move Validation**
- [ ] **Structure Validation**: Validate target directory structure
- [ ] **Import Validation**: Check for forbidden imports
- [ ] **Dependency Validation**: Verify dependency compatibility
- [ ] **Functionality Testing**: Test basic functionality

## 🎓 **Educational Benefits**

### **Learning Objectives**
- **Systematic Transitions**: Learn systematic approach to code moves
- **Architecture Preservation**: Understand how to maintain clean architecture
- **Dependency Management**: Learn proper dependency handling
- **Validation Processes**: Understand validation and testing

### **Best Practices**
- **Automated Processes**: Use automation for consistency
- **Validation**: Always validate before and after moves
- **Documentation**: Document move processes and changes
- **Testing**: Test functionality after moves

---

**Extension move guidelines ensure clean, systematic transitions between extension versions while maintaining educational value and SUPREME_RULES compliance.**

## 🔗 **See Also**

- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`standalone.md`**: Standalone principle and extension independence
- **`conceptual-clarity.md`**: Conceptual clarity guidelines for extensions