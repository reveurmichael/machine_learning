# Extension Evolution Rules

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines extension evolution rules with strict SUPREME_RULES compliance.

> **See also:** `final-decision-10.md`, `factory-design-pattern.md`, `standalone.md`, `project-structure-plan.md`.

## 🎯 **Core Philosophy: SUPREME_RULES Compliance**

Extension evolution follows the **SUPREME_RULES** established in `final-decision-10.md`, ensuring that all extensions maintain:
- **Canonical `create()` method** for all factories
- **Simple logging** (print statements only, no complex logging frameworks)
- **Lightweight, OOP-based, extensible, non-over-engineered** design
- **Standalone principle** for extensions
- **Single source of truth** compliance

### **Educational Value**
- **Consistent Evolution**: All extensions follow identical evolution patterns
- **Canonical Patterns**: Factory methods use `create()` method consistently
- **Simple Logging**: All components use print() statements only
- **Architectural Integrity**: Maintains project-wide consistency

## 🏗️ **Factory Pattern: Canonical Method is create()**

All extension factories MUST use the canonical method name `create()` for instantiation, not `create_agent()` or any other variant. This ensures consistency and aligns with the KISS principle.

### **Reference Implementation**

A generic, educational `SimpleFactory` is provided in `extensions/common/utils/factory_utils.py`:

```python
from extensions.common.utils.factory_utils import SimpleFactory

class MyExtension:
    def __init__(self, name):
        self.name = name

factory = SimpleFactory()
factory.register("myextension", MyExtension)
extension = factory.create("myextension", name="TestExtension")  # CANONICAL create() method
print(f"[Factory] Created extension: {extension.name}")  # Simple logging
```

### **Example Extension Factory**
```python
class ExtensionFactory:
    _registry = {
        "HEURISTICS": HeuristicsExtension,
        "SUPERVISED": SupervisedExtension,
        "REINFORCEMENT": ReinforcementExtension,
    }
    @classmethod
    def create(cls, extension_type: str, **kwargs):  # CANONICAL create() method
        extension_class = cls._registry.get(extension_type.upper())
        if not extension_class:
            raise ValueError(f"Unknown extension type: {extension_type}")
        print(f"[ExtensionFactory] Creating extension: {extension_type}")  # Simple logging
        return extension_class(**kwargs)
```

## 📋 **Extension Evolution Standards**

### **Version Progression Rules**
1. **v0.01**: Basic implementation with canonical factory patterns
2. **v0.02**: Enhanced features while maintaining v0.01 compatibility
3. **v0.03**: Streamlit dashboard integration with canonical patterns
4. **v0.04**: Advanced capabilities (heuristics only) with canonical patterns

### **Mandatory Requirements for All Versions**
- [ ] **Canonical Factory**: All factories use `create()` method exactly
- [ ] **Simple Logging**: Uses print() statements only for all operations
- [ ] **SUPREME_RULES Reference**: References `final-decision-10.md` in documentation
- [ ] **Standalone Principle**: Extension + common folder = standalone unit
- [ ] **No Cross-Extension Dependencies**: Only share code via common folder

### **Quality Standards**
- **OOP Design**: Proper inheritance and composition patterns
- **Educational Value**: Clear examples and explanations
- **Extensibility**: Easy to extend without breaking existing functionality
- **Documentation**: Comprehensive docstrings and comments

## 🔧 **Implementation Guidelines**

### **Extension Structure**
```
extensions/{algorithm}-v0.0N/
├── __init__.py
├── agents/                    # Agent implementations
├── app.py                     # Streamlit app (v0.03+)
├── dashboard/                 # UI components (v0.03+)
├── scripts/                   # CLI entry points
└── config.py                  # Extension-specific configuration
```

### **Common Integration**
```python
# All extensions use common utilities
from extensions.common.utils.factory_utils import SimpleFactory
from extensions.common.utils.path_utils import get_extension_path

# Simple logging throughout
print(f"[{extension_name}] Initializing extension")  # Simple logging
```

## 📊 **Evolution Compliance Checklist**

### **v0.01 Requirements**
- [ ] Basic agent implementation with canonical factory
- [ ] Simple logging with print() statements only
- [ ] Standalone operation (extension + common)
- [ ] Reference to `final-decision-10.md`

### **v0.02 Requirements**
- [ ] All v0.01 requirements maintained
- [ ] Enhanced features with canonical patterns
- [ ] Backward compatibility with v0.01
- [ ] Additional agent types with canonical factory

### **v0.03 Requirements**
- [ ] All v0.02 requirements maintained
- [ ] Streamlit dashboard with canonical patterns
- [ ] Script integration with subprocess
- [ ] Multi-tab interface following dashboard standards

### **v0.04 Requirements (Heuristics Only)**
- [ ] All v0.03 requirements maintained
- [ ] JSONL generation capability
- [ ] Advanced data formats with canonical patterns
- [ ] Enhanced visualization and analysis tools

## 🎓 **Educational Integration**

### **Learning Progression**
- **v0.01**: Learn basic canonical patterns and simple logging
- **v0.02**: Understand extension evolution and backward compatibility
- **v0.03**: Master Streamlit integration with canonical patterns
- **v0.04**: Experience advanced capabilities with canonical patterns

### **Best Practices**
- **Consistency**: Same patterns across all extensions and versions
- **Simplicity**: Avoid over-engineering and complex logging
- **Documentation**: Clear explanations of canonical patterns
- **Examples**: Working code examples with canonical factory usage

---

**Extension evolution rules ensure consistent, educational, and maintainable development across all Snake Game AI extensions while maintaining strict compliance with `final-decision-10.md` SUPREME_RULES.**

## 🔗 **See Also**

- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`factory-design-pattern.md`**: Canonical factory implementation for all systems
- **`standalone.md`**: Standalone principle and extension independence
- **`project-structure-plan.md`**: Project structure and organization 