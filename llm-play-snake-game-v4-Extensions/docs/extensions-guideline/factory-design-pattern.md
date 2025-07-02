# Factory Design Pattern Implementation

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines factory design pattern implementation with strict SUPREME_RULES compliance.

> **See also:** `agents.md`, `core.md`, `standalone.md`, `final-decision-10.md`.

## 🎯 **Core Philosophy: Canonical Factory Patterns + SUPREME_RULES**

The Factory Design Pattern in the Snake Game AI project follows **SUPREME_RULES** established in SUPREME_RULES from `final-decision-10.md`, ensuring:
- **Canonical `create()` method** for all factories (never `create_agent()`, `create_model()`, etc.)
- **Simple logging** (print statements only, no complex logging frameworks)
- **Lightweight, OOP-based, extensible, non-over-engineered** design
- **Educational value** through clear examples and explanations

### **Educational Value**
- **Consistent Patterns**: All factories use identical `create()` method
- **Simple Logging**: All components use print() statements only
- **OOP Principles**: Demonstrates inheritance, polymorphism, and encapsulation
- **Extensibility**: Easy to add new types without modifying existing code

## 🏗️ **SUPREME_RULES: Canonical Method is create()**

**CRITICAL REQUIREMENT**: All factory classes MUST use the canonical method name `create()` for instantiation, not `create_agent()`, `create_model()`, `make_agent()`, or any other variant. This ensures consistency and aligns with the KISS principle.

### **Reference Implementation**

A generic, educational `SimpleFactory` is provided in `utils/factory_utils.py`:

```python
class SimpleFactory:
    """
    Generic factory implementation following final-decision-10.md SUPREME_RULES
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for all factory types
    Educational Value: Shows how SUPREME_RULES enable consistent patterns
    across simple heuristics and complex AI systems.
    
    Reference: `final-decision-10.md` SUPREME_RULES for canonical method naming
    """
    
    def __init__(self):
        self._registry = {}
        print(f"[SimpleFactory] Initialized")  # SUPREME_RULES compliant logging
    
    def register(self, name: str, cls):
        """Register a class with the factory"""
        self._registry[name] = cls
        print(f"[SimpleFactory] Registered: {name}")  # SUPREME_RULES compliant logging
    
    def create(self, name: str, **kwargs):  # CANONICAL create() method - SUPREME_RULES
        """Create instance using canonical create() method following `final-decision-10.md`"""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown type: {name}. Available: {available}")
        print(f"[SimpleFactory] Creating: {name}")  # SUPREME_RULES compliant logging
        return self._registry[name](**kwargs)

# ❌ FORBIDDEN: Non-canonical method names (violates SUPREME_RULES)
# Only use create() as the canonical factory method name per SUPREME_RULES from `final-decision-10.md`.
def create_agent(self, name: str):  # FORBIDDEN - not canonical
    pass
    
    def build_model(self, name: str):  # FORBIDDEN - not canonical
        pass
    
    def make_instance(self, name: str):  # FORBIDDEN - not canonical
        pass
```

### **Usage Example**
```python
from utils.factory_utils import SimpleFactory

class MyAgent:
    def __init__(self, name):
        self.name = name

# Create factory and register types
factory = SimpleFactory()
factory.register("myagent", MyAgent)

# Use canonical create() method
agent = factory.create("myagent", name="TestAgent")  # CANONICAL create() method
print(f"Created agent: {agent.name}")  # SUPREME_RULES compliant logging
```

## 🔧 **Factory Pattern Implementation Examples**

### **Agent Factory (Canonical Implementation)**

# TODO

## 🎓 **Educational Applications with Canonical Patterns**

### **Factory Pattern Benefits**
- **Loose Coupling**: Client code doesn't need to know concrete classes
- **Easy Extension**: Add new types without modifying existing code
- **Centralized Creation**: All object creation logic in one place
- **Consistent Interface**: Same `create()` method across all factories

### **Pattern Consistency**
- **Canonical Method**: All factories use `create()` method consistently
- **Simple Logging**: Print statements provide clear operation visibility
- **Educational Value**: Canonical patterns enable predictable learning
- **SUPREME_RULES**: Advanced systems follow same standards as simple ones

## 📋 **SUPREME_RULES Implementation Checklist for Factory Patterns**

### **Mandatory Requirements**
- [ ] **Canonical Method**: All factories use `create()` method exactly (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses print() statements only for all factory operations (final-decision-10.md compliance)
- [ ] **GOOD_RULES Reference**: References `final-decision-10.md` in all factory documentation
- [ ] **Pattern Consistency**: Follows canonical patterns across all factory implementations

### **Factory-Specific Standards**
- [ ] **Registration**: Canonical factory patterns for all type registration
- [ ] **Creation**: Canonical factory patterns for all instance creation
- [ ] **Error Handling**: Canonical patterns for all error conditions
- [ ] **Validation**: Simple logging for all validation operations

### **Educational Integration**
- [ ] **Clear Examples**: Simple examples using canonical `create()` method
- [ ] **Pattern Documentation**: Clear explanation of factory pattern benefits
- [ ] **SUPREME_RULES Compliance**: All examples follow `final-decision-10.md` standards
- [ ] **Cross-Reference**: Links to related patterns and principles

## 🔗 **Cross-References and Integration**

### **Related Documents**
- **`final-decision-10.md`**: SUPREME_RULES for canonical factory patterns
- **`agents.md`**: Agent factory implementations and patterns
- **`core.md`**: Core architecture and factory integration
- **`standalone.md`**: Standalone extension principles with factory patterns

### **Implementation Files**
- **`utils/factory_utils.py`**: Canonical factory utilities
- **`extensions/common/utils/path_utils.py`**: Path management with factory patterns
- **`extensions/common/utils/csv_schema_utils.py`**: Schema utilities with factory patterns

### **Educational Resources**
- **Design Patterns**: Factory pattern as foundation for all object creation
- **SUPREME_RULES**: Canonical patterns ensure consistency across all extensions
- **Simple Logging**: Print statements provide clear operation visibility
- **OOP Principles**: Factory pattern demonstrates encapsulation and polymorphism

---

**Factory Design Pattern implementation ensures consistent, educational, and maintainable object creation across all Snake Game AI extensions while maintaining strict compliance with `final-decision-10.md` SUPREME_RULES.**

## 🔗 **See Also**

- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`agents.md`**: Authoritative reference for agent implementation with canonical patterns
- **`core.md`**: Base class architecture following canonical principles
- **`standalone.md`**: Standalone principle and extension independence