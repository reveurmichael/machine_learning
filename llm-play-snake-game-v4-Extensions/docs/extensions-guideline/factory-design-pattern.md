# Factory Design Pattern Implementation

## 🎯 **Core Philosophy: Canonical Factory Patterns**

The Factory Design Pattern in the Snake Game AI project follows **SUPREME_RULES** established in SUPREME_RULES from `final-decision.md`, ensuring:
- **Canonical `create()` method** for all factories (never `create_agent()`, `create_model()`, etc.)
- **Simple logging** (use only the simple print functions from `ROOT/utils/print_utils.py` such as `print_info`, `print_warning`, `print_error`, `print_success`, `print_important` — never use raw `print()` or any complex logging frameworks)
- **Lightweight, OOP-based, extensible, non-over-engineered** design
- **Educational value** through clear examples and explanations

### **Educational Value**
- **Consistent Patterns**: All factories use identical `create()` method
- **Simple Logging**: Use only the print functions from `ROOT/utils/print_utils.py` (e.g., `print_info`, `print_warning`, etc.) for all operation visibility. Never use raw `print()`.
- **OOP Principles**: Demonstrates inheritance, polymorphism, and encapsulation
- **Extensibility**: Easy to add new types without modifying existing code

## 🏗️ **SUPREME_RULES: Canonical Method is create()**

**CRITICAL REQUIREMENT**: All factory classes MUST use the canonical method name `create()` for instantiation, not `create_agent()`, `create_model()`, `make_agent()`, or any other variant. This ensures consistency and aligns with the KISS principle.

### **Reference Implementation**

A generic, educational `SimpleFactory` is provided in `utils/factory_utils.py`:

```python
class SimpleFactory:
    """
    Generic factory implementation following final-decision.md SUPREME_RULES
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for all factory types
    Educational Value: Shows how SUPREME_RULES enable consistent patterns
    across simple heuristics and complex AI systems.
    
    Reference: `final-decision.md` SUPREME_RULES for canonical method naming
    """
    
    def __init__(self):
        self._registry = {}
        print_info(f"[SimpleFactory] Initialized")  # SUPREME_RULES compliant logging
    
    def register(self, name: str, cls):
        """Register a class with the factory"""
        self._registry[name] = cls
        print_info(f"[SimpleFactory] Registered: {name}")  # SUPREME_RULES compliant logging
    
    def create(self, name: str, **kwargs):  # CANONICAL create() method - SUPREME_RULES
        """Create instance using canonical create() method following `final-decision.md`"""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown type: {name}. Available: {available}")
        print_info(f"[SimpleFactory] Creating: {name}")  # SUPREME_RULES compliant logging
        return self._registry[name](**kwargs)

# ❌ FORBIDDEN: Non-canonical method names (violates SUPREME_RULES)
# Only use create() as the canonical factory method name per SUPREME_RULES from `final-decision.md`.
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
print_info(f"Created agent: {agent.name}")  # SUPREME_RULES compliant logging
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
- **Simple Logging**: Use only the print functions from `ROOT/utils/print_utils.py` (e.g., `print_info`, `print_warning`, etc.) for all operation visibility. Never use raw `print()`.
- **Educational Value**: Canonical patterns enable predictable learning
- **SUPREME_RULES**: Advanced systems follow same standards as simple ones
