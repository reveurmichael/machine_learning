# MVC Architecture for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines MVC architecture patterns for extensions.

> **See also:** `final-decision-10.md`, `core.md`, `project-structure-plan.md`.

## 🎯 **MVC Architecture Overview**

The project implements a clean MVC architecture where:
- **Models** handle game state, data persistence, and business logic
- **Views** manage user interface rendering and user interaction
- **Controllers** coordinate between models and views, handling user input and orchestrating operations

### **Educational Value**
- **Separation of Concerns**: Clear boundaries between data, presentation, and control logic
- **Extensibility**: Easy to add new game modes, UI types, and agent implementations
- **Testability**: Each component can be tested independently
- **Canonical Patterns**: Demonstrates factory patterns and simple logging throughout

## 🏗️ **Project MVC Structure**

### **Model Layer (`core/`, `extensions/`)**
Handles game state, business logic, and data operations:

```python
# Core Models
class GameData:
    """
    Model: Manages game state and statistics
    
    Design Pattern: Model Pattern (MVC Architecture)
    Purpose: Encapsulates game state and data operations
    Educational Value: Shows how canonical patterns work with
    data management while maintaining simple logging.
    
    Reference: final-decision-10.md for simple logging standards
    """
    
    def __init__(self):
        self.game_state = {}
        print(f"[GameData] Initialized")  # Simple logging - SUPREME_RULES
    
    def update_state(self, new_state: dict):
        """Update game state with simple logging"""
        self.game_state.update(new_state)
        print(f"[GameData] State updated: {len(new_state)} items")  # Simple logging

class GameLogic:
    """
    Model: Encapsulates game rules and mechanics
    
    Design Pattern: Model Pattern (MVC Architecture)
    Purpose: Handles game rules and business logic
    Educational Value: Demonstrates canonical patterns for
    game logic while maintaining simple logging.
    
    Reference: final-decision-10.md for canonical patterns
    """
    
    def __init__(self):
        self.rules = {}
        print(f"[GameLogic] Initialized")  # Simple logging - SUPREME_RULES
    
    def validate_move(self, move: str) -> bool:
        """Validate game move with simple logging"""
        is_valid = move in ["UP", "DOWN", "LEFT", "RIGHT"]
        print(f"[GameLogic] Move validation: {move} -> {is_valid}")  # Simple logging
        return is_valid
```

### **View Layer (`gui/`, `web/`, `dashboard/`)**
Manages presentation and user interaction:

```python
# GUI Views (pygame)
class BaseGUI:
    """
    View: Abstract base for all graphical interfaces
    
    Design Pattern: Template Method Pattern (MVC Architecture)
    Purpose: Provides common interface for all GUI implementations
    Educational Value: Shows how canonical patterns work with
    view management while maintaining simple logging.
    
    Reference: final-decision-10.md for canonical patterns
    """
    
    def __init__(self):
        print(f"[BaseGUI] Initialized")  # Simple logging - SUPREME_RULES
    
    def render(self, game_state: dict):
        """Render game state (override in subclasses)"""
        print(f"[BaseGUI] Rendering game state")  # Simple logging
        raise NotImplementedError("Subclasses must implement render")

class GameGUI(BaseGUI):
    """View: Real-time game visualization with pygame"""
    
    def render(self, game_state: dict):
        """Render real-time game visualization"""
        print(f"[GameGUI] Rendering real-time game")  # Simple logging
        # Pygame rendering logic here
```

### **Controller Layer (`core/`, `web/controllers/`, `scripts/`)**
Coordinates between models and views:

```python
# Core Controllers
class GameController:
    """
    Controller: Manages game state updates and agent interaction
    
    Design Pattern: Controller Pattern (MVC Architecture)
    Purpose: Coordinates between models and views
    Educational Value: Shows how canonical patterns work with
    controller logic while maintaining simple logging.
    
    Reference: final-decision-10.md for canonical patterns
    """
    
    def __init__(self, game_data, game_logic, view):
        self.game_data = game_data
        self.game_logic = game_logic
        self.view = view
        print(f"[GameController] Initialized with MVC components")  # Simple logging
    
    def process_move(self, move: str):
        """Process agent move using MVC pattern"""
        print(f"[GameController] Processing move: {move}")  # Simple logging
        
        # Validate move using model
        if self.game_logic.validate_move(move):
            # Update state using model
            self.game_data.update_state({"last_move": move})
            # Update view
            self.view.render(self.game_data.game_state)
            print(f"[GameController] Move processed successfully")  # Simple logging
        else:
            print(f"[GameController] Invalid move rejected")  # Simple logging
```

## 🎓 **Educational Applications with Canonical Patterns**

### **MVC Pattern Benefits**
- **Separation of Concerns**: Clear boundaries between data, presentation, and control
- **Extensibility**: Easy to add new components without affecting others
- **Testability**: Each component can be tested independently
- **Canonical Patterns**: Factory patterns ensure consistent component creation

### **Pattern Consistency**
- **Canonical Method**: All MVC components use consistent patterns
- **Simple Logging**: Print statements provide clear operation visibility
- **Educational Value**: Canonical patterns enable predictable learning
- **SUPREME_RULES**: MVC systems follow same standards as other components

## 📋 **SUPREME_RULES Implementation Checklist for MVC Patterns**

### **Mandatory Requirements**
- [ ] **Canonical Method**: All MVC components use consistent patterns (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses print() statements only for all MVC operations (final-decision-10.md compliance)
- [ ] **GOOD_RULES Reference**: References `final-decision-10.md` in all MVC documentation
- [ ] **Pattern Consistency**: Follows canonical patterns across all MVC implementations

### **MVC-Specific Standards**
- [ ] **Model Layer**: Handles data and business logic with simple logging
- [ ] **View Layer**: Manages presentation with simple logging
- [ ] **Controller Layer**: Coordinates components with simple logging
- [ ] **Component Communication**: Clean interfaces between layers

### **Educational Integration**
- [ ] **Clear Examples**: Simple examples using canonical patterns
- [ ] **Pattern Documentation**: Clear explanation of MVC pattern benefits
- [ ] **SUPREME_RULES Compliance**: All examples follow final-decision-10.md standards
- [ ] **Cross-Reference**: Links to related patterns and principles

## 🔗 **Cross-References and Integration**

### **Related Documents**
- **`final-decision-10.md`**: SUPREME_RULES for canonical MVC patterns
- **`core.md`**: Core architecture and MVC integration
- **`project-structure-plan.md`**: Project structure standards

### **Implementation Files**
- **`extensions/common/utils/factory_utils.py`**: Canonical factory utilities
- **`extensions/common/utils/path_utils.py`**: Path management with factory patterns
- **`extensions/common/utils/csv_schema_utils.py`**: Schema utilities with factory patterns

### **Educational Resources**
- **Design Patterns**: MVC pattern as foundation for clean architecture
- **SUPREME_RULES**: Canonical patterns ensure consistency across all extensions
- **Simple Logging**: Print statements provide clear operation visibility
- **OOP Principles**: MVC pattern demonstrates effective separation of concerns
