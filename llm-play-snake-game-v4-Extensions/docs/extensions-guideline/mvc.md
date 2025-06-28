# MVC Architecture for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines MVC architecture patterns for extensions.

> **See also:** `core.md`, `final-decision-10.md`, `project-structure-plan.md`.

# MVC Architecture in Snake Game AI Project

This document explains how the Model-View-Controller (MVC) architectural pattern is implemented throughout the Snake Game AI project, providing separation of concerns and extensibility for all tasks (Task-0 through Task-5).

## 🎯 **MVC Architecture Overview**

The project implements a clean MVC architecture where:
- **Models** handle game state, data persistence, and business logic
- **Views** manage user interface rendering and user interaction
- **Controllers** coordinate between models and views, handling user input and orchestrating operations

### **Core MVC Benefits for Snake AI Project:**
- **Separation of Concerns**: Clear boundaries between data, presentation, and control logic
- **Extensibility**: Easy to add new game modes, UI types, and agent implementations
- **Testability**: Each component can be tested independently
- **Multiple Views**: Same game logic can support pygame, web, and headless modes
- **Agent Agnostic**: Works equally well for LLM, heuristic, RL, and supervised learning agents

## 🏗️ **Project MVC Structure**

### **Model Layer (`core/`, `extensions/`)**
Handles game state, business logic, and data operations:

```python
# Core Models
class GameData:
    """
    Model: Manages game state and statistics
    - Snake positions, apple location, score
    - Game history and move sequences
    - Performance metrics and timing
    """

class GameLogic:
    """
    Model: Encapsulates game rules and mechanics
    - Move validation and collision detection
    - Board state management and updates
    - Apple placement and scoring logic
    """

class GameManager:
    """
    Model: Orchestrates overall game lifecycle
    - Session management and game counting
    - Save/load operations and persistence
    - Agent coordination and move execution
    """
```

### **View Layer (`gui/`, `web/`, `dashboard/`)**
Manages presentation and user interaction:

```python
# GUI Views (pygame)
class BaseGUI:
    """View: Abstract base for all graphical interfaces"""
    
class GameGUI(BaseGUI):
    """View: Real-time game visualization with pygame"""
    
class ReplayGUI(BaseGUI):
    """View: Game replay with step-through controls"""

# Web Views (Flask + templates)
class WebViewRenderer:
    """View: Web-based game visualization and controls"""
    
class TemplateEngine:
    """View: HTML template rendering for web interface"""

# Dashboard Views (Streamlit)
class TrainingDashboard:
    """View: Interactive training and evaluation interface"""
```

### **Controller Layer (`core/`, `web/controllers/`, `scripts/`)**
Coordinates between models and views:

```python
# Core Controllers
class GameController:
    """
    Controller: Manages game state updates and agent interaction
    - Processes agent moves and updates game state
    - Coordinates between GameLogic and GameData
    - Handles game events and state transitions
    """

class GameRunner:
    """
    Controller: Orchestrates complete game sessions
    - Manages multiple games and statistics
    - Controls GUI updates and user interaction
    - Handles save/load operations and replay
    """

# Web Controllers
class BaseWebController:
    """Controller: Abstract base for web request handling"""
    
class GamePlayController(BaseWebController):
    """Controller: Handles live game web requests"""
    
class ReplayController(BaseWebController):
    """Controller: Manages replay navigation and controls"""
```

**The MVC architecture in this project provides a solid foundation for building extensible, maintainable, and testable AI game systems while supporting multiple agent types and interface modes.**
