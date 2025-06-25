# Web GUI Architecture for Extensions

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ and extension guidelines. Web GUI components follow the same architectural patterns established in the GOODFILES.

## 🎯 **Core Philosophy: Web as Universal Interface**

The web architecture demonstrates perfect base class design where generic controllers provide foundation services while extension-specific implementations add specialized functionality. This follows the established architectural patterns from the Final Decision series.

### **Design Philosophy**
- **Universal Base Classes**: Generic web controllers for all extensions
- **Strategy Pattern**: Pluggable state providers and rendering strategies
- **Factory Pattern**: Dynamic controller creation and registration
- **Educational Value**: Clear demonstration of web MVC architecture

## 🏗️ **Perfect Base Class Architecture**

### **BaseWebController: Universal Foundation**
Following Final Decision patterns for base class design:

```python
class BaseWebController:
    """Abstract controller with Template Method pattern"""
    
    def __init__(self, model_manager, view_renderer):
        # Generic attributes (NO Task-0 specific code)
        self.model_manager = model_manager
        self.view_renderer = view_renderer
        self.request_handlers = {}  # Strategy pattern
        
    # Template method defining request handling pipeline
    def handle_request(self, request_type, context):
        """Generic request processing workflow"""
        if request_type == "state":
            return self.handle_state_request(context)
        elif request_type == "control":
            return self.handle_control_request(context)
        
    # Extension points for subclasses
    def handle_control_request(self, context): pass  # Abstract
    def handle_state_request(self, context): pass    # Abstract
```

### **Generic State Management**
```python
class StateProvider:
    """Abstract interface for data sources"""
    # Can wrap any game engine (live, replay, simulation)
    # NO LLM-specific dependencies
    
class GameStateModel:
    """Generic state management using Observer pattern"""
    # Works with any StateProvider implementation
    # Uses Observer pattern for state change notifications
```

## 🔧 **Extension Integration Patterns**

### **Heuristics Web Controller**
```python
class HeuristicGameController(BaseWebController):
    """Heuristic-based game controller using A*, BFS, etc."""
    
    def __init__(self, model_manager, view_renderer, pathfinder):
        super().__init__(model_manager, view_renderer)
        self.pathfinder = pathfinder  # A*, BFS, Hamiltonian cycle
        self.algorithm_name = pathfinder.get_algorithm_name()
    
    def handle_control_request(self, context):
        """Handle move requests using pathfinding algorithms"""
        current_state = self.model_manager.get_current_state()
        next_move = self.pathfinder.find_next_move(current_state)
        return {
            "action": next_move,
            "algorithm": self.algorithm_name,
            "path_length": len(self.pathfinder.current_path)
        }
    
    def handle_state_request(self, context):
        """Return state with heuristic-specific data"""
        base_state = super().handle_state_request(context)
        base_state.update({
            "algorithm_info": self.pathfinder.get_stats(),
            "current_path": self.pathfinder.current_path
        })
        return base_state
```

### **Supervised Learning Web Controller**
```python
class SupervisedGameController(BaseWebController):
    """ML-based game controller using trained models"""
    
    def __init__(self, model_manager, view_renderer, ml_model):
        super().__init__(model_manager, view_renderer)
        self.ml_model = ml_model  # MLP, XGBoost, etc.
        self.model_name = ml_model.get_model_name()
    
    def handle_control_request(self, context):
        """Handle actions using ML model predictions"""
        current_state = self.model_manager.get_current_state()
        prediction, confidence = self.ml_model.predict(current_state)
        return {
            "action": prediction,
            "confidence": confidence,
            "model_type": self.model_name
        }
    
    def handle_state_request(self, context):
        """Return state with ML-specific metrics"""
        base_state = super().handle_state_request(context)
        base_state.update({
            "model_metrics": self.ml_model.get_performance_stats(),
            "prediction_history": self.ml_model.recent_predictions
        })
        return base_state
```

### **Reinforcement Learning Web Controller**
```python
class RLGameController(BaseWebController):
    """RL-based game controller using DQN, PPO, etc."""
    
    def __init__(self, model_manager, view_renderer, rl_agent):
        super().__init__(model_manager, view_renderer)
        self.rl_agent = rl_agent  # DQN, PPO, A3C
        self.training_mode = True
    
    def handle_control_request(self, context):
        """Handle actions using RL agent"""
        current_state = self.model_manager.get_current_state()
        action, q_values = self.rl_agent.select_action(current_state)
        return {
            "action": action,
            "q_values": q_values.tolist(),
            "epsilon": self.rl_agent.epsilon
        }
    
    def handle_state_request(self, context):
        """Return state with RL-specific metrics"""
        base_state = super().handle_state_request(context)
        base_state.update({
            "training_metrics": self.rl_agent.get_training_stats(),
            "episode_reward": self.rl_agent.current_episode_reward
        })
        return base_state
```

## 🏭 **Factory Pattern Integration**

### **Controller Registration System**
Following Final Decision 7-8 factory patterns:

```python
class ControllerFactory:
    """Factory for creating web controllers"""
    
    _controller_registry = {
        "llm": GamePlayController,        # Task-0
        "heuristic": HeuristicGameController,  # Task-1
        "supervised": SupervisedGameController,  # Task-2
        "rl": RLGameController,           # Task-3
    }
    
    @classmethod
    def register_controller_type(cls, name: str, controller_class):
        """Register new controller type"""
        cls._controller_registry[name] = controller_class
    
    @classmethod
    def create_controller(cls, controller_type: str, *args, **kwargs):
        """Create controller by type name"""
        controller_class = cls._controller_registry.get(controller_type)
        if not controller_class:
            raise ValueError(f"Unknown controller type: {controller_type}")
        return controller_class(*args, **kwargs)
```

## 🎯 **Architectural Benefits**

### **Perfect Separation of Concerns**
- **BaseWebController**: Generic request handling (NO LLM dependencies)
- **StateProvider**: Universal data interface
- **GameStateModel**: Generic state management
- **Event System**: Handles any type of game events

### **Extension Benefits**
- **Code Reuse**: All extensions inherit core web functionality
- **Consistent API**: Same REST endpoints across all algorithm types
- **Easy Integration**: New algorithms plug into existing web infrastructure
- **Scalable Architecture**: Factory pattern enables dynamic extension loading

### **Educational Value**
- **Design Patterns**: Demonstrates Strategy, Factory, Observer, Template Method
- **Clean Architecture**: Clear separation between presentation, business logic, and data
- **Progressive Enhancement**: From simple Task-0 to sophisticated multi-algorithm systems
- **Real-world Patterns**: Standard web application architecture principles

---

**The web GUI architecture exemplifies the architectural principles established in the Final Decision series, providing a scalable foundation that supports all extension types while maintaining consistent interface patterns and demonstrating professional web application design principles.**

