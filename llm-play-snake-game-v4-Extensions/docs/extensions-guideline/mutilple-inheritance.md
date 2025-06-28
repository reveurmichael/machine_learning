# Multiple Inheritance Guidelines for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines multiple inheritance guidelines for extensions.

> **See also:** `core.md`, `final-decision-10.md`, `project-structure-plan.md`.

# Multiple Inheritance Guidelines

## 🎯 **Core Philosophy: Careful Use of Multiple Inheritance**

Multiple inheritance should be used judiciously in the Snake Game AI project. While Python supports it, the architecture primarily relies on single inheritance with composition patterns for complex behaviors.

## 🚫 **General Recommendation: Avoid Multiple Inheritance**

The project architecture is designed around **single inheritance hierarchies** with **composition patterns** for complex functionality:

```python
# ✅ PREFERRED: Single inheritance with composition
class RLGameManager(BaseGameManager):
    """Reinforcement Learning game manager using composition"""
    
    def __init__(self, agent_type: str):
        super().__init__()
        self.experience_replay = ExperienceReplayBuffer()  # Composition
        self.neural_network = PolicyNetwork()             # Composition
        self.optimizer = AdamOptimizer()                  # Composition

# ❌ AVOID: Multiple inheritance for complex behaviors
class RLGameManager(BaseGameManager, ExperienceReplayMixin, 
                   PolicyNetworkMixin, OptimizerMixin):
    """Complex multiple inheritance - harder to understand and maintain"""
    pass
```

## ✅ **Limited Acceptable Uses**

### **1. Mixin Classes for Shared Utilities**
Small, focused mixin classes that provide specific utilities:

```python
class LoggingMixin:
    """Simple mixin for debug output (simple logging: simple print statements)"""
    
    def debug_print(self, name: str, message: str):
        """Simple debug output for the component"""
        # Simple print-based output instead of complex file logging
        print(f"[{name}] {message}")

class HeuristicAgent(BaseAgent, LoggingMixin):
    """Agent with logging capabilities via mixin"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.debug_print(f"heuristic_{name}", "Agent initialized")
```

### **2. Interface Implementation**
Multiple inheritance for implementing multiple interfaces:

```python
from abc import ABC, abstractmethod

class Trainable(ABC):
    @abstractmethod
    def train(self, data): pass

class Evaluable(ABC):
    @abstractmethod
    def evaluate(self, test_data): pass

class MLAgent(BaseAgent, Trainable, Evaluable):
    """Agent implementing multiple interfaces"""
    
    def train(self, data):
        # Training implementation
        pass
    
    def evaluate(self, test_data):
        # Evaluation implementation
        pass
```

## 🔧 **Design Patterns to Prefer**

### **Composition over Inheritance**
```python
# ✅ PREFERRED: Composition pattern
class SupervisedGameManager(BaseGameManager):
    """Uses composition for complex functionality"""
    
    def __init__(self, model_type: str):
        super().__init__()
        self.model = ModelFactory.create(model_type)      # Composition
        self.trainer = TrainerFactory.create(model_type)  # Composition
        self.evaluator = EvaluatorFactory.create()        # Composition
```

### **Strategy Pattern for Variants**
```python
# ✅ PREFERRED: Strategy pattern for algorithm variants
class PathfindingAgent(BaseAgent):
    """Uses strategy pattern instead of inheritance hierarchy"""
    
    def __init__(self, strategy_name: str):
        super().__init__()
        self.strategy = PathfindingFactory.create(strategy_name)
    
    def plan_move(self, game_state):
        return self.strategy.find_path(game_state)
```

## 🚨 **Warning Signs to Avoid**

### **Diamond Problem**
```python
# ❌ AVOID: Diamond inheritance patterns
class A:
    def method(self): pass

class B(A):
    def method(self): pass

class C(A):
    def method(self): pass

class D(B, C):  # Diamond problem - which method()?
    pass
```

### **Complex MRO (Method Resolution Order)**
```python
# ❌ AVOID: Complex inheritance chains
class ComplexAgent(BaseAgent, PathfindingMixin, LearningMixin, 
                  EvaluationMixin, LoggingMixin, VisualizationMixin):
    """Too many mixins - difficult to understand behavior"""
    pass
```

## 📋 **Guidelines for Multiple Inheritance**

### **When to Consider Multiple Inheritance**
- **Small, focused mixins** that provide specific utilities
- **Interface implementation** where multiple protocols are needed
- **Cross-cutting concerns** like logging or monitoring

### **When to Avoid Multiple Inheritance**
- **Complex behavior composition** - use composition instead
- **Deep inheritance hierarchies** - prefer flat structures
- **Conflicting method names** - indicates design problems
- **Educational complexity** - keep learning examples simple

### **Implementation Checklist**
- [ ] Is the mixin small and focused on one concern?
- [ ] Are method names unlikely to conflict?
- [ ] Is the MRO clear and predictable?
- [ ] Does it improve code reuse without adding complexity?
- [ ] Is it easy to understand for educational purposes?

---

**The project architecture prioritizes clarity and maintainability over complex inheritance patterns. Use single inheritance with composition as the primary design approach, reserving multiple inheritance for simple, well-defined cases only.**