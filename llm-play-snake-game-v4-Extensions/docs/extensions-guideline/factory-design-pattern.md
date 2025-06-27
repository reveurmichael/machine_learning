# Factory Design Pattern for Snake Game AI Extensions

## 🎯 **Core Philosophy: Simple, Lightweight Factories**

Following **SUPREME_RULE NO.3**, factory patterns are deliberately simple and lightweight, avoiding over-engineering while maintaining educational value and flexibility.

## 🚫 **EXPLICIT DECISION: NO BaseFactory or factory_utils.py**

**CRITICAL ARCHITECTURAL DECISION**: This project **explicitly rejects**:
- ❌ **BaseFactory abstract class** in `extensions/common/utils/`
- ❌ **factory_utils.py module** in `extensions/common/utils/`
- ❌ **Any shared factory inheritance hierarchy**

**Rationale**: Simple dictionary-based factories work perfectly and follow SUPREME_RULE NO.3. Each extension creates its own simple factory without shared infrastructure or inheritance.

### **KISS Principle Applied**
- **Simple Registry**: Use basic dictionaries instead of complex class hierarchies
- **Clear Interface**: Straightforward `create()` method pattern
- **No Over-Engineering**: Avoid unnecessary abstraction layers
- **Easy Extension**: Adding new types is straightforward

### **Design Benefits**
- **Loose Coupling**: Client code doesn't depend on specific implementation classes
- **Open/Closed Principle**: Easy to add new types without modifying existing code
- **Educational Clarity**: Simple enough to understand immediately
- **Practical Utility**: Solves real problems without complexity

## 🏗️ **Simple Factory Template (SUPREME_RULE NO.3)**

Following the KISS principle from `kiss.md`, factories are deliberately simple:

```python
class HeuristicAgentFactory:
    """Simple factory for heuristic agents"""
    
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, **kwargs) -> BaseAgent:
        """Create agent by name"""
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        return agent_class(**kwargs)
    
    @classmethod
    def list_algorithms(cls):
        """List available algorithms"""
        return list(cls._registry.keys())
```

### **Key SUPREME_RULE NO.3 Principles**
- **Simple Dictionary Registry**: No complex inheritance hierarchies
- **Clear Error Messages**: Helpful feedback when algorithm not found
- **Easy Extension**: Adding new algorithms is trivial
- **No Over-Engineering**: Just what's needed, nothing more

## 🧠 **Extension-Specific Implementations**

### **Heuristics Factory (Simple & Effective)**
```python
# extensions/heuristics-v0.02/agents/__init__.py

from .agent_bfs import BFSAgent
from .agent_astar import AStarAgent
from .agent_dfs import DFSAgent
from .agent_hamiltonian import HamiltonianAgent

class HeuristicAgentFactory:
    """Simple factory for heuristic agents - SUPREME_RULE NO.3"""
    
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
        "HAMILTONIAN": HamiltonianAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, grid_size: int = 10, **kwargs) -> BaseAgent:
        """Create heuristic agent by algorithm name"""
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
        return agent_class(algorithm, grid_size, **kwargs)
    
    @classmethod
    def list_algorithms(cls):
        """List available algorithms"""
        return list(cls._registry.keys())
```

### **Supervised Learning Factory (Lightweight)**
```python
# extensions/supervised-v0.02/models/__init__.py

class SupervisedModelFactory:
    """Simple factory for ML models - SUPREME_RULE NO.3"""
    
    _registry = {
        # Neural Network Models
        "MLP": MLPAgent,
        "CNN": CNNAgent,
        
        # Tree-Based Models  
        "XGBOOST": XGBoostAgent,
        "LIGHTGBM": LightGBMAgent,
        "RANDOMFOREST": RandomForestAgent,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs) -> BaseAgent:
        """Create ML model by type"""
        model_class = cls._registry.get(model_type.upper())
        if not model_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model: {model_type}. Available: {available}")
        return model_class(**kwargs)
    
    @classmethod
    def list_models(cls):
        """List available models"""
        return list(cls._registry.keys())
```

## 🎯 **SUPREME_RULE NO.3 Design Principles**

### **1. Keep It Simple**
- Use basic dictionary registries instead of complex hierarchies
- Simple `create()` and `list_*()` methods
- Clear, minimal error handling

### **2. Easy Extension**
- Adding new algorithms/models = adding one line to registry
- No complex configuration or setup required
- Print statements for simple debugging

### **3. Educational Clarity**
- Code is immediately understandable
- No hidden complexity or abstraction layers
- Demonstrates factory pattern without over-engineering

## 🔧 **Simple Integration Examples**

### **Command-Line Integration (SUPREME_RULE NO.3)**
```python
# main.py - Simple factory usage
from agents import HeuristicAgentFactory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="BFS")
    args = parser.parse_args()
    
    print(f"Available algorithms: {HeuristicAgentFactory.list_algorithms()}")
    
    # Simple factory usage
    try:
        agent = HeuristicAgentFactory.create(args.algorithm, grid_size=10)
        print(f"Created {args.algorithm} agent")
    except ValueError as e:
        print(f"Error: {e}")
        return
```

### **Streamlit Integration (Simple)**
```python
# app.py - Streamlit dashboard integration
import streamlit as st
from agents import HeuristicAgentFactory

# Simple algorithm selection
algorithm = st.selectbox(
    "Choose Algorithm",
    HeuristicAgentFactory.list_algorithms()
)

# Simple agent creation
if st.button("Run Algorithm"):
    agent = HeuristicAgentFactory.create(algorithm, grid_size=10)
    print(f"Running {algorithm} algorithm")
```

## 🚀 **SUPREME_RULE NO.3 Benefits**

### **Simplicity & Clarity**
- **Easy to Understand**: Anyone can immediately understand how factories work
- **No Hidden Complexity**: What you see is what you get
- **Quick to Implement**: Add new extensions without learning complex patterns

### **Flexibility & Extension**
- **Easy to Add New Types**: Just add one line to the registry
- **No Framework Lock-in**: Simple Python dictionaries and functions
- **Encourages Experimentation**: Low barrier to trying new algorithms

### **Educational Value**
- **Clear Design Pattern Example**: Shows factory pattern without distractions
- **Practical Implementation**: Solves real problems simply
- **No Over-Engineering**: Demonstrates restraint in design

---

**Simple factories following SUPREME_RULE NO.3 provide all the benefits of the factory pattern while remaining lightweight, understandable, and easily extensible for new algorithms and extensions.** 