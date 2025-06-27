# Factory Design Pattern for Snake Game AI Extensions

## 🎯 **Core Philosophy: Consistent Creation Patterns**

The Factory Pattern is a cornerstone design pattern in the Snake Game AI project, enabling:
- **Flexible agent/model creation** without tight coupling to concrete classes
- **Plugin-style architecture** where new algorithms can be added seamlessly  
- **Consistent interfaces** across different extension types
- **Educational demonstration** of fundamental design patterns

### **Design Benefits**
- **Loose Coupling**: Client code doesn't depend on specific implementation classes
- **Open/Closed Principle**: Easy to add new types without modifying existing code
- **Single Responsibility**: Creation logic centralized in factory classes
- **Consistent Interface**: Uniform creation patterns across all extensions

### **SUPREME_RULE NO.3 Integration**

Factory patterns in `extensions/common/` follow **SUPREME_RULE NO.3**: "The common folder provides lightweight OOP bases; extensions can subclass them if specialised behaviour is needed."

**Key OOP Features for Extensibility:**
- **Inheritance-Ready Base Factories**: All factory classes designed for extension through inheritance
- **Protected Extension Points**: Methods marked for selective customization by subclasses
- **Virtual Registry Methods**: Complete behavior replacement when needed for specialized requirements
- **Composition Support**: Pluggable registry and validation components

**Example - Extensible Factory:**
```python
# Note: Following SUPREME_RULE NO.3, factory patterns are kept simple
# Each extension implements its own lightweight factory
class BaseFactory(ABC):
    def _initialize_factory_settings(self):
        """SUPREME_RULE NO.3: Extension point for specialized factories"""
        pass
    
    def _setup_extension_specific_agents(self):
        """SUPREME_RULE NO.3: Override to register custom agents"""
        pass

# In a specialized extension
class CustomRLFactory(BaseFactory):
    def _initialize_factory_settings(self):
        # Add RL-specific factory configuration
        self.rl_model_validator = RLModelValidator()
        self.environment_integration = True
    
    def _setup_extension_specific_agents(self):
        # Register specialized RL agents
        self._agent_registry["multi_objective_dqn"] = {
            "class": MultiObjectiveDQNAgent,
            "required_params": ["state_size", "action_size", "objectives"],
            "extensions": ["reinforcement"],
            "description": "Multi-objective Deep Q-Network"
        }
```

## 🏗️ **Standard Factory Template**

> **Authoritative Reference**: See `unified-factory-pattern-guide.md` for the complete, standardized factory implementation.

### **Universal Factory Pattern**
All extensions MUST use the standardized factory implementation from the unified guide:

```python
class {Type}Factory:
    """
    Factory Pattern Implementation for {Type} Creation
    
    Design Pattern: Factory Pattern
    Purpose: Create {type} instances without exposing instantiation logic
    Educational Note: Demonstrates how factory patterns enable plugin architectures
    and support the Open/Closed Principle by allowing new types to be added
    without modifying existing client code.
    
    Benefits:
    - Decouples client code from concrete implementations
    - Centralizes creation logic for easy maintenance
    - Enables dynamic type selection at runtime
    - Supports plugin-style architecture
    """
    
    _registry = {
        # Map string identifiers to implementation classes
        "TYPE1": Type1Implementation,
        "TYPE2": Type2Implementation,
        "TYPE3": Type3Implementation,
    }
    
    @classmethod
    def create(cls, type_name: str, **kwargs) -> BaseType:
        """Create instance by type name"""
        # Implementation details in unified guide
        pass
```

## 🧠 **Extension-Specific Implementations**

> **Authoritative Reference**: See `unified-factory-pattern-guide.md` for complete extension-specific factory examples.

### **Heuristics Factory**
```python
# extensions/heuristics-v0.02/agents/__init__.py

from .agent_bfs import BFSAgent
from .agent_astar import AStarAgent
from .agent_dfs import DFSAgent
from .agent_hamiltonian import HamiltonianAgent

class HeuristicAgentFactory:
    """
    Factory Pattern Implementation for Heuristic Agents
    
    Design Pattern: Factory Pattern
    Purpose: Create heuristic agent instances without exposing instantiation logic
    Educational Note: Demonstrates how factory patterns enable algorithm selection
    at runtime, supporting the Strategy Pattern for different pathfinding approaches.
    """
    
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
        "HAMILTONIAN": HamiltonianAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, grid_size: int = 10, **kwargs) -> BaseAgent:
        """Create heuristic agent by algorithm name"""
        # Implementation details in unified guide
        pass
```

### **Supervised Learning Factory**
```python
# extensions/supervised-v0.02/models/__init__.py

from .neural_networks.agent_mlp import MLPAgent
from .neural_networks.agent_cnn import CNNAgent
from .tree_models.agent_xgboost import XGBoostAgent
from .tree_models.agent_lightgbm import LightGBMAgent

class SupervisedModelFactory:
    """
    Factory Pattern Implementation for Supervised Learning Models
    
    Design Pattern: Factory Pattern + Abstract Factory Pattern
    Purpose: Create ML model instances across different frameworks and architectures
    Educational Note: Shows how factory patterns can handle multiple model types
    (neural networks, tree models, etc.) with a unified interface.
    """
    
    _registry = {
        # Neural Network Models
        "MLP": MLPAgent,
        "CNN": CNNAgent,
        "LSTM": LSTMAgent,
        
        # Tree-Based Models  
        "XGBOOST": XGBoostAgent,
        "LIGHTGBM": LightGBMAgent,
        "RANDOMFOREST": RandomForestAgent,
        
        # Graph Neural Networks
        "GCN": GCNAgent,
        "GRAPHSAGE": GraphSAGEAgent,
    }
    
    @classmethod
    def create(cls, model_type: str, input_dim: int, **kwargs) -> BaseMLAgent:
        """Create supervised learning model by type"""
        # Implementation details in unified guide
        pass
```

## 🎯 **Key Design Principles**

### **1. Consistent Naming Conventions**
- Factory classes follow `{Type}Factory` naming pattern
- Registry keys use UPPERCASE for consistency
- Create methods follow `create(type_name, **kwargs)` signature

### **2. Error Handling**
- Clear error messages when unknown types are requested
- List available types in error messages
- Graceful degradation for missing implementations

### **3. Extensibility**
- Easy to add new types without modifying existing code
- Plugin-style registration for third-party extensions
- Backward compatibility for existing factory users

## 🔧 **Integration with Extension Architecture**

### **Command-Line Integration**
```python
# main.py - Using factory for dynamic algorithm selection
from agents import HeuristicAgentFactory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=HeuristicAgentFactory.list_algorithms())
    args = parser.parse_args()
    
    # Factory creates the appropriate agent
    agent = HeuristicAgentFactory.create(args.algorithm, grid_size=10)
```

### **Configuration Integration (Following SUPREME_RULE NO.3)**
```python
# config.py - Flexible factory-aware configuration
from agents import HeuristicAgentFactory

# Educational Note (SUPREME_RULE NO.3): 
# We should be able to add new extensions easily and try out new ideas.
# Dynamic algorithm listing enables flexibility without hard-coded restrictions.

def get_available_algorithms():
    """Get currently available algorithms dynamically"""
    return HeuristicAgentFactory.list_algorithms()

# Flexible defaults that adapt to available algorithms
def get_default_algorithm():
    """Get sensible default algorithm from available options"""
    available = get_available_algorithms()
    # Prefer BFS if available, otherwise use first available
    return "BFS" if "BFS" in available else available[0] if available else None
```

## 🚀 **Benefits for Extension Development**

### **Educational Value**
- **Design Pattern Demonstration**: Shows factory pattern in real-world context
- **Plugin Architecture**: Demonstrates extensible system design
- **Separation of Concerns**: Clear boundaries between creation and usage

### **Technical Benefits**
- **Consistent Interfaces**: Uniform agent/model creation across extensions
- **Easy Testing**: Mock factories for unit testing
- **Runtime Flexibility**: Dynamic algorithm/model selection

### **Maintenance Benefits**
- **Centralized Logic**: All creation logic in one place
- **Easy Updates**: Add new types without touching client code
- **Clear Dependencies**: Explicit registration of available types

## 🔗 **See Also**

- **`extension-evolution-rules.md`**: How factory patterns evolve across versions
- **`final-decision-7.md`**: Factory pattern architectural decisions
- **`final-decision-8.md`**: Factory pattern implementation standards

---

**This factory pattern implementation ensures consistent, extensible, and maintainable agent/model creation across all Snake Game AI extensions while serving as an educational example of fundamental design patterns.** 