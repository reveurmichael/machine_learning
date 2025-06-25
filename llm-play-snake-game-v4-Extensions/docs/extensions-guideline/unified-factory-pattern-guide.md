# Unified Factory Pattern Guide

> **Authoritative Reference**: This document provides the **single canonical factory pattern implementation** for all extensions. It replaces all duplicate factory examples in other guideline files.

## 🎯 **Core Factory Pattern Philosophy**

The factory pattern enables **dynamic agent/model creation** without tight coupling between client code and concrete implementations. This is essential for:
- **Algorithm selection** via command-line arguments
- **Plugin architecture** for easy extension
- **Consistent interfaces** across all extension types
- **Educational demonstration** of the Factory design pattern

## 🏭 **Universal Factory Interface**

All extensions MUST implement this standardized factory interface:

```python
# extensions/{algorithm}-v0.0N/agents/__init__.py (or models/__init__.py)

from abc import ABC, abstractmethod
from typing import Dict, Type, Any
from core.game_agents import BaseAgent  # or appropriate base class

class BaseAgentFactory(ABC):
    """
    Universal Factory Pattern Implementation
    
    Design Pattern: Factory Pattern
    Purpose: Create appropriate agent instances based on configuration
    Educational Note: Demonstrates loose coupling and plugin architecture
    
    Benefits:
    - Eliminates tight coupling between client code and concrete agents
    - Easy to add new agent types without modifying existing code
    - Centralized agent creation logic with consistent error handling
    - Supports dynamic agent selection and configuration
    """
    
    _registry: Dict[str, Type[BaseAgent]] = {}
    
    @classmethod
    @abstractmethod
    def get_registry(cls) -> Dict[str, Type[BaseAgent]]:
        """Return the agent registry for this factory"""
        pass
    
    @classmethod
    def create_agent(cls, algorithm: str, **kwargs) -> BaseAgent:
        """
        Create agent by algorithm name with consistent error handling
        
        Args:
            algorithm: Algorithm name (case-insensitive)
            **kwargs: Additional arguments passed to agent constructor
            
        Returns:
            Configured agent instance
            
        Raises:
            ValueError: If algorithm is not registered
        """
        registry = cls.get_registry()
        algorithm_upper = algorithm.upper()
        
        if algorithm_upper not in registry:
            available = ", ".join(registry.keys())
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
        
        agent_class = registry[algorithm_upper]
        return agent_class(**kwargs)
    
    @classmethod
    def list_available_algorithms(cls) -> list[str]:
        """Return list of available algorithm names"""
        return list(cls.get_registry().keys())
    
    @classmethod
    def register_agent(cls, name: str, agent_class: Type[BaseAgent]) -> None:
        """Register a new agent class (for plugin architecture)"""
        cls._registry[name.upper()] = agent_class
```

## 🔧 **Extension-Specific Implementations**

### **Heuristics Factory**
```python
# extensions/heuristics-v0.0N/agents/__init__.py

from .agent_bfs import BFSAgent
from .agent_astar import AStarAgent
from .agent_dfs import DFSAgent
from .agent_hamiltonian import HamiltonianAgent

class HeuristicAgentFactory(BaseAgentFactory):
    """
    Factory for Heuristic Pathfinding Agents
    
    Algorithms:
    - BFS: Breadth-First Search for shortest path
    - ASTAR: A* with Manhattan distance heuristic
    - DFS: Depth-First Search exploration
    - HAMILTONIAN: Hamiltonian path-based approach
    """
    
    @classmethod
    def get_registry(cls) -> Dict[str, Type[BaseAgent]]:
        return {
            "BFS": BFSAgent,
            "ASTAR": AStarAgent,
            "DFS": DFSAgent,
            "HAMILTONIAN": HamiltonianAgent,
        }

# Convenience factory instance
factory = HeuristicAgentFactory()
```

### **Supervised Learning Factory**
```python
# extensions/supervised-v0.0N/models/__init__.py

from .neural_networks.agent_mlp import MLPAgent
from .neural_networks.agent_cnn import CNNAgent
from .neural_networks.agent_lstm import LSTMAgent
from .tree_models.agent_xgboost import XGBoostAgent
from .tree_models.agent_lightgbm import LightGBMAgent
from .graph_models.agent_gcn import GCNAgent

class SupervisedModelFactory(BaseAgentFactory):
    """
    Factory for Supervised Learning Models
    
    Model Categories:
    - Neural Networks: MLP, CNN, LSTM for different data types
    - Tree Models: XGBoost, LightGBM for tabular data
    - Graph Models: GCN for graph-structured data
    """
    
    @classmethod
    def get_registry(cls) -> Dict[str, Type[BaseAgent]]:
        return {
            # Neural Networks
            "MLP": MLPAgent,
            "CNN": CNNAgent,
            "LSTM": LSTMAgent,
            
            # Tree Models
            "XGBOOST": XGBoostAgent,
            "LIGHTGBM": LightGBMAgent,
            
            # Graph Models
            "GCN": GCNAgent,
        }
```

### **Reinforcement Learning Factory**
```python
# extensions/reinforcement-v0.0N/agents/__init__.py

from .agent_dqn import DQNAgent
from .agent_ppo import PPOAgent
from .agent_a3c import A3CAgent

class RLAgentFactory(BaseAgentFactory):
    """
    Factory for Reinforcement Learning Agents
    
    Algorithms:
    - DQN: Deep Q-Network for value-based learning
    - PPO: Proximal Policy Optimization for policy gradients
    - A3C: Asynchronous Actor-Critic for distributed learning
    """
    
    @classmethod
    def get_registry(cls) -> Dict[str, Type[BaseAgent]]:
        return {
            "DQN": DQNAgent,
            "PPO": PPOAgent,
            "A3C": A3CAgent,
        }
```

## 🎮 **Usage Patterns**

### **Command-Line Integration**
```python
# extensions/{algorithm}-v0.0N/main.py

import argparse
from agents import factory  # or appropriate factory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', required=True, 
                       choices=factory.list_available_algorithms(),
                       help='Algorithm to run')
    parser.add_argument('--grid-size', type=int, default=10)
    args = parser.parse_args()
    
    # Factory creates appropriate agent
    agent = factory.create_agent(
        algorithm=args.algorithm,
        grid_size=args.grid_size
    )
    
    # Use agent in game manager
    game_manager = GameManager(agent=agent, args=args)
    game_manager.run()
```

### **Streamlit Integration**
```python
# extensions/{algorithm}-v0.03/dashboard/tab_main.py

import streamlit as st
from agents import factory

def render_algorithm_selection():
    """Render algorithm selection widget"""
    
    available_algorithms = factory.list_available_algorithms()
    selected_algorithm = st.selectbox(
        "Select Algorithm",
        options=available_algorithms,
        help="Choose the algorithm to run"
    )
    
    if st.button("Run Algorithm"):
        agent = factory.create_agent(
            algorithm=selected_algorithm,
            grid_size=st.session_state.grid_size
        )
        # Launch via subprocess or direct execution
        run_algorithm(agent)
```

## 🔄 **Evolution Across Versions**

### **v0.01: No Factory (Simple Direct Creation)**
```python
# extensions/heuristics-v0.01/main.py
from agent_bfs import BFSAgent  # Direct import, no factory

def main():
    agent = BFSAgent(grid_size=10)  # Direct instantiation
    # Simple single-algorithm approach
```

### **v0.02: Factory Introduction**
```python
# extensions/heuristics-v0.02/agents/__init__.py
class HeuristicAgentFactory:  # Factory pattern introduced
    _registry = {"BFS": BFSAgent, "ASTAR": AStarAgent}
    
    @classmethod
    def create_agent(cls, algorithm: str, **kwargs):
        return cls._registry[algorithm.upper()](**kwargs)
```

### **v0.03+: Stable Factory (Unchanged)**
```python
# extensions/heuristics-v0.03/agents/__init__.py
# 🔒 IDENTICAL to v0.02 - factory stability maintained
class HeuristicAgentFactory:  # Exactly same implementation
    _registry = {"BFS": BFSAgent, "ASTAR": AStarAgent}  # No changes
```

## 🏗️ **Design Pattern Benefits**

### **Educational Value**
- **Factory Pattern**: Clear demonstration of creational design pattern
- **Registry Pattern**: Plugin architecture for extensibility
- **Strategy Pattern**: Interchangeable algorithms with consistent interface
- **Template Method**: Common factory structure across all extensions

### **Technical Benefits**
- **Loose Coupling**: Client code doesn't depend on concrete classes
- **Easy Extension**: New algorithms require minimal code changes
- **Consistent Interface**: All agents created through same mechanism
- **Error Handling**: Centralized validation and error messages

### **Maintenance Benefits**
- **Single Source of Truth**: One place to manage agent registration
- **Testable Components**: Factory can be easily unit tested
- **Version Stability**: Factory interface remains constant across versions
- **Clear Dependencies**: Explicit imports and registrations

## 🚫 **Anti-Patterns to Avoid**

### **Don't Use Different Factory Interfaces**
```python
# ❌ WRONG: Each extension using different factory interface
class HeuristicFactory:
    def make_agent(self, name):  # Different method name
        pass

class SupervisedFactory:
    def get_model(self, algorithm):  # Different method name
        pass
```

### **Don't Hardcode Agent Creation**
```python
# ❌ WRONG: Hardcoded agent selection
if algorithm == "BFS":
    agent = BFSAgent()
elif algorithm == "ASTAR":
    agent = AStarAgent()
# This becomes unmaintainable with many algorithms
```

### **Don't Break Factory Stability**
```python
# ❌ WRONG: Changing factory in v0.03
class HeuristicAgentFactory:
    _registry = {
        "BFS_NEW": BFSAgent,  # FORBIDDEN! Was "BFS" in v0.02
    }
```

## 📊 **Factory Validation**

```python
# extensions/common/validation/factory_validator.py

def validate_factory_compliance(factory_class):
    """Validate factory follows standard interface"""
    
    # Check required methods exist
    assert hasattr(factory_class, 'create_agent')
    assert hasattr(factory_class, 'list_available_algorithms')
    assert hasattr(factory_class, 'get_registry')
    
    # Check registry is not empty
    registry = factory_class.get_registry()
    assert len(registry) > 0, "Factory registry cannot be empty"
    
    # Check all registered classes inherit from BaseAgent
    for name, agent_class in registry.items():
        assert issubclass(agent_class, BaseAgent), f"Agent {name} must inherit from BaseAgent"
```

---

**This unified factory pattern ensures consistent, maintainable, and educational agent creation across all extensions while demonstrating fundamental design patterns.** 