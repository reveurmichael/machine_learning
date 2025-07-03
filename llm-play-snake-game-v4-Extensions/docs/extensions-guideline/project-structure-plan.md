# Project Structure Plan for Snake Game AI Extensions

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for project structure and supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`).

> **See also:** `core.md`, `standalone.md`, `final-decision-10.md`, `config.md`.

## 🏗️ **High-Level Architecture Overview**

```
ROOT/                                    # Task-0 (LLM-based Snake AI)
├── core/                               # Base classes for all tasks
├── config/                             # Universal constants
├── extensions/                         # Task 1-5 implementations
│   ├── common/                         # Shared utilities (SUPREME_RULES compliant per final-decision-10.md)
│   ├── {algorithm}-v{version}/         # Flexible extension naming
│   └── [any new extension type]/       # Unlimited extensibility
├── gui/                                # GUI components
├── web/                                # Web interface
├── replay/                             # Replay system
├── scripts/                            # Entry points
└── docs/                               # Documentation
```

**Architectural Note**: Extensions can be any algorithm type - heuristics, ML, RL, evolutionary, custom approaches, experimental ideas, or novel research directions, all following SUPREME_RULES from `final-decision-10.md`.

## 📁 **Extension Structure Template (Flexible)**

### **Universal Extension Pattern**
```
extensions/{extension_type}-v{version}/
├── __init__.py
├── agents/                             # Algorithm implementations (v0.02+)
│   ├── __init__.py                     # Agent factory with canonical create() method
│   ├── agent_{algorithm1}.py           # Any algorithm approach
│   ├── agent_{algorithm2}.py           # Following SUPREME_RULES
│   └── agent_{algorithmN}.py           # Unlimited algorithm variety
├── scripts/                            # CLI entry points (v0.03+)
├── game_logic.py                       # Extension-specific logic
├── game_manager.py                     # Session management
├── game_runner.py                      # Quick-play helper (RECOMMENDED)
└── README.md                           # Documentation
```

**Extension Note**: 
- `{extension_type}`: Any descriptive name (heuristics, supervised, custom, experimental)
- `{algorithm}`: Any algorithm following `agent_{name}.py` pattern
- No restrictions on algorithm types or approaches
- All factories must use canonical `create()` method per SUPREME_RULES from `final-decision-10.md`
- `game_runner.py`: Extension-specific quick-play utilities (see Game Runner Pattern below)

## 🎯 **Core Design Principles**

### **1. Educational Flexibility**
- **No Algorithm Restrictions**: Extensions can implement any approach
- **Rapid Prototyping**: Easy to create and test new ideas  
- **Experimental Freedom**: Encourage novel and creative solutions
- **Learning-Focused**: Structure supports educational exploration

### **2. Inheritance-Based Architecture**
```python
# Perfect inheritance hierarchy enabling any algorithm type
BaseGameManager → YourCustomGameManager
BaseGameLogic → YourCustomGameLogic  
BaseAgent → YourCustomAgent

# Architectural Note:
# Base classes are designed to support ANY algorithm approach:
# - Pathfinding algorithms
# - Machine learning models
# - Evolutionary algorithms
# - Rule-based systems
# - Hybrid approaches
# - Novel research ideas
```

### **3. Shared Utilities (SUPREME_RULES Compliant)**
```python
# extensions/common/ - Following SUPREME_RULES from `final-decision-10.md`

from utils.factory_utils import SimpleFactory
from extensions.common.utils.path_utils import get_dataset_path
from extensions.common.utils.dataset_utils import load_dataset_for_training

# SUPREME_RULES Compliance (`final-decision-10.md`):
# - Lightweight, reusable foundation for all extensions
# - Simple, preferably OOP but never over-engineered
# - Supports experimentation and flexibility
# - Uses utils/print_utils.py functions only for logging (no .log files)

# Example: Standard usage (most extensions)
factory = SimpleFactory()
factory.register("myagent", CustomAgent)
agent = factory.create("myagent", config=params)  # CANONICAL create() method per SUPREME_RULES

# Example: Specialized usage (exceptional needs)
class QuantumDatasetLoader(BaseDatasetLoader):
    def _initialize_loader_specific_settings(self):
        self.quantum_validator = QuantumValidator()
        print_info(f"[QuantumLoader] Initialized quantum settings")  # SUPREME_RULES compliant logging
    
    def _generate_extension_specific_metadata(self, data, file_path):
        print_info(f"[QuantumLoader] Generating quantum metadata")  # SUPREME_RULES compliant logging
        return {"quantum_entanglement_score": self._measure_entanglement(data)}
```

## 🔧 **Implementation Guidelines**

### **Extension Creation (SUPREME_RULES Compliant)**
```python
# ✅ ENCOURAGED: Any algorithm approach following final-decision-10.md
class NovelAlgorithmAgent(BaseAgent):
    """
    Implement any algorithm you can imagine:
    - Traditional pathfinding
    - Machine learning approaches
    - Evolutionary computation
    - Swarm intelligence
    - Quantum algorithms
    - Hybrid methods
    - Your own novel ideas
    
    All implementations must follow SUPREME_RULES from `final-decision-10.md`.
    """
    
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)
        print_info(f"[{name}] Novel algorithm agent initialized")  # SUPREME_RULES compliant logging
    
    def plan_move(self, game_state):
        print_info(f"[{self.name}] Planning move with novel algorithm")  # SUPREME_RULES compliant logging
        # Your creative algorithm implementation
        return self.your_innovative_approach(game_state)
```

### **Factory Pattern Implementation (CANONICAL)**
```python
class CustomAgentFactory:
    """
    Factory following SUPREME_RULES from `final-decision-10.md`
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create agents using the canonical create() method
    Educational Value: Shows how SUPREME_RULES apply to custom implementations
    """
    
    _registry = {
        "CUSTOM": CustomAgent,
        "NOVEL": NovelAlgorithmAgent,
        "EXPERIMENTAL": ExperimentalAgent,
    }
    
    @classmethod
    def create(cls, agent_type: str, **kwargs):  # CANONICAL create() method per SUPREME_RULES
        """Create agent using canonical create() method following SUPREME_RULES from `final-decision-10.md`"""
        agent_class = cls._registry.get(agent_type.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {available}")
        print_info(f"[CustomAgentFactory] Creating agent: {agent_type}")  # SUPREME_RULES compliant logging
        return agent_class(**kwargs)
```

### **Flexible Configuration**
```python
# Configuration supports any extension type
EXTENSION_CONFIG = {
    'type': 'your_custom_type',           # Any descriptive name
    'algorithms': ['any', 'approach'],    # No restrictions
    'parameters': {                       # Flexible parameters
        'custom_param1': 'any_value',
        'experimental_setting': True,
        'novel_hyperparameter': 42
    }
}
```

## 📊 **Extension Evolution Path**

### **Version Progression (Flexible)**
```
v0.01: Proof of Concept
├── Single algorithm implementation
├── Basic functionality
├── Educational clarity
└── Foundation for growth

v0.02: Multi-Algorithm Expansion
├── Multiple related algorithms
├── Agent factory patterns with canonical create() method

v0.03: Interactive Dashboards
├── Streamlit web interface

v0.04: Advanced Features (Optional)
├── Language-rich datasets (if applicable)
```

**Version Note**: Version progression is flexible - extensions can skip versions if not needed, add custom versions (v0.05, v0.06, etc.), implement features in any order, or focus on their specific research goals, all while maintaining `final-decision-10.md` compliance.

## 🎓 **Educational Benefits**

### **Learning Objectives**
- **Algorithm Design**: Practice implementing any algorithm approach
- **Software Architecture**: Understand inheritance and design patterns
- **Comparative Analysis**: Easy comparison between different methods
- **Research Skills**: Platform for novel algorithm development
- **SUPREME_RULES Compliance**: Learn professional software engineering standards

### **Research Applications**
- **Algorithm Innovation**: Test new algorithmic ideas
- **Performance Benchmarking**: Compare across different approaches
- **Educational Demonstrations**: Clear examples for teaching
- **Experimental Platform**: Rapid prototyping and iteration

### **Extension-Specific Patterns**

#### **Heuristics game_runner.py**
```python
# extensions/heuristics-v0.03/game_runner.py
"""Quick-play helper for pathfinding algorithms."""

from agents import BaseHeuristicAgent
from game_logic import HeuristicGameLogic

def play_heuristic(agent: BaseHeuristicAgent, **kwargs) -> List[dict]:
    """Execute pathfinding agent and return trajectory."""
    game = HeuristicGameLogic(use_gui=kwargs.get('render', False))
    # Implementation specific to heuristic algorithms
    
def play_bfs(**kwargs):
    """Convenience function for BFS."""
    from agents.agent_bfs import BFSAgent
    return play_heuristic(BFSAgent(), **kwargs)
```

#### **Machine Learning game_runner.py**
```python
# extensions/supervised-v0.02/game_runner.py  
"""Quick-play helper for trained ML models."""

from agents import BaseMLAgent
from game_logic import MLGameLogic

def play_ml(model_path: str, **kwargs) -> List[dict]:
    """Execute trained model and return trajectory."""
    game = MLGameLogic(use_gui=kwargs.get('render', False))
    agent = load_trained_model(model_path)
    # Implementation specific to ML inference
```

#### **Reinforcement Learning game_runner.py**
```python
# extensions/reinforcement-v0.02/game_runner.py
"""Quick-play helper for RL agents."""

from agents import BaseRLAgent
from game_logic import RLGameLogic

def play_rl(agent_or_path: Union[BaseRLAgent, str], **kwargs) -> List[dict]:
    """Execute RL agent (training or inference mode)."""
    game = RLGameLogic(use_gui=kwargs.get('render', False))
    # Implementation specific to RL agents

def train_rl(agent: BaseRLAgent, episodes: int = 1000) -> dict:
    """Train RL agent and return statistics."""
    # Training-specific implementation
```

### **Benefits of Extension game_runner.py**

1. **Simplified Testing**: `python -c "from game_runner import play_bfs; play_bfs(render=True)"`
2. **Clear Documentation**: Shows exactly how each agent type works
3. **Research Efficiency**: Quick validation without complex setup
4. **Educational Value**: Demonstrates agent capabilities clearly

---

**This project structure plan ensures maximum flexibility and educational value while maintaining clean architecture and strict compliance with `final-decision-10.md` SUPREME_RULES. It encourages innovation, experimentation, and creative problem-solving in the Snake Game AI domain.**