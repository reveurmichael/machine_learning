# Extensions v0.01 - Foundation Patterns

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and follows SUPREME_RULE NO.3 for educational flexibility.

## 🎯 **SUPREME_RULES: Educational Flexibility & OOP Design**

**SUPREME_RULE NO.3**: We should be able to add new extensions easily and try out new ideas. Therefore, code in the "extensions/common/" folder should NOT be too restrictive.

**SUPREME_RULE NO.4**: While things in the folder "extensions/common/" are expected to be shared across all extensions, we expect exceptions to be made for certain extensions, as we have a very future-proof mindset. Therefore, whenever possible, make things in the "extensions/common/" folder OOP, so that, if exceptions are to be made, they can extend those classes in the "extensions/common/" folder, to adapt to the exceptions and some exceptional needs for those certain extensions.

**Educational Philosophy**: v0.01 extensions serve as proof-of-concept implementations that encourage experimentation and learning. The architecture supports any algorithm type without artificial restrictions. The OOP design in common utilities enables both standard usage and specialized customization when needed.

## 🏗️ **Foundation Architecture**

### **Core Philosophy: Proof of Concept**
v0.01 extensions demonstrate:
- **Single Algorithm Focus**: One primary algorithm per extension
- **Minimal Viable Implementation**: Core functionality without complexity
- **Educational Clarity**: Clear, understandable code structure
- **Extensibility Foundation**: Prepared for future algorithm additions

### **Flexible Algorithm Support (Following SUPREME_RULE NO.3)**
```python
# ✅ FLEXIBLE: Extensions can implement any algorithm
class AnyAlgorithmAgent(BaseAgent):
    """
    Educational Note (SUPREME_RULE NO.3):
    Extensions can implement any algorithm without restrictions.
    This encourages experimentation and creative problem-solving.
    """
    
    def plan_move(self, game_state):
        # Any algorithm implementation welcome
        return self.your_creative_algorithm(game_state)
```

**No Algorithm Restrictions**: Following SUPREME_RULE NO.3, extensions can implement any algorithm approach - pathfinding, machine learning, evolutionary, rule-based, or novel approaches.

### **Common Utilities Usage (Following SUPREME_RULE NO.4)**
```python
# ✅ STANDARD USAGE: Most v0.01 extensions use common utilities as-is
from extensions.common.path_utils import ensure_project_root, get_dataset_path
from extensions.common.csv_schema import TabularFeatureExtractor
from extensions.common.dataset_loader import BaseDatasetLoader

# Standard usage - works for most extensions
project_root = ensure_project_root()
extractor = TabularFeatureExtractor()
loader = BaseDatasetLoader(config)

# ✅ SPECIALIZED USAGE: If needed, inherit for exceptional requirements
class CustomFeatureExtractor(TabularFeatureExtractor):
    """
    Educational Note (SUPREME_RULE NO.4):
    When standard features aren't sufficient, inherit and customize.
    Most extensions won't need this - only for exceptional cases.
    """
    
    def _extract_extension_specific_features(self, game_state):
        # Add custom features for specialized algorithms
        return {"custom_metric": self._calculate_custom_metric(game_state)}
```

## 📁 **Directory Structure Template**

### **Standard v0.01 Pattern**
```
extensions/{algorithm}-v0.01/
├── __init__.py
├── agent_{algorithm}.py        # Primary algorithm implementation
├── game_logic.py              # Extension-specific game logic
├── game_manager.py            # Session management
├── main.py                    # CLI entry point
└── README.md                  # Algorithm documentation
```

### **Flexible Naming (Following SUPREME_RULE NO.3)**
- **Extension Types**: Any descriptive name (heuristics, supervised, reinforcement, evolutionary, custom, experimental, etc.)
- **Algorithm Names**: Any algorithm identifier following `agent_{name}.py` pattern
- **Version Numbers**: Any version following `v0.01` pattern

**Educational Value**: This simple structure makes it easy to understand the core components while providing a foundation for future expansion.

## 🔧 **Implementation Patterns**

### **Agent Implementation Template**
```python
from core.game_agents import BaseAgent

class YourAlgorithmAgent(BaseAgent):
    """
    Proof-of-concept implementation for [Your Algorithm].
    
    Educational Note (SUPREME_RULE NO.3):
    This template supports any algorithm approach. Feel free to:
    - Implement novel pathfinding strategies
    - Try machine learning approaches
    - Experiment with rule-based systems
    - Create hybrid algorithms
    - Test theoretical concepts
    
    The system is designed to be flexible and encourage innovation.
    """
    
    def __init__(self, name: str, grid_size: int):
        super().__init__(name, grid_size)
        self.algorithm_name = name
        # Add your algorithm-specific initialization
        
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """
        Plan next move using your algorithm.
        
        Following SUPREME_RULE NO.3, this can be any algorithm:
        - Pathfinding (BFS, A*, Dijkstra, custom)
        - Machine learning (neural networks, decision trees)
        - Evolutionary (genetic algorithms, swarm intelligence)
        - Rule-based (expert systems, fuzzy logic)
        - Hybrid approaches combining multiple techniques
        """
        # Your algorithm implementation here
        return self.your_algorithm_logic(game_state)
    
    def reset(self) -> None:
        """Reset algorithm state for new game."""
        # Reset any algorithm-specific state
        pass
```

### **Game Logic Integration**
```python
from core.game_logic import BaseGameLogic

class YourExtensionGameLogic(BaseGameLogic):
    """
    Extension-specific game logic for [Your Algorithm].
    
    Educational Note (SUPREME_RULE NO.3):
    Customize game logic for your specific algorithm needs:
    - Add algorithm-specific state tracking
    - Implement custom move validation
    - Include performance metrics
    - Add debugging capabilities
    """
    
    def __init__(self, grid_size=10, use_gui=True):
        super().__init__(grid_size, use_gui)
        # Add your extension-specific initialization
        
    def plan_next_moves(self):
        """Plan moves using your algorithm agent."""
        # Integration with your agent implementation
        pass
```

## 🚀 **Benefits of v0.01 Pattern**

### **Educational Advantages**
- **Clear Learning Path**: Simple structure easy to understand
- **Minimal Complexity**: Focus on algorithm implementation
- **Rapid Prototyping**: Quick proof-of-concept development
- **Foundation Building**: Prepares for more complex versions

### **Technical Benefits**
- **Clean Architecture**: Inherits from proven base classes
- **Consistent Interface**: Works with existing game framework
- **Easy Testing**: Simple structure enables focused testing
- **Future-Ready**: Designed for expansion in v0.02+

### **Flexibility (SUPREME_RULE NO.3)**
- **No Algorithm Restrictions**: Implement any approach
- **Custom Extensions**: Create entirely new extension types
- **Experimental Freedom**: Try novel ideas without constraints
- **Rapid Iteration**: Easy to modify and improve

## 📊 **Configuration and Usage**

### **Flexible Configuration Pattern**
```python
# Following SUPREME_RULE NO.3: No restrictive configuration
def create_agent(algorithm: str, grid_size: int, **kwargs):
    """
    Create agent with flexible configuration.
    
    Educational Note (SUPREME_RULE NO.3):
    This function accepts any algorithm name and configuration,
    encouraging experimentation with new approaches.
    """
    try:
        # Dynamic agent creation - no hard-coded restrictions
        agent_class = import_agent_class(algorithm)
        return agent_class(grid_size=grid_size, **kwargs)
    except ImportError:
        available = list_available_algorithms()
        raise ValueError(f"Algorithm '{algorithm}' not found. "
                        f"Available: {available}. "
                        f"Following SUPREME_RULE NO.3, you can easily add new algorithms!")
```

### **Example Usage**
```bash
# Any algorithm can be implemented and used
python main.py --algorithm your_custom_algorithm --grid-size 10
python main.py --algorithm experimental_approach --grid-size 12
python main.py --algorithm novel_pathfinding --grid-size 16
```

## 🎓 **Educational Value**

### **Learning Objectives**
- **Algorithm Implementation**: Practice implementing algorithms from scratch
- **Software Architecture**: Understand inheritance and design patterns
- **Problem Solving**: Apply algorithms to concrete problems
- **Experimentation**: Encourage trying new approaches

### **Research Applications**
- **Algorithm Comparison**: Easy to compare different approaches
- **Performance Analysis**: Built-in metrics and logging
- **Educational Demonstrations**: Clear examples for teaching
- **Innovation Platform**: Foundation for novel algorithm development

## 🔮 **Evolution Path**

### **v0.01 → v0.02 Evolution**
- **Single Algorithm**: v0.01 focuses on one primary algorithm
- **Multi-Algorithm**: v0.02 expands to multiple related algorithms
- **Agent Organization**: v0.02 introduces `agents/` directory structure
- **Factory Patterns**: v0.02 adds agent factory for dynamic creation

### **Preparation for Growth**
v0.01 extensions are designed to evolve naturally:
- **Inheritance Ready**: Base classes support extension
- **Modular Design**: Easy to add new components
- **Consistent Patterns**: Smooth transition to complex versions
- **Educational Continuity**: Learning builds progressively

## 🔗 **See Also**

- **`extensions-v0.02.md`**: Multi-algorithm expansion patterns
- **`core.md`**: Base class architecture documentation
- **`final-decision-10.md`**: SUPREME_RULE NO.3 specification
- **`config.md`**: Configuration architecture guidelines

---

**v0.01 extensions provide the perfect starting point for algorithm exploration, combining educational clarity with the flexibility needed for innovation. Following SUPREME_RULE NO.3, they encourage experimentation and creative problem-solving while establishing solid architectural foundations.**






