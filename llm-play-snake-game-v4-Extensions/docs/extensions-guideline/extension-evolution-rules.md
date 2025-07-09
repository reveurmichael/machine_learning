# Extension Evolution Rules

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision.md`) and defines extension evolution rules.

> **See also:** `final-decision.md`, `standalone.md`, `conceptual-clarity.md`.

## 🎯 **Core Philosophy: Progressive Enhancement**

Extension evolution follows a **progressive enhancement** model where each version builds upon the previous one, adding new capabilities while maintaining backward compatibility within the same extension family, strictly following SUPREME_RULES from `final-decision.md`.

### **Educational Value**
- **Incremental Learning**: Each version introduces new concepts gradually
- **Software Evolution**: Understanding how systems evolve over time
- **Backward Compatibility**: Learning to maintain compatibility while adding features
- **Version Management**: Understanding versioning strategies and best practices

## 📈 **Version Evolution Pattern**

### **v0.01: Foundation**
```python
# v0.01 establishes the basic foundation
class BasicAgent:
    """Basic agent implementation for v0.01."""
    
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        print_info(f"[BasicAgent] Initialized with grid size {grid_size}")  # SUPREME_RULES compliant logging
    
    def plan_move(self, game_state: dict) -> str:
        """Basic move planning."""
        # Simple heuristic: move toward apple
        head_pos = game_state['head_position']
        apple_pos = game_state['apple_position']
        
        if apple_pos[0] > head_pos[0]:
            return 'RIGHT'
        elif apple_pos[0] < head_pos[0]:
            return 'LEFT'
        elif apple_pos[1] > head_pos[1]:
            return 'DOWN'
        else:
            return 'UP'
```

### **v0.02: Multi-Algorithm Support**
```python
# v0.02 adds multiple algorithms using canonical factory pattern
class AgentFactory:
    """Factory for creating different agent types."""
    
    _registry = {
        'BFS': BFSAgent,
        'ASTAR': AStarAgent,
        'DFS': DFSAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, **kwargs):  # CANONICAL create() method per SUPREME_RULES
        """Create agent using canonical factory pattern following SUPREME_RULES from final-decision.md."""
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        print_info(f"[AgentFactory] Creating {algorithm} agent")  # SUPREME_RULES compliant logging
        return agent_class(**kwargs)

class BFSAgent(BasicAgent):
    """BFS agent extending basic agent."""
    
    def plan_move(self, game_state: dict) -> str:
        """Plan move using BFS algorithm."""
        path = self._bfs_pathfinding(game_state)
        if path and len(path) > 1:
            return self._direction_from_path(path[0], path[1])
        return super().plan_move(game_state)  # Fallback to basic behavior
```

### **v0.03: Advanced Features**
```python
# v0.03 adds advanced features like dataset generation and evaluation
class AdvancedAgent(BFSAgent):
    """Advanced agent with dataset generation capabilities."""
    
    def __init__(self, grid_size: int, generate_dataset: bool = False):
        super().__init__(grid_size)
        self.generate_dataset = generate_dataset
        self.dataset_generator = DatasetGenerator() if generate_dataset else None
        
        print_info(f"[AdvancedAgent] Initialized with dataset generation: {generate_dataset}")  # SUPREME_RULES compliant logging
    
    def plan_move(self, game_state: dict) -> str:
        """Plan move and optionally generate dataset."""
        move = super().plan_move(game_state)
        
        if self.generate_dataset:
            self.dataset_generator.add_game_state(game_state, move)
        
        return move
    
    def save_dataset(self, output_path: str):
        """Save generated dataset."""
        if self.dataset_generator:
            self.dataset_generator.save_to_csv(output_path)
            print_success(f"[AdvancedAgent] Saved dataset to {output_path}")  # SUPREME_RULES compliant logging
```

## 🔄 **Evolution Rules**

### **1. Backward Compatibility Within Family**
```python
# ✅ CORRECT: v0.02 extends v0.01 functionality
class BFSAgent(BasicAgent):  # Inherits from v0.01
    def plan_move(self, game_state: dict) -> str:
        # Enhanced implementation
        pass

# ✅ CORRECT: v0.03 extends v0.02 functionality
class AdvancedAgent(BFSAgent):  # Inherits from v0.02
    def plan_move(self, game_state: dict) -> str:
        # Further enhanced implementation
        pass

# ❌ INCORRECT: Breaking changes within same extension family
class BFSAgent:  # Should inherit from BasicAgent
    def plan_move(self, game_state: dict) -> str:
        # Completely different interface
        pass
```

### **2. Canonical Pattern Consistency**
```python
# ✅ CORRECT: Consistent factory pattern across versions
class AgentFactory:
    @classmethod
    def create(cls, algorithm: str, **kwargs):  # CANONICAL create() method
        """Create agent using canonical factory pattern."""
        pass

# ✅ CORRECT: Consistent logging pattern
class GameManager:
    def start_game(self):
        print_info(f"[GameManager] Starting game {self.game_count}")  # SUPREME_RULES compliant logging
        # Game logic here
        print_success(f"[GameManager] Game completed, score: {self.score}")  # SUPREME_RULES compliant logging

# ❌ INCORRECT: Inconsistent patterns across versions
class AgentFactory:
    def create_agent(self, algorithm: str, **kwargs):  # FORBIDDEN - not canonical
        """This violates SUPREME_RULES: factory methods must be named create()"""
        pass

# ✅ CORRECT: Consistent patterns across versions
class AgentFactory:
    @classmethod
    def create(cls, algorithm: str, **kwargs):  # CANONICAL create() method per SUPREME_RULES
        """Create agent using canonical factory pattern following SUPREME_RULES from final-decision.md."""
        pass
```

### **3. Feature Addition Strategy**
```python
# ✅ CORRECT: Additive feature enhancement
class v0_01_Agent:
    def plan_move(self, game_state: dict) -> str:
        """Basic move planning."""
        pass

class v0_02_Agent(v0_01_Agent):
    def plan_move(self, game_state: dict) -> str:
        """Enhanced move planning with multiple algorithms."""
        pass
    
    def get_algorithm_info(self) -> dict:
        """New feature in v0.02."""
        pass

class v0_03_Agent(v0_02_Agent):
    def plan_move(self, game_state: dict) -> str:
        """Further enhanced move planning with dataset generation."""
        pass
    
    def get_algorithm_info(self) -> dict:
        """Enhanced algorithm info in v0.03."""
        pass
    
    def generate_dataset(self) -> pd.DataFrame:
        """New feature in v0.03."""
        pass

# ❌ INCORRECT: Removing features in newer versions
class v0_03_Agent(v0_02_Agent):
    def plan_move(self, game_state: dict) -> str:
        """Removed get_algorithm_info method - breaks compatibility."""
        pass
    # Missing get_algorithm_info method
```

## 📋 **Version-Specific Requirements**

### **v0.01 Requirements**
- [ ] **Basic Functionality**: Core algorithm implementation
- [ ] **Simple Interface**: Basic move planning interface
- [ ] **Canonical Patterns**: Factory pattern with `create()` method
- [ ] **Simple Logging**: Print statements only (SUPREME_RULES compliance)

### **v0.02 Requirements**
- [ ] **Multi-Algorithm Support**: Multiple algorithm implementations
- [ ] **Inheritance Structure**: Extends v0.01 functionality
- [ ] **Factory Enhancement**: Enhanced factory with multiple algorithms
- [ ] **Backward Compatibility**: Maintains v0.01 interface

### **v0.03 Requirements**
- [ ] **Dataset Generation**: Generate training datasets
- [ ] **Evaluation Tools**: Performance evaluation capabilities
- [ ] **Advanced Features**: Enhanced functionality beyond v0.02
- [ ] **Backward Compatibility**: Maintains v0.02 interface

### **v0.04 Requirements (Heuristics Only)**
- [ ] **JSONL Generation**: Generate JSONL datasets for LLM fine-tuning
- [ ] **Enhanced Data Formats**: Support for multiple data formats
- [ ] **Advanced Evaluation**: Comprehensive evaluation metrics
- [ ] **Backward Compatibility**: Maintains v0.03 interface

## 🎯 **Evolution Best Practices**

### **1. Incremental Enhancement**
```python
# ✅ CORRECT: Incremental feature addition
class BaseAgent:
    def plan_move(self, game_state: dict) -> str:
        """Base move planning."""
        pass

class EnhancedAgent(BaseAgent):
    def plan_move(self, game_state: dict) -> str:
        """Enhanced move planning."""
        # Call parent method for basic functionality
        basic_move = super().plan_move(game_state)
        # Add enhancement
        enhanced_move = self._enhance_move(basic_move, game_state)
        return enhanced_move
    
    def _enhance_move(self, basic_move: str, game_state: dict) -> str:
        """Enhance basic move with additional logic."""
        pass
```

### **2. Configuration Evolution**
```python
# ✅ CORRECT: Evolving configuration with backward compatibility
class AgentConfig:
    def __init__(self, **kwargs):
        # v0.01 parameters
        self.grid_size = kwargs.get('grid_size', 10)
        
        # v0.02 parameters (new)
        self.algorithm = kwargs.get('algorithm', 'BFS')
        
        # v0.03 parameters (new)
        self.generate_dataset = kwargs.get('generate_dataset', False)
        
        print_info(f"[AgentConfig] Initialized with algorithm: {self.algorithm}")  # SUPREME_RULES compliant logging
```

### **3. Documentation Evolution**
```python
class AdvancedAgent:
    """
    Advanced agent with dataset generation capabilities.
    
    Evolution History:
    - v0.01: Basic move planning
    - v0.02: Multi-algorithm support with factory pattern
    - v0.03: Dataset generation and evaluation tools
    - v0.04: JSONL generation for LLM fine-tuning (heuristics only)
    
    Design Patterns:
    - Factory Pattern: Algorithm selection and creation
    - Template Method Pattern: Consistent move planning workflow
    - Strategy Pattern: Pluggable algorithm implementations
    """
```

## 🎓 **Educational Benefits**

### **Learning Objectives**
- **Software Evolution**: Understanding how software systems evolve
- **Backward Compatibility**: Learning to maintain compatibility
- **Version Management**: Understanding versioning strategies
- **Incremental Development**: Learning progressive enhancement

### **Best Practices**
- **Additive Changes**: Add features without removing existing ones
- **Consistent Patterns**: Maintain canonical patterns across versions
- **Clear Documentation**: Document evolution history and changes
- **Testing Strategy**: Test backward compatibility and new features

---

**Extension evolution rules ensure progressive enhancement while maintaining backward compatibility and educational value across all Snake Game AI extensions.**

## 🔗 **See Also**

- **`final-decision.md`**: SUPREME_RULES governance system and canonical standards
- **`standalone.md`**: Standalone principle and extension independence
- **`conceptual-clarity.md`**: Conceptual clarity guidelines for extensions 