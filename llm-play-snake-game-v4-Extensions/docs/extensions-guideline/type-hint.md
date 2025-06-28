# Type Hinting Standards for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and establishes comprehensive type hinting standards across all extensions.

## 🎯 **Core Philosophy: Type Safety for AI Development**

Type hints serve as both **documentation** and **development aid**, enabling better code understanding, IDE support, and automated error detection. In the Snake Game AI project, type hints are particularly valuable for ensuring consistency across multiple algorithm implementations and extensions.

### **Design Philosophy**
- **Educational Clarity**: Type hints make code self-documenting for learning purposes
- **Development Efficiency**: Enable superior IDE support and automated error detection
- **Cross-Extension Consistency**: Ensure uniform interfaces across all algorithm types
- **AI-Friendly Code**: Support AI development assistants with explicit type information

## 🏗️ **Type Hinting Standards**

### **Universal Type Definitions**
```python
# Standard type imports for all extensions
from typing import Dict, List, Tuple, Optional, Union, Any, Protocol
from typing import Callable, Iterator, Generator, TypeVar, Generic
from pathlib import Path
import numpy.typing as npt

# Project-specific type aliases
Position = Tuple[int, int]
Direction = str  # Literal["UP", "DOWN", "LEFT", "RIGHT", "NONE"]
GameState = Dict[str, Any]
NDArray = npt.NDArray[Any]
ScoreType = Union[int, float]
```

### **Base Class Type Annotations**
```python
# Core base classes with comprehensive type hints
class BaseAgent:
    """Base agent class with complete type annotations"""
    
    def __init__(self, name: str, grid_size: int) -> None:
        self.name: str = name
        self.grid_size: int = grid_size
        self.algorithm_name: str = name
    
    def plan_move(self, game_state: GameState) -> Direction:
        """Plan next move based on current game state"""
        raise NotImplementedError
    
    def reset(self) -> None:
        """Reset agent state for new game"""
        raise NotImplementedError

class BaseGameManager:
    """Base game manager with factory pattern types"""
    
    GAME_LOGIC_CLS: Optional[type] = None  # Factory type hint
    
    def __init__(self, args: Any) -> None:
        self.game_count: int = 0
        self.total_score: int = 0
        self.round_count: int = 0
        self.args: Any = args
```

## 📚 **Extension-Specific Type Patterns**

### **Heuristics Extensions**
```python
# Pathfinding-specific types
PathList = List[Direction]
SearchNode = Tuple[Position, int, Optional['SearchNode']]  # position, cost, parent
HeuristicFunction = Callable[[Position, Position], float]

class PathfindingAgent(BaseAgent):
    """Pathfinding agent with algorithm-specific types"""
    
    def __init__(self, name: str, grid_size: int, 
                 heuristic: Optional[HeuristicFunction] = None) -> None:
        super().__init__(name, grid_size)
        self.heuristic: HeuristicFunction = heuristic or self.manhattan_distance
        self.path_cache: Dict[Tuple[Position, Position], PathList] = {}
    
    def find_path(self, start: Position, goal: Position, 
                  obstacles: List[Position]) -> Optional[PathList]:
        """Find optimal path with comprehensive type annotations"""
        # Implementation with type safety
        pass
    
    @staticmethod
    def manhattan_distance(pos1: Position, pos2: Position) -> float:
        """Calculate Manhattan distance between positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
```

### **Supervised Learning Extensions**
```python
# Machine learning specific types
import torch
from torch import Tensor
import pandas as pd

ModelInput = Union[Tensor, NDArray, pd.DataFrame]
ModelOutput = Union[Tensor, NDArray, List[float]]
TrainingMetrics = Dict[str, Union[float, List[float]]]

class MLAgent(BaseAgent):
    """Machine learning agent with ML-specific types"""
    
    def __init__(self, model_type: str, input_dim: int, 
                 output_dim: int, device: str = "cpu") -> None:
        super().__init__(model_type, 10)  # grid_size default
        self.model: torch.nn.Module = self.create(input_dim, output_dim)
        self.device: torch.device = torch.device(device)
        self.training_history: TrainingMetrics = {}
    
    def predict(self, features: ModelInput) -> ModelOutput:
        """Make prediction with type-safe input/output"""
        # Implementation with type conversion handling
        pass
    
    def train_model(self, X_train: ModelInput, y_train: ModelOutput, 
                   epochs: int, batch_size: int = 32) -> TrainingMetrics:
        """Train model with comprehensive type annotations"""
        # Training implementation
        pass
```

### **Reinforcement Learning Extensions**
```python
# RL-specific types
Action = int  # Discrete action space
Reward = float
State = NDArray
Experience = Tuple[State, Action, Reward, State, bool]  # s, a, r, s', done

class RLAgent(BaseAgent):
    """RL agent with experience replay types"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 memory_size: int = 10000) -> None:
        super().__init__("RL", 10)
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.memory: List[Experience] = []
        self.epsilon: float = 1.0
    
    def select_action(self, state: State, training: bool = True) -> Action:
        """Select action with epsilon-greedy exploration"""
        # Implementation with type safety
        pass
    
    def store_experience(self, experience: Experience) -> None:
        """Store experience in replay buffer"""
        self.memory.append(experience)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
```

## 🔧 **Configuration Type Safety**

### **Configuration Classes with Types**
```python
# Configuration with comprehensive type annotations
from dataclasses import dataclass
from typing import Literal

@dataclass
class GameConfig:
    """Game configuration with type safety"""
    grid_size: int = 10
    max_steps: int = 1000
    max_games: int = 1
    visualization: bool = False
    seed: Optional[int] = None

@dataclass
class LLMConfig:
    """LLM configuration with literal types"""
    provider: Literal["gpt", "claude", "deepseek", "hunyuan", "local"] = "gpt"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 150
    timeout: float = 30.0

@dataclass
class MLConfig:
    """Machine learning configuration"""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping: bool = True
```

## 🚀 **Factory Pattern Type Safety**

### **Type-Safe Factory Implementation**
```python
from typing import TypeVar, Type, ClassVar, Protocol

AgentType = TypeVar('AgentType', bound=BaseAgent)

class AgentFactory(Protocol):
    """Protocol for type-safe agent factories"""
    
    @classmethod
    def create(cls, algorithm: str, **kwargs: Any) -> BaseAgent:
        """Create agent with type safety"""
        ...

class HeuristicAgentFactory:
    """Type-safe factory for heuristic agents"""
    
    _registry: ClassVar[Dict[str, Type[BaseAgent]]] = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, grid_size: int = 10, 
               **kwargs: Any) -> BaseAgent:
        """Create heuristic agent with full type safety"""
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        return agent_class(algorithm, grid_size, **kwargs)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """Get list of available algorithms"""
        return list(cls._registry.keys())
```

## 📊 **Data Processing Type Safety**

### **Dataset Type Annotations**
```python
# Data processing with comprehensive types
import pandas as pd
from sklearn.model_selection import train_test_split

DataSplit = Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]  # X_train, X_val, X_test, y_train, y_val, y_test

class DatasetLoader:
    """Type-safe dataset loading and processing"""
    
    def __init__(self, grid_size: int) -> None:
        self.grid_size: int = grid_size
        self.feature_columns: List[str] = self._get_feature_columns()
    
    def load_csv_dataset(self, dataset_path: Path) -> pd.DataFrame:
        """Load CSV dataset with type validation"""
        df = pd.read_csv(dataset_path)
        self._validate_dataset_schema(df)
        return df
    
    def prepare_features_and_targets(self, df: pd.DataFrame, 
                                   scale_features: bool = True) -> Tuple[NDArray, NDArray]:
        """Prepare features and targets with type safety"""
        # Implementation with type checking
        pass
    
    def split_dataset(self, X: NDArray, y: NDArray, 
                     test_size: float = 0.2, 
                     val_size: float = 0.1) -> DataSplit:
        """Split dataset with comprehensive type annotations"""
        # Implementation with proper type returns
        pass
```

## 🔍 **Type Checking and Validation**

### **Runtime Type Validation**
```python
from typing import get_type_hints, get_origin, get_args

def validate_types(func: Callable) -> Callable:
    """Decorator for runtime type checking"""
    type_hints = get_type_hints(func)
    
    def wrapper(*args, **kwargs):
        # Type validation logic
        return func(*args, **kwargs)
    
    return wrapper

# Usage example
class TypeSafeAgent(BaseAgent):
    """Agent with runtime type validation"""
    
    @validate_types
    def plan_move(self, game_state: GameState) -> Direction:
        """Type-validated move planning"""
        # Implementation with runtime type checking
        pass
```

## 📋 **Type Hinting Best Practices**

### **Required Annotations**
- [ ] **Public methods**: Always include type hints for all parameters and return values
- [ ] **Class attributes**: Annotate important instance variables in `__init__`
- [ ] **Factory methods**: Use proper generic types and protocols
- [ ] **Configuration classes**: Use dataclasses with type annotations
- [ ] **Complex data structures**: Define type aliases for clarity

### **Optional Annotations**
- **Private methods**: Annotate if complex or reused
- **Simple variables**: Use sparingly for obvious types
- **Lambda functions**: Usually not necessary
- **Temporary variables**: Only if type is non-obvious

### **Type Hint Quality Guidelines**
```python
# ✅ GOOD: Clear, specific types
def calculate_distance(pos1: Position, pos2: Position) -> float:
    """Calculate distance between two positions"""
    pass

# ✅ GOOD: Union types for multiple acceptable types
def load_config(config: Union[Path, str, Dict[str, Any]]) -> GameConfig:
    """Load configuration from various sources"""
    pass

# ❌ AVOID: Over-broad Any types
def process_data(data: Any) -> Any:
    """Too generic - not helpful"""
    pass

# ❌ AVOID: Complex nested types without aliases
def complex_function(data: Dict[str, List[Tuple[int, Optional[str]]]]) -> bool:
    """Define type alias instead"""
    pass
```

## 🔗 **Tools and Integration**

### **Recommended Type Checking Tools**
- **mypy**: Static type checking for Python
- **pyright**: Microsoft's Python type checker
- **PyCharm**: IDE with built-in type checking
- **VS Code**: Python extension with type checking support

### **Configuration for Type Checking**
```ini
# mypy.ini
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
```

---

**Type hints in the Snake Game AI project serve as both documentation and development aid, ensuring consistency across extensions while making the codebase more accessible to both human developers and AI assistants. Use type hints judiciously where they add clarity and safety, avoiding over-annotation of obvious types.**