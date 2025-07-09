# Standalone Principle for Extensions

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for standalone principles and supplements the _Final Decision Series_ (`` → `final-decision.md`).

> **See also:** `core.md`, `project-structure-plan.md`, `final-decision.md`, `config.md`.

## 🎯 **Core Philosophy: The Golden Rule of Modularity**

This document defines the single most important architectural principle for extensions: the **Standalone Principle**. It is a non-negotiable rule that governs the entire `extensions/` directory and ensures that the project remains modular, maintainable, and educational, strictly following SUPREME_RULES from `final-decision.md`.

> **The Golden Rule:**
> **An extension directory, when combined with the `extensions/common/` directory, must be a completely self-contained, standalone unit.**

This means you should be able to delete all other extension folders, and the remaining extension would still function perfectly.

## 🚧 **Defining the Boundary**

Think of each extension as being inside a protective "bubble." The only things allowed to cross into this bubble are dependencies on the **core framework** and the shared **common utilities**.

## 🚫 **Forbidden vs. ✅ Allowed Imports**

This principle translates into a very clear set of rules about what an extension can and cannot import.

### **Absolutely Forbidden**

An extension **must never** import code from another extension. This is the most critical rule.

```python
# ❌ COMPLETELY FORBIDDEN IN: extensions/supervised-v0.02/main.py

# Cannot import from another extension type
from extensions.heuristics_v0_03.agents import BFSAgent

# Cannot import from a different version of the same extension
from extensions.supervised_v0_01.models import OldMLPAgent

# Cannot import from a sibling extension, no matter how helpful
from extensions.reinforcement_v0_01 import DQNAgent
```

### **Perfectly Allowed**

An extension is **expected** to import from the core framework and the common utilities.

```python
# ✅ PERFECTLY ALLOWED IN: extensions/supervised-v0.02/main.py

# Import the foundational base classes
from core.game_manager import BaseGameManager
from core.game_data import BaseGameData

# Import the shared utility belt
from extensions.common.path_utils import ensure_project_root
from extensions.common.dataset_utils import load_heuristic_dataset
```

## 🧰 **The Role of the `extensions/common/` Directory**

The `common/` directory is designed to support the Standalone Principle, not violate it. It is a **shared utility belt**, not a shared brain.

*   **It provides TOOLS, not CONCEPTS.** `common/` contains helper functions for paths, file I/O, data validation, and other tasks that are conceptually neutral.
*   **It has no algorithmic knowledge.** The `common/` directory knows nothing about BFS, A*, DQN, or neural networks. It only knows how to handle common data structures and files.
*   **It reduces boilerplate, not thinking.** Using `common/` prevents you from rewriting the same `ensure_project_root()` function in every extension, but it does not provide the core logic for your extension.

By centralizing these non-essential utilities, the `common/` directory allows each extension's code to focus purely on what makes it unique: its algorithms and its approach to solving the Snake game.

---

> **Adherence to the Standalone Principle is what makes our project scalable and understandable. It allows for independent development, isolated testing, and clear conceptual separation, which are essential for long-term success.**

# Standalone Extension Architecture

This document provides guidelines for implementing the standalone principle across all extensions in the Snake Game AI project, ensuring modularity, independence, and clear conceptual boundaries.

## 🏗️ **Architecture Overview**

### **Standalone Units**
```
Standalone Unit = Extension + Common Folder

Examples:
- heuristics-v0.01 + common = Standalone
- heuristics-v0.02 + common = Standalone  
- heuristics-v0.03 + common = Standalone
- heuristics-v0.04 + common = Standalone
- supervised-v0.01 + common = Standalone
- supervised-v0.02 + common = Standalone
- supervised-v0.03 + common = Standalone
- reinforcement-v0.01 + common = Standalone
- reinforcement-v0.02 + common = Standalone
└── reinforcement-v0.03 + common = Standalone
```

### **Directory Structure**
```
extensions/
├── common/                    # Shared utilities (part of every standalone unit)
│   ├── __init__.py
│   ├── config/               # Configuration constants folder
│   ├── csv_schema.py         # Dataset schema definitions
│   ├── dataset_loader.py     # Dataset loading utilities
│   ├── file_utils.py         # File management utilities
│   ├── grid_utils.py         # Grid size validation and utilities
│   ├── path_utils.py         # Path management utilities
│   └── validation/           # Validation functions and utilities
│       ├── __init__.py
│       ├── dataset_validator.py  # Dataset format validation
│       ├── model_validator.py    # Model format validation
│       └── path_validator.py     # Path structure validation
├── heuristics-v0.01/         # Standalone with common
├── heuristics-v0.02/         # Standalone with common
├── heuristics-v0.03/         # Standalone with common
├── heuristics-v0.04/         # Standalone with common
├── supervised-v0.01/         # Standalone with common
├── supervised-v0.02/         # Standalone with common
├── supervised-v0.03/         # Standalone with common
├── reinforcement-v0.01/      # Standalone with common
├── reinforcement-v0.02/      # Standalone with common
└── reinforcement-v0.03/      # Standalone with common
```

## 🚫 **Forbidden Dependencies**

### **What Extensions CANNOT Do**
```python
# ❌ FORBIDDEN: Cross-extension imports
from heuristics_v0_01 import BFSAgent
from heuristics_v0_02 import AStarAgent
from supervised_v0_01 import MLPAgent
from reinforcement_v0_01 import DQNAgent

# ❌ FORBIDDEN: Version-specific imports
from extensions.heuristics_v0_02.agents import agent_bfs
from extensions.supervised_v0_01 import neural_agent

# ❌ FORBIDDEN: Direct extension-to-extension communication
heuristic_result = heuristics_v0_02.run_algorithm()
ml_model = supervised_v0_01.train_model(heuristic_result)
```

### **What Extensions CAN Do**
```python
# ✅ ALLOWED: Core framework imports
from core.game_manager import BaseGameManager
from core.game_logic import BaseGameLogic
from core.game_data import BaseGameData

# ✅ ALLOWED: Common utilities imports
from extensions.common.config import get_grid_size_config
from extensions.common.dataset_loader import load_csv_dataset
from extensions.common.csv_schema import generate_csv_schema

# Simple validation instead of complex utils
def validate_game_state(state):
    """Simple validation function"""
    if not state or 'snake_positions' not in state:
        raise ValueError("Invalid game state")
    return True
```

## 📁 **Common Folder Design**

### **Purpose of Common Folder**
The `common/` folder serves as a **shared utility library** that enhances the standalone principle rather than violating it:

### **Design Patterns in Common**
- **Utility Pattern**: Stateless helper functions
- **Template Pattern**: Common workflows
- **Factory Pattern**: Creation utilities for common objects
- **Configuration Pattern**: Centralized settings management
- **Validation Pattern**: Data integrity and format checking

## 🔄 **Standalone Workflow Examples**

### **Heuristics Extension Workflow**
```python
# heuristics-v0.03/main.py
"""
Standalone heuristics extension with common utilities

This extension + common folder = Complete standalone unit
"""

# ✅ Core framework imports (allowed)
from core.game_manager import BaseGameManager
from core.game_logic import BaseGameLogic

# ✅ Common utilities imports (allowed)
from extensions.common.config import ExtensionConfig
from extensions.common.csv_schema import CSVSchemaManager

# ✅ Extension-specific imports (allowed)
from .agents.agent_bfs import BFSAgent
from .agents.agent_astar import AStarAgent
from .game_logic import HeuristicGameLogic

class HeuristicGameManager(BaseGameManager):
    """
    Standalone heuristic game manager
    
    Uses only:
    - Core framework components
    - Common utilities
    - Extension-specific components
    """
    
    def __init__(self, algorithm: str, grid_size: int):
        # Validate using common utilities
        if grid_size < 5 or grid_size > 50:
            raise ValueError(f"Invalid grid size: {grid_size}")
        
        super().__init__(grid_size=grid_size)
        
        # Simple logging
        print_info(f"[HeuristicGameManager] Initializing with algorithm: {algorithm}, grid: {grid_size}x{grid_size}")  # SUPREME_RULES compliant logging
        
        # Create agent using extension-specific factory
        self.agent = self.create(algorithm)
    
    def create(self, algorithm: str):
        """Create agent using extension-specific factory"""
        agents = {
            'BFS': BFSAgent,
            'ASTAR': AStarAgent
        }
        
        if algorithm not in agents:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return agents[algorithm]()
    
    def generate_dataset(self, num_games: int) -> str:
        """Generate dataset using common schema"""
        
        # Run games and collect data
        game_data = self._run_games(num_games)
        
        # Convert to standard CSV format using common utilities
        csv_data = CSVSchemaManager.convert_to_standard_format(game_data)
        
        # Save to standard location using simple path construction
        base_path = "logs/extensions/datasets"
        output_path = f"{base_path}/grid-size-{self.grid_size}/heuristics_v0.03_{timestamp}"
        
        csv_file = f"{output_path}/tabular_{self.agent.name.lower()}_data.csv"
        csv_data.to_csv(csv_file, index=False)
        
        print_success(f"[HeuristicGameManager] Dataset saved to: {csv_file}")  # SUPREME_RULES compliant logging
        return csv_file
```

### **Supervised Learning Extension Workflow**
```python
# supervised-v0.03/main.py
"""
Standalone supervised learning extension with common utilities

This extension + common folder = Complete standalone unit
"""

# ✅ Core framework imports (allowed)
from core.game_manager import BaseGameManager

# ✅ Common utilities imports (allowed)
from extensions.common.dataset_loader import DatasetLoader

# ✅ Extension-specific constants
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
VALID_MODEL_TYPES = ["MLP", "CNN", "XGBOOST", "LIGHTGBM"]

# ✅ Extension-specific imports (allowed)
from .models.neural_networks.agent_mlp import MLPAgent
from .models.tree_models.agent_xgboost import XGBoostAgent
from .training.train_neural import NeuralTrainer

class SupervisedGameManager(BaseGameManager):
    """
    Standalone supervised learning game manager
    
    Loads datasets generated by heuristics extensions
    but maintains complete independence
    """
    
    def __init__(self, model_type: str, grid_size: int):
        # Simple validation
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model type: {model_type}. Supported: {VALID_MODEL_TYPES}")
        if grid_size < 5 or grid_size > 50:
            raise ValueError(f"Invalid grid size: {grid_size}")
        
        super().__init__(grid_size=grid_size)
        
        # Simple logging
        print_info(f"[SupervisedGameManager] Initializing {model_type} model for {grid_size}x{grid_size} grid")  # SUPREME_RULES compliant logging
        
        # Load training data using common utilities
        self.dataset_loader = DatasetLoader(grid_size)
        self.model = self.create(model_type)
    
    def train_model(self, algorithms: List[str] = None):
        """Train model using datasets from heuristic extensions"""
        
        # Load datasets using common utilities (standalone operation)
        training_data = self.dataset_loader.load_heuristic_datasets(algorithms)
        
        # Prepare data using common utilities
        X, y, label_encoder = self.dataset_loader.prepare_for_training(training_data)
        
        # Train using extension-specific logic
        self.model.train(X, y)
        
        # Save model to standard location using simple path construction
        base_path = "logs/extensions/models"
        model_path = f"{base_path}/grid-size-{self.grid_size}/supervised_v0.03_{timestamp}"
        
        self.model.save(f"{model_path}/{self.model.name}_model.pkl")
        print_success(f"[SupervisedGameManager] Model saved to: {model_path}")  # SUPREME_RULES compliant logging
```

## 📊 **Standalone Validation**

### **Validation Checklist**
Each extension must pass the standalone validation:

```python
def validate_extension_standalone(extension_path: str) -> bool:
    """Simple validation function for standalone principles"""
            print_info(f"[Validator] Checking extension: {extension_path}")  # SUPREME_RULES compliant logging
    
    # Simple checks using basic file operations
    python_files = [f for f in os.listdir(extension_path) if f.endswith('.py')]
    
    for file_path in python_files:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # Simple pattern checks
            forbidden_patterns = ['heuristics_v0_', 'supervised_v0_', 'reinforcement_v0_']
            for pattern in forbidden_patterns:
                if pattern in content:
                    print_error(f"[Validator] Found forbidden import: {pattern} in {file_path}")  # SUPREME_RULES compliant logging
                    return False
    
            print_success(f"[Validator] Extension {extension_path} is standalone compliant")  # SUPREME_RULES compliant logging
    return True
```

## 🎯 **Benefits of Standalone Architecture**

### **Educational Benefits**
- **Clear Conceptual Boundaries**: Each extension represents one AI approach
- **Independent Study**: Students can learn extensions in isolation
- **Progressive Learning**: Natural progression from simple to complex
- **Comparative Analysis**: Easy to compare different approaches

### **Technical Benefits**
- **Maintenance Simplicity**: Changes don't cascade across extensions
- **Deployment Flexibility**: Extensions can be deployed independently
- **Testing Isolation**: Bugs in one extension don't affect others
- **Development Speed**: Teams can work on extensions in parallel

### **Research Benefits**
- **Experimental Safety**: New features remain contained
- **Version Control**: Different versions coexist without conflicts
- **Reproducibility**: Standalone units ensure reproducible results
- **Modularity**: Easy to add new extensions without breaking existing ones

## 🔗 **See Also**

- **`final-decision.md`**: final-decision.md governance system
- **`core.md`**: Base class architecture and inheritance patterns
- **`project-structure-plan.md`**: Project structure and organization
- **`config.md`**: Configuration architecture standards

---

**The standalone principle ensures that each extension represents a clear, independent implementation of an AI approach while leveraging shared utilities for consistency and efficiency. This architecture promotes educational clarity, technical maintainability, and research flexibility.**

