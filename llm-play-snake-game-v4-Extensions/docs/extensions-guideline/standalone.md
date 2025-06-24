# Standalone Extension Architecture for Snake Game AI

This document provides comprehensive guidelines for implementing the standalone principle across all extensions in the Snake Game AI project, ensuring modularity, independence, and clear conceptual boundaries.

## 🎯 **Core Standalone Principle**

The standalone principle is fundamental to the Snake Game AI extension architecture:

### **Definition of Standalone**
- **Extension + Common = Standalone**: Each extension (`blabla-v0.0N`) combined with the `common/` folder forms a completely self-contained unit
- **No Cross-Extension Dependencies**: Extensions cannot import from other extensions
- **Conceptual Clarity**: Each extension represents a distinct AI approach (heuristics, RL, supervised learning, etc.)
- **Version Independence**: Different versions of the same extension are completely independent . # TODO: Though, ideally, for things like Supervized, Heuristics, Reinforcement Learning,StableBaseline, the agents folder between v0.02 and v0.03 should be exactly the same. This forces us to conceive really good code architecture in v0.02 in the first place. # TODO

### **Why Standalone Architecture?**
- **Educational Clarity**: Each extension showcases a specific AI concept without confusion
- **Maintenance Simplicity**: Changes to one extension don't break others
- **Deployment Flexibility**: Extensions can be deployed independently
- **Learning Progression**: Students can study extensions in isolation
- **Research Isolation**: Experimental features remain contained

> **Important — Authoritative Reference:** This guide is **supplementary** to the _Final Decision Series_ (`final-decision-0` → `final-decision-10`). **If any statement here conflicts with a Final Decision document, the latter always prevails.**

# The Standalone Principle

## 🎯 **Core Philosophy: The Golden Rule of Modularity**

This document defines the single most important architectural principle for extensions: the **Standalone Principle**. It is a non-negotiable rule that governs the entire `extensions/` directory and ensures that the project remains modular, maintainable, and educational.

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
- reinforcement-v0.03 + common = Standalone
```

### **Directory Structure**
```
extensions/
├── common/                    # Shared utilities (part of every standalone unit)
│   ├── __init__.py
│   ├── config.py             # Common configuration utilities
│   ├── csv_schema.py         # Dataset schema definitions
│   ├── dataset_loader.py     # Dataset loading utilities
│   ├── file_utils.py         # File management utilities
│   ├── grid_utils.py         # Grid size validation and utilities
│   ├── path_utils.py         # Path management utilities
│   └── validation_utils.py   # Common validation functions # TODO: or, maybe, make this into a folder "validation" because we will have a lot of validation functionalities. Validation is important for us to make sure that we are not going everywhere which knowing that we are already doing things very wrong.
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
from extensions.common.validation_utils import validate_game_state

```

## 📁 **Common Folder Design**

### **Purpose of Common Folder**
The `common/` folder serves as a **shared utility library** that enhances the standalone principle rather than violating it:

### **Design Patterns in Common**
- **Utility Pattern**: Stateless helper functions
- **Template Pattern**: Common workflows
- TODO: maybe some other stuffs?

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
from extensions.common.logging_utils import setup_extension_logging

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
        if not ExtensionConfig.validate_grid_size(grid_size):
            raise ValueError(f"Invalid grid size: {grid_size}")
        
        super().__init__(grid_size=grid_size)
        
        # Setup logging using common utilities
        self.logger = setup_extension_logging('heuristics-v0.03')
        
        # Create agent using extension-specific factory
        self.agent = self._create_agent(algorithm)
    
    def _create_agent(self, algorithm: str):
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
        
        # Save to standard location using common utilities
        output_path = ExtensionConfig.create_grid_size_path(
            ExtensionConfig.get_dataset_base_path(),
            self.grid_size
        )
        
        csv_file = f"{output_path}/tabular_{self.agent.name.lower()}_data.csv"
        csv_data.to_csv(csv_file, index=False)
        
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
from extensions.common.config import ExtensionConfig
from extensions.common.validation_utils import validate_model_config

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
        # Validate using common utilities
        if not ExtensionConfig.validate_grid_size(grid_size):
            raise ValueError(f"Invalid grid size: {grid_size}")
        
        super().__init__(grid_size=grid_size)
        
        # Load training data using common utilities
        self.dataset_loader = DatasetLoader(grid_size)
        self.model = self._create_model(model_type)
    
    def train_model(self, algorithms: List[str] = None):
        """Train model using datasets from heuristic extensions"""
        
        # Load datasets using common utilities (standalone operation)
        training_data = self.dataset_loader.load_heuristic_datasets(algorithms)
        
        # Prepare data using common utilities
        X, y, label_encoder = self.dataset_loader.prepare_for_training(training_data)
        
        # Train using extension-specific logic
        self.model.train(X, y)
        
        # Save model to standard location using common utilities
        model_path = ExtensionConfig.create_grid_size_path(
            ExtensionConfig.get_models_base_path(),
            self.grid_size
        )
        
        self.model.save(f"{model_path}/{self.model.name}_model.pkl")
```

## 📊 **Standalone Validation**

### **Validation Checklist**
Each extension must pass the standalone validation:

```python
# extensions/common/validation_utils.py # TODO: or, maybe, make this into a folder "validation" because we will have a lot of validation functionalities. Validation is important for us to make sure that we are not going everywhere which knowing that we are already doing things very wrong.
class StandaloneValidator:
    """
    Validates that extensions follow standalone principles
    
    Features:
    - Import dependency analysis
    - Cross-extension reference detection
    - Common folder usage validation
    - Version independence verification
    """
    
    @staticmethod
    def validate_extension(extension_path: str) -> Dict[str, bool]:
        """Validate extension follows standalone principles"""
        
        results = {
            'no_cross_extension_imports': False,
            'uses_common_utilities': False,
            'no_version_specific_code': False,
            'proper_core_usage': False
        }
        
        # Analyze Python files in extension
        python_files = glob.glob(f"{extension_path}/**/*.py", recursive=True)
        
        for file_path in python_files:
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Check for forbidden imports
                if StandaloneValidator._has_cross_extension_imports(content):
                    results['no_cross_extension_imports'] = False
                    return results
                
                # Check for common utilities usage
                if StandaloneValidator._uses_common_utilities(content):
                    results['uses_common_utilities'] = True
                
                # Check for proper core usage
                if StandaloneValidator._uses_core_framework(content):
                    results['proper_core_usage'] = True
        
        results['no_cross_extension_imports'] = True
        results['no_version_specific_code'] = True
        
        return results
    
    @staticmethod
    def _has_cross_extension_imports(content: str) -> bool:
        """Check for forbidden cross-extension imports"""
        
        forbidden_patterns = [
            r'from\s+heuristics_v0_\d+',
            r'from\s+supervised_v0_\d+',
            r'from\s+reinforcement_v0_\d+',
            r'from\s+extensions\.heuristics-v0\.\d+',
            r'from\s+extensions\.supervised-v0\.\d+',
            r'from\s+extensions\.reinforcement-v0\.\d+'
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, content):
                return True
        
        return False
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

---

**The standalone principle ensures that each extension represents a clear, independent implementation of an AI approach while leveraging shared utilities for consistency and efficiency. This architecture promotes educational clarity, technical maintainability, and research flexibility.**

