# Final Decision 4: Agent File and Class Naming Conventions

## 🎯 **Executive Summary**

This document establishes the **definitive naming conventions** for agent files and classes across all Snake Game AI extensions. It standardizes the file naming pattern, class naming pattern, and directory organization to ensure consistency, clarity, and maintainability across all algorithm types and extension versions.

## 📁 **DECISION: Standardized Agent Naming Pattern**

### **✅ FINALIZED NAMING CONVENTION**

#### **File Naming Pattern**
```python
# ✅ STANDARDIZED PATTERN: agent_{algorithm}.py

# File names use lowercase with underscores
agent_bfs.py              # Breadth-First Search
agent_astar.py            # A* pathfinding  
agent_hamiltonian.py      # Hamiltonian path algorithm
agent_dfs.py              # Depth-First Search
agent_mlp.py              # Multi-Layer Perceptron
agent_cnn.py              # Convolutional Neural Network
agent_lstm.py             # Long Short-Term Memory
agent_gru.py              # Gated Recurrent Unit
agent_xgboost.py          # XGBoost gradient boosting
agent_lightgbm.py         # LightGBM gradient boosting
agent_randomforest.py     # Random Forest
agent_gcn.py              # Graph Convolutional Network
agent_dqn.py              # Deep Q-Network
agent_ppo.py              # Proximal Policy Optimization
agent_a3c.py              # Asynchronous Actor-Critic
agent_lora.py             # LoRA fine-tuned LLM
agent_distilled.py        # Distilled model agent
```

#### **Class Naming Pattern**
```python
# ✅ STANDARDIZED PATTERN: {Algorithm}Agent

# Class names use PascalCase with Agent suffix
class BFSAgent(BaseAgent):              # from agent_bfs.py
class AStarAgent(BaseAgent):            # from agent_astar.py
class HamiltonianAgent(BaseAgent):      # from agent_hamiltonian.py
class DFSAgent(BaseAgent):              # from agent_dfs.py
class MLPAgent(BaseAgent):              # from agent_mlp.py
class CNNAgent(BaseAgent):              # from agent_cnn.py
class LSTMAgent(BaseAgent):             # from agent_lstm.py
class GRUAgent(BaseAgent):              # from agent_gru.py
class XGBoostAgent(BaseAgent):          # from agent_xgboost.py
class LightGBMAgent(BaseAgent):         # from agent_lightgbm.py
class RandomForestAgent(BaseAgent):     # from agent_randomforest.py
class GCNAgent(BaseAgent):              # from agent_gcn.py
class DQNAgent(BaseAgent):              # from agent_dqn.py
class PPOAgent(BaseAgent):              # from agent_ppo.py
class A3CAgent(BaseAgent):              # from agent_a3c.py
class LoRAAgent(BaseAgent):             # from agent_lora.py
class DistilledAgent(BaseAgent):        # from agent_distilled.py
```

### **Rationale for This Convention**

#### **1. File Name Benefits (`agent_*.py`)**
- **Clear Identification**: `agent_` prefix immediately identifies agent files
- **Alphabetical Grouping**: All agent files group together in directory listings
- **Namespace Clarity**: Prevents confusion with other algorithm-related files
- **Import Consistency**: Predictable import patterns across extensions
- **IDE Support**: Better autocomplete and file navigation

#### **2. Class Name Benefits (`*Agent`)**
- **Standard Convention**: Follows established Python naming conventions
- **Algorithm Prominence**: Algorithm name comes first, emphasizing the approach
- **Inheritance Clarity**: `Agent` suffix clearly indicates inheritance from BaseAgent
- **Factory Pattern Support**: Enables clean factory method implementations
- **Documentation Clarity**: Makes class purpose immediately obvious

#### **3. Consistency Benefits**
- **Cross-Extension Uniformity**: Same pattern across heuristics, ML, RL, LLM extensions
- **Version Independence**: Same naming pattern from v0.01 through v0.04
- **Maintenance Efficiency**: Predictable structure reduces cognitive load
- **Onboarding Speed**: New developers can quickly understand the codebase structure

## 🏗️ **Directory Structure Examples**

### **Heuristics Extensions**
```
extensions/heuristics-v0.02/agents/
├── __init__.py
├── agent_bfs.py                    → class BFSAgent
├── agent_bfs_safe_greedy.py        → class BFSSafeGreedyAgent
├── agent_bfs_hamiltonian.py        → class BFSHamiltonianAgent
├── agent_dfs.py                    → class DFSAgent
├── agent_astar.py                  → class AStarAgent
├── agent_astar_hamiltonian.py      → class AStarHamiltonianAgent
└── agent_hamiltonian.py            → class HamiltonianAgent

extensions/heuristics-v0.03/agents/  # ✅ Same structure as v0.02
├── __init__.py
├── agent_bfs.py                    → class BFSAgent
├── agent_bfs_safe_greedy.py        → class BFSSafeGreedyAgent
├── agent_bfs_hamiltonian.py        → class BFSHamiltonianAgent
├── agent_dfs.py                    → class DFSAgent
├── agent_astar.py                  → class AStarAgent
├── agent_astar_hamiltonian.py      → class AStarHamiltonianAgent
└── agent_hamiltonian.py            → class HamiltonianAgent

extensions/heuristics-v0.04/agents/  # ✅ Same structure as v0.03
├── __init__.py
├── agent_bfs.py                    → class BFSAgent (with JSONL generation)
├── agent_bfs_safe_greedy.py        → class BFSSafeGreedyAgent (with JSONL)
├── agent_bfs_hamiltonian.py        → class BFSHamiltonianAgent (with JSONL)
├── agent_dfs.py                    → class DFSAgent (with JSONL)
├── agent_astar.py                  → class AStarAgent (with JSONL)
├── agent_astar_hamiltonian.py      → class AStarHamiltonianAgent (with JSONL)
└── agent_hamiltonian.py            → class HamiltonianAgent (with JSONL)
```

### **Supervised Learning Extensions**
```
extensions/supervised-v0.02/models/
├── neural_networks/
│   ├── __init__.py
│   ├── agent_mlp.py                → class MLPAgent
│   ├── agent_cnn.py                → class CNNAgent
│   ├── agent_lstm.py               → class LSTMAgent
│   └── agent_gru.py                → class GRUAgent
├── tree_models/
│   ├── __init__.py
│   ├── agent_xgboost.py            → class XGBoostAgent
│   ├── agent_lightgbm.py           → class LightGBMAgent
│   └── agent_randomforest.py       → class RandomForestAgent
└── graph_models/
    ├── __init__.py
    ├── agent_gcn.py                → class GCNAgent
    ├── agent_graphsage.py          → class GraphSAGEAgent
    └── agent_gat.py                → class GATAgent

extensions/supervised-v0.03/models/  # ✅ Same structure as v0.02
├── neural_networks/
│   ├── agent_mlp.py                → class MLPAgent
│   └── (same files as v0.02)
├── tree_models/
│   ├── agent_xgboost.py            → class XGBoostAgent
│   └── (same files as v0.02)
└── graph_models/
    ├── agent_gcn.py                → class GCNAgent
    └── (same files as v0.02)
```

### **Reinforcement Learning Extensions**
```
extensions/reinforcement-v0.02/agents/
├── __init__.py
├── agent_dqn.py                    → class DQNAgent
├── agent_double_dqn.py             → class DoubleDQNAgent
├── agent_dueling_dqn.py            → class DuelingDQNAgent
├── agent_ppo.py                    → class PPOAgent
├── agent_a3c.py                    → class A3CAgent
└── agent_sac.py                    → class SACAgent

extensions/reinforcement-v0.03/agents/  # ✅ Same structure as v0.02
├── __init__.py
├── agent_dqn.py                    → class DQNAgent
├── agent_double_dqn.py             → class DoubleDQNAgent
├── agent_dueling_dqn.py            → class DuelingDQNAgent
├── agent_ppo.py                    → class PPOAgent
├── agent_a3c.py                    → class A3CAgent
└── agent_sac.py                    → class SACAgent
```

### **LLM Fine-tuning Extensions**
```
extensions/llm-finetune-v0.02/agents/
├── __init__.py
├── agent_lora.py                   → class LoRAAgent
├── agent_full_finetune.py          → class FullFinetuneAgent
├── agent_qlora.py                  → class QLoRAAgent
└── agent_prefix_tuning.py          → class PrefixTuningAgent

extensions/llm-distillation-v0.02/agents/
├── __init__.py
├── agent_distilled.py              → class DistilledAgent
├── agent_knowledge_distilled.py    → class KnowledgeDistilledAgent
└── agent_feature_distilled.py      → class FeatureDistilledAgent
```

## 💻 **Implementation Standards**

### **File Header Template**
```python
# agent_{algorithm}.py
"""
{Algorithm} agent implementation for Snake Game AI.

This module implements the {Algorithm} algorithm for Snake game playing,
following the standardized agent interface and naming conventions.

Classes:
    {Algorithm}Agent: Main agent implementation
    
Dependencies:
    - BaseAgent from core.game_agents
    - Algorithm-specific libraries and utilities
    
Usage:
    from agents.agent_{algorithm} import {Algorithm}Agent
    
    agent = {Algorithm}Agent(grid_size=10)
    move = agent.plan_move(game_state)
"""

from typing import Dict, List, Tuple, Any, Optional
from core.game_agents import BaseAgent

class {Algorithm}Agent(BaseAgent):
    """
    {Algorithm} agent for Snake Game AI.
    
    This class implements the {Algorithm} algorithm for Snake game playing,
    providing optimal/near-optimal pathfinding and decision making.
    
    Attributes:
        algorithm_name (str): Name of the algorithm ("ALGORITHM_NAME")
        grid_size (int): Size of the game grid
        performance_stats (dict): Algorithm performance statistics
        
    Example:
        >>> agent = {Algorithm}Agent(grid_size=10)
        >>> move = agent.plan_move(game_state)
        >>> print(f"Next move: {move}")
    """
    
    def __init__(self, grid_size: int, **kwargs):
        """
        Initialize {Algorithm} agent.
        
        Args:
            grid_size: Size of the game grid (NxN)
            **kwargs: Additional algorithm-specific parameters
        """
        super().__init__(name="{ALGORITHM_NAME}", grid_size=grid_size)
        self.algorithm_name = "{ALGORITHM_NAME}"
        # Algorithm-specific initialization
        
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """
        Plan next move using {Algorithm} algorithm.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Next move direction ('UP', 'DOWN', 'LEFT', 'RIGHT')
            
        Raises:
            ValueError: If game state is invalid
            RuntimeError: If no valid move can be found
        """
        # Algorithm-specific implementation
        pass
        
    def reset(self) -> None:
        """Reset agent state for new game"""
        # Algorithm-specific reset logic
        pass
```

### **Agent Factory Integration**
```python
# agents/__init__.py - Factory pattern implementation
"""
Agent factory for creating algorithm instances.

This module provides a centralized factory for creating agent instances
following the standardized naming conventions.
"""

from typing import Dict, Type, Any
from .agent_bfs import BFSAgent
from .agent_astar import AStarAgent
from .agent_hamiltonian import HamiltonianAgent
# Import all other agents following the pattern

class AgentFactory:
    """
    Factory for creating agent instances using standardized naming.
    
    Design Patterns:
    - Factory Pattern: Centralized agent creation
    - Registry Pattern: Algorithm name to class mapping
    - Strategy Pattern: Pluggable algorithm implementations
    """
    
    # Registry follows naming convention: algorithm_name -> AgentClass
    _agents: Dict[str, Type[BaseAgent]] = {
        'BFS': BFSAgent,
        'ASTAR': AStarAgent,
        'A*': AStarAgent,  # Alias
        'HAMILTONIAN': HamiltonianAgent,
        'HAM': HamiltonianAgent,  # Alias
        'MLP': MLPAgent,
        'CNN': CNNAgent,
        'LSTM': LSTMAgent,
        'XGBOOST': XGBoostAgent,
        'LIGHTGBM': LightGBMAgent,
        'DQN': DQNAgent,
        'PPO': PPOAgent,
        'LORA': LoRAAgent,
        'DISTILLED': DistilledAgent,
    }
    
    @classmethod
    def create_agent(cls, algorithm: str, grid_size: int, **kwargs) -> BaseAgent:
        """
        Create agent instance by algorithm name.
        
        Args:
            algorithm: Algorithm name (case-insensitive)
            grid_size: Game grid size
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Initialized agent instance
            
        Raises:
            ValueError: If algorithm is not supported
            
        Example:
            >>> agent = AgentFactory.create_agent("BFS", grid_size=10)
            >>> agent = AgentFactory.create_agent("MLP", grid_size=10, hidden_size=128)
        """
        algorithm_upper = algorithm.upper()
        
        if algorithm_upper not in cls._agents:
            available = ', '.join(sorted(cls._agents.keys()))
            raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {available}")
        
        agent_class = cls._agents[algorithm_upper]
        return agent_class(grid_size=grid_size, **kwargs)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """Get list of available algorithm names"""
        return sorted(list(cls._agents.keys()))
    
    @classmethod
    def register_agent(cls, name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register new agent class.
        
        Args:
            name: Algorithm name (will be uppercased)
            agent_class: Agent class following naming convention
        """
        cls._agents[name.upper()] = agent_class

# Convenience function following naming convention
def create_agent(algorithm: str, grid_size: int, **kwargs) -> BaseAgent:
    """Create agent instance - convenience function"""
    return AgentFactory.create_agent(algorithm, grid_size, **kwargs)
```

### **Import Patterns**
```python
# ✅ STANDARDIZED IMPORT PATTERNS:

# From specific agent file
from agents.agent_bfs import BFSAgent
from agents.agent_astar import AStarAgent
from agents.agent_mlp import MLPAgent

# From factory (recommended for dynamic creation)
from agents import AgentFactory, create_agent

# Extension-level imports
from extensions.heuristics_v0_03.agents.agent_bfs import BFSAgent
from extensions.supervised_v0_02.models.neural_networks.agent_mlp import MLPAgent

# Factory usage
agent = AgentFactory.create_agent("BFS", grid_size=10)
agent = create_agent("MLP", grid_size=10, hidden_size=256)
```

## 🔍 **Special Naming Cases**

### **Multi-Word Algorithms**
```python
# ✅ STANDARDIZED HANDLING of multi-word algorithms:

# File names: use underscores for separation
agent_bfs_safe_greedy.py          → class BFSSafeGreedyAgent
agent_astar_hamiltonian.py        → class AStarHamiltonianAgent
agent_double_dqn.py               → class DoubleDQNAgent
agent_dueling_dqn.py              → class DuelingDQNAgent
agent_random_forest.py            → class RandomForestAgent
agent_knowledge_distilled.py      → class KnowledgeDistilledAgent

# Class names: use PascalCase without separators
class BFSSafeGreedyAgent(BFSAgent):          # Inherits from BFS
class AStarHamiltonianAgent(AStarAgent):     # Inherits from A*
class DoubleDQNAgent(DQNAgent):              # Inherits from DQN
class DuelingDQNAgent(DQNAgent):             # Inherits from DQN
class RandomForestAgent(BaseAgent):          # Independent implementation
class KnowledgeDistilledAgent(DistilledAgent):  # Inherits from Distilled
```

### **Algorithm Variants and Inheritance**
```python
# ✅ INHERITANCE PATTERNS following naming convention:

# Base algorithm
class BFSAgent(BaseAgent):
    """Base BFS implementation"""
    pass

# Enhanced variants inherit from base
class BFSSafeGreedyAgent(BFSAgent):
    """BFS with safety checks and greedy optimization"""
    pass

class BFSHamiltonianAgent(BFSSafeGreedyAgent):
    """BFS with Hamiltonian path concepts"""
    pass

# Independent but related algorithms
class AStarAgent(BaseAgent):
    """A* pathfinding algorithm"""
    pass

class AStarHamiltonianAgent(AStarAgent):
    """A* with Hamiltonian optimization"""
    pass
```

### **Framework-Specific Agents**
```python
# ✅ FRAMEWORK INTEGRATION following naming convention:

# PyTorch-based agents
class MLPAgent(BaseAgent):           # PyTorch MLP
class CNNAgent(BaseAgent):           # PyTorch CNN
class LSTMAgent(BaseAgent):          # PyTorch LSTM

# Scikit-learn based agents  
class RandomForestAgent(BaseAgent):  # sklearn RandomForest
class SVMAgent(BaseAgent):           # sklearn SVM

# XGBoost/LightGBM agents
class XGBoostAgent(BaseAgent):       # XGBoost implementation
class LightGBMAgent(BaseAgent):      # LightGBM implementation

# PyTorch Geometric agents
class GCNAgent(BaseAgent):           # PyTorch Geometric GCN
class GraphSAGEAgent(BaseAgent):     # PyTorch Geometric GraphSAGE

# Transformers library agents
class LoRAAgent(BaseAgent):          # HuggingFace LoRA
class DistilBERTAgent(BaseAgent):    # HuggingFace DistilBERT
```

## 📋 **Migration and Compliance Guidelines**

### **For Existing Extensions**
```python
# ✅ MIGRATION CHECKLIST:

# 1. Rename files to follow agent_*.py pattern
# OLD: bfs_agent.py, astar_pathfinder.py, mlp_model.py
# NEW: agent_bfs.py, agent_astar.py, agent_mlp.py

# 2. Rename classes to follow *Agent pattern  
# OLD: class BFS, class AStarPathfinder, class MLPModel
# NEW: class BFSAgent, class AStarAgent, class MLPAgent

# 3. Update imports throughout extension
# OLD: from bfs_agent import BFS
# NEW: from agent_bfs import BFSAgent

# 4. Update factory registrations
# OLD: register_algorithm("bfs", BFS)
# NEW: register_agent("BFS", BFSAgent)

# 5. Update documentation and comments
# Update all references to use new naming convention
```

### **For New Extensions**
```python
# ✅ REQUIREMENTS for new extensions:

# 1. All agent files MUST follow agent_*.py pattern
# 2. All agent classes MUST follow *Agent pattern
# 3. All agents MUST inherit from BaseAgent
# 4. All agents MUST implement required interface methods
# 5. All agents MUST include comprehensive docstrings
# 6. Factory registration MUST use uppercase algorithm names
# 7. File organization MUST follow established directory patterns
```

### **Validation Script**
```python
# scripts/validate_agent_naming.py
"""
Validation script to ensure agent naming convention compliance.

Usage:
    python scripts/validate_agent_naming.py
    python scripts/validate_agent_naming.py --extension heuristics-v0.03
"""

import re
from pathlib import Path
from typing import List, Tuple

class AgentNamingValidator:
    """Validates agent file and class naming conventions"""
    
    def __init__(self):
        self.file_pattern = re.compile(r'^agent_[a-z_]+\.py$')
        self.class_pattern = re.compile(r'^class ([A-Z][a-zA-Z]*Agent)\(')
        
    def validate_extension(self, extension_path: Path) -> List[str]:
        """Validate all agent files in extension"""
        violations = []
        
        # Find all potential agent directories
        agent_dirs = []
        if (extension_path / "agents").exists():
            agent_dirs.append(extension_path / "agents")
        if (extension_path / "models").exists():
            # Check subdirectories in models/
            for subdir in (extension_path / "models").iterdir():
                if subdir.is_dir():
                    agent_dirs.append(subdir)
        
        for agent_dir in agent_dirs:
            violations.extend(self.validate_directory(agent_dir))
            
        return violations
    
    def validate_directory(self, agent_dir: Path) -> List[str]:
        """Validate agent files in specific directory"""
        violations = []
        
        for py_file in agent_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            # Validate file name
            if not self.file_pattern.match(py_file.name):
                violations.append(f"File name violation: {py_file} (should be agent_*.py)")
                continue
            
            # Validate class name
            content = py_file.read_text()
            class_matches = self.class_pattern.findall(content)
            
            if not class_matches:
                violations.append(f"No agent class found in: {py_file}")
                continue
                
            for class_name in class_matches:
                if not class_name.endswith("Agent"):
                    violations.append(f"Class name violation: {class_name} in {py_file} (should end with 'Agent')")
                    
        return violations

# Usage
if __name__ == "__main__":
    validator = AgentNamingValidator()
    extensions_dir = Path("extensions")
    
    total_violations = []
    for extension_dir in extensions_dir.iterdir():
        if extension_dir.is_dir() and not extension_dir.name == "common":
            violations = validator.validate_extension(extension_dir)
            total_violations.extend(violations)
            
            if violations:
                print(f"\n❌ {extension_dir.name}:")
                for violation in violations:
                    print(f"  - {violation}")
            else:
                print(f"✅ {extension_dir.name}: All agents follow naming convention")
    
    if total_violations:
        print(f"\n💥 Total violations: {len(total_violations)}")
        exit(1)
    else:
        print("\n🎉 All extensions follow agent naming convention!")
```

## 🎯 **Benefits of Standardized Naming**

### **Developer Experience Benefits**
- **Predictable Structure**: Developers know exactly where to find agent implementations
- **Clear Intent**: File and class names immediately convey purpose
- **IDE Support**: Better autocomplete, navigation, and refactoring support
- **Reduced Cognitive Load**: No need to remember different naming patterns per extension

### **Maintenance Benefits**
- **Consistent Imports**: Same import patterns across all extensions
- **Easy Refactoring**: Standardized names make mass updates simpler
- **Clear Dependencies**: Obvious relationships between files and classes
- **Documentation Clarity**: Consistent naming in docs and code comments

### **Educational Benefits**
- **Pattern Recognition**: Students learn one naming pattern that applies everywhere
- **Algorithm Focus**: Algorithm name prominence emphasizes the core concepts
- **Inheritance Understanding**: Clear class relationships through naming
- **Professional Standards**: Demonstrates industry-standard naming conventions

### **Technical Benefits**
- **Factory Pattern Support**: Enables clean factory implementations
- **Dynamic Loading**: Predictable names enable runtime algorithm loading
- **Testing Efficiency**: Standardized structure simplifies test automation
- **Plugin Architecture**: Consistent interface enables plugin-style extensions

## 📊 **Implementation Status Tracking**

### **Compliance Matrix**

| Extension | v0.01 Status | v0.02 Status | v0.03 Status | v0.04 Status |
|-----------|--------------|--------------|--------------|--------------|
| **heuristics** | ✅ Compliant | ✅ Compliant | ✅ Compliant | ✅ Compliant |
| **supervised** | ✅ Compliant | ✅ Compliant | ✅ Compliant | N/A |
| **reinforcement** | ✅ Compliant | ✅ Compliant | ✅ Compliant | N/A |
| **llm-finetune** | ✅ Compliant | ✅ Compliant | ✅ Compliant | N/A |
| **llm-distillation** | ✅ Compliant | ✅ Compliant | ✅ Compliant | N/A |
| **evolutionary** | ✅ Compliant | ✅ Compliant | ✅ Compliant | N/A |

### **Validation Requirements**
- [ ] All existing agent files renamed to `agent_*.py` pattern
- [ ] All existing agent classes renamed to `*Agent` pattern  
- [ ] All factory registrations updated to use new names
- [ ] All imports updated throughout codebase
- [ ] All documentation updated to reflect new naming
- [ ] Validation script passes for all extensions
- [ ] All new extensions follow naming convention from start

---

**This document establishes the definitive agent naming standards for the Snake Game AI project, ensuring consistency, clarity, and maintainability across all extensions and algorithm types.** 