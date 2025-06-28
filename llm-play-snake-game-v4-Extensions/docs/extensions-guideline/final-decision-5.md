# Final Decision 5: Extension Directory Structure & Evolution Standards


## 🎯 **Core Philosophy: Progressive Extension Evolution**



```python
# ✅ CORRECT: Simple logging as per SUPREME_RULE NO.3
def setup_extension_environment(extension_path: str, version: str):
    """Setup extension environment with proper structure"""
    print(f"[ExtensionManager] Setting up {extension_path} v{version}")  # SUPREME_RULE NO.3
    
    # Validate extension structure
    required_files = ["__init__.py", "game_logic.py", "game_manager.py"]
    for file in required_files:
        if not os.path.exists(os.path.join(extension_path, file)):
            print(f"[ExtensionManager] Missing required file: {file}")  # SUPREME_RULE NO.3
            return False
    
    # Version-specific validation
    if version == "0.02" and not os.path.exists(os.path.join(extension_path, "agents")):
        print(f"[ExtensionManager] v0.02 requires agents/ directory")  # SUPREME_RULE NO.3
        return False
    
    print(f"[ExtensionManager] Extension structure validated successfully")  # SUPREME_RULE NO.3
    return True

def create_extension_directory(extension_type: str, version: str):
    """Create new extension directory with proper structure"""
    extension_name = f"{extension_type}-v{version}"
    extension_path = f"extensions/{extension_name}"
    
    os.makedirs(extension_path, exist_ok=True)
    print(f"[ExtensionManager] Created extension directory: {extension_path}")  # SUPREME_RULE NO.3
    
    # Create version-specific structure
    if version == "0.02":
        os.makedirs(os.path.join(extension_path, "agents"), exist_ok=True)
        print(f"[ExtensionManager] Created agents/ directory for v0.02")  # SUPREME_RULE NO.3
    
    return extension_path
```

The extension directory structure implements a sophisticated evolution model that ensures consistency, educational progression, and maintainability across all Snake Game AI extensions.

## 🎯 **Executive Summary**

This document establishes the **definitive directory structure templates** for all Snake Game AI extensions, ensuring consistency across different algorithm types (heuristics, supervised learning, reinforcement learning, evolutionary algorithms, LLM fine-tuning, distillation, etc.) while supporting the natural evolution from v0.01 to v0.03.

## 📁 **Universal Extension Template**

### **Base Template Pattern**

All extensions follow the same structural evolution:

```
extensions/{algorithm}-v{version}/
├── __init__.py                    # Extension package initialization
├── main.py                        # Primary entry point (CLI for v0.01-v0.02, may be minimal for v0.03)
├── game_logic.py                  # Algorithm-specific game logic (extends BaseGameLogic)
├── game_manager.py                # Algorithm-specific manager (extends BaseGameManager)
├── game_data.py                   # Algorithm-specific data handling (v0.02+)
├── {algorithm}_config.py          # Extension-specific configuration (v0.03+)
├── replay_engine.py               # Replay processing (v0.03+)
├── replay_gui.py                  # PyGame replay interface (v0.03+)
└── [version-specific additions]    # See detailed templates below
```

## 🌱 **v0.01 Template: Proof of Concept**

### **Purpose**: Single algorithm, minimal complexity, proof that base classes work

```
extensions/{algorithm}-v0.01/
├── __init__.py                    # Package initialization
├── main.py                        # Simple entry point, minimal arguments
├── agent_{primary}.py             # Single primary algorithm implementation
├── game_logic.py                  # Extends BaseGameLogic for algorithm
├── game_manager.py                # Extends BaseGameManager for algorithm
└── README.md                      # Basic documentation
```

### **Characteristics**:
- **Single algorithm only** (e.g., BFS for heuristics, MLP for supervised)
- **No command-line arguments** for algorithm selection
- **No GUI components** - console output only
- **Direct file placement** - no organized subfolders
- **Proof of concept** - demonstrates base class integration

### **Examples**:

#### **Heuristics v0.01**
```
extensions/heuristics-v0.01/
├── __init__.py
├── main.py                        # python main.py (no args)
├── agent_bfs.py                   # BFSAgent class
├── game_logic.py                  # HeuristicGameLogic(BaseGameLogic)
├── game_manager.py                # HeuristicGameManager(BaseGameManager)
└── README.md
```

#### **Supervised v0.01**
```
extensions/supervised-v0.01/
├── __init__.py
├── main.py                        # python main.py (minimal args)
├── agent_neural.py                # MLP, CNN, LSTM implementations
├── train.py                       # Training script
├── game_logic.py                  # NeuralGameLogic(BaseGameLogic)
├── game_manager.py                # NeuralGameManager(BaseGameManager)
└── README.md
```


#### **Reinforcement v0.01**
```
extensions/reinforcement-v0.01/
├── __init__.py
├── main.py                        # python main.py (basic DQN)
├── agent_dqn.py                   # DQN implementation
├── train.py                       # RL training script
├── game_logic.py                  # RLGameLogic(BaseGameLogic)
├── game_manager.py                # RLGameManager(BaseGameManager)
└── README.md
```

## 🚀 **v0.02 Template: Multi-Algorithm Expansion**

### **Purpose**: Multiple algorithms, organized structure, algorithm selection

```
extensions/{algorithm}-v0.02/
├── __init__.py                    # Enhanced package initialization
├── main.py                        # Multi-algorithm entry point with --algorithm arg
├── game_logic.py                  # Enhanced algorithm-specific logic
├── game_manager.py                # Multi-algorithm manager with factory patterns
├── game_data.py                   # Algorithm-specific data extensions
├── agents/                        # ✨ NEW: Organized algorithm implementations
│   ├── __init__.py               # Agent factory and exports
│   ├── agent_{type1}.py          # First algorithm variant
│   ├── agent_{type2}.py          # Second algorithm variant
│   ├── agent_{type3}.py          # Third algorithm variant
│   └── [additional agents]       # More algorithms as needed
└── README.md                      # Enhanced documentation
```

### **Characteristics**:
- **Multiple algorithms** within the same domain
- **`--algorithm` command-line argument** for selection
- **Organized agents/ folder** with factory patterns
- **Enhanced base class usage** with more sophisticated patterns
- **No GUI yet** - still CLI only


## 🌐 **v0.03 Template: Web Interface & Dataset Generation**

### **Purpose**: Streamlit web interface, dataset generation, replay capabilities

```
extensions/{algorithm}-v0.03/
├── __init__.py                    # Package initialization
├── app.py                         # ✨ NEW: Streamlit web interface
├── {algorithm}_config.py          # ✨ NEW: Renamed from config.py for clarity
├── game_logic.py                  # Enhanced with dataset generation
├── game_manager.py                # Enhanced manager
├── game_data.py                   # Enhanced data handling
├── replay_engine.py               # ✨ NEW: Replay processing engine
├── replay_gui.py                  # ✨ NEW: PyGame replay interface
├── agents/                        # ✨ COPIED: Exact copy from v0.02
│   ├── __init__.py               # Same as v0.02
│   ├── agent_{type1}.py          # Same as v0.02
│   ├── agent_{type2}.py          # Same as v0.02
│   └── [all other agents]        # Same as v0.02
├── dashboard/                     # ✨ NEW: Streamlit tabs
│   ├── __init__.py
│   ├── tab_{algorithm1}.py       # Tab for each algorithm
│   ├── tab_{algorithm2}.py
│   └── tab_comparison.py         # Algorithm comparison tab
├── scripts/                       # ✨ NEW: CLI and automation
│   ├── __init__.py
│   ├── main.py                   # Moved from root (enhanced CLI)
│   ├── generate_dataset.py       # Dataset generation CLI
│   ├── replay.py                 # PyGame replay script
│   └── replay_web.py             # Flask web replay
└── README.md                      # Comprehensive documentation
```

### **Characteristics**:
- **Streamlit app.py** with algorithm tabs for launching scripts with adjustable parameters
- **Dataset generation** capabilities for other extensions to consume
- **PyGame + Flask web replay** systems
- **Enhanced configuration** management
- **Agents folder exactly copied** from v0.02 (no changes to algorithm implementations)

### **Web Interface Pattern**:

#### **Streamlit App Structure (OOP)**
```python
# app.py - Universal pattern for all extensions
import streamlit as st
from abc import ABC, abstractmethod
from typing import List

class BaseExtensionApp(ABC):
    """
    Base class for all extension Streamlit applications
    
    Design Patterns:
    - Template Method Pattern: Defines common app structure
    - Strategy Pattern: Algorithm-specific implementations
    - Factory Pattern: Tab creation based on algorithms
    """
    
    def __init__(self):
        self.setup_page_config()
        self.main()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=f"{self.get_extension_name()} Snake AI",
            page_icon="🐍",
            layout="wide"
        )
    
    @abstractmethod
    def get_extension_name(self) -> str:
        """Return extension name for display"""
        raise NotImplementedError("Subclasses must implement get_extension_name")
    
    @abstractmethod
    def get_algorithms(self) -> List[str]:
        """Return list of available algorithms"""
        raise NotImplementedError("Subclasses must implement get_algorithms")
    
    def main(self):
        """Main application flow"""
        st.title(f"{self.get_extension_name()} Snake Game AI - v0.03")
        
        # Create tabs for each algorithm
        algorithms = self.get_algorithms()
        tabs = st.tabs(algorithms)
        
        for tab, algorithm in zip(tabs, algorithms):
            with tab:
                self.render_algorithm_interface(algorithm)
    
    @abstractmethod
    def render_algorithm_interface(self, algorithm: str):
        """Render interface for specific algorithm"""
        raise NotImplementedError("Subclasses must implement render_algorithm_interface")
    
    def render_common_controls(self, algorithm: str):
        """Common controls for all algorithms"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_games = st.slider("Max Games", 1, 100, 10, key=f"{algorithm}_games")
        with col2:
            grid_size = st.selectbox("Grid Size", [8, 10, 12, 16, 20], index=1, key=f"{algorithm}_grid")
        with col3:
            verbose = st.checkbox("Verbose Output", key=f"{algorithm}_verbose")
        
        return max_games, grid_size, verbose

# Extension-specific implementations
class HeuristicSnakeApp(BaseExtensionApp):
    def get_extension_name(self) -> str:
        return "Heuristic"
    
    def get_algorithms(self) -> List[str]:
        return ["BFS", "BFS Safe Greedy", "BFS Hamiltonian", "DFS", "A*", "A* Hamiltonian", "Hamiltonian"]
    
    def render_algorithm_interface(self, algorithm: str):
        st.subheader(f"{algorithm} Algorithm")
        max_games, grid_size, verbose = self.render_common_controls(algorithm)
        
        # Algorithm-specific controls
        if algorithm.startswith("A*"):
            heuristic = st.selectbox("Heuristic Function", ["Manhattan", "Euclidean"], key=f"{algorithm}_heuristic")
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button(f"Run {algorithm}", key=f"{algorithm}_run"):
                self.run_algorithm(algorithm, max_games, grid_size, verbose)
        with col2:
            if st.button(f"Generate Dataset", key=f"{algorithm}_dataset"):
                self.generate_dataset(algorithm, max_games, grid_size)
        with col3:
            if st.button(f"Replay (PyGame)", key=f"{algorithm}_pygame"):
                self.replay_pygame(algorithm, grid_size)
        with col4:
            if st.button(f"Replay (Web)", key=f"{algorithm}_web"):
                self.replay_web(algorithm, grid_size)

if __name__ == "__main__":
    HeuristicSnakeApp()
```

### **Examples**:

#### **Heuristics v0.03**
```
extensions/heuristics-v0.03/
├── __init__.py
├── app.py                         # HeuristicSnakeApp(BaseExtensionApp)
├── heuristic_config.py            # Heuristic-specific configuration
├── game_logic.py                  # Enhanced with CSV dataset generation
├── game_manager.py                # Enhanced manager
├── game_data.py                   # Enhanced data with export capabilities
├── replay_engine.py               # Heuristic replay processing
├── replay_gui.py                  # PyGame heuristic visualization
├── agents/                        # Exact copy from v0.02
│   ├── __init__.py
│   ├── agent_bfs.py
│   ├── agent_bfs_safe_greedy.py
│   ├── agent_bfs_hamiltonian.py
│   ├── agent_dfs.py
│   ├── agent_astar.py
│   ├── agent_astar_hamiltonian.py
│   └── agent_hamiltonian.py
├── dashboard/
│   ├── __init__.py
│   ├── tab_bfs.py
│   ├── tab_astar.py
│   ├── tab_hamiltonian.py
│   └── tab_comparison.py
├── scripts/
│   ├── __init__.py
│   ├── main.py                    # Enhanced CLI from v0.02
│   ├── generate_dataset.py        # CSV/NPZ/Parquet generation
│   ├── replay.py                  # PyGame replay script
│   └── replay_web.py              # Flask web replay
└── README.md
```

## 📋 **Template Implementation Guidelines**

### **1. Version Evolution Rules**

- **v0.01**: Keep it simple, single algorithm, proof of concept
- **v0.02**: Add algorithm diversity, organized structure, no GUI
- **v0.03**: Add web interface, dataset generation, replay capabilities

### **2. File Naming Conventions**

- **Agent files**: `agent_{algorithm}.py` (e.g., `agent_bfs.py`, `agent_mlp.py`)
- **Config files**: `{extension}_config.py` (e.g., `heuristic_config.py`, `supervised_config.py`)
- **Main entry**: Always `main.py` for CLI, `app.py` for Streamlit
- **Core files**: `game_logic.py`, `game_manager.py`, `game_data.py` (consistent across all)

### **3. Inheritance Patterns**

All extensions must extend base classes:
```python
# Required base class extensions
class {Algorithm}GameLogic(BaseGameLogic):
    """Algorithm-specific game logic"""
    def plan_next_moves(self):
        # Algorithm-specific planning logic
        print(f"[{Algorithm}GameLogic] Planning next moves")  # SUPREME_RULE NO.3
        
        # Get current game state
        current_state = self.get_state_snapshot()
        
        # Algorithm-specific pathfinding
        path = self._calculate_path(current_state)
        
        # Update planned moves
        self.planned_moves = path
        print(f"[{Algorithm}GameLogic] Planned {len(path)} moves")  # SUPREME_RULE NO.3

class {Algorithm}GameManager(BaseGameManager):
    """Algorithm-specific game manager"""
    GAME_LOGIC_CLS = {Algorithm}GameLogic  # Factory pattern
    
    def initialize_task_specific_components(self):
        # Extension-specific initialization
        print(f"[{Algorithm}GameManager] Initializing task-specific components")  # SUPREME_RULE NO.3
        
        # Initialize algorithm-specific components
        self.algorithm_config = self._load_algorithm_config()
        self.performance_tracker = self._create_performance_tracker()
        
        print(f"[{Algorithm}GameManager] Task-specific components initialized")  # SUPREME_RULE NO.3

class {Algorithm}GameData(BaseGameData):
    """Algorithm-specific data handling"""
    def __init__(self):
        super().__init__()
        # Add algorithm-specific data fields
        self.algorithm_metrics = {}
        self.performance_history = []
        print(f"[{Algorithm}GameData] Initialized with algorithm-specific fields")  # SUPREME_RULE NO.3
```

### **4. Factory Pattern Implementation**

Each extension must implement agent factories:
```python
# agents/__init__.py - Universal pattern
from typing import Dict, Type
from core.game_agents import BaseAgent

class {Algorithm}AgentFactory:
    """Factory for {Algorithm} agents"""
    
    _registry = {
        'TYPE1': Agent1Class,
        'TYPE2': Agent2Class,
        'TYPE3': Agent3Class,
    }
    
    @classmethod
    def create(cls, algorithm: str, **kwargs) -> BaseAgent:
        """Create agent by algorithm name"""
        if algorithm.upper() not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {available}")
        
        agent_class = cls._registry[algorithm.upper()]
        print(f"[{Algorithm}AgentFactory] Creating agent: {algorithm}")  # SUPREME_RULE NO.3
        return agent_class(**kwargs)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """Get list of available algorithm names"""
        return list(cls._registry.keys())

# Convenience function
def create_agent(algorithm: str, **kwargs) -> BaseAgent:
    """Create {algorithm} agent - convenience function"""
    return {Algorithm}AgentFactory.create(algorithm, **kwargs)
```
