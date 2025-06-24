# Final Decision: Extension Directory Structure Templates

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

#### **Evolutionary v0.01**
```
extensions/evolutionary-v0.01/
├── __init__.py
├── main.py                        # python main.py (basic GA)
├── agent_ga.py                    # Basic genetic algorithm
├── chromosome.py                  # Chromosome representation
├── game_logic.py                  # EvolutionaryGameLogic(BaseGameLogic)
├── game_manager.py                # EvolutionaryGameManager(BaseGameManager)
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

### **Examples**:

#### **Heuristics v0.02**
```
extensions/heuristics-v0.02/
├── __init__.py
├── main.py                        # --algorithm BFS|ASTAR|DFS|HAMILTONIAN
├── game_logic.py                  # HeuristicGameLogic with algorithm switching
├── game_manager.py                # Multi-algorithm manager
├── game_data.py                   # Heuristic-specific data tracking
├── agents/
│   ├── __init__.py               # HeuristicAgentFactory
│   ├── agent_bfs.py              # BFS algorithm
│   ├── agent_bfs_safe_greedy.py  # BFS with safety heuristics
│   ├── agent_bfs_hamiltonian.py  # BFS + Hamiltonian concepts
│   ├── agent_dfs.py              # Depth-First Search
│   ├── agent_astar.py            # A* pathfinding
│   ├── agent_astar_hamiltonian.py # A* + Hamiltonian
│   └── agent_hamiltonian.py      # Pure Hamiltonian path
└── README.md
```

#### **Supervised v0.02**
```
extensions/supervised-v0.02/
├── __init__.py
├── main.py                        # Model selection and evaluation
├── game_logic.py                  # ML-specific game logic
├── game_manager.py                # Multi-model manager
├── game_data.py                   # ML game data with prediction tracking
├── models/                        # ✨ Different from agents/ - algorithm dependent
│   ├── neural_networks/
│   │   ├── __init__.py
│   │   ├── agent_mlp.py
│   │   ├── agent_cnn.py
│   │   ├── agent_lstm.py
│   │   └── agent_gru.py
│   ├── tree_models/
│   │   ├── __init__.py
│   │   ├── agent_xgboost.py
│   │   ├── agent_lightgbm.py
│   │   └── agent_randomforest.py
│   └── graph_models/
│       ├── __init__.py
│       ├── agent_gcn.py
│       ├── agent_graphsage.py
│       └── agent_gat.py
├── training/                      # Training scripts per model type
│   ├── train_neural.py
│   ├── train_tree.py
│   └── train_graph.py
└── README.md
```

#### **Evolutionary v0.02**
```
extensions/evolutionary-v0.02/
├── __init__.py
├── main.py                        # --algorithm GA|ES|GP with framework choice
├── game_logic.py                  # Enhanced evolutionary logic
├── game_manager.py                # Population management
├── game_data.py                   # Evolutionary data tracking
├── agents/
│   ├── __init__.py               # EvolutionaryAgentFactory
│   ├── agent_ga_custom.py        # Hand-coded genetic algorithm
│   ├── agent_ga_deap.py          # DEAP framework implementation
│   ├── agent_es.py               # Evolution Strategies
│   └── agent_gp.py               # Genetic Programming
├── frameworks/                    # Framework-specific utilities
│   ├── deap_utils.py
│   └── custom_ga_utils.py
└── README.md
```

#### **Reinforcement v0.02**
```
extensions/reinforcement-v0.02/
├── __init__.py
├── main.py                        # --algorithm DQN|PPO|A3C
├── game_logic.py                  # RL-specific game logic
├── game_manager.py                # RL training manager
├── game_data.py                   # RL data with experience tracking
├── agents/
│   ├── __init__.py               # RLAgentFactory
│   ├── agent_dqn.py              # Deep Q-Network
│   ├── agent_ppo.py              # Proximal Policy Optimization
│   ├── agent_a3c.py              # Asynchronous Actor-Critic
│   └── agent_sac.py              # Soft Actor-Critic
├── training/
│   ├── train_dqn.py
│   ├── train_ppo.py
│   └── train_a3c.py
└── README.md
```

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
        pass
    
    @abstractmethod
    def get_algorithms(self) -> List[str]:
        """Return list of available algorithms"""
        pass
    
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
        pass
    
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
                self.launch_pygame_replay(algorithm)
        with col4:
            if st.button(f"Replay (Web)", key=f"{algorithm}_web"):
                self.launch_web_replay(algorithm)

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

#### **Supervised v0.03**
```
extensions/supervised-v0.03/
├── __init__.py
├── app.py                         # SupervisedSnakeApp(BaseExtensionApp)
├── supervised_config.py           # ML-specific configuration
├── game_logic.py                  # Enhanced ML game logic
├── game_manager.py                # Enhanced model evaluation manager
├── game_data.py                   # Enhanced ML data with prediction tracking
├── replay_engine.py               # Model decision replay processing
├── replay_gui.py                  # PyGame model visualization
├── models/                        # Same structure as v0.02
│   ├── neural_networks/
│   ├── tree_models/
│   └── graph_models/
├── training/                      # Enhanced training scripts
│   ├── train_neural.py
│   ├── train_tree.py
│   └── train_graph.py
├── evaluation/                    # NEW: Evaluation and comparison
│   ├── __init__.py
│   ├── model_comparison.py
│   ├── performance_analysis.py
│   └── visualization.py
├── dashboard/
│   ├── __init__.py
│   ├── tab_training.py
│   ├── tab_evaluation.py
│   ├── tab_comparison.py
│   └── tab_visualization.py
├── scripts/
│   ├── __init__.py
│   ├── train.py                   # Enhanced training CLI
│   ├── evaluate.py                # Model evaluation script
│   ├── replay.py                  # PyGame model replay
│   └── replay_web.py              # Flask model replay
└── README.md
```

## 🧬 **Special Extension Templates**

### **LLM Fine-tuning Extensions**

#### **LLM-Finetune v0.01**
```
extensions/llm-finetune-v0.01/
├── __init__.py
├── main.py                        # Basic fine-tuning script
├── finetune.py                    # Core fine-tuning logic
├── game_logic.py                  # LLM-specific game logic
├── game_manager.py                # LLM evaluation manager
└── README.md
```

#### **LLM-Finetune v0.02**
```
extensions/llm-finetune-v0.02/
├── __init__.py
├── main.py                        # Multi-approach fine-tuning
├── pipeline.py                    # Fine-tuning pipeline
├── game_logic.py                  # Enhanced LLM logic
├── game_manager.py                # Enhanced manager
├── game_data.py                   # LLM-specific data tracking
├── approaches/
│   ├── __init__.py
│   ├── lora_finetuning.py         # LoRA approach
│   ├── full_finetuning.py         # Full model fine-tuning
│   └── qlora_finetuning.py        # QLoRA approach
├── training/
│   ├── train_lora.py
│   ├── train_full.py
│   └── train_qlora.py
└── README.md
```

#### **LLM-Finetune v0.03**
```
extensions/llm-finetune-v0.03/
├── __init__.py
├── app.py                         # LLMFinetuneApp(BaseExtensionApp)
├── llm_config.py                  # LLM-specific configuration
├── game_logic.py
├── game_manager.py
├── game_data.py
├── replay_engine.py
├── replay_gui.py
├── approaches/                    # Same as v0.02
├── training/                      # Enhanced training
├── dashboard/
│   ├── __init__.py
│   ├── tab_lora.py
│   ├── tab_full_finetune.py
│   ├── tab_evaluation.py
│   └── tab_comparison.py
├── scripts/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   ├── replay.py
│   └── replay_web.py
└── README.md
```

### **Eureka Extensions**

#### **Eureka v0.01**
```
extensions/eureka-v0.01/
├── __init__.py
├── main.py                        # Basic reward evolution
├── eureka_engine.py               # Core Eureka evolution engine
├── reward_generator.py            # LLM-based reward generation
├── game_logic.py                  # Eureka-specific game logic
├── game_manager.py                # Reward evolution manager
└── README.md
```

#### **Eureka v0.02**
```
extensions/eureka-v0.02/
├── __init__.py
├── main.py                        # Multi-strategy evolution
├── eureka_engine.py               # Enhanced evolution engine
├── game_logic.py
├── game_manager.py
├── game_data.py
├── strategies/
│   ├── __init__.py
│   ├── genetic_evolution.py       # Genetic algorithm for rewards
│   ├── gradient_evolution.py      # Gradient-based evolution
│   └── llm_evolution.py           # LLM-guided evolution
├── reward_functions/
│   ├── __init__.py
│   ├── base_rewards.py
│   ├── shaped_rewards.py
│   └── custom_rewards.py
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
    pass

class {Algorithm}GameManager(BaseGameManager):
    """Algorithm-specific game manager"""
    GAME_LOGIC_CLS = {Algorithm}GameLogic  # Factory pattern
    pass

class {Algorithm}GameData(BaseGameData):
    """Algorithm-specific data handling"""
    pass
```

### **4. Factory Pattern Implementation**

Each extension must implement agent factories:
```python
# agents/__init__.py - Universal pattern
from typing import Dict, Type
from core.game_agents import BaseAgent

class {Algorithm}AgentFactory:
    """Factory for creating {algorithm} agents"""
    
    _agents: Dict[str, Type[BaseAgent]] = {
        'TYPE1': Agent1Class,
        'TYPE2': Agent2Class,
        'TYPE3': Agent3Class,
    }
    
    @classmethod
    def create_agent(cls, algorithm: str, **kwargs) -> BaseAgent:
        """Create agent by algorithm name"""
        if algorithm.upper() not in cls._agents:
            available = ', '.join(cls._agents.keys())
            raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {available}")
        
        agent_class = cls._agents[algorithm.upper()]
        return agent_class(**kwargs)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """Get list of available algorithm names"""
        return list(cls._agents.keys())

# Convenience function
def create_{algorithm}_agent(algorithm: str, **kwargs) -> BaseAgent:
    """Create {algorithm} agent - convenience function"""
    return {Algorithm}AgentFactory.create_agent(algorithm, **kwargs)
```

## 🎯 **Benefits of Standardized Templates**

### **1. Consistency**
- **Predictable structure** across all extensions
- **Easy navigation** for developers and AI assistants
- **Consistent patterns** for learning and extension

### **2. Maintainability**
- **Clear evolution path** from v0.01 to v0.03
- **Organized code** with proper separation of concerns
- **Standardized interfaces** for easy integration

### **3. Educational Value**
- **Progressive complexity** from simple to sophisticated
- **Clear design patterns** demonstrated consistently
- **Learning pathway** for students and researchers

### **4. Extensibility**
- **Easy to add** new algorithm types
- **Template-driven development** for consistency
- **Future-proof architecture** for new versions

## 📋 **Implementation Checklist**

### **For Any New Extension**:
- [ ] Follow appropriate version template (v0.01, v0.02, or v0.03)
- [ ] Extend required base classes (GameLogic, GameManager, GameData)
- [ ] Implement agent factory pattern
- [ ] Use standardized file naming conventions
- [ ] Include comprehensive README documentation
- [ ] Follow inheritance patterns from established extensions
- [ ] Use TaskAwarePathManager for dataset/model paths (v0.03+)
- [ ] Implement Streamlit BaseExtensionApp pattern (v0.03+)

### **Migration from Existing Extensions**:
- [ ] Verify current structure matches templates
- [ ] Rename files to follow naming conventions
- [ ] Reorganize agents into proper folder structure
- [ ] Update imports and factory patterns
- [ ] Ensure base class extensions are correct

---

**This standardized template system ensures consistent, maintainable, and educational extension development across all Snake Game AI algorithm types while supporting natural evolution from proof-of-concept to production-ready implementations.** 