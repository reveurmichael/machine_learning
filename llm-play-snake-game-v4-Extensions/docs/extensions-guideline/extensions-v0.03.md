> **Important — Authoritative Reference:** This guidance supplements the _Final Decision Series_ (`final-decision-0` → `final-decision-10`). Where conflicts exist, defer to the Final Decision documents.

> **SUPREME_RULES**: Both `heuristics-v0.03` and `heuristics-v0.04` are widely used depending on use cases and scenarios. For supervised learning and other general purposes, both versions can be used. For LLM fine-tuning, only `heuristics-v0.04` will be used. The CSV format is **NOT legacy** - it's actively used and valuable for supervised learning.

# Extensions v0.03: Web Interface & Dataset Generation

## 🎯 **Core Philosophy: User-Friendly Interface + Data Generation**

v0.03 represents the transition from command-line tools to **user-friendly web interfaces** while adding **dataset generation capabilities** for other extensions. This version demonstrates how to build upon stable algorithmic foundations (v0.02) with modern web technologies.

## 🏗️ **Architectural Transformation: UI, Scripts, and Stability**

A `v0.03` extension introduces a significant architectural refactoring to cleanly separate concerns, as defined in `final-decision-5.md`.

### **The New Directory Structure**
```
extensions/{algorithm_type}-v0.03/
├── app.py                   # 👈 NEW: The main OOP Streamlit application
├── dashboard/               # 👈 NEW: All modular UI components (tabs, views)
│   └── ...
├── scripts/                 # 👈 NEW: All command-line entry points are moved here
│   ├── main.py              # The CLI runner (formerly at the root)
│   ├── generate_dataset.py  # A dedicated script for generating datasets
│   └── replay.py            # A script for launching a game replay
├── agents/                  # ✅ IDENTICAL to v0.02's agents folder
│   └── ...
└── {algorithm}_config.py    # 👈 NEW: A dedicated config file for the extension
```

### **Key Architectural Changes**

1.  **`app.py` is the New Entry Point:** The primary way to interact with a `v0.03` extension is through the Streamlit web application. It **must** follow the OOP architecture defined in `final-decision-9.md`.
2.  **`scripts/` Consolidates CLI Tools:** All command-line functionality is moved into the `scripts/` folder. These scripts are self-contained and can be run independently of the UI.
3.  **`dashboard/` Organizes the UI:** All UI components used by `app.py` are organized into the `dashboard/` folder for modularity and clarity.
4.  **The `agents/` Folder is Stable:** The `agents/` directory contains the same core algorithms from `v0.02`. This is a critical principle, demonstrating that the core agent logic is stable and can be reused without modification.

### **🔒 Evolution Rules for Agents Folder**

> **Important**: For complete evolution rules, see `extension-evolution-rules.md` - the authoritative reference for all version transitions.

**Core Stability Principle**: The `agents/` folder follows strict evolution rules to maintain algorithmic integrity:

**✅ Required: Copy Exactly from v0.02** (See evolution rules guide for details)
- All core algorithm files (`agent_bfs.py`, `agent_astar.py`, etc.)
- Factory registration system (`__init__.py`)
- Base agent interfaces and method signatures

**➕ Allowed: Enhancements and Extensions** (See evolution rules guide for patterns)
- Enhanced algorithm variants (`agent_bfs_web_optimized.py`)
- Monitoring and metrics collection utilities
- Web interface integration helpers
- Performance optimization wrappers

**❌ Forbidden: Breaking Changes** (See evolution rules guide for complete list)
- Modifying core algorithm behavior or logic
- Changing factory registration names or signatures
- Removing or renaming existing agent files
- Breaking backward compatibility with v0.02

### **Exception Cases for New Agents**

New agents can be added in v0.03 **only** if they:
1. **Extend existing algorithms** without modifying originals
2. **Add web-specific functionality** (real-time monitoring, progress tracking)
3. **Maintain interface compatibility** with existing factory patterns
4. **Follow naming conventions** (`agent_{algorithm}_{enhancement}.py`)

## 🔧 **The "UI as a Script-Runner" Pattern**

The single most important concept in `v0.03` is that the **Streamlit application's main purpose is to be a user-friendly frontend for the scripts in the `scripts/` folder.**

The UI should not re-implement any core logic. Instead, it should:
1.  Use interactive widgets (sliders, buttons) to gather parameters from the user.
2.  Construct a valid command-line string based on the user's input.
3.  Use Python's `subprocess` module to execute the appropriate script (e.g., `scripts/main.py` or `scripts/generate_dataset.py`).
4.  (Optional) Stream the output from the script back to the web interface.

This pattern ensures that all functionality remains accessible and automatable from the command line, while the UI provides a convenient layer of interactivity.

## 📊 **The Role of Data Producer**

A key responsibility of a `v0.03` extension is to generate high-quality, structured data for other parts of the ecosystem (like training supervised models). This is formalized through the `scripts/generate_dataset.py` script. This script must produce data that adheres to the standardized directory structure and format required by the project.

## 📋 **Compliance Checklist: The Definition of Done**

A `v0.03` extension is considered complete and successful if it meets these criteria:

- [ ] Does it have a primary `app.py` that follows the mandatory OOP `BaseExtensionApp` architecture?
- [ ] Have all command-line entry points been moved into a `scripts/` directory?
- [ ] Are all modular UI components organized within a `dashboard/` directory?
- [ ] Is the `agents/` folder an identical copy of the one from `v0.02`?
- [ ] Does the Streamlit UI function primarily as a "script-runner" using `subprocess`?
- [ ] Does it include a `scripts/generate_dataset.py` for data production?
- [ ] **Flask web replay** system (extends ROOT/web infrastructure)

---

> **The `v0.03` extension represents a fully realized component: it is accessible via a web interface, powerful via the command line, and contributes value back to the ecosystem through data generation. It is the pinnacle of an extension's evolution.**

## 🎯 **Core Philosophy: Web Interface & Data Generation**

v0.03 builds upon v0.02's multi-algorithm foundation to demonstrate:
- **Web interface evolution**: From CLI-only to Streamlit web applications
- **Dataset generation**: Creating training data for other extensions
- **Replay capabilities**: Both pygame and web-based replay systems

## 🔧 **Universal v0.03 Template**

Following Final Decision 5 and Final Decision 9, all v0.03 extensions follow this structure:

```
extensions/{algorithm}-v0.03/
├── __init__.py                    # Enhanced package initialization
├── app.py                         # 🆕 OOP Streamlit web interface
├── {algorithm}_config.py          # 🆕 Extension-specific configuration
├── game_logic.py                  # Enhanced algorithm-specific logic
├── game_manager.py                # Multi-algorithm manager with dataset generation
├── game_data.py                   # Algorithm-specific data with export capabilities
├── replay_engine.py               # 🆕 Replay processing engine
├── replay_gui.py                  # 🆕 PyGame replay interface
├── dashboard/                     # 🆕 Streamlit tab components
│   ├── __init__.py
│   ├── tab_main.py               # Main algorithm interface
│   ├── tab_training.py           # Training interface (if applicable)
│   ├── tab_evaluation.py         # Evaluation interface
│   ├── tab_replay.py             # Replay interface
│   └── tab_comparison.py         # Comparison interface
├── agents/                        # 🔒 Copied exactly from v0.02 + allowed enhancements
│   ├── __init__.py               # 🔒 Stable factory (unchanged)
│   ├── agent_{type1}.py          # 🔒 Core algorithm (unchanged)
│   ├── agent_{type2}.py          # 🔒 Core algorithm (unchanged)
│   ├── agent_{type1}_enhanced.py # ➕ Allowed: Enhanced variants
│   └── [monitoring utilities]    # ➕ Allowed: Web interface support
└── scripts/                       # 🆕 Script organization
    ├── main.py                   # Moved from root
    ├── generate_dataset.py       # 🆕 Dataset generation CLI
    ├── replay.py                 # 🆕 PyGame replay script
    └── replay_web.py             # Flask web replay (extends ROOT/web infrastructure)
```

### **Key Characteristics:**
- **Streamlit web interface** with OOP architecture (Final Decision 9)
- **Dataset generation** for other extensions
- **Replay capabilities** (PyGame + Flask web following ROOT/web architecture)
- **Organized dashboard** components
- **Script launching** via subprocess with adjustable parameters

## 🧠 **Algorithm-Specific Examples**

### **Heuristics v0.03**
```
extensions/heuristics-v0.03/
├── __init__.py
├── app.py                         # HeuristicStreamlitApp
├── heuristic_config.py            # Heuristic-specific configuration
├── game_logic.py                  # Enhanced heuristic logic
├── game_manager.py                # Multi-algorithm manager
├── game_data.py                   
├── replay_engine.py               # Replay processing
├── replay_engine.py          # 🆕 Replay processing engine
├── replay_gui.py             # 🆕 PyGame replay interface
├── dashboard/
│   ├── __init__.py
│   ├── tab_main.py               # Algorithm selection and execution
│   ├── tab_evaluation.py         # Performance evaluation
│   ├── tab_replay.py             # Replay interface
│   └── tab_comparison.py         # Algorithm comparison
├── agents/                   # 🔒 Core algorithms from v0.02 + ➕ allowed enhancements
│   ├── __init__.py           # 🔒 Stable factory (copied exactly)
│   ├── agent_bfs.py          # 🔒 Core BFS (copied exactly)
│   ├── agent_bfs_safe_greedy.py    # 🔒 Core variant (copied exactly)
│   ├── agent_bfs_hamiltonian.py    # 🔒 Core variant (copied exactly)
│   ├── agent_dfs.py          # 🔒 Core DFS (copied exactly)
│   ├── agent_astar.py        # 🔒 Core A* (copied exactly)
│   ├── agent_astar_hamiltonian.py  # 🔒 Core variant (copied exactly)
│   ├── agent_hamiltonian.py # 🔒 Core algorithm (copied exactly)
│   ├── agent_bfs_web_optimized.py  # ➕ New: Web interface optimization
│   └── web_monitoring_utils.py     # ➕ New: Real-time monitoring
└── scripts/
    ├── main.py                   # CLI interface
    ├── generate_dataset.py       # CSV dataset generation
    ├── replay.py                 # PyGame replay
    └── replay_web.py             # Flask web replay (extends ROOT/web infrastructure)
```

### **Supervised v0.03**
```
extensions/supervised-v0.03/
├── __init__.py
├── app.py                         # SupervisedStreamlitApp
├── supervised_config.py           # ML-specific configuration
├── game_logic.py                  # ML-specific game logic
├── game_manager.py                # Multi-model evaluation manager
├── game_data.py                   # ML data with prediction tracking
├── replay_engine.py               # Replay engine
├── replay_gui.py                  # PyGame replay
├── replay_web.py                  # Flask web replay (extends ROOT/web infrastructure)
├── dashboard/
│   ├── __init__.py
│   ├── tab_training.py           # Model training interface
│   ├── tab_evaluation.py         # Model evaluation
│   ├── tab_comparison.py         # Model comparison
│   └── tab_replay.py             # Model decision replay
├── models/                        # 🔒 Core models from v0.02 + ➕ web enhancements
│   ├── neural_networks/          # 🔒 Core neural models (copied exactly)
│   │   ├── __init__.py           # 🔒 Stable factory
│   │   ├── agent_mlp.py          # 🔒 Core MLP (unchanged)
│   │   ├── agent_cnn.py          # 🔒 Core CNN (unchanged)
│   │   ├── agent_lstm.py         # 🔒 Core LSTM (unchanged)
│   │   └── agent_mlp_web_monitor.py # ➕ New: Web training visualization
│   ├── tree_models/              # 🔒 Core tree models (copied exactly)
│   │   ├── agent_xgboost.py      # 🔒 Core XGBoost (unchanged)
│   │   ├── agent_lightgbm.py     # 🔒 Core LightGBM (unchanged)
│   │   └── tree_web_explainer.py # ➕ New: Interactive feature importance
│   └── graph_models/             # 🔒 Core graph models (copied exactly)
├── training/                      # Enhanced training scripts
│   ├── train_neural.py
│   ├── train_tree.py
│   └── train_graph.py
└── scripts/
    ├── train.py                  # CLI training interface
    ├── evaluate.py               # Model evaluation
    ├── replay.py                 # PyGame model replay
    └── replay_web.py             # Flask model replay (extends ROOT/web infrastructure)
```

### **Reinforcement v0.03**
```
extensions/reinforcement-v0.03/
├── __init__.py
├── app.py                         # ReinforcementStreamlitApp
├── reinforcement_config.py        # RL-specific configuration
├── game_logic.py                  # RL-specific game logic
├── game_manager.py                # Multi-algorithm RL manager
├── game_data.py                   # RL data with experience tracking
├── replay_engine.py               # RL agent replay
├── replay_gui.py                  # PyGame RL visualization
├── dashboard/
│   ├── __init__.py
│   ├── tab_training.py           # RL training interface
│   ├── tab_evaluation.py         # RL evaluation
│   ├── tab_comparison.py         # RL algorithm comparison
│   └── tab_replay.py             # RL agent replay
├── agents/                        # 🔒 Core RL algorithms from v0.02 + ➕ enhancements
│   ├── __init__.py               # 🔒 Stable factory (copied exactly)
│   ├── agent_dqn.py              # 🔒 Core DQN (copied exactly)
│   ├── agent_ppo.py              # 🔒 Core PPO (copied exactly)
│   ├── agent_a3c.py              # 🔒 Core A3C (copied exactly)
│   ├── agent_dqn_web_monitor.py  # ➕ New: Web training monitoring
│   └── rl_metrics_collector.py   # ➕ New: Real-time metrics
├── training/                      # RL training scripts
│   ├── train_dqn.py
│   ├── train_ppo.py
│   └── train_a3c.py
└── scripts/
    ├── train.py                  # CLI RL training
    ├── evaluate.py               # RL evaluation
    ├── replay.py                 # PyGame RL replay
    └── replay_web.py             # Flask RL replay (extends ROOT/web infrastructure)
```

## 🏗️ **Streamlit OOP Architecture**

Following Final Decision 9, all v0.03 extensions use Object-Oriented Programming architecture.
## 🌐 **Web Infrastructure & Replay Systems**

> **SUPREME_RULES Reference**: All Flask integration follows `flask.md` - extensions must leverage existing ROOT/web infrastructure and follow MVC patterns.

### **Common Web Components**
- **Streamlit frontend**: Interactive parameter control and visualization
- **Flask replay backend**: Extends ROOT/web infrastructure with RESTful API for game state management
- **JavaScript visualization**: Real-time board state rendering using ROOT/web static assets
- **WebSocket support**: Live algorithm/model execution updates

### **Flask Integration Pattern (SUPREME_RULES)**
```python
# All extension Flask apps must follow this pattern
from web.controllers.base_controller import BaseController
from web.views.template_engines import render_template
from web.models.game_state_model import GameStateModel

class ExtensionController(BaseController):
    """Extension-specific Flask controller following ROOT/web patterns"""
    pass
```

### **Replay Features**
- **PyGame replay**: Algorithm/model step-through with performance metrics
- **Web replay**: Browser-based replay using ROOT/web templates and static assets
## 📊 **Dataset Generation System**

### **Dataset Storage Structure**
Following Final Decision 1 with standardized format:
```
logs/extensions/datasets/
├── grid-size-8/
│   └── {extension}_v0.03_{timestamp}/
├── grid-size-10/
│   └── {extension}_v0.03_{timestamp}/
└── grid-size-12/
    └── {extension}_v0.03_{timestamp}/
```

**🎯 Enforced Path Format**: `logs/extensions/datasets/grid-size-N/{extension}_v{version}_{timestamp}/`

### **Data Formats Supported**
- **CSV**: Tabular data for XGBoost, LightGBM, simple neural networks
- **NPZ**: NumPy arrays for sequential/temporal models (LSTM, GRU)
- **Parquet**: Efficient storage for large datasets with complex structures
- **JSON**: Human-readable format for debugging and analysis

### **Data Structures Supported**
- **Tabular**: Flattened board state + engineered features
- **Sequential**: Time-series data for RNN/LSTM models
- **Graph**: Node/edge representations for Graph Neural Networks
- **Image**: Board state as images for CNN models

## 🚀 **Evolution Summary**

### **v0.02 → v0.03 Standardized Evolution:**

**All Extensions (Heuristics, Supervised, Reinforcement):**
- ✅ **CLI only** → **Streamlit web application**
- ✅ **No replay** → **PyGame + Flask web replay (following ROOT/web patterns)**
- ✅ **Basic logging** → **Dataset generation with standardized paths**
- 🔒 **Core algorithms** → **Stable (copied exactly from v0.02)**
- ➕ **Web enhancements** → **Allowed additions for UI integration**

### **v0.03 → v0.04 Evolution Rules:**
- **Heuristics only**: Numerical datasets → **Language-rich datasets for LLM fine-tuning**
- **All other extensions**: No v0.04 (v0.03 is the final mature version)
- **Path structure**: Remains identical across all versions
- **Core stability**: v0.04 maintains same 🔒 stability rules as v0.03

### **Version Compatibility Guarantee:**
| Component | v0.02 → v0.03 | v0.03 → v0.04 |
|-----------|---------------|---------------|
| **Core Algorithm Files** | 🔒 **Copy exactly** | 🔒 **Copy exactly** |
| **Factory Registration** | 🔒 **Unchanged** | 🔒 **Unchanged** |
| **Path Structure** | ✅ **Standardized** | 🔒 **Stable** |
| **Web Interface** | ➕ **Added** | ➕ **Enhanced** |
| **Dataset Formats** | ➕ **CSV/NPZ** | ➕ **Add JSONL** |

## 📋 **Implementation Checklist**

### **For All v0.03 Extensions:**
- [ ] **OOP Streamlit app.py** following Final Decision 9
- [ ] **Dashboard folder** with organized tab components
- [ ] **Dataset generation** scripts and CLI
- [ ] **PyGame replay** system
- [ ] **Flask web replay** system (extends ROOT/web infrastructure)
- [ ] **Agent/model folder** copied exactly from v0.02
- [ ] **Configuration renamed** to avoid conflicts

### **Streamlit App Requirements:**
- [ ] **Inherits from ExtensionStreamlitApp** base class
- [ ] **Implements required abstract methods** (setup_page, main, cleanup)
- [ ] **Uses dashboard components** for modular organization
- [ ] **Launches scripts via subprocess** with adjustable parameters
- [ ] **Consistent user experience** across all extensions

## 🎯 **Success Criteria**

### **Dataset Generation Goals:**
- Multiple data formats for different model types
- Configurable grid sizes and game parameters
- High-quality labeled training data
- Efficient storage and loading mechanisms

### **Replay System Goals:**
- Smooth visualization of game progression
- Clear display of algorithm reasoning or model decisions
- Export capabilities for analysis and presentation
- Cross-platform compatibility (pygame desktop and flask web following ROOT/web patterns)

### **Web Interface Goals:**
- Consistent user experience across all extensions
- Intuitive parameter adjustment and control
- Seamless integration between different algorithm types

## 🎯 **SUPREME_RULES: Version Selection Guidelines**

- **For heuristics**: Use v0.04 instead of v0.03 - it's a superset with no downsides
- **For supervised learning**: Use CSV from heuristics-v0.04
- **For LLM fine-tuning**: Use JSONL from heuristics-v0.04
- **For research**: Use both formats from heuristics-v0.04
- **CSV is ACTIVE**: Not legacy - actively used for supervised learning
- **JSONL is ADDITIONAL**: New capability for LLM fine-tuning

---

**Remember**: v0.03 is about **user experience** and **data production**. Create polished interfaces that make algorithms/models accessible and generate high-quality datasets for the ML ecosystem. However, for heuristics specifically, prefer v0.04 as it provides everything v0.03 does and more.
