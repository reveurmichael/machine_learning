> **Important — Authoritative Reference:** This guidance supplements the _Final Decision Series_ (`final-decision-0` → `final-decision-10`). Where conflicts exist, defer to the Final Decision documents.

# Extensions v0.03: The Application & Data Phase

## 🎯 **Core Philosophy: From Tool to Application**

The `v0.03` extension represents the final and most mature stage of development. It marks the transition from a command-line tool (`v0.02`) into a polished, user-facing **web application** and a reliable **producer of high-quality data**.

This version answers the question:

> "How do we make our powerful algorithms accessible, interactive, and useful to other parts of the ecosystem?"

It achieves this by introducing a clear separation between the user interface and the underlying scriptable logic.

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
4.  **The `agents/` Folder is Stable:** The `agents/` directory is copied **identically** from `v0.02`. This is a critical principle, demonstrating that the core agent logic is stable and can be reused without modification.

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
├── agents/                        # Same as v0.02 (copied exactly)
│   ├── __init__.py
│   ├── agent_{type1}.py
│   ├── agent_{type2}.py
│   └── [additional agents]
└── scripts/                       # 🆕 Script organization
    ├── main.py                   # Moved from root
    ├── generate_dataset.py       # 🆕 Dataset generation CLI
    ├── replay.py                 # 🆕 PyGame replay script
    └── replay_web.py             # 🆕 Flask web replay
```

### **Key Characteristics:**
- **Streamlit web interface** with OOP architecture (Final Decision 9)
- **Dataset generation** for other extensions
- **Replay capabilities** (PyGame + Flask web)
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
├── agents/                   # Same as v0.02 (copied exactly)
│   ├── __init__.py
│   ├── agent_bfs.py
│   ├── agent_bfs_safe_greedy.py
│   ├── agent_bfs_hamiltonian.py
│   ├── agent_dfs.py
│   ├── agent_astar.py
│   ├── agent_astar_hamiltonian.py
│   └── agent_hamiltonian.py
└── scripts/
    ├── main.py                   # CLI interface
    ├── generate_dataset.py       # CSV dataset generation
    ├── replay.py                 # PyGame replay
    └── replay_web.py             # Flask web replay
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
├── replay_web.py                  # Flask web replay
├── dashboard/
│   ├── __init__.py
│   ├── tab_training.py           # Model training interface
│   ├── tab_evaluation.py         # Model evaluation
│   ├── tab_comparison.py         # Model comparison
│   └── tab_replay.py             # Model decision replay
├── models/                        # Same as v0.02
│   ├── neural_networks/
│   ├── tree_models/
│   └── graph_models/
├── training/                      # Enhanced training scripts
│   ├── train_neural.py
│   ├── train_tree.py
│   └── train_graph.py
└── scripts/
    ├── train.py                  # CLI training interface
    ├── evaluate.py               # Model evaluation
    ├── replay.py                 # PyGame model replay
    └── replay_web.py             # Flask model replay
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
├── agents/                        # Same as v0.02
│   ├── agent_dqn.py
│   ├── agent_ppo.py
│   └── [other RL agents]
├── training/                      # RL training scripts
│   ├── train_dqn.py
│   ├── train_ppo.py
│   └── train_a3c.py
└── scripts/
    ├── train.py                  # CLI RL training
    ├── evaluate.py               # RL evaluation
    ├── replay.py                 # PyGame RL replay
    └── replay_web.py             # Flask RL replay
```

## 🏗️ **Streamlit OOP Architecture**

Following Final Decision 9, all v0.03 extensions use Object-Oriented Programming architecture.
## 🌐 **Web Infrastructure & Replay Systems**

### **Common Web Components**
- **Streamlit frontend**: Interactive parameter control and visualization
- **Flask replay backend**: RESTful API for game state management
- **JavaScript visualization**: Real-time board state rendering
- **WebSocket support**: Live algorithm/model execution updates

### **Replay Features**
- **PyGame replay**: Algorithm/model step-through with performance metrics
- **Web replay**: Browser-based replay with responsive design
## 📊 **Dataset Generation System**

### **Dataset Storage Structure**
Following Final Decision 1:
```
ROOT/logs/extensions/datasets/
├── grid-size-8/
│   └── {algorithm}_v0.03_{timestamp}/
├── grid-size-10/
│   └── {algorithm}_v0.03_{timestamp}/
└── grid-size-12/
    └── {algorithm}_v0.03_{timestamp}/
```

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

### **v0.02 → v0.03 Progression:**

**All Extensions:**
- ✅ **CLI only** → **Streamlit web application**
- ✅ **No replay** → **PyGame + Flask web replay**
- ✅ **Basic logging** → **Dataset generation**

### **v0.03 → v0.04 Preview:**
- **Heuristics only**: Numerical datasets → **Language-rich datasets for LLM fine-tuning**
- **Other algorithms**: No v0.04 (v0.03 is sufficient)

## 📋 **Implementation Checklist**

### **For All v0.03 Extensions:**
- [ ] **OOP Streamlit app.py** following Final Decision 9
- [ ] **Dashboard folder** with organized tab components
- [ ] **Dataset generation** scripts and CLI
- [ ] **PyGame replay** system
- [ ] **Flask web replay** system
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
- Cross-platform compatibility (pygame desktop and flask web)

### **Web Interface Goals:**
- Consistent user experience across all extensions
- Intuitive parameter adjustment and control
- Seamless integration between different algorithm types

---

**Remember**: v0.03 is about **user experience** and **data production**. Create polished interfaces that make algorithms/models accessible and generate high-quality datasets for the ML ecosystem.
