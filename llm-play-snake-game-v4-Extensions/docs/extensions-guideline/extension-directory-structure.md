# Extension Directory Structure & Evolution Standards

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
- **No GUI components by default** - console output only (GUI optional per SUPREME_RULE NO.5)
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
- **No GUI by default** - still CLI only (GUI optional per SUPREME_RULE NO.5)

## 🌐 **v0.03 Template: Web Interface & Dataset Generation**

### **Purpose**: Streamlit app.py for script launching, dataset generation, optional replay capabilities (SUPREME_RULE NO.5)

```
extensions/{algorithm}-v0.03/
├── __init__.py                    # Package initialization
├── app.py                         # ✨ NEW: Streamlit app for launching scripts with adjustable parameters (SUPREME_RULE NO.5)
├── {algorithm}_config.py          # ✨ NEW: Renamed from config.py for clarity
├── game_logic.py                  # Enhanced with dataset generation
├── game_manager.py                # Enhanced manager
├── game_data.py                   # Enhanced data handling
├── replay_engine.py               # ✨ NEW: Replay processing engine
├── replay_gui.py                  # ✨ NEW: PyGame replay interface (optional) - GUI optional per SUPREME_RULE NO.5
├── agents/                        # ✨ COPIED: Exact copy from v0.02
│   ├── __init__.py               # Same as v0.02
│   ├── agent_{type1}.py          # Same as v0.02
│   ├── agent_{type2}.py          # Same as v0.02
│   └── [all other agents]        # Same as v0.02
├── scripts/                       # ✨ NEW: CLI and automation
│   ├── __init__.py
│   ├── main.py                   # Moved from root (enhanced CLI)
│   ├── generate_dataset.py       # Dataset generation CLI
│   ├── replay.py                 # PyGame replay script (optional) - GUI optional per SUPREME_RULE NO.5
│   └── replay_web.py             # Flask web replay (optional) - GUI optional per SUPREME_RULE NO.5
└── README.md                      # Comprehensive documentation
```
