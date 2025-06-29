# Final Decision 5: Extension Directory Structure & Evolution Standards

> **SUPREME AUTHORITY**: This document establishes the definitive directory structure templates and evolution standards for all Snake Game AI extensions.

> **See also:** `project-structure-plan.md` (Master blueprint), `extension-evolution-rules.md` (Evolution guidelines), `extensions-move-guidelines.md` (Move standards), `final-decision-10.md` (SUPREME_RULES).

## 🎯 **Core Philosophy: Progressive Extension Evolution**

The extension directory structure implements a sophisticated evolution model that ensures consistency, educational progression, and maintainability across all Snake Game AI extensions, strictly following `final-decision-10.md` SUPREME_RULES.

### **SUPREME_RULES Integration**
- **SUPREME_RULE NO.1**: Enforces reading all GOOD_RULES before making extension structure changes to ensure comprehensive understanding
- **SUPREME_RULE NO.2**: Uses precise `final-decision-N.md` format consistently when referencing architectural decisions
- **SUPREME_RULE NO.3**: Enables lightweight common utilities with OOP extensibility while maintaining extension patterns through inheritance rather than tight coupling
- **SUPREME_RULE NO.4**: Ensures all markdown files are coherent and aligned through nuclear diffusion infusion process

### **GOOD_RULES Integration**
This document integrates with the **GOOD_RULES** governance system established in `final-decision-10.md`:
- **`project-structure-plan.md`**: Authoritative reference for overall project structure
- **`extension-evolution-rules.md`**: Authoritative reference for extension evolution patterns
- **`extensions-move-guidelines.md`**: Authoritative reference for extension move standards
- **`single-source-of-truth.md`**: Ensures structure consistency across all extensions

### **Simple Logging Examples (SUPREME_RULE NO.3)**
All code examples in this document follow **SUPREME_RULE NO.3** by using simple print() statements rather than complex logging mechanisms:

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

## 🏭 **Factory Pattern Integration**

### **Extension Factory Pattern**
```python
# extensions/common/utils/extension_factory.py
class ExtensionFactory:
    """
    Factory for creating extension instances
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create appropriate extension instances based on type and version
    Educational Value: Shows how canonical factory patterns work with extensions
    
    Reference: final-decision-10.md for canonical method naming
    """
    
    _registry = {
        "HEURISTICS": {
            "0.01": HeuristicsV001Extension,
            "0.02": HeuristicsV002Extension,
            "0.03": HeuristicsV003Extension,
        },
        "SUPERVISED": {
            "0.01": SupervisedV001Extension,
            "0.02": SupervisedV002Extension,
            "0.03": SupervisedV003Extension,
        },
        "REINFORCEMENT": {
            "0.01": ReinforcementV001Extension,
            "0.02": ReinforcementV002Extension,
            "0.03": ReinforcementV003Extension,
        },
    }
    
    @classmethod
    def create(cls, extension_type: str, version: str, **kwargs):  # CANONICAL create() method
        """Create extension using canonical create() method (SUPREME_RULES compliance)"""
        extension_class = cls._registry.get(extension_type.upper(), {}).get(version)
        if not extension_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown extension type: {extension_type} v{version}. Available: {available}")
        print(f"[ExtensionFactory] Creating extension: {extension_type} v{version}")  # Simple logging
        return extension_class(**kwargs)
```

## 🎓 **Educational Value and Learning Path**

### **Learning Objectives**
- **Extension Evolution**: Understanding how extensions evolve from simple to complex
- **Directory Organization**: Learning to organize code with clear progression
- **Factory Patterns**: Understanding canonical factory pattern implementation
- **Version Management**: Learning to manage multiple versions of extensions

### **Implementation Examples**
- **Extension Creation**: How to create extensions following evolution patterns
- **Version Migration**: How to migrate from one version to the next
- **Directory Organization**: How to organize extension files consistently
- **Factory Integration**: How to integrate extensions with factory patterns

## 🔗 **Integration with Other Documentation**

### **GOOD_RULES Alignment**
This document aligns with:
- **`project-structure-plan.md`**: Detailed project structure standards
- **`extension-evolution-rules.md`**: Extension evolution patterns
- **`extensions-move-guidelines.md`**: Extension move standards
- **`single-source-of-truth.md`**: Architectural principles

### **Extension Guidelines**
This directory structure supports:
- All extension types (heuristics, supervised, reinforcement, LLM)
- All evolution stages (v0.01, v0.02, v0.03)
- Consistent file and directory organization
- Predictable extension progression patterns

---

**This extension directory structure ensures consistent, progressive, and maintainable extension organization across all Snake Game AI extensions while maintaining SUPREME_RULES compliance.**

> **SUPREME_RULES COMPLIANCE**: This document strictly follows the SUPREME_RULES established in `final-decision-10.md`, ensuring coherence, educational value, and architectural integrity across the entire project.


