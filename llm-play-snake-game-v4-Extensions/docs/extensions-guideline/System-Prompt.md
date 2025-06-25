# System Prompt Guidelines

> **Important — Authoritative Reference:** This document establishes core principles for the Snake Game AI project. All other guideline documents must align with these foundational principles.

## 📚 **Documentation as First-Class Citizen**

Documentation, docstrings, and comments are paramount in this project. Each refactoring must preserve and enhance existing documentation. Every bug fix must be recorded in related code comments and docstrings.

## 🎯 **Core Refactoring Philosophy**

We are refactoring the codebase to make it more generic and reusable while keeping Task-0 (LLM playing snake game) functionality unchanged.

### **VITAL Reference**
Check `ROOT/docs/extensions-guideline/project-structure-plan.md` for the complete refactoring objectives.

## 🏗️ **Architectural Principles**

### **Single Source of Truth**
Each extension plus the common folder is regarded as standalone. We maintain single source of truth, especially from:
- `ROOT/config/` folder
- `ROOT/extensions/common/` folder

### **OOP, SOLID, and DRY Principles**
- **OOP First**: Make everything object-oriented for easier extension
- **SOLID Compliance**: Follow all SOLID principles rigorously
- **DRY Principle**: Eliminate code duplication through proper abstraction

Future tasks (Task 1-5) can be implemented as subclasses of base classes using inheritance, adapters, or composition. We tolerate that future tasks may not use all base class attributes/functions, as long as they don't pollute output data files.

## 🎮 **Task-0 Reference**

If you lose sight of Task-0, check these reference files:
- `logs/hunyuan-t1-latest_20250617_223807/game_1.json`
- `logs/hunyuan-t1-latest_20250617_223807/game_8.json`
- `logs/hunyuan-t1-latest_20250617_223807/summary.json`

These provide concrete examples of Task-0 output schema and behavior.

## 🏷️ **Naming Conventions**

### **Class Naming Philosophy**
- **Task-0 Classes**: No need for `Task0` prefix. Names like `GameController`, `FileManager`, `GameData`, `GameLogic` are Task-0 specific by default
- **Base Classes**: Use `Base` prefix for classes extended by multiple tasks (e.g., `BaseFileManager`, `BaseGameData`, `BaseGameLogic`)
- **Extension Classes**: Use descriptive prefixes (e.g., `HeuristicGameManager`, `RLGameManager`)

### **Singleton Pattern**
`BaseFileManager` and `FileManager` should use the Singleton pattern. Consider other classes for singleton implementation as appropriate.

## 🧠 **Design Patterns**

This project emphasizes design patterns for educational value and maintainability:
- Use appropriate design patterns extensively
- Document each pattern with detailed comments explaining:
  - Why the pattern was chosen
  - Its philosophical approach
  - Trade-offs and alternatives
  - Implementation rationale

## 🔗 **Inheritance Strategy**

Classes in the `extensions/` folder should primarily inherit from base classes in `core/`, `replay/`, and `gui/` folders. In rare cases, they might inherit from Task-0 derived classes instead of base classes.

## 🚀 **Forward-Looking Architecture**

### **No Backward Compatibility**
We refactor with a future-proof mindset, creating fresh, self-consistent, and self-contained systems. We do not maintain backward compatibility - deprecated code should be removed entirely. No legacy considerations for extensions.

For Task-0, maintain output schema compliance as defined in the reference files above.

### **No Code Pollution**
No pollution from extensions (Task 1-5) into the ROOT folder. Extension-specific terminology (e.g., "heuristics", "reinforcement learning") should not appear in the ROOT folder - only in the `extensions/` folder.

### **No Over-Preparation**
Let future tasks implement their own required code/functions. Avoid overkill and over-preparation in the ROOT folder.

## 🔒 **Critical Constraints**

### **Core and Replay Folder Protection**
**VITAL**: Do not remove any classes in `./core/` or `./replay/` folders. You can add functions or classes, but never remove existing classes. These are already being used by extensions.

### **Extension Evolution Stability**
**VITAL**: Maintain exact consistency across extension versions:

- Keep the `agents/` folder identical between `./extensions/heuristics-v0.02` and `./extensions/heuristics-v0.03`
- Keep the `agents/` folder identical between `./extensions/supervised-v0.02` and `./extensions/supervised-v0.03`
- Keep the `agents/` folder identical between `./extensions/reinforcement-v0.02` and `./extensions/reinforcement-v0.03`
- Keep the `agents/` folder identical between `./extensions/evolutionary-v0.02` and `./extensions/evolutionary-v0.03`

Each extension version plus the common folder forms a standalone unit.

## 📁 **Common Folder Philosophy**

The `./extensions/common/` folder contains shared utilities that are:
- Common for current needs
- Potentially useful for future extensions
- Non-essential to core extension concepts

This approach ensures that each extension `{algorithm}-v0.0N` represents important conceptual ideas (e.g., heuristics, supervised learning, RL) while keeping these concepts highly visible. Moving non-essential code to the common folder enhances conceptual clarity.

**Standalone Principle**: Each extension `{algorithm}-v0.0N` plus the common folder forms a standalone unit. No code sharing between extensions is allowed.

## 🏷️ **Type Hints**

Use type hints where you are confident about types. Don't add type hints for the sake of type hinting alone.

## 📝 **Documentation Standards**

Never reduce the clarity, verbosity, or detail of docstrings and comments. Embrace OOP and inheritance extensively. Use basic but effective design patterns extensively with comprehensive documentation.

## 🚫 **Forbidden Import Patterns**

These import patterns are strictly forbidden:
```python
# ❌ NEVER DO THIS
from heuristics_v0.03 import some_module
from {algorithm}_v0.0N import some_module
from extensions.distillation_v0_03 import some_module
from extensions.{algorithm}_v0_0N import some_module
```

## 🛣️ **Path Management**

Use `chdir()` extensively, preferably through utility functions from:
- `./extensions/common/path_utils.py`
- `./utils/path_utils.py`

## 🔄 **DRY Principles for Extensions**

Apply DRY principles extensively in extensions, but only for common utilities in the `./extensions/common/` folder. Never share code between extensions directly.

## 📈 **Extension Version Evolution**

### **v0.02 Requirements**
v0.02 should not break v0.01 functionalities.

### **v0.03 Requirements**
v0.03 should not break v0.02 functionalities.

### **v0.04 Requirements (Heuristics Only)**
v0.04 is exclusive to heuristics extensions. Other extensions only support v0.01, v0.02, and v0.03.

For heuristics v0.04:
- Generate JSONL files in addition to CSV files (from v0.03)
- Preserve all v0.03 functionality
- Use OOP or adapters for extensions
- Implement validation pipeline to ensure JSONL generation works for all heuristic agents

## 🔧 **Implementation Guidelines**

### **No Backward Compatibility**
Refactor with a future-proof mindset for fresh, newly shipped, self-consistent, and self-contained systems. No backward compatibility maintenance.

### **Extension Data Generation**
Extensions `{algorithm}-v0.0N` should generate:
- JSON files (standard)
- PTH or NPZ files (for RL/supervised learning)
- Parquet files (as appropriate)

For transforming JSON files to CSV, use shared tools in the "common" folder. For JSONL generation, place in heuristics-v0.04 folder or common folder based on clarity requirements.

### **File Naming Clarity**
Avoid confusing file names like `generate_dataset_v03.py`, `generate_dataset.py`, `generate_jsonl_dataset.py` in the same folder. Use clear, descriptive names and consider moving utilities to the "common" folder.

## 🚫 **Breaking Changes in Extensions**

In the extensions folder, if things need to break, break them cleanly. No need for class name aliases or adapters unless absolutely necessary.

### **Import Aliases**
No import aliases unless truly necessary.

## 📊 **Dataset and Model Organization**

### **Grid Size Flexibility**
Grid size should not be fixed to 10. Generated datasets (JSON, CSV, JSONL files/folders) are stored in:
```
./logs/extensions/datasets/grid-size-N/{algorithm}_v0.0N_{timestamp}/
```

### **Extension Coverage**
Ensure this structure works for:
- **Heuristics**: v0.01, v0.02, v0.03, v0.04
- **Supervised Learning**: v0.01, v0.02, v0.03
- **Reinforcement Learning**: v0.01, v0.02

### **Model Storage**
Models trained by ML/DL/RL are stored in:
```
./logs/extensions/models/grid-size-N/{algorithm}_v0.0N_{timestamp}/
```

## 🎨 **Streamlit App Philosophy**

Streamlit `app.py` is **NOT** for:
- Game state visualization
- Real-time progress display
- Snake move visualization

Its primary purpose is to launch scripts in the "scripts" folder with adjustable parameters using subprocess. This is why extensions v0.03 have a "dashboard" folder.

## 🔒 **Final Decision Protection**

**VITAL**: Never edit files in the pattern `ROOT/docs/extensions-guideline/final-decision-N.md` as these are final decisions and single sources of truth.

## 🧠 **Deep Learning Framework**

**No TensorFlow**: Use PyTorch and PyTorch ecosystem exclusively:
- torchvision
- torchtext
- torchaudio
- torch-geometric / PyG (for graph neural networks)
- Other PyTorch family frameworks