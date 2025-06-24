> **Important — Authoritative Reference:** This planning document is subordinate to the _Final Decision Series_ (`final-decision-0` → `final-decision-10`). Any conflicting guidance must follow the Final Decision documents.

# Project Structure Plan – Extensions & Multi-Task Architecture

This document provides the blueprint for building robust, extensible Snake AI extensions from heuristics to neural networks to LLM fine-tuning.

## 🎯 **Core Philosophy: First-Citizen vs Second-Citizen**

**Guiding Principle: "Task-0 first, everything else second."**

1. **Task-0 (LLM Snake)** is the *first-citizen* – our flagship production system that must always compile, run, and deliver the core "LLM plays Snake" experience. It owns the `master` branch and defines architectural patterns.

2. **Task 1-5** are *second-citizens* – experimental research tracks that live alongside Task-0 but can never break or degrade its stability. They import from Task-0 modules (`core/`, `gui/`, `utils/`) but never the reverse.

3. **Dependency Direction**: Second-citizens extend Task-0's base classes but Task-0 remains completely unaware of their existence.

## 🎮 **Task Overview & Data Flow**

| Task | Role | Key Output | Feeds Into |
|------|------|------------|------------|
| **0** | LLM Snake (First-Citizen) | Game sessions, replays | *Foundation for all* |
| **1** | Heuristic Agents | CSV datasets, **JSONL trajectories** | Task 2, 4, 5 |
| **2** | Supervised Learning | Trained models (PyTorch, XGBoost, etc.) | *Performance baselines* |
| **3** | Reinforcement Learning | RL agents (DQN, PPO, SAC) | *Alternative policies* |
| **4** | LLM Fine-tuning | LoRA-tuned models | Task 5 |
| **5** | Knowledge Distillation | Compressed student models | *Deployment-ready agents* |

**Critical Data Dependencies:**
- **Task 4 & 5** depend entirely on **heuristics-v0.04** JSONL outputs
- **Task 2** trains on heuristics CSV datasets
- **Task 3** can use heuristics for curriculum learning

## 📁 **Repository Architecture**

```
ROOT/
├── core/                    # 🏛️ First-citizen engine (Task-0)
│   ├── game_*.py           # BaseGameManager, BaseGameLogic, etc.
│   └── agents.py           # BaseAgent protocol (universal interface)
│
├── gui/                     # 🎨 First-citizen visualization  
│   ├── base_gui.py         # BaseGUI (extensible by second-citizens)
│   ├── game_gui.py         # Task-0 PyGame implementation
│   └── replay_gui.py       # Task-0 replay viewer
│
├── llm/                     # 🤖 First-citizen LLM integration
│   ├── agent_llm.py        # LLM implementation of BaseAgent
│   └── providers/          # LLM provider abstractions
│
├── web/                     # 🌐 First-citizen Flask site
│   ├── templates/          # Task-0 web interface
│   └── static/             # Task-0 assets
│
├── extensions/              # 🧪 Second-citizen research tracks
│   ├── common/             # Shared utilities (non-essential)
│   ├── heuristics-v0.0X/   # Task-1: Classical algorithms
│   ├── supervised-v0.0X/   # Task-2: Neural networks
│   ├── reinforcement-v0.0X/# Task-3: RL agents  
│   ├── llm-finetune-v0.0X/ # Task-4: LLM fine-tuning
│   └── distillation-v0.0X/ # Task-5: Knowledge distillation
│
└── logs/                    # 📊 Data & artifacts
    ├── [task-0-sessions]/  # First-citizen logs  
    └── extensions/         # Second-citizen outputs
        ├── datasets/grid-size-N/{extension_type}_v{version}_{timestamp}/{algorithm_name}/processed_data/
        └── models/grid-size-N/{extension_type}_v{version}_{timestamp}/{model_name}/model_artifacts/
```

## 🏗️ **Mandatory Extension Components**

Every standalone extension (v0.02+) **must** include:

| Component | Versions | Purpose |
|-----------|----------|---------|
| **`README.md`** | v0.02+ | Documentation entrypoint, quick-start guide |
| **`agents/`** | v0.02+ | Policy implementations (`agent_<algo>.py`) |
| **`dashboard/`** | v0.03+ | Streamlit UI components |
| **`scripts/`** | v0.03+ | CLI tools + `app.py` launcher |

**Evolution Pattern:**
- **v0.01**: Proof-of-concept (minimal structure)
- **v0.02**: Multi-algorithm support (`agents/` folder)  
- **v0.03**: Web dashboards (`dashboard/`, `scripts/`)
- **v0.04**: JSONL generation (*heuristics only*)

## 🔄 **Data Lineage & Storage**

### **Standardized Paths**
```
logs/extensions/
├── datasets/grid-size-N/           # 📊 Training datasets  
│   ├── heuristics_v0.03_20250625_143022/bfs/processed_data/tabular_data.csv
│   ├── heuristics_v0.03_20250625_143022/bfs/processed_data/sequential_data.npz
│   └── heuristics_v0.04_20250625_143022/bfs/processed_data/reasoning_data.jsonl
│
└── models/grid-size-N/             # 🧠 Trained models
    ├── pytorch/                    # Neural networks
    ├── lightgbm/                   # Tree models  
    └── transformers/               # Fine-tuned LLMs
```

### **Critical Dependencies**
**heuristics-v0.04** → **JSONL trajectories** → **Task 4 & 5**


Without heuristics-v0.04 JSONL output, Task 4 (LLM fine-tuning) and Task 5 (distillation) cannot begin. This creates a clear dependency chain that ensures data quality and consistency.


## 🎯 Extension Deep Dive

### 🧠 Task 1: Heuristic Agents

**Algorithms**: BFS, A*, Hamiltonian paths, wall-following
**Key Output**: CSV datasets + **JSONL with reasoning explanations**
**Success Metrics**: >80% apple efficiency, 100% survival (Hamiltonian)

### 🎓 Task 2: Supervised Learning  

**Models**: MLP, CNN, LSTM, XGBoost, LightGBM, Graph Neural Networks
**Data Source**: Heuristics CSV datasets exclusively
**Training**: Multi-framework support (PyTorch, scikit-learn, etc.)

### 🎮 Task 3: Reinforcement Learning

**Algorithms**: DQN, PPO, A3C, SAC (with/without Gymnasium)
**Environment**: OpenAI Gym wrapper around Snake core
**Features**: Live Q-value visualization, training curve monitoring

### 🤖 Task 4: LLM Fine-tuning

**Technique**: LoRA/QLoRA via Hugging Face PEFT  
**Data Source**: heuristics-v0.04 JSONL trajectories
**Models**: Instruction-tuned LLMs adapted for Snake gameplay

### 📦 Task 5: Knowledge Distillation

**Approach**: Teacher (fine-tuned LLM) → Student (smaller model)
**Loss**: α·CrossEntropy + β·KL divergence
**Output**: Deployment-ready compressed models

## 🎨 **Multi-Modal Interface Strategy**

Each extension provides **two presentation layers**:

1. **Flask Blueprint** → Integration with main web interface  
2. **PyGame** (`gui_*.py`) → Desktop visualization with overlays

**Performance Modes:**
- `--use-gui` (default): Real-time PyGame rendering
- `--no-gui`: Headless mode for high-speed training/dataset generation

## 🔧 **Technical Standards**

### **Architecture Patterns**
- **SOLID principles** throughout
- **Base classes** in `core/` extended by second-citizens
- **Factory pattern** for agent creation
- **Observer pattern** for GUI updates
- **Singleton pattern** for file managers

### **Code Quality**
- **Python 3.10+** with type hints
- **Black** formatting, **Ruff** linting, **Mypy** type checking
- **Comprehensive docstrings** explaining design patterns used

### **Sentinel Values**
- **`EMPTY`**: Task-0 only (LLM parsing failures)
- **`SOMETHING_IS_WRONG`**: Task-0 only
- **`INVALID_REVERSALS`**: Shared across all tasks
- **`NO_PATH_FOUND`**: Shared across all tasks. E.g. Task-0 LLM tells us that there is no path found. Or, in the case of heuristics, the heuristics tells us that there is no path found.

## 📋 **Success Criteria**

✅ **Architectural Integrity**: Task-0 never breaks due to extension changes  
✅ **Data Quality**: Grid-size aware storage, proper metadata  
✅ **Interface Consistency**: All extensions follow component requirements  
✅ **Educational Value**: Rich docstrings demonstrating design patterns

---

**Task-0 first, everything else second.**