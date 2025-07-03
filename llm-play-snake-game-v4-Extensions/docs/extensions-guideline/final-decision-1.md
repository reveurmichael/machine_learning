# Final Decision 1: Directory Structure & Data Organization Architecture

> **SUPREME AUTHORITY**: This document establishes the definitive directory structure and data organization standards for the Snake Game AI project.

> **See also:** `datasets-folder.md` (Dataset organization), `data-format-decision-guide.md` (Format selection), `unified-path-management-guide.md` (Path standards), `final-decision-10.md` (SUPREME_RULES).

## 🎯 **Core Philosophy: Grid-Size Agnostic Multi-Directional Data Ecosystem**

The directory structure implements a sophisticated **multi-directional data ecosystem** where:
- **All tasks generate datasets** during training/evaluation
- **Better models create better datasets** through positive feedback loops
- **Cross-task pollination** improves overall system performance
- **Training produces both models AND datasets simultaneously**

### **SUPREME_RULES Integration**
- **SUPREME_RULE NO.1**: Enforces reading all GOOD_RULES before making directory structure changes to ensure comprehensive understanding
- **SUPREME_RULE NO.2**: Uses precise `final-decision-N.md` format consistently when referencing architectural decisions
- **SUPREME_RULE NO.3**: Enables lightweight common utilities with OOP extensibility while maintaining directory structure patterns through inheritance rather than tight coupling
- **SUPREME_RULE NO.4**: Ensures all markdown files are coherent and aligned through nuclear diffusion infusion process

### **GOOD_RULES Integration**
This document integrates with the **GOOD_RULES** governance system established in `final-decision-10.md`:
- **`datasets-folder.md`**: Authoritative reference for dataset directory organization standards
- **`data-format-decision-guide.md`**: Authoritative reference for format selection criteria
- **`unified-path-management-guide.md`**: Authoritative reference for path management standards
- **`single-source-of-truth.md`**: Ensures no duplication across extension guidelines

## 🚫 **CRITICAL: NO singleton_utils.py in extensions/common/**

**STOP! READ THIS FIRST**: This project **explicitly FORBIDS**:
- ❌ **singleton_utils.py in extensions/common/utils/**
- ❌ **Any wrapper around ROOT/utils/singleton_utils.py**
- ✅ **USE ROOT/utils/singleton_utils.py** directly when truly needed (it's already generic)

## 🎯 **Executive Summary**

This document establishes the **definitive directory structure** for organizing datasets and models in the `./logs/extensions/` folder across all Snake Game AI tasks (1-5). The structure reflects the **multi-directional data ecosystem** where all tasks can both **consume and generate** high-quality datasets and models, strictly following SUPREME_RULES from `final-decision-10.md`.

### **Simple Logging Examples (SUPREME_RULE NO.3)**
All code examples in this document follow **SUPREME_RULE NO.3** by using ROOT/utils/print_utils.py functions rather than complex logging mechanisms:

```python
from utils.print_utils import print_info, print_warning, print_error, print_success

# ✅ CORRECT: Simple logging as per SUPREME_RULE NO.3
def create_dataset_directory(extension_type: str, version: str, grid_size: int):
    """Create dataset directory with proper structure using canonical factory pattern"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/extensions/datasets/grid-size-{grid_size}/{extension_type}_v{version}_{timestamp}"
    
    os.makedirs(path, exist_ok=True)
    print_info(f"[DatasetManager] Created dataset directory: {path}")  # SUPREME_RULE NO.3
    
    return path

def validate_directory_structure(path: str):
    """Validate directory follows required structure"""
    if not os.path.exists(path):
        print_error(f"[Validator] Directory does not exist: {path}")  # SUPREME_RULE NO.3
        return False
    
    required_files = ["processed_data", "game_logs"]
    for file in required_files:
        if not os.path.exists(os.path.join(path, file)):
            print_error(f"[Validator] Missing required directory: {file}")  # SUPREME_RULE NO.3
            return False
    
    print_success(f"[Validator] Directory structure is valid: {path}")  # SUPREME_RULE NO.3
    return True
```

## 🧠 **Key Architectural Insights**

### **1. Multi-Directional Data Flow**
Unlike traditional linear pipelines, our ecosystem recognizes that:
- **All tasks generate datasets** during training/evaluation
- **Better models create better datasets** (positive feedback loop)
- **Cross-task pollination** improves overall system performance
- **Training produces both models AND datasets simultaneously**

### **2. Task Performance Hierarchy**
Expected performance progression (generally):
1. **Heuristics** (Task 1): Baseline, interpretable, deterministic
2. **Supervised** (Task 2): Better than heuristics, fast inference
3. **Reinforcement Learning** (Task 3): **Potentially optimal**, learned through exploration
4. **LLM Fine-tuned** (Task 4): High performance + natural language reasoning
5. **LLM Distilled** (Task 5): Efficient while maintaining quality

### **3. Dataset Quality Features by Source**

| Task Type | Performance | Interpretability | Speed | Special Features |
|-----------|-------------|------------------|-------|------------------|
| **Heuristics** | Baseline | Highest | Fast | Algorithm traces, search paths |
| **Supervised** | Good | Medium | Very Fast | Confidence scores, feature importance |
| **Reinforcement** | Potentially Optimal | Low | Fast | Q-values, policy distributions, exploration data |
| **LLM Fine-tuned** | High | Highest | Slow | Natural language explanations, step-by-step reasoning |
| **LLM Distilled** | Good | High | Medium | Compressed reasoning, efficiency optimized |

## 📁 **Final Directory Structure**

### **Datasets Organization**

```
logs/extensions/datasets/
└── grid-size-N/
    ├── heuristics_v0.03_{timestamp}/          # Task 1 → Task 2 (Traditional ML)
    │   ├── bfs/
    │   │   ├── game_logs/                     # Original game_N.json, summary.json
    │   │   └── processed_data/
    │   │       ├── tabular_data.csv           # For supervised learning
    │   │       ├── sequential_data.npz        # For RNN/LSTM
    │   │       └── metadata.json
    │   └── astar/ [same structure]
    │
    ├── heuristics_v0.04_{timestamp}/          # Task 1 → Task 4 (LLM Fine-tuning)
    │   ├── bfs/
    │   │   ├── game_logs/                     # Original game_N.json, summary.json
    │   │   └── processed_data/
    │   │       ├── tabular_data.csv           # Active format for supervised learning
    │   │       ├── reasoning_data.jsonl       # LLM fine-tuning format
    │   │       └── metadata.json
    │   └── astar/ [same structure]
    │
    ├── supervised_v0.02_{timestamp}/          # Task 2 → Others (Improved datasets)
    │   ├── mlp_generated/
    │   │   ├── game_logs/                     # MLP-generated games
    │   │   └── processed_data/
    │   │       ├── tabular_data.csv           # Higher quality features
    │   │       ├── sequential_data.npz        # Better temporal patterns
    │   │       ├── confidence_scores.csv      # Model confidence per move
    │   │       └── metadata.json
    │   ├── xgboost_generated/
    │   └── ensemble_generated/                # Best supervised performance
    │
    ├── reinforcement_v0.02_{timestamp}/       # Task 3 → Others (Optimal datasets)
    │   ├── dqn_generated/
    │   │   ├── game_logs/                     # DQN-generated games
    │   │   ├── experience_replay/
    │   │   │   ├── transitions.npz            # High-quality state transitions
    │   │   │   └── episode_data.json
    │   │   └── processed_data/
    │   │       ├── tabular_data.csv           # Optimal policy features
    │   │       ├── q_values.npz               # Q-value annotations
    │   │       ├── action_probabilities.npz   # Policy distributions
    │   │       └── metadata.json
    │   ├── ppo_generated/
    │   └── curriculum_generated/              # Curriculum learning datasets
    │
    ├── llm_finetune_v0.02_{timestamp}/        # Task 4 → Others (Language-grounded)
    │   ├── lora_generated/
    │   │   ├── game_logs/                     # LLM-generated games
    │   │   └── processed_data/
    │   │       ├── reasoning_data.jsonl       # Rich explanations
    │   │       ├── tabular_data.csv           # Human-interpretable features
    │   │       ├── language_features.npz      # Embedding-based features
    │   │       └── metadata.json
    │   └── full_model_generated/
    │
    └── llm_distillation_v0.02_{timestamp}/    # Task 5 → Others (Efficient)
        ├── student_generated/
        │   ├── game_logs/                     # Distilled model games
        │   └── processed_data/
        │       ├── reasoning_data.jsonl       # Compressed explanations
        │       ├── efficiency_metrics.csv     # Speed/quality trade-offs
        │       └── metadata.json
        └── ensemble_generated/
```

### **Models Organization (Integrated with Dataset Generation)**

```
logs/extensions/models/
└── grid-size-N/
    ├── supervised_v0.02_{timestamp}/
    │   ├── mlp/
    │   │   ├── model_artifacts/
    │   │   │   ├── model.pth                      # Primary model output
    │   │   │   ├── model.onnx                     # Deployment format
    │   │   │   ├── config.json                    # Model configuration
    │   │   │   └── feature_importance.json        # Model interpretability
    │   │   ├── training_process/
    │   │   │   ├── training_history/
    │   │   │   │   ├── loss_curves.json
    │   │   │   │   ├── metrics_per_epoch.json
    │   │   │   │   └── checkpoints/
    │   │   │   └── generated_datasets/            # 🔥 Datasets created during training
    │   │   │       ├── validation_games/
    │   │   │       │   ├── game_N.json            # Games played during validation
    │   │   │       │   └── summary.json
    │   │   │       ├── evaluation_datasets/
    │   │   │       │   ├── tabular_data.csv       # Features from evaluation games
    │   │   │       │   ├── confidence_scores.csv  # Model confidence per move
    │   │   │       │   └── prediction_analysis.npz
    │   │   │       └── dataset_metadata.json
    │   │   └── deployment_ready/
    │   │       ├── optimized_model.onnx           # Production-ready
    │   │       └── inference_config.json
    │   └── xgboost/ [same structure]
    │
    ├── reinforcement_v0.02_{timestamp}/
    │   ├── dqn/
    │   │   ├── model_artifacts/
    │   │   │   ├── policy_network.pth             # Primary RL model
    │   │   │   ├── target_network.pth             # Target network
    │   │   │   ├── config.json                    # RL hyperparameters
    │   │   │   └── training_metrics.json
    │   │   ├── training_process/
    │   │   │   ├── training_history/
    │   │   │   │   ├── loss_curves.json
    │   │   │   │   ├── reward_curves.json
    │   │   │   │   └── exploration_metrics.json
    │   │   │   └── generated_datasets/            # 🔥 High-quality RL datasets
    │   │   │       ├── optimal_games/
    │   │   │       │   ├── game_N.json            # Games with optimal policy
    │   │   │       │   └── summary.json
    │   │   │       ├── experience_datasets/
    │   │   │       │   ├── transitions.npz        # State-action-reward tuples
    │   │   │       │   ├── q_values.npz           # Q-value annotations
    │   │   │       │   └── policy_distributions.npz
    │   │   │       └── dataset_metadata.json
    │   │   └── deployment_ready/
    │   │       ├── optimized_policy.onnx          # Production-ready
    │   │       └── inference_config.json
    │   └── ppo/ [same structure]
    │
    ├── llm_finetune_v0.02_{timestamp}/
    │   ├── lora/
    │   │   ├── model_artifacts/
    │   │   │   ├── adapter_model.safetensors      # LoRA adapter weights
    │   │   │   ├── base_model_config.json         # Base model configuration
    │   │   │   ├── lora_config.json               # LoRA-specific config
    │   │   │   └── training_metrics.json
    │   │   ├── training_process/
    │   │   │   ├── training_history/
    │   │   │   │   ├── loss_curves.json
    │   │   │   │   ├── learning_rate_schedule.json
    │   │   │   │   └── validation_metrics.json
    │   │   │   └── generated_datasets/            # 🔥 Language-grounded datasets
    │   │   │       ├── reasoning_games/
    │   │   │       │   ├── game_N.json            # Games with explanations
    │   │   │       │   └── summary.json
    │   │   │       ├── language_datasets/
    │   │   │       │   ├── reasoning_data.jsonl   # Rich explanations
    │   │   │       │   ├── language_features.npz  # Embedding-based features
    │   │   │       │   └── explanation_quality.csv
    │   │   │       └── dataset_metadata.json
    │   │   └── deployment_ready/
    │   │       ├── merged_model.safetensors        # Production-ready
    │   │       └── inference_config.json
    │   └── full_model/ [same structure]
    │
    └── llm_distillation_v0.02_{timestamp}/
        ├── student/
        │   ├── model_artifacts/
        │   │   ├── student_model.pth               # Distilled student model
        │   │   ├── teacher_model.pth               # Reference teacher model
        │   │   ├── config.json                     # Distillation config
        │   │   └── efficiency_metrics.json
        │   ├── training_process/
        │   │   ├── training_history/
        │   │   │   ├── distillation_loss.json
        │   │   │   ├── efficiency_curves.json
        │   │   │   └── quality_metrics.json
        │   │   └── generated_datasets/            # 🔥 Efficient datasets
        │   │       ├── efficient_games/
        │   │       │   ├── game_N.json            # Games with efficiency focus
        │   │       │   └── summary.json
        │   │       ├── efficiency_datasets/
        │   │       │   ├── reasoning_data.jsonl   # Compressed explanations
        │   │       │   ├── efficiency_metrics.csv # Speed/quality trade-offs
        │   │       │   └── compression_analysis.npz
        │   │       └── dataset_metadata.json
        │   └── deployment_ready/
        │       ├── optimized_student.onnx          # Production-ready
        │       └── inference_config.json
        └── ensemble/ [same structure]
```

### **Extension Guidelines**
This directory structure supports:
- All extension types (heuristics, supervised, reinforcement, LLM)
- All grid sizes (8x8, 10x10, 12x12, 16x16, 20x20)
- All data formats (CSV, NPZ, JSONL)
- Cross-extension data sharing and reuse

---

**This directory structure ensures consistent, scalable, and educational data organization across all Snake Game AI extensions while enabling the multi-directional data ecosystem that drives continuous improvement.**

> **SUPREME_RULES COMPLIANCE**: This document strictly follows the SUPREME_RULES established in `final-decision-10.md`, ensuring coherence, educational value, and architectural integrity across the entire project.
