> **Important — Authoritative Reference:** This document supplements Final Decision 1 (Directory Structure). **For conflicts, Final Decision 1 prevails.**

# Dataset and Model Directory Structure

This document defines the **standardized directory organization** for datasets and models across all Snake Game AI extensions, following the architecture established in Final Decision 1.

## 🎯 **Core Design Philosophy**

### **Grid-Size Agnostic Organization**
Directory structure is designed to be **grid-size independent** while maintaining clear separation between different grid configurations, enabling flexible experimentation across different board sizes.

### **Multi-Directional Data Ecosystem**
Unlike traditional linear pipelines, our architecture recognizes that:
- **All tasks generate datasets** during training/evaluation
- **Better models create better datasets** through positive feedback loops
- **Cross-task pollination** improves overall system performance
- **Training produces both models AND datasets simultaneously**

## 📁 **Standardized Directory Structure**

### **Datasets Organization**
```
logs/extensions/datasets/
└── grid-size-N/                          # Grid-size specific organization
    ├── heuristics_v0.03_{timestamp}/      # Task 1 → Tasks 2-5 (Foundation)
    │   ├── bfs/
    │   │   ├── game_logs/                 # Original game data
    │   │   └── processed_data/
    │   │       ├── tabular_data.csv       # For supervised learning
    │   │       └── metadata.json
    │   └── astar/ [same structure]
    │
    ├── heuristics_v0.04_{timestamp}/      # Task 1 → Task 4 (LLM Fine-tuning)
    │   ├── bfs/
    │   │   └── processed_data/
    │   │       ├── tabular_data.csv       # Legacy format
    │   │       ├── reasoning_data.jsonl   # 🔥 For LLM fine-tuning
    │   │       └── metadata.json
    │   └── astar/ [same structure]
    │
    ├── supervised_v0.02_{timestamp}/      # Task 2 → Others (Improved datasets)
    ├── reinforcement_v0.02_{timestamp}/   # Task 3 → Others (Optimal datasets)
    ├── llm_finetune_v0.02_{timestamp}/    # Task 4 → Others (Language-grounded)
    └── llm_distillation_v0.02_{timestamp}/ # Task 5 → Others (Efficient)
```

**Standard Format**: `logs/extensions/datasets/grid-size-N/{extension}_v{version}_{timestamp}/`

### **Models Organization**
```
logs/extensions/models/
└── grid-size-N/
    ├── supervised_v0.02_{timestamp}/
    │   ├── mlp/
    │   │   ├── model_artifacts/           # Primary model outputs
    │   │   │   ├── model.pth
    │   │   │   ├── model.onnx
    │   │   │   └── config.json
    │   │   └── training_process/
    │   │       └── generated_datasets/    # 🔥 Datasets created during training
    │   └── xgboost/ [same structure]
    │
    ├── reinforcement_v0.02_{timestamp}/
    └── llm_finetune_v0.02_{timestamp}/
```

## 🔄 **Data Flow Benefits**

### **Performance Hierarchy Integration**
Expected progression generally follows: **Heuristics** → **Supervised** → **Reinforcement** → **LLM Fine-tuned** → **LLM Distilled**

### **Cross-Task Data Enhancement**
- **Heuristics** provide baseline datasets with algorithmic traces
- **Supervised** generate confidence-scored datasets with faster inference
- **Reinforcement** create potentially optimal datasets with exploration data
- **LLM Fine-tuned** produce language-grounded datasets with explanations
- **LLM Distilled** generate efficient datasets optimized for deployment

## 📊 **Path Management Integration**

All extensions **MUST** use standardized path utilities from Final Decision 6:

```python
from extensions.common.path_utils import get_dataset_path, get_model_path

# Grid-size agnostic path generation
dataset_path = get_dataset_path(
    extension_type="heuristics", 
    version="0.03",
    grid_size=grid_size,  # Any supported size
    algorithm="bfs",
    timestamp=timestamp
)
```

## 🎯 **Extension Compliance Requirements**

### **All Extensions Must:**
- Use the standardized `grid-size-N/` hierarchy
- Generate paths using `extensions/common/path_utils.py`
- Follow dataset format specifications from Final Decision 2
- Maintain version-specific naming conventions
- Support multi-directional data consumption and generation

### **Supported Extensions:**
- **Heuristics**: v0.01, v0.02, v0.03, v0.04
- **Supervised**: v0.01, v0.02, v0.03
- **Reinforcement**: v0.01, v0.02, v0.03
- **LLM Fine-tuning**: v0.01, v0.02, v0.03
- **LLM Distillation**: v0.01, v0.02, v0.03

---

**This directory structure ensures consistent, scalable organization while supporting the multi-directional data ecosystem that enables continuous improvement across all algorithm types.**

