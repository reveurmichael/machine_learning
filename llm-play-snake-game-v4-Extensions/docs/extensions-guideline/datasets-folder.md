# Dataset and Model Directory Structure

> **SUPREME_RULES**: Both `heuristics-v0.03` and `heuristics-v0.04` are widely used depending on use cases and scenarios. For supervised learning and other general purposes, both versions can be used. For LLM fine-tuning, only `heuristics-v0.04` will be used. The CSV format is **NOT legacy** - it's actively used and valuable for supervised learning.

This document defines the **standardized directory organization** for datasets and models across all Snake Game AI extensions.

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
    ├── heuristics_v0.04_{timestamp}/      # 🎯 DEFINITIVE VERSION - Use this!
    │   ├── bfs/
    │   │   ├── game_logs/                 # Original game data
    │   │   └── processed_data/
    │   │       ├── tabular_data.csv       # ✅ ACTIVE: For supervised learning
    │   │       ├── reasoning_data.jsonl   # 🔥 NEW: For LLM fine-tuning
    │   │       └── metadata.json
    │   └── astar/ [same structure]
    │
    ├── heuristics_v0.03_{timestamp}/      # STILL SUPPORTED: Widely used in production; v0.04 recommended for new work
    │   ├── bfs/
    │   │   ├── game_logs/                 # Original game data
    │   │   └── processed_data/
    │   │       ├── tabular_data.csv       # ✅ Still valuable, but v0.04 is better
    │   │       └── metadata.json
    │   └── astar/ [same structure]
    │
    ├── supervised_v0.02_{timestamp}/      # Task 2 → Others (Improved datasets)
    ├── reinforcement_v0.02_{timestamp}/   # Task 3 → Others (Optimal datasets)
    ├── llm_finetune_v0.02_{timestamp}/    # Task 4 → Others (Language-grounded)
    └── llm_distillation_v0.02_{timestamp}/ # Task 5 → Others (Efficient)
```

**🎯 Standardized Directory Format**: `logs/extensions/datasets/grid-size-N/{extension}_v{version}_{timestamp}/`

### **SUPREME_RULES: Version Selection**
- **Prefer `heuristics-v0.04`** for new datasets: it is a strict superset of v0.03 and adds JSONL generation.
- **`heuristics-v0.03` remains fully supported and widely used** in existing pipelines.  Continue to use it when backward-compatibility or comparison with historical results is required.
- **CSV format is ACTIVE**: Not legacy – essential for supervised learning across both v0.03 and v0.04.
- **JSONL format is ADDITIONAL**: Available only in v0.04 for LLM fine-tuning use-cases.

### **Path Validation Utilities**
```python
def validate_dataset_path_structure(path: str) -> bool:
    """Validate dataset path follows standardized format"""
    import re
    pattern = r"logs/extensions/datasets/grid-size-\d+/\w+_v\d+\.\d+_\d{8}_\d{6}/"
    return bool(re.match(pattern, path))

def enforce_path_structure(extension_type: str, version: str, grid_size: int, timestamp: str) -> str:
    """Enforce standardized path structure"""
    return f"logs/extensions/datasets/grid-size-{grid_size}/{extension_type}_v{version}_{timestamp}/"
```

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

**🎯 Standardized Model Path Format**: `logs/extensions/models/grid-size-N/{extension}_v{version}_{timestamp}/`

## 🔄 **Data Flow Benefits**

### **Performance Hierarchy Integration**
Expected progression generally follows: **Heuristics v0.04** → **Supervised** → **Reinforcement** → **LLM Fine-tuned** → **LLM Distilled**

### **Cross-Task Data Enhancement**
- **Heuristics v0.04** provide baseline datasets with algorithmic traces + language explanations
- **Supervised** generate confidence-scored datasets with faster inference
- **Reinforcement** create potentially optimal datasets with exploration data
- **LLM Fine-tuned** produce language-grounded datasets with explanations
- **LLM Distilled** generate efficient datasets optimized for deployment

## 📊 **Path Management Integration**

All extensions **MUST** use standardized path utilities from `unified-path-management-guide.md`:

```python
from extensions.common.path_utils import get_dataset_path, get_model_path

# Standardized path generation with validation
dataset_path = get_dataset_path(
    extension_type="heuristics", 
    version="0.04",  # 🎯 Use v0.04 - it's the definitive version
    grid_size=grid_size,  # Any supported size
    algorithm="bfs",
    timestamp=timestamp  # Format: YYYYMMDD_HHMMSS
)
# Enforced result: logs/extensions/datasets/grid-size-{grid_size}/heuristics_v0.04_{timestamp}/
```

## 🎯 **Extension Compliance Requirements**

### **All Extensions Must:**
- Use the standardized `grid-size-N/` hierarchy
- Generate paths using `extensions/common/path_utils.py`
- Follow dataset format specifications from final-decision-2.md
- Maintain version-specific naming conventions
- Support multi-directional data consumption and generation

### **Supported Extensions:**
- **Heuristics**: v0.01, v0.02, v0.03, **v0.04 (DEFINITIVE)**
- **Supervised**: v0.01, v0.02, v0.03
- **Reinforcement**: v0.01, v0.02, v0.03
- **LLM Fine-tuning**: v0.01, v0.02, v0.03
- **LLM Distillation**: v0.01, v0.02, v0.03

### **SUPREME_RULES: Version Selection Guidelines**
- **For heuristics**: Prefer v0.04 for new experiments, but v0.03 is still valid and maintained.
- **For supervised learning**: Use CSV from either heuristics-v0.03 **or** heuristics-v0.04 (both widely used).
- **For LLM fine-tuning**: Use JSONL available in heuristics-v0.04.
- **For research & ablation studies**: Consider using **both** versions to assess the impact of language-rich explanations introduced in v0.04.

---

**This directory structure ensures consistent, scalable organization while supporting the multi-directional data ecosystem that enables continuous improvement across all algorithm types. Remember: heuristics-v0.04 is the definitive version to use.**

