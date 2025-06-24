> **Important — Authoritative Reference:** This document supplements the Final Decision Series (final-decision-0 → final-decision-10). If conflict arises, the Final Decisions prevail.

# Dataset and Model Directory Structure

**✅ FINAL STRUCTURE ESTABLISHED**: Directory structure has been decided as per Final Decision 1.

This document defines the **single source of truth** for dataset and model organization across **all** extensions.

## 🎯 **Core Principle: Grid-Size Agnostic Organization**

The directory structure is designed to be **grid-size independent** while maintaining clear separation between different grid configurations:

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
    │   │       ├── tabular_data.csv           # Legacy format
    │   │       ├── reasoning_data.jsonl       # 🔥 For LLM fine-tuning
    │   │       └── metadata.json
    │   └── astar/ [same structure]
    │
    ├── supervised_v0.02_{timestamp}/          # Task 2 → Others (Improved datasets)
    ├── reinforcement_v0.02_{timestamp}/       # Task 3 → Others (Optimal datasets)
    ├── llm_finetune_v0.02_{timestamp}/        # Task 4 → Others (Language-grounded)
    └── llm_distillation_v0.02_{timestamp}/    # Task 5 → Others (Efficient)
```

### **Models Organization**
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
    │   │   └── training_process/
    │   │       ├── training_history/
    │   │       └── generated_datasets/            # 🔥 Datasets created during training
    │   └── xgboost/ [same structure]
    │
    ├── reinforcement_v0.02_{timestamp}/
    └── llm_finetune_v0.02_{timestamp}/
```

## 🔄 **Multi-Directional Data Flow**

Unlike traditional linear pipelines, our ecosystem recognizes that:
- **All tasks generate datasets** during training/evaluation
- **Better models create better datasets** (positive feedback loop)
- **Cross-task pollination** improves overall system performance
- **Training produces both models AND datasets simultaneously**

## 📊 **Extension Compliance**

### **Heuristics Extensions**
- heuristics-v0.01, v0.02, v0.03, v0.04

### **Supervised Learning Extensions**  
- supervised-v0.01, v0.02, v0.03

### **Reinforcement Learning Extensions**
- reinforcement-v0.01, v0.02

### **All Extensions Must**
- Use the standardized directory structure
- Generate grid-size specific paths using `extensions/common/path_utils.py`
- Follow the dataset format specifications from Final Decision 2
- Maintain backward compatibility within their version lineage

---

**This structure ensures consistent, scalable dataset and model organization across all extensions while supporting the multi-directional data ecosystem.**

