# Unified Data Format Decision Guide

> **Authoritative Reference**: This document is the **single source of truth** for data format decisions across all extensions. It replaces all conflicting format recommendations in other documents.

## 🎯 **Core Philosophy: Right Format for Right Purpose**

Different algorithm types have fundamentally different data representation needs. This guide provides a clear decision matrix for selecting the optimal format based on your specific use case.

## 📊 **Authoritative Format Decision Matrix**

| Extension Type | Version | Primary Format | Use Case | File Extension |
|---------------|---------|---------------|----------|----------------|
| **Heuristics** | v0.03 | CSV (16-feature) | Training supervised models | `.csv` |
| **Heuristics** | v0.04 | JSONL | LLM fine-tuning datasets | `.jsonl` |
| **Supervised** | v0.02+ | CSV (16-feature) | Tree models (XGBoost, LightGBM) | `.csv` |
| **Supervised** | v0.02+ | NPZ Arrays | Sequential models (LSTM, GRU) | `.npz` |
| **Supervised** | v0.02+ | 2D Arrays | Spatial models (CNN) | `.npz` |
| **Reinforcement** | v0.02+ | NPZ Arrays | Experience replay buffers | `.npz` |
| **Evolutionary** | v0.02+ | Raw Arrays | Population-based optimization | `.npz` |

## 🧠 **Detailed Format Specifications**

### **CSV (16-Feature Tabular)**
**Best For**: XGBoost, LightGBM, Random Forest, Simple MLP
```python
# Fixed 16 features work for any grid size
features = [
    'head_x', 'head_y', 'apple_x', 'apple_y', 'snake_length',
    'apple_dir_up', 'apple_dir_down', 'apple_dir_left', 'apple_dir_right',
    'danger_straight', 'danger_left', 'danger_right',
    'free_space_up', 'free_space_down', 'free_space_left', 'free_space_right'
]
# Plus metadata: game_id, step_in_game, target_move
```

**When to Use**:
- ✅ Tree-based models requiring engineered features
- ✅ Simple neural networks with tabular input
- ✅ Fast inference with minimal preprocessing
- ❌ NOT for CNNs, RNNs, or GNNs

### **JSONL (JSON Lines)**
**Best For**: LLM Fine-tuning, Language-Grounded Learning
```json
{"prompt": "Current state description...", "completion": "Move RIGHT because..."}
```

**When to Use**:
- ✅ Training language models on heuristic reasoning
- ✅ Creating language-rich datasets with explanations
- ✅ Fine-tuning LLMs with prompt-completion pairs
- ❌ NOT for numerical optimization or traditional ML

### **NPZ (NumPy Arrays)**
**Best For**: Deep Learning, Sequential Models, RL
```python
# Sequential data for RNNs
np.savez('dataset.npz', 
         sequences=game_sequences,      # (batch, time, features)
         targets=move_sequences,        # (batch, time, 1)
         metadata=game_metadata)

# Experience replay for RL
np.savez('experience.npz',
         states=state_array,            # (batch, state_dim)
         actions=action_array,          # (batch, 1)
         rewards=reward_array,          # (batch, 1)
         next_states=next_state_array)  # (batch, state_dim)
```

**When to Use**:
- ✅ Sequential/temporal models (LSTM, GRU)
- ✅ Reinforcement learning experience buffers
- ✅ Large numerical datasets requiring compression
- ✅ Evolutionary algorithm population data

### **2D Arrays (Spatial)**
**Best For**: Computer Vision, CNNs, Spatial Analysis
```python
# Board state as image-like 2D array
board_representation = np.zeros((grid_size, grid_size, channels))
# channels: [snake_body, snake_head, apple, walls, ...]
```

**When to Use**:
- ✅ Convolutional neural networks
- ✅ Spatial pattern recognition
- ✅ Computer vision approaches to game state
- ❌ NOT for tabular or sequential models

## 🔄 **Format Selection Decision Tree**

```
Are you training an LLM?
├─ YES → Use JSONL format
└─ NO → What type of model?
    ├─ Tree-based (XGBoost, LightGBM) → Use CSV (16-feature)
    ├─ CNN (spatial patterns) → Use 2D Arrays (NPZ)
    ├─ RNN (temporal sequences) → Use Sequential NPZ
    ├─ RL (experience replay) → Use NPZ with state-action pairs
    └─ Evolutionary → Use Raw Arrays (NPZ)
```

## 📁 **Storage Path Standards**

All formats follow the standardized path structure:
```
logs/extensions/datasets/grid-size-N/{extension}_v{version}_{timestamp}/
├── {algorithm}/
│   ├── processed_data/
│   │   ├── tabular_data.csv        # For supervised learning
│   │   ├── reasoning_data.jsonl    # For LLM fine-tuning (heuristics-v0.04 only)
│   │   ├── sequential_data.npz     # For RNN/LSTM models
│   │   └── spatial_data.npz        # For CNN models
│   └── raw_data/
│       └── game_logs/              # Original game data
```

## 🚫 **Common Anti-Patterns to Avoid**

### **Don't Mix Formats Within Version**
```python
# ❌ WRONG: Same extension using multiple primary formats
heuristics-v0.03 → CSV AND JSONL  # Confusing!

# ✅ CORRECT: Clear format per version purpose
heuristics-v0.03 → CSV only       # For supervised learning
heuristics-v0.04 → JSONL only     # For LLM fine-tuning
```

### **Don't Use Wrong Format for Algorithm Type**
```python
# ❌ WRONG: Using CSV for CNNs
cnn_model.train(csv_data)  # Loses spatial structure!

# ✅ CORRECT: Use 2D arrays for CNNs
cnn_model.train(spatial_arrays)  # Preserves spatial relationships
```

### **Don't Create Custom Formats**
```python
# ❌ WRONG: Inventing new formats
pickle.dump(custom_format, file)  # Non-standard!

# ✅ CORRECT: Use established formats
np.savez(file, **standard_npz_format)  # Interoperable
```

## 🎯 **Format Validation**

Extensions MUST validate their data format compliance:
```python
# Required validation for all extensions
from extensions.common.validation import validate_dataset_format

def generate_dataset():
    dataset = create_dataset()
    
    # Validate format matches extension type and version
    validate_dataset_format(
        dataset_path=output_path,
        extension_type="heuristics",
        version="0.03",
        expected_format="csv"
    )
```

## 📚 **Migration Guide**

If you need to convert between formats:

### **CSV → NPZ (for deep learning)**
```python
from extensions.common.converters import csv_to_sequential_npz
sequential_data = csv_to_sequential_npz(csv_file, sequence_length=10)
```

### **Game Logs → JSONL (for LLM training)**
```python
from extensions.common.converters import game_logs_to_jsonl
jsonl_data = game_logs_to_jsonl(game_logs, include_reasoning=True)
```

---

**This unified guide eliminates format confusion and ensures each extension uses the optimal data representation for its intended purpose.** 