# CSV Schema for Snake Game Extensions

## 🎯 **Core Philosophy: Grid-Size Agnostic Design**

The CSV schema uses a **fixed set of 16 engineered features** that work for any grid size (8x8, 10x10, 12x12, 16x16, 20x20, etc.), ensuring consistency across all extensions and enabling cross-grid-size comparisons, strictly following forward-looking architecture principles.

**When to Use CSV**: This format is optimal for tree-based models (XGBoost, LightGBM) and simple MLPs. For other model types, see the data format decision guide.

## 📊 **Standardized Schema Structure**

### **Fixed Feature Set (16 features)**

| Feature Category | Features | Description |
|------------------|----------|-------------|
| **Position** | `head_x`, `head_y`, `apple_x`, `apple_y` | Absolute coordinates |
| **Game State** | `snake_length` | Current snake length |
| **Apple Direction** | `apple_dir_up`, `apple_dir_down`, `apple_dir_left`, `apple_dir_right` | Binary directional flags |
| **Danger Detection** | `danger_straight`, `danger_left`, `danger_right` | Binary collision risk flags |
| **Free Space** | `free_space_up`, `free_space_down`, `free_space_left`, `free_space_right` | Free cell counts per direction |

### **Metadata & Target (3 columns)**
- `game_id`: Unique game session identifier
- `step_in_game`: Step number within the game
- `target_move`: The move taken (UP, DOWN, LEFT, RIGHT)

**Total: 19 columns** (2 metadata + 16 features + 1 target)

## 🧠 **Design Benefits**

### **Grid-Size Independence**
- **Universal Features**: Same 16 features work for any board size
- **Scalable Performance**: Efficient feature extraction across all grid configurations
- **Cross-Grid Compatibility**: Enables training on one size, testing on another

### **Format Selection Decision Matrix**
> **Authoritative Reference**: See `data-format-decision-guide.md` for complete format selection criteria.

The 16-feature tabular schema is **specifically designed** for certain algorithm types:

| Representation Type | Best For | Data Format | Use Cases |
|-------------------|----------|-------------|-----------|
| **16-Feature Tabular** | XGBoost, LightGBM, Random Forest, Simple MLP | CSV (from heuristics-v0.04) | Tree-based models, traditional ML |
| **Sequential (NPZ)** | LSTM, GRU, Temporal Models | NPZ arrays | Time-series analysis, RNN architectures |
| **Spatial (2D Arrays)** | CNN, Computer Vision Models | 2D numpy arrays | Image-like processing, spatial patterns |
| **Graph Structures** | GNN, Relationship-based Models | Graph formats | Complex relationships, network analysis |
| **Raw Board State** | Evolutionary Algorithms, GP | Full grid matrices | Population-based optimization |

### **When to Use 16-Feature Schema**
- ✅ **Tree-based models**: XGBoost, LightGBM, Random Forest
- ✅ **Simple MLPs**: Fully connected networks with tabular input
- ✅ **Traditional ML**: SVM, Logistic Regression, Naive Bayes
- ✅ **Fast inference**: Minimal preprocessing required
- ❌ **CNNs**: Use spatial 2D arrays instead
- ❌ **RNNs**: Use sequential NPZ format instead
- ❌ **GNNs**: Use graph representation instead
- ❌ **Evolutionary Algorithms**: Use specialized NPZ raw arrays instead

### **Cross-Extension Integration**
- **Heuristics v0.04 (DEFINITIVE)**: Generates standardized CSV datasets for supervised learning
- **Supervised v0.02+**: Consumes CSV datasets from heuristics-v0.04 for training all model types
- **Evaluation**: Consistent comparison framework across all algorithm types

## 📁 **Path Integration**

> **Authoritative Reference**: See `unified-path-management-guide.md` for complete path management standards.

Uses standardized path management from `unified-path-management-guide.md`:

```python
from extensions.common.utils.path_utils import get_dataset_path

# Standardized path generation with enforced format
dataset_path = get_dataset_path(
    extension_type="heuristics", 
    version="0.04",  # 🎯 Use v0.04 - it's the definitive version
    grid_size=grid_size,  # Any supported size
    algorithm="bfs",
    timestamp=timestamp  # Format: YYYYMMDD_HHMMSS
)
# Result: logs/extensions/datasets/grid-size-{grid_size}/heuristics_v0.04_{timestamp}/
```

## 🔧 **Usage Examples**

### **Dataset Generation**
```python
from extensions.common.utils.csv_utils import create_csv_record

csv_row = create_csv_record(
    game_state=current_state,
    move="RIGHT", 
    step_number=5
)
```

### **Dataset Loading**
```python
from extensions.common.utils.dataset_utils import load_dataset_for_training

X_train, X_val, X_test, y_train, y_val, y_test, info = load_dataset_for_training(
    dataset_paths=["path/to/heuristics_v0.04_dataset.csv"],  # Use v0.04
    grid_size=grid_size  # Validates compatibility
)
```

## 🎯 **Extension Benefits**

### **Heuristics Extensions**
- **v0.04 (DEFINITIVE)**: Consistent dataset generation across all algorithms (BFS, A*, Hamiltonian)
- Grid-size independence enables flexible experimentation
- Rich feature capture of algorithmic decision patterns
- CSV format actively used for supervised learning

### **Supervised Learning Extensions**
- Standardized input format across all model types (MLP, CNN, XGBoost, etc.)
- Efficient training with appropriately sized feature vectors
- Cross-algorithm comparison using identical feature space
- Use CSV from heuristics-v0.04 for optimal results

### **Research Applications**
- Reproducible experiments across different grid sizes
- Consistent evaluation metrics and feature interpretability
- Transfer learning possibilities between different configurations
- Use heuristics-v0.04 as the definitive data source

## 🏗️ **Architecture Overview**

### **File Structure**
```
extensions/common/
├── config/
│   └── csv_formats.py          # Centralized CSV constants and schema
└── utils/
    └── csv_utils.py            # Consolidated CSV utilities
```

### **Key Components**

#### **1. CSV Formats (`csv_formats.py`)**
```python
from extensions.common.config.csv_formats import (
    CSV_ALL_COLUMNS, CSV_FEATURE_COLUMNS, CSV_TARGET_COLUMN
)

# Core schema definition
CSV_FEATURE_COLUMNS = [
    'head_x', 'head_y', 'apple_x', 'apple_y',  # Position (4)
    'snake_length',                              # Game state (1)
    'apple_dir_up', 'apple_dir_down', 'apple_dir_left', 'apple_dir_right',  # Direction (4)
    'danger_straight', 'danger_left', 'danger_right',  # Danger (3)
    'free_space_up', 'free_space_down', 'free_space_left', 'free_space_right'  # Space (4)
]
```

#### **2. CSV Utilities (`csv_utils.py`)**
```python
from extensions.common.utils.csv_utils import CSVFeatureExtractor, create_csv_record

# Feature extraction
extractor = CSVFeatureExtractor()
features = extractor.extract_features(game_state, move, step_number)

# Simple record creation
csv_record = create_csv_record(game_state, move, step_number)
```

### **Design Patterns**

#### **Strategy Pattern**
- Different feature extraction strategies for different model types
- Pluggable preprocessing pipelines
- Consistent interface across extensions

#### **Template Method Pattern**
- Base classes define structure
- Extensions implement specific logic
- Consistent behavior across implementations

#### **Factory Pattern**
- Schema generation based on grid size
- Agent creation based on model type
- Dataset loading based on format

## 📈 **Performance Considerations**

- **Feature count is fixed**: 16 features regardless of grid size
- **Efficient lookups**: Uses sets for snake body collision detection
- **Memory efficient**: Processes data in chunks
- **Scalable**: Works with datasets of any size

## 🔮 **Future Extensions**

The schema is designed to be extensible:

1. **Additional features**: Can add new engineered features without breaking existing models
2. **Different formats**: Support for sequential data (LSTM) using NPZ and graph data (GNN) using specialized formats

## 🔗 **See Also**

- **`data-format-decision-guide.md`**: Authoritative reference for all format decisions
- **`evolutionary.md`**: Alternative state representations for evolutionary algorithms
- **`datasets-folder.md`**: Directory structure and organization standards
- **`unified-path-management-guide.md`**: Path management standards

## 🎯 **Important Guidelines: Version Selection Guidelines**

- **For supervised learning**: Use CSV from heuristics-v0.04 (definitive version)
- **For LLM fine-tuning**: Use JSONL from heuristics-v0.04 only
- **For research**: Use both formats from heuristics-v0.04
- **CSV is ACTIVE**: Not legacy - actively used for supervised learning
- **JSONL is ADDITIONAL**: New capability for LLM fine-tuning (heuristics-v0.04 only)

---

**This grid-size agnostic CSV schema ensures consistent, scalable datasets for supervised learning across all Snake Game AI extensions while maintaining cross-extension compatibility and following forward-looking architecture principles.** 