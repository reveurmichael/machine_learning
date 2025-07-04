# CSV Schema Utilities for Snake Game Extensions


## 🎯 Overview

The CSV schema utilities provide:
- **Flexible grid size support**: Works with any grid size (8x8, 10x10, 12x12, 16x16, 20x20, etc.)
- **Consistent feature engineering**: Same feature set across all grid sizes
- **Centralized data processing**: Used by both heuristics (for dataset generation) and supervised learning (for training)
- **Validation and error handling**: Ensures data integrity and compatibility

## 📊 **State Representation Selection Guide**

> **Authoritative Reference**: See `data-format-decision-guide.md` for complete format selection criteria.

**Choose the right representation for your algorithm type:**

### **16-Feature Tabular (CSV) - This Schema**
```python
# Best for: Tree-based models, simple MLPs, traditional ML
# Source: heuristics-v0.04 (definitive version) - actively used, NOT legacy
features = extract_tabular_features(game_state)  # Returns 16 features
model = XGBoostAgent(features)
```
**Use when**: XGBoost, LightGBM, Random Forest, simple neural networks

### **Sequential (NPZ) - Time Series**
```python
# Best for: LSTM, GRU, temporal analysis
sequence = extract_temporal_sequence(game_states)  # Multiple time steps
model = LSTMAgent(sequence)
```
**Use when**: RNN architectures, temporal pattern recognition

### **Spatial (2D Arrays) - Computer Vision**
```python
# Best for: CNN, computer vision approaches
board_image = game_state_to_2d_array(game_state)  # Grid as image
model = CNNAgent(board_image)
```
**Use when**: Convolutional networks, spatial pattern recognition

### **Graph Structures - Relationship Modeling**
```python
# Best for: GNN, relationship-based models
graph = game_state_to_graph(game_state)  # Nodes and edges
model = GCNAgent(graph)
```
**Use when**: Graph neural networks, complex relationship modeling

## 📊 CSV Schema Structure

### Fixed Feature Set (16 features)

The schema uses a fixed set of 16 engineered features that work for any grid size:

| Feature Category | Features | Description |
|------------------|----------|-------------|
| **Position** | `head_x`, `head_y`, `apple_x`, `apple_y` | Absolute coordinates |
| **Game State** | `snake_length` | Current snake length |
| **Apple Direction** | `apple_dir_up`, `apple_dir_down`, `apple_dir_left`, `apple_dir_right` | Binary flags for apple position relative to head |
| **Danger Detection** | `danger_straight`, `danger_left`, `danger_right` | Binary flags for immediate collision risk |
| **Free Space** | `free_space_up`, `free_space_down`, `free_space_left`, `free_space_right` | Count of free cells in each direction |

### Metadata Columns (2 columns)
- `game_id`: Unique game session identifier
- `step_in_game`: Step number within the game

### Target Column (1 column)
- `target_move`: The move taken (UP, DOWN, LEFT, RIGHT)

**Total: 19 columns** (2 metadata + 16 features + 1 target)

## 🔧 Key Components

### 1. CSVSchema Class
```python
from extensions.common.utils.csv_schema_utils import generate_csv_schema

# Generate schema for any grid size
schema = generate_csv_schema(grid_size=10)
print_info(f"Features: {schema.get_feature_count()}")
print_info(f"Columns: {schema.get_column_names()}")
```

### 2. TabularFeatureExtractor
```python
from extensions.common.utils.csv_schema_utils import TabularFeatureExtractor

extractor = TabularFeatureExtractor()
features = extractor.extract_features(game_state, grid_size=10)
```

### 3. DatasetLoader
```python
from extensions.common.utils.dataset_utils import load_dataset_for_training

# Load and prepare dataset for training
X_train, X_val, X_test, y_train, y_val, y_test, info = load_dataset_for_training(
    dataset_paths=["path/to/heuristics_v0.04_dataset.csv"],  # Use v0.04
    grid_size=10
)
```

## 🚀 Usage Examples

### For Heuristics Extensions (Dataset Generation)

```python
from extensions.common.utils.csv_schema_utils import create_csv_row

# During game execution, create CSV rows
game_state = {
    "head_position": [5, 5],
    "apple_position": [7, 3],
    "snake_positions": [[5, 5], [5, 6], [5, 7]],
    "current_direction": "UP",
    "score": 10,
    "steps": 50,
    "snake_length": 3
}

csv_row = create_csv_row(
    game_state=game_state,
    target_move="RIGHT",
    game_id=1,
    step_in_game=5,
    grid_size=10
)
```

### For Supervised Learning Extensions (Training)

```python
from extensions.common.utils.dataset_utils import DatasetLoader

# Load dataset from heuristics-v0.04
loader = DatasetLoader(grid_size=10)
df = loader.load_csv_dataset("path/to/heuristics_v0.04_dataset.csv")  # Use v0.04

# Prepare for training
X, y = loader.prepare_features_and_targets(df, scale_features=True)

# Split dataset
X_train, X_val, X_test, y_train, y_val, y_test = loader.split_dataset(X, y)
```

### Multi-Grid Size Support

```python
# Works with any grid size
for grid_size in [8, 10, 12, 16, 20]:
    schema = generate_csv_schema(grid_size)
    print_info(f"Grid {grid_size}x{grid_size}: {schema.get_feature_count()} features")
```

## 📁 File Structure

```
extensions/common/utils/
├── csv_schema.py      # Core schema and feature extraction
├── dataset_utils.py         # Dataset loading and preprocessing
└── test_csv_schema.py       # Test suite
```

## 🧪 Testing

Run the test suite to verify functionality:

```bash
cd extensions/common/utils
python test_csv_schema.py
```

Tests cover:
- Schema generation for different grid sizes
- Feature extraction accuracy
- CSV row creation
- Data validation

## 🔄 Integration with Extensions

### Heuristics v0.04 (DEFINITIVE)
- Uses `create_csv_row()` to generate training datasets
- Stores datasets in `ROOT/logs/extensions/datasets/grid-size-N/heuristics_v0.04_{timestamp}/`
- Supports multiple data formats (CSV, JSONL) for different use cases
- CSV format actively used for supervised learning

### Supervised Learning v0.02
- Uses `DatasetLoader` to load and preprocess datasets from heuristics-v0.04
- Supports multiple model types (MLP, CNN, LSTM, XGBoost, etc.)
- Automatic grid size detection and validation
- Uses CSV from heuristics-v0.04 for optimal results

## 🎯 Design Patterns

### Strategy Pattern
- Different feature extraction strategies for different model types
- Pluggable preprocessing pipelines
- Consistent interface across extensions

### Factory Pattern
- Schema generation based on grid size
- Agent creation based on model type
- Dataset loading based on format

### Template Method
- Base classes define structure
- Extensions implement specific logic
- Consistent behavior across implementations

## 📈 Performance Considerations

- **Feature count is fixed**: 16 features regardless of grid size
- **Efficient lookups**: Uses sets for snake body collision detection
- **Memory efficient**: Processes data in chunks
- **Scalable**: Works with datasets of any size

## 🔮 Future Extensions

The schema is designed to be extensible:

1. **Additional features**: Can add new engineered features without breaking existing models
2. **Different formats**: Support for sequential data (LSTM) using NPZ and graph data (GNN) using specialized formats

## 🔗 **See Also**

- **`data-format-decision-guide.md`**: Authoritative reference for all format decisions
- **`csv-schema-1.md`**: Core schema documentation
- **`unified-path-management-guide.md`**: Path management standards

## 🎯 **Important Guidelines: Version Selection Guidelines**

- **For supervised learning**: Use CSV from either heuristics-v0.03 or heuristics-v0.04 (both widely used)
- **For LLM fine-tuning**: Use JSONL from heuristics-v0.04 only
- **For research**: Use both formats from heuristics-v0.04
- **CSV is ACTIVE**: Not legacy - actively used for supervised learning
- **JSONL is ADDITIONAL**: New capability for LLM fine-tuning (heuristics-v0.04 only)
