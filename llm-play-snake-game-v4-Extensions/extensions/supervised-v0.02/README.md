# Supervised Learning v0.02 - Multi-Model Framework

Multi-model extension for supervised learning, supporting all ML model types with organized structure.

## 🎯 Overview

Supervised Learning v0.02 demonstrates the natural evolution from v0.01 to a comprehensive multi-model framework. It follows the same pattern as heuristics v0.02 - multiple algorithms with `--model` argument, organized structure, and perfect base class reuse.

## 🔧 Key Characteristics

- **All ML model types**: Neural networks, tree models, and graph neural networks
- **`--model` argument**: Choose specific model type from command line
- **Organized structure**: Models folder with clear categorization
- **Multiple frameworks**: PyTorch, XGBoost, LightGBM, PyTorch Geometric
- **No GUI by default**: CLI only, focused on evaluation and comparison

## 📁 File Structure

```
./extensions/supervised-v0.02/
├── __init__.py
├── main.py              # Multi-model entry point with --model argument
├── game_logic.py        # Extends BaseGameLogic for multi-model support
├── game_manager.py      # Extends BaseGameManager with model factory
├── game_data.py         # Extends BaseGameData for model tracking
├── models/              # Organized model implementations
│   ├── neural_networks/ # PyTorch implementations
│   │   ├── agent_mlp.py
│   │   ├── agent_cnn.py
│   │   └── agent_lstm.py
│   ├── tree_models/     # Tree-based models
│   │   ├── agent_xgboost.py
│   │   ├── agent_lightgbm.py
│   │   └── agent_randomforest.py
│   └── graph_models/    # Graph neural networks
│       └── agent_gcn.py
├── training/            # Training scripts
│   ├── train_neural.py  # PyTorch neural networks
│   ├── train_tree.py    # XGBoost, LightGBM, RandomForest
│   └── train_graph.py   # Graph neural networks
└── README.md            # This file
```

## 🧠 Model Portfolio

### Neural Networks (PyTorch)
- **MLP**: Multi-Layer Perceptron for tabular data
- **CNN**: Convolutional Neural Network for board data
- **LSTM**: Long Short-Term Memory for sequential data
- **GRU**: Gated Recurrent Unit for sequential data

### Tree Models
- **XGBoost**: Gradient boosting with XGBoost framework
- **LightGBM**: Gradient boosting with LightGBM framework
- **RandomForest**: Ensemble of decision trees

### Graph Models (PyTorch Geometric)
- **GCN**: Graph Convolutional Network
- **GraphSAGE**: GraphSAGE for large graphs
- **GAT**: Graph Attention Network

## 🚀 Usage

### List Available Models
```bash
cd extensions/supervised-v0.02
python main.py --list-models
```

### Run Specific Models
```bash
# Neural networks
python main.py --model MLP --max-games 10
python main.py --model CNN --max-games 5
python main.py --model LSTM --max-games 3

# Tree models
python main.py --model XGBOOST --max-games 10
python main.py --model LIGHTGBM --max-games 5
python main.py --model RANDOMFOREST --max-games 3

# Graph models
python main.py --model GCN --max-games 3
python main.py --model GRAPHSAGE --max-games 2
python main.py --model GAT --max-games 2
```

### Training Models
```bash
# Train neural networks
python training/train_neural.py --model MLP --dataset-path ../../logs/extensions/datasets/grid-size-10/

# Train tree models
python training/train_tree.py --model XGBOOST --dataset-path ../../logs/extensions/datasets/grid-size-10/

# Train graph models
python training/train_graph.py --model GCN --dataset-path ../../logs/extensions/datasets/grid-size-10/
```

## 📊 Output

Generates the same output structure as heuristics extensions:
- `game_N.json` files with game histories
- `supervised_summary.json` with multi-model specific metrics
- **No LLM-specific fields** (removed from Task-0 schema)
- **Model-specific metadata** for tracking and comparison

### Example Summary Structure
```json
{
  "extension_type": "supervised_learning",
  "version": "v0.02",
  "model_type": "XGBOOST",
  "grid_size": 10,
  "model_category": "Tree",
  "multi_model_metrics": {
    "model_architecture": "XGBOOST",
    "model_category": "Tree",
    "input_features": 16,
    "is_trained": false,
    "framework": "XGBoost"
  },
  "game_statistics": {
    "total_games": 10,
    "average_score": 18.5,
    "average_steps": 145.2
  }
}
```

## 🏗️ Design Patterns

### Factory Pattern
- `SupervisedGameManager` creates agents based on model type
- Pluggable model selection with `--model` argument
- Easy extension to new model types

### Strategy Pattern
- Different model types as strategies
- Same interface, different implementations
- Easy comparison and evaluation

### Template Method Pattern
- Base classes define common structure
- Subclasses implement model-specific logic
- Consistent training and evaluation interface

## 🔄 Base Class Integration

Perfect reuse of Task-0 base classes:

```python
# Extends base classes from Task-0
class SupervisedGameManager(BaseGameManager):
    GAME_LOGIC_CLS = SupervisedGameLogic  # Factory pattern
    
    def __init__(self, args):
        super().__init__(args)  # Inherits all base functionality
        # Add multi-model-specific extensions only

class SupervisedGameLogic(BaseGameLogic):
    def __init__(self, agent, grid_size=10, max_steps=1000):
        super().__init__(grid_size=grid_size, max_steps=max_steps)  # Inherits all base functionality
        # Add multi-model-specific extensions only

class SupervisedGameData(BaseGameData):
    def __init__(self, grid_size=10):
        super().__init__(grid_size=grid_size)  # Inherits all base functionality
        # Add multi-model-specific extensions only
```

### What Base Classes Provide
- ✅ Generic game loop and session management
- ✅ Round counting and statistics tracking
- ✅ File I/O and logging infrastructure
- ✅ Game state management and validation
- ✅ Error handling and safety checks

### What Extensions Add
- Multi-model decision making
- Model training and evaluation pipelines
- Feature extraction and preprocessing
- Model persistence and loading
- Framework-specific optimizations

## 📈 Evolution from v0.01

### v0.01 → v0.02 Progression
- ✅ **Single model type** → **All ML model types**
- ✅ **No arguments** → **`--model` parameter**
- ✅ **Simple structure** → **Organized models folder**
- ✅ **Basic training** → **Advanced training pipelines**
- ✅ **Neural networks only** → **Neural, Tree, Graph models**

### Key Improvements
- **Model diversity**: Support for all major ML model types
- **Framework integration**: Multiple ML frameworks (PyTorch, XGBoost, LightGBM)
- **Organized structure**: Clear separation of model categories
- **Enhanced evaluation**: Model-specific metrics and comparison
- **Better documentation**: Comprehensive model information

## 🎯 Success Criteria

A successful v0.02 extension should:
1. **Support all model types** (neural, tree, graph)
2. **Demonstrate clear evolution** from v0.01
3. **Maintain organized structure** with models folder
4. **Provide consistent interface** across all model types
5. **Enable model comparison** and evaluation
6. **Reuse maximum infrastructure** from Task-0

## 🔧 Technical Details

### Model Categories
- **Neural Networks**: PyTorch-based implementations for different data types
- **Tree Models**: XGBoost, LightGBM, RandomForest for tabular data
- **Graph Models**: PyTorch Geometric for graph-structured data

### Framework Integration
- **PyTorch**: Neural networks and graph neural networks
- **XGBoost**: Gradient boosting for tabular data
- **LightGBM**: Light gradient boosting machine
- **Scikit-learn**: Random forest and utility functions
- **PyTorch Geometric**: Graph neural networks

### Training Pipelines
- **Neural networks**: PyTorch training with validation and early stopping
- **Tree models**: Framework-specific training with hyperparameter tuning
- **Graph models**: PyTorch Geometric training with graph data processing

## 📚 Related Documentation

- [Extensions v0.02 Guidelines](../docs/extensions-v0.02.md)
- [CSV Schema Documentation](../common/README_CSV_SCHEMA.md)
- [Base Classes Documentation](../../core/)

---

**Remember**: v0.02 is about **model diversity** and **natural evolution**. Show how systems grow from simple to sophisticated while maintaining clean architecture. 