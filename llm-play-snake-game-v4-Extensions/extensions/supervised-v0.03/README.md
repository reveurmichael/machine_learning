# Supervised Learning v0.03 - Web Interface & Multi-Model Framework

A comprehensive supervised learning framework for Snake game AI with web interface, supporting multiple model types and interactive training/evaluation.

## 🎯 **Design Philosophy**

- **Web Interface Evolution**: Streamlit-based interactive training and evaluation
- **Multi-Model Support**: Neural Networks, Tree Models, and Graph Neural Networks
- **Standardized Model Management**: Cross-platform, time-proof model saving/loading
- **Grid Size Flexibility**: Support for arbitrary grid sizes (8x8 to 20x20)
- **No Backward Compatibility**: Fresh, future-proof codebase

## 🏗️ **Architecture Overview**

### **Design Patterns Used**

1. **Template Method Pattern**: Standardized training and evaluation pipelines
2. **Strategy Pattern**: Pluggable model implementations
3. **Factory Pattern**: Model creation and management
4. **Singleton Pattern**: Model utilities and path management

### **Key Components**

```
supervised-v0.03/
├── app.py                    # Streamlit web interface
├── scripts/
│   ├── train.py             # CLI training script
│   └── evaluate.py          # CLI evaluation script
├── models/
│   ├── neural_networks/     # PyTorch implementations
│   │   ├── agent_mlp.py     # Multi-Layer Perceptron
│   │   ├── agent_cnn.py     # Convolutional Neural Network
│   │   └── agent_lstm.py    # Long Short-Term Memory
│   ├── tree_models/         # Tree-based models
│   │   ├── agent_xgboost.py # XGBoost gradient boosting
│   │   └── agent_lightgbm.py # LightGBM gradient boosting
│   └── graph_models/        # Graph neural networks
│       └── agent_gcn.py     # Graph Convolutional Network
└── evaluation/              # Evaluation utilities
```

## 🚀 **Quick Start**

### **Web Interface**

```bash
# Start the Streamlit web interface
streamlit run app.py
```

The web interface provides:
- **Interactive Training**: Real-time parameter adjustment and training progress
- **Model Comparison**: Side-by-side performance analysis
- **Evaluation Dashboard**: Comprehensive metrics and visualization
- **Multi-Model Support**: Train and compare different model types

### **Command Line Training**

```bash
# Train MLP neural network
python scripts/train.py --model MLP --grid-size 15 --epochs 200

# Train XGBoost tree model
python scripts/train.py --model XGBoost --grid-size 10 --max-games 500

# Train GCN graph neural network
python scripts/train.py --model GCN --grid-size 12 --hidden-channels 64
```

### **Model Evaluation**

```bash
# Evaluate trained model
python scripts/evaluate.py --model MLP --grid-size 15

# Compare multiple models
python scripts/evaluate.py --model-path logs/extensions/models/grid-size-15/
```

## 🧠 **Supported Models**

### **Neural Networks (PyTorch)**

| Model | Architecture | Use Case | Key Features |
|-------|-------------|----------|--------------|
| **MLP** | Multi-Layer Perceptron | Tabular features | Dropout, Xavier initialization |
| **CNN** | Convolutional Neural Network | Board state images | 2D convolutions, pooling |
| **LSTM** | Long Short-Term Memory | Sequential data | Memory cells, attention |
| **GRU** | Gated Recurrent Unit | Sequential data | Simplified LSTM |

### **Tree Models**

| Model | Framework | Use Case | Key Features |
|-------|-----------|----------|--------------|
| **XGBoost** | XGBoost | Tabular features | Gradient boosting, JSON format |
| **LightGBM** | LightGBM | Tabular features | Gradient boosting, text format |
| **RandomForest** | Scikit-learn | Tabular features | Ensemble learning |

### **Graph Neural Networks**

| Model | Framework | Use Case | Key Features |
|-------|-----------|----------|--------------|
| **GCN** | PyTorch Geometric | Graph data | Graph convolutions |
| **GraphSAGE** | PyTorch Geometric | Graph data | Neighborhood sampling |
| **GAT** | PyTorch Geometric | Graph data | Attention mechanisms |

## 📊 **Model Management**

### **Standardized Saving/Loading**

All models follow standardized saving/loading patterns:

```python
# Save model with metadata
agent.save_model("my_model", export_onnx=True)

# Load model
agent.load_model("my_model")
```

### **Model Directory Structure**

```
logs/extensions/models/
├── grid-size-8/
│   ├── pytorch/
│   │   ├── mlp_model.pth
│   │   ├── mlp_model.onnx
│   │   └── mlp_model_metadata.json
│   ├── xgboost/
│   │   ├── xgboost_model.json
│   │   └── xgboost_model_metadata.json
│   └── lightgbm/
│       ├── lightgbm_model.txt
│       └── lightgbm_model_metadata.json
├── grid-size-10/
└── grid-size-15/
```

### **Model Metadata**

Each saved model includes rich metadata:

```json
{
  "model_type": "MLP",
  "grid_size": 15,
  "input_size": 229,
  "hidden_size": 256,
  "learning_rate": 0.001,
  "torch_version": "2.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "training_metrics": {
    "final_accuracy": 0.85,
    "epochs_trained": 200
  }
}
```

## 🔧 **Configuration**

### **Grid Size Support**

The framework supports arbitrary grid sizes:

```python
# Different grid sizes
agent_8x8 = MLPAgent(grid_size=8)    # 8x8 grid
agent_10x10 = MLPAgent(grid_size=10) # 10x10 grid (default)
agent_15x15 = MLPAgent(grid_size=15) # 15x15 grid
agent_20x20 = MLPAgent(grid_size=20) # 20x20 grid
```

### **Feature Engineering**

Automatic feature extraction based on grid size:

```python
# Feature count scales with grid size
grid_size_8 = 85 features   # 8x8 + 4 position features
grid_size_10 = 104 features # 10x10 + 4 position features  
grid_size_15 = 229 features # 15x15 + 4 position features
grid_size_20 = 404 features # 20x20 + 4 position features
```

## 📈 **Training & Evaluation**

### **Training Metrics**

All models track comprehensive training metrics:

- **Accuracy**: Prediction accuracy on validation set
- **Loss**: Training and validation loss curves
- **Feature Importance**: For tree-based models
- **Training Time**: Wall-clock training duration
- **Model Size**: Parameter count and file size

### **Evaluation Metrics**

Standardized evaluation across all models:

- **Game Performance**: Average score, win rate, steps
- **Prediction Accuracy**: Direction prediction accuracy
- **Inference Speed**: Prediction latency
- **Memory Usage**: Model memory footprint

### **Model Comparison**

```python
# Compare multiple models
results = compare_models(["MLP", "XGBoost", "GCN"], grid_size=15)
```

## 🌐 **Web Interface Features**

### **Interactive Training**

- **Real-time Progress**: Live training progress visualization
- **Parameter Tuning**: Interactive hyperparameter adjustment
- **Model Selection**: Easy switching between model types
- **Grid Size Control**: Dynamic grid size selection

### **Evaluation Dashboard**

- **Performance Metrics**: Comprehensive model evaluation
- **Visualization**: Training curves and performance plots
- **Model Comparison**: Side-by-side model analysis
- **Export Capabilities**: Save results and visualizations

### **Model Management**

- **Model Browser**: Browse and load saved models
- **Metadata Viewer**: View model configuration and metrics
- **Version Control**: Track model versions and changes
- **Deployment**: Export models for production use

## 🔄 **Integration with Other Extensions**

### **Dataset Integration**

Load training data from heuristics extensions:

```python
# Load datasets from heuristics-v0.03
dataset_path = "logs/extensions/datasets/grid-size-15/tabular_bfs_data.csv"
X_train, y_train = load_dataset(dataset_path)
```

### **Cross-Extension Compatibility**

- **Heuristics v0.03**: Dataset generation and export
- **Reinforcement v0.02**: Model comparison and benchmarking
- **Common Utils**: Shared utilities and standards

## 🛠️ **Development**

### **Adding New Models**

1. **Create Model Class**: Implement the model interface
2. **Add Training Logic**: Implement training pipeline
3. **Add to CLI**: Update training script
4. **Add to Web UI**: Update Streamlit interface
5. **Add Tests**: Create unit tests

### **Model Interface**

```python
class BaseModel:
    def __init__(self, grid_size: int, **kwargs):
        pass
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def save_model(self, model_name: str) -> str:
        pass
    
    def load_model(self, model_path: str) -> None:
        pass
```

## 📚 **Documentation**

- **API Reference**: Complete model and utility documentation
- **Tutorials**: Step-by-step training and evaluation guides
- **Examples**: Code examples for all model types
- **Best Practices**: Recommended configurations and workflows

## 🤝 **Contributing**

1. **Fork the Repository**: Create your own fork
2. **Create Feature Branch**: Work on new features
3. **Follow Standards**: Adhere to coding standards and patterns
4. **Add Tests**: Include comprehensive tests
5. **Submit Pull Request**: Submit for review

## 📄 **License**

This extension follows the same license as the main project.

---

**Supervised Learning v0.03** - Modern, extensible, and future-proof supervised learning framework for Snake game AI. 