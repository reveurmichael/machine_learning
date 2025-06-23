# Supervised Learning v0.01 - Neural Networks

Simple proof-of-concept extension for supervised learning, focusing on neural networks only.

## 🎯 Overview

Supervised Learning v0.01 demonstrates how to extend the Snake game with neural network agents. It follows the same pattern as heuristics v0.01 - minimal complexity, focused on a single approach (neural networks), and perfect base class reuse.

## 🔧 Key Characteristics

- **Neural networks only**: PyTorch-based implementations (MLP, CNN, LSTM)
- **No command-line arguments**: Simple CLI with no complex options
- **No GUI by default**: Pure console output, no pygame/web UI
- **Minimal complexity**: Just proof that base classes work for supervised learning

## 📁 File Structure

```
./extensions/supervised-v0.01/
├── __init__.py
├── main.py              # Simple entry point, no arguments
├── agent_neural.py      # Neural network agents (MLP, CNN, LSTM)
├── game_logic.py        # Extends BaseGameLogic
├── game_manager.py      # Extends BaseGameManager
├── train.py             # Simple training script
└── README.md            # This file
```

## 🧠 Neural Network Agents

### MLPAgent
- **Purpose**: Multi-Layer Perceptron for tabular feature data
- **Input**: 16 engineered features from game state
- **Output**: 4 classes (UP, DOWN, LEFT, RIGHT)
- **Architecture**: 3 hidden layers with ReLU and dropout

### CNNAgent
- **Purpose**: Convolutional Neural Network for board data
- **Input**: 3-channel board representation (snake, apple, empty)
- **Output**: 4 classes (UP, DOWN, LEFT, RIGHT)
- **Architecture**: 2 conv layers + pooling + fully connected

### LSTMAgent
- **Purpose**: Long Short-Term Memory for sequential data
- **Input**: Feature sequences from game state
- **Output**: 4 classes (UP, DOWN, LEFT, RIGHT)
- **Architecture**: LSTM + fully connected layer

## 🚀 Usage

### Simple Evaluation
```bash
cd extensions/supervised-v0.01
python main.py
```

This runs the default MLP agent for 10 games on a 10x10 grid.

### Training Models
```bash
# Train MLP model
python train.py --model MLP --dataset-path ../../logs/extensions/datasets/grid-size-10/tabular_bfs_data.csv

# Train CNN model
python train.py --model CNN --dataset-path ../../logs/extensions/datasets/grid-size-10/tabular_bfs_data.csv

# Train LSTM model
python train.py --model LSTM --dataset-path ../../logs/extensions/datasets/grid-size-10/tabular_bfs_data.csv
```

## 📊 Output

Generates the same output structure as heuristics extensions:
- `game_N.json` files with game histories
- `supervised_summary.json` with neural network specific metrics
- **No LLM-specific fields** (removed from Task-0 schema)
- **Neural network metadata** for tracking

### Example Summary Structure
```json
{
  "extension_type": "supervised_learning",
  "version": "v0.01",
  "model_type": "MLP",
  "grid_size": 10,
  "training_mode": false,
  "neural_network_metrics": {
    "model_architecture": "MLP",
    "input_features": 16,
    "is_trained": false
  },
  "game_statistics": {
    "total_games": 10,
    "average_score": 15.2,
    "average_steps": 120.5
  }
}
```

## 🏗️ Design Patterns

### Template Method Pattern
- `BaseNeuralAgent` defines common interface
- Subclasses implement specific architectures
- Consistent training and prediction interface

### Factory Pattern
- `SupervisedGameManager` creates agents based on type
- Pluggable agent selection
- Easy extension to new architectures

### Strategy Pattern
- Different neural network architectures as strategies
- Same interface, different implementations
- Easy comparison and evaluation

## 🔄 Base Class Integration

Perfect reuse of Task-0 base classes:

```python
# Extends base classes from Task-0
class SupervisedGameManager(BaseGameManager):
    GAME_LOGIC_CLS = SupervisedGameLogic  # Factory pattern
    
    def __init__(self, args):
        super().__init__(args)  # Inherits all base functionality
        # Add neural network-specific extensions only

class SupervisedGameLogic(BaseGameLogic):
    def __init__(self, agent, grid_size=10, max_steps=1000):
        super().__init__(grid_size=grid_size, max_steps=max_steps)  # Inherits all base functionality
        # Add neural network-specific extensions only
```

### What Base Classes Provide
- ✅ Generic game loop and session management
- ✅ Round counting and statistics tracking
- ✅ File I/O and logging infrastructure
- ✅ Game state management and validation
- ✅ Error handling and safety checks

### What Extensions Add
- Neural network decision making
- Model training and evaluation
- Feature extraction and preprocessing
- Model persistence and loading

## 📈 Evolution Path

### v0.01 → v0.02 Progression
- **Single model type** → **Multiple model types** (XGBoost, LightGBM, etc.)
- **No arguments** → **`--model` parameter**
- **Simple structure** → **Organized models folder**
- **Basic training** → **Advanced training pipelines**

### v0.02 → v0.03 Preview
- **CLI only** → **Streamlit web interface**
- **No replay** → **PyGame + Flask web replay**
- **Basic logging** → **Interactive training visualization**

## 🎯 Success Criteria

A successful v0.01 extension should:
1. **Prove base class abstraction works** for supervised learning
2. **Generate valid output** that follows established schemas
3. **Demonstrate clear evolution path** to v0.02
4. **Maintain code simplicity** and readability
5. **Reuse maximum infrastructure** from Task-0
6. **Document design decisions** clearly

## 🔧 Technical Details

### Feature Engineering
Uses the common CSV schema with 16 engineered features:
- Position features (head_x, head_y, apple_x, apple_y)
- Game state features (snake_length)
- Apple direction features (4 binary flags)
- Danger detection features (3 binary flags)
- Free space features (4 integer counts)

### Model Architecture
- **Input layer**: 16 features (from CSV schema)
- **Hidden layers**: Configurable size (default: 256)
- **Output layer**: 4 classes (UP, DOWN, LEFT, RIGHT)
- **Activation**: ReLU with dropout for regularization

### Training Pipeline
- **Data loading**: Uses common dataset loader
- **Preprocessing**: Feature scaling and encoding
- **Training**: PyTorch with Adam optimizer
- **Evaluation**: Accuracy on validation/test sets
- **Persistence**: Model checkpointing and loading

## 📚 Related Documentation

- [Extensions v0.01 Guidelines](../docs/extensions-v0.01.md)
- [CSV Schema Documentation](../common/README_CSV_SCHEMA.md)
- [Base Classes Documentation](../../core/)

---

**Remember**: v0.01 is about **proving the concept works**. Keep it simple, focused, and extensible. The sophistication comes in later versions. 