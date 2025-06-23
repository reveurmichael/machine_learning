# 🚀 LLM Fine-tuning Integration v0.03 - Interactive Web Interface

**Evolution:** v0.02 CLI-only → v0.03 Interactive Web Interface with Real-time Monitoring

This extension provides a comprehensive web-based interface for fine-tuning language models using heuristic datasets from the Snake game. It combines the robust backend capabilities of v0.02 with an intuitive, interactive web frontend.

## 🌟 Key Features v0.03

### 🌐 **Interactive Web Interface**
- **Streamlit Application**: Modern, responsive web interface
- **Multi-tab Navigation**: Organized workflow with dedicated tabs
- **Real-time Updates**: Live progress monitoring during training
- **Interactive Charts**: Training loss curves and performance metrics

### 📊 **Enhanced Dataset Management**
- **Visual Dataset Browser**: Browse datasets with rich metadata
- **Interactive Preprocessing**: Configure and run preprocessing pipelines
- **Format Conversion**: Support for CSV, JSONL, Parquet formats
- **Quality Validation**: Automated dataset health checks

### 🎯 **Advanced Training Configuration**
- **Point-and-Click Setup**: No command-line expertise required
- **Strategy Selection**: LoRA, QLoRA, Full fine-tuning options
- **Parameter Tuning**: Interactive sliders and controls
- **Progress Visualization**: Real-time training metrics

### 📈 **Model Comparison Dashboard**
- **Side-by-side Comparison**: Compare multiple trained models
- **Statistical Analysis**: Performance metrics with significance testing
- **Visual Charts**: Interactive plots and graphs
- **Export Results**: Download comparison reports

### 🔍 **Comprehensive Evaluation Suite**
- **Web-based Testing**: Run evaluations through the browser
- **Multiple Metrics**: Win rate, score, accuracy, and more
- **Interactive Results**: Drill down into evaluation details
- **Game Replay**: Visual replay of model decisions

## 📋 Quick Start Guide

### 1. **Launch Web Interface (Recommended)**

```bash
# Navigate to extension directory
cd extensions/heuristics-llm-fine-tuning-integration-v0.03

# Launch Streamlit web app
python cli.py web --port 8501

# Open browser to: http://localhost:8501
```

### 2. **Alternative: Flask Dashboard**

```bash
# Launch Flask dashboard with real-time updates
python cli.py dashboard --port 5000

# Open browser to: http://localhost:5000
```

### 3. **Command Line Interface (Advanced)**

```bash
# Show system information
python cli.py info

# List available datasets
python cli.py datasets --list

# Run training from CLI
python cli.py train --strategy LoRA --model gpt2 --epochs 3
```

## 🎮 Web Interface Guide

### **Tab 1: 🎯 Training Configuration**

1. **Select Training Strategy**
   - LoRA (Low-Rank Adaptation) - Recommended
   - QLoRA (Quantized LoRA) - Memory efficient
   - Full Fine-tuning - Maximum customization

2. **Choose Base Model**
   - microsoft/DialoGPT-small (Fast)
   - microsoft/DialoGPT-medium (Balanced)
   - gpt2 (Standard)
   - distilgpt2 (Lightweight)

3. **Configure Parameters**
   - Grid Size: 8, 10, 12, 16, 20
   - Number of Epochs: 1-20
   - Learning Rate: Auto or custom

4. **Select Datasets**
   - Multi-select from available heuristic datasets
   - Automatic compatibility checking
   - Preview dataset statistics

### **Tab 2: 📊 Training Monitoring**

- **Real-time Progress**: Live epoch and loss updates
- **Training Curves**: Interactive loss and learning rate plots
- **System Metrics**: Memory usage, GPU utilization
- **Training Controls**: Start, stop, pause functionality
- **Log Viewer**: Real-time training logs

### **Tab 3: 📈 Model Comparison**

- **Model Selection**: Choose models to compare
- **Metric Comparison**: Win rate, average score, accuracy
- **Statistical Tests**: T-tests, effect size calculations
- **Visual Charts**: Bar charts, scatter plots, box plots
- **Export Options**: PDF reports, CSV data

### **Tab 4: 💾 Dataset Management**

- **Dataset Browser**: Visual exploration of available datasets
- **Metadata Viewer**: Samples, size, algorithm, creation date
- **Preprocessing Tools**: Format conversion, cleaning, splitting
- **Quality Checks**: Validation and health reports
- **Batch Operations**: Process multiple datasets

### **Tab 5: 🔍 Model Evaluation**

- **Model Testing**: Run comprehensive evaluations
- **Metric Selection**: Choose evaluation criteria
- **Game Simulation**: Test on actual Snake game scenarios
- **Result Analysis**: Detailed performance breakdown
- **Replay System**: Visualize model decisions

### **Tab 6: ⚙️ Settings & Configuration**

- **System Information**: Extension status, dependencies
- **Logging Configuration**: Set verbosity levels
- **Performance Settings**: Memory limits, parallelization
- **Export/Import**: Save and load configurations

## 🛠️ Installation & Dependencies

### **Required Dependencies**

```bash
pip install streamlit pandas numpy matplotlib plotly
pip install transformers torch datasets accelerate
pip install flask flask-socketio  # For dashboard
```

### **Optional Dependencies**

```bash
pip install peft  # For LoRA/QLoRA support
pip install bitsandbytes  # For quantization
pip install wandb  # For experiment tracking
pip install tensorboard  # For additional monitoring
```

### **Verify Installation**

```bash
python cli.py info
```

This will show the status of all dependencies and v0.02 components.

## 📊 Usage Examples

### **Example 1: Quick Training Setup**

1. Launch web interface: `python cli.py web`
2. Go to **Training** tab
3. Select "LoRA" strategy and "gpt2" model
4. Choose grid size 10 and 3 epochs
5. Select BFS and A* datasets
6. Click "▶️ Start Training"
7. Monitor progress in **Monitoring** tab

### **Example 2: Dataset Preprocessing**

1. Go to **Datasets** tab
2. Select datasets for preprocessing
3. Choose output format (JSONL recommended)
4. Set max samples (10,000 default)
5. Configure train/validation/test splits
6. Click "🔄 Preprocess Datasets"

### **Example 3: Model Comparison**

1. Train multiple models with different configurations
2. Go to **Comparison** tab
3. Select two models for comparison
4. Click "🔍 Run Comparison"
5. View side-by-side metrics
6. Export results as PDF

### **Example 4: CLI Training**

```bash
# Create config file
cat > training_config.json << EOF
{
    "strategy": "LoRA",
    "model_name": "microsoft/DialoGPT-small",
    "num_epochs": 5,
    "grid_size": 12,
    "datasets": ["bfs_dataset.csv", "astar_dataset.csv"]
}
EOF

# Run training
python cli.py train --config training_config.json
```

## 🔧 Configuration Options

### **Training Configuration**

```python
# Advanced training config (for developers)
from extensions.heuristics_llm_fine_tuning_integration_v0_02.training_config import TrainingConfigBuilder

config = TrainingConfigBuilder() \
    .with_model("microsoft/DialoGPT-medium") \
    .with_strategy("LoRA") \
    .with_epochs(10) \
    .with_learning_rate(1e-4) \
    .with_batch_size(8) \
    .with_lora_config(r=16, alpha=32) \
    .build()
```

### **Dataset Preprocessing**

```python
from .dataset_manager import PreprocessingConfig

config = PreprocessingConfig(
    output_format="jsonl",
    max_samples=50000,
    train_split=0.8,
    validation_split=0.1,
    test_split=0.1,
    shuffle=True,
    remove_duplicates=True
)
```

## 📈 Architecture & Design Patterns

### **Design Patterns Used**

1. **Model-View-Controller (MVC)**
   - Separates UI (Streamlit) from business logic
   - Clean separation of concerns

2. **Observer Pattern**
   - Real-time progress updates
   - WebSocket notifications

3. **Repository Pattern**
   - Centralized dataset access
   - Abstracted storage operations

4. **Strategy Pattern**
   - Different training strategies (LoRA, QLoRA, Full)
   - Pluggable preprocessing options

5. **Factory Pattern**
   - Configuration builders
   - Dataset format creation

6. **Facade Pattern**
   - Simplified web interface
   - Complex backend operations

### **Component Architecture**

```
v0.03/
├── app.py              # Main Streamlit application
├── web_interface.py    # Flask dashboard & WebSocket
├── dataset_manager.py  # Dataset operations & preprocessing
├── cli.py             # Command-line interface
└── README.md          # This documentation

Dependencies:
├── v0.02/             # Core training & evaluation logic
├── common/            # Shared utilities
└── core/              # Base game components
```

## 🚀 Evolution from v0.02

| Feature | v0.02 | v0.03 |
|---------|-------|-------|
| **Interface** | CLI only | Web + CLI |
| **Monitoring** | Log files | Real-time dashboard |
| **Configuration** | JSON files | Interactive forms |
| **Visualization** | None | Charts & graphs |
| **Dataset Management** | Manual | Visual browser |
| **Model Comparison** | CLI reports | Interactive dashboard |
| **Progress Tracking** | Text logs | Progress bars & metrics |
| **User Experience** | Expert users | All skill levels |

## 🔍 Troubleshooting

### **Common Issues**

1. **Streamlit won't start**
   ```bash
   pip install streamlit
   python -m streamlit --version
   ```

2. **v0.02 components missing**
   - Ensure v0.02 extension is properly installed
   - Check Python path includes project root

3. **No datasets found**
   - Run heuristics extensions first to generate datasets
   - Check logs/extensions/datasets/ directory

4. **Training fails**
   - Verify GPU/CPU availability
   - Check model compatibility
   - Ensure sufficient memory

### **Performance Tips**

1. **For large datasets**: Use preprocessing to limit samples
2. **For slow training**: Try LoRA instead of full fine-tuning
3. **For memory issues**: Use QLoRA or smaller models
4. **For better UI performance**: Close unused browser tabs

## 📝 Development Notes

### **Extension Guidelines Compliance**

- ✅ Follows v0.03 evolution pattern (CLI → Web interface)
- ✅ Reuses v0.02 components without modification
- ✅ Maintains backward compatibility
- ✅ Uses common utilities from extensions/common/
- ✅ Implements proper design patterns with documentation
- ✅ Provides comprehensive user experience

### **Future Enhancements**

1. **Authentication & Multi-user Support**
2. **Experiment Tracking Integration (W&B, MLflow)**
3. **Distributed Training Support**
4. **Advanced Hyperparameter Tuning**
5. **Model Deployment Integration**
6. **Mobile-responsive Interface**

## 📞 Support & Contributing

For issues, questions, or contributions:

1. Check existing heuristics extensions for reference
2. Follow OOP and SOLID principles
3. Add comprehensive docstrings and comments
4. Test with multiple datasets and configurations
5. Ensure web interface responsiveness

**Version:** 0.03  
**Type:** Interactive Web Interface  
**Dependencies:** v0.02 components, Streamlit, Flask  
**Evolution Path:** v0.01 (Proof of Concept) → v0.02 (Production CLI) → v0.03 (Web Interface) 