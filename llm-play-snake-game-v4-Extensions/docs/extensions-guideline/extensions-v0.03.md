# Extensions v0.03 - Web Interface & Dataset Generation

This document serves as the **definitive guideline** for implementing v0.03 extensions across different algorithm types. It demonstrates the evolution from CLI-only v0.02 to web-enabled systems with dataset generation capabilities.

## 🎯 **Core Philosophy: Web Interface & Data Generation**

v0.03 builds upon v0.02's multi-algorithm foundation to demonstrate:
- **Web interface evolution**: From CLI-only to Streamlit web applications
- **Dataset generation**: Creating training data for other extensions
- **Replay capabilities**: Both pygame and web-based replay systems
- **Interactive visualization**: Real-time algorithm/model performance monitoring

## 🔧 **Heuristics v0.03 - Web Interface & Dataset Production**

### **Location:** `./extensions/heuristics-v0.03`

### **Key Evolution from v0.02:**
- **CLI only** → **Streamlit web application**
- **No replay** → **PyGame + Flask web replay**
- **Basic logging** → **CSV dataset generation for ML training**
- **Console output** → **Interactive web visualization**

### **New Features Added:**
- **`app.py`**: Streamlit web interface with algorithm selection tabs
- **Dataset generation**: CSV output for supervised learning training
- **Web replay**: Flask-based replay system
- **PyGame replay**: Desktop replay visualization
- **Interactive controls**: Real-time parameter adjustment

### **File Structure:**
```
./extensions/heuristics-v0.03/
├── __init__.py
├── app.py                    # 🆕 Streamlit web interface
├── heuristic_config.py       # 🆕 Configuration (renamed from config.py)
├── game_logic.py             # Extends BaseGameLogic
├── game_manager.py           # Multi-algorithm manager
├── game_data.py              # Heuristic game data with dataset export
├── replay_engine.py          # 🆕 Replay processing engine
├── replay_gui.py             # 🆕 PyGame replay interface
├── agents/                   # Same as v0.02 (copied exactly)
│   ├── __init__.py
│   ├── agent_bfs.py
│   ├── agent_bfs_safe_greedy.py
│   ├── agent_bfs_hamiltonian.py
│   ├── agent_dfs.py
│   ├── agent_astar.py
│   ├── agent_astar_hamiltonian.py
│   └── agent_hamiltonian.py
└── scripts/                  # 🆕 Script organization
    ├── main.py               # Moved from root
    ├── generate_dataset.py   # 🆕 Dataset generation CLI
    ├── replay.py             # 🆕 PyGame replay script
    └── replay_web.py         # 🆕 Flask web replay
```

### **Streamlit Web Interface:**
```python
# app.py structure
import streamlit as st

def main():
    st.title("Heuristic Snake Algorithms - v0.03")
    
    # Algorithm selection tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "BFS", "BFS Safe Greedy", "BFS Hamiltonian", 
        "DFS", "A*", "A* Hamiltonian", "Hamiltonian"
    ])
    
    with tab1:
        run_algorithm_interface("BFS")
    with tab2:
        run_algorithm_interface("BFS_SAFE_GREEDY")
    # ... etc for each algorithm
    
def run_algorithm_interface(algorithm: str):
    st.subheader(f"{algorithm} Algorithm")
    
    # Parameters
    max_games = st.slider("Max Games", 1, 100, 10)
    grid_size = st.selectbox("Grid Size", [8, 10, 12, 16, 20], index=1)
    
    # Run controls
    if st.button(f"Run {algorithm}"):
        run_heuristic_games(algorithm, max_games, grid_size)
    
    # Replay controls
    if st.button(f"Replay {algorithm} (PyGame)"):
        launch_pygame_replay(algorithm)
    
    if st.button(f"Replay {algorithm} (Web)"):
        launch_web_replay(algorithm)
```

### **Dataset Generation:**
```bash
# Generate training datasets for supervised learning
python scripts/generate_dataset.py --algorithm BFS --games 1000 --format csv --structure tabular
python scripts/generate_dataset.py --algorithm ASTAR --games 500 --format npz --structure sequential
python scripts/generate_dataset.py --algorithm mixed --games 2000 --format parquet --structure graph

# Output location: ROOT/logs/extensions/datasets/grid-size-N/
```

### **Web Replay System:**
- **Flask backend**: Serves game data and replay controls
- **JavaScript frontend**: Interactive replay visualization
- **RESTful API**: State management and control endpoints
- **Real-time updates**: WebSocket support for live algorithm execution

## 🧠 **Supervised Learning v0.03 - Interactive Training & Evaluation**

### **Location:** `./extensions/supervised-v0.03`

### **Key Evolution from v0.02:**
- **CLI training only** → **Streamlit training interface**
- **No visualization** → **Interactive training progress & model comparison**
- **Basic evaluation** → **Web-based model performance dashboard**
- **Static training** → **Dynamic hyperparameter tuning**

### **New Features Added:**
- **`app.py`**: Streamlit interface for training, evaluation, and comparison
- **Interactive training**: Real-time loss/accuracy plots
- **Model comparison**: Side-by-side performance analysis
- **Web replay**: Visualize model decision-making process
- **Dataset integration**: Load datasets from heuristics-v0.03

### **File Structure:**
```
./extensions/supervised-v0.03/
├── __init__.py
├── app.py                    # 🆕 Streamlit training/evaluation interface
├── supervised_config.py      # 🆕 ML-specific configuration
├── game_logic.py             # ML-specific game logic
├── game_manager.py           # Multi-model evaluation manager
├── game_data.py              # ML game data with prediction tracking
├── replay_engine.py          # 🆕 Model decision replay
├── replay_gui.py             # 🆕 PyGame model visualization
├── models/                   # Same as v0.02 (all model types)
│   ├── neural_networks/
│   ├── tree_models/
│   └── graph_models/
├── training/                 # Enhanced training scripts
│   ├── train_neural.py       # Interactive PyTorch training
│   ├── train_tree.py         # XGBoost/LightGBM with live metrics
│   ├── train_graph.py        # GNN training with visualization
│   └── train_ensemble.py     # Ensemble with web monitoring
├── evaluation/               # 🆕 Evaluation and comparison
│   ├── model_comparison.py   # Cross-model benchmarking
│   ├── performance_analysis.py # Detailed performance metrics
│   └── visualization.py      # Model behavior visualization
└── scripts/                  # 🆕 Script organization
    ├── train.py              # CLI training interface
    ├── evaluate.py           # Model evaluation script
    ├── replay.py             # PyGame model replay
    └── replay_web.py         # Flask model replay
```

### **Streamlit Training Interface:**
```python
# app.py structure
import streamlit as st

def main():
    st.title("Supervised Learning Snake Models - v0.03")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Training", "Evaluation", "Comparison", "Replay"
    ])
    
    with tab1:
        training_interface()
    with tab2:
        evaluation_interface()
    with tab3:
        model_comparison_interface()
    with tab4:
        replay_interface()

def training_interface():
    st.subheader("Model Training")
    
    # Model selection
    model_type = st.selectbox("Model Type", [
        "MLP", "CNN", "LSTM", "GRU", 
        "XGBoost", "LightGBM", "RandomForest",
        "GCN", "GraphSAGE", "GAT"
    ])
    
    # Dataset selection
    dataset_paths = st.multiselect("Dataset Paths", [
        "logs/extensions/datasets/grid-size-8/",
        "logs/extensions/datasets/grid-size-10/",
        "logs/extensions/datasets/grid-size-12/"
    ])
    
    # Training parameters
    if model_type in ["MLP", "CNN", "LSTM", "GRU"]:
        epochs = st.slider("Epochs", 10, 1000, 100)
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256])
        learning_rate = st.select_slider("Learning Rate", 
            options=[0.001, 0.01, 0.1], value=0.01)
    
    # Live training with progress bars and plots
    if st.button("Start Training"):
        train_model_with_live_updates(model_type, dataset_paths, {
            'epochs': epochs, 'batch_size': batch_size, 'lr': learning_rate
        })
```

### **Model Evaluation Dashboard:**
```python
def evaluation_interface():
    st.subheader("Model Evaluation")
    
    # Load trained models
    available_models = list_available_models()
    selected_models = st.multiselect("Select Models", available_models)
    
    # Evaluation metrics
    for model_name in selected_models:
        with st.expander(f"{model_name} Performance"):
            model = load_model(model_name)
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{model.accuracy:.2f}%")
            with col2:
                st.metric("Avg Score", f"{model.avg_score:.1f}")
            with col3:
                st.metric("Win Rate", f"{model.win_rate:.1f}%")
            
            # Performance plots
            st.plotly_chart(create_performance_plot(model))
            
            # Decision visualization
            st.plotly_chart(create_decision_heatmap(model))
```

### **Dataset Integration:**
```bash
# Load datasets generated by heuristics-v0.03
python scripts/train.py --model MLP \
    --dataset-path ../../logs/extensions/datasets/grid-size-10/tabular_bfs_data.csv \
    --dataset-path ../../logs/extensions/datasets/grid-size-10/tabular_astar_data.csv

# Multi-dataset training for robust models
python scripts/train.py --model XGBoost \
    --dataset-path ../../logs/extensions/datasets/grid-size-8/ \
    --dataset-path ../../logs/extensions/datasets/grid-size-10/ \
    --dataset-path ../../logs/extensions/datasets/grid-size-12/
```

## 🌐 **Web Infrastructure & Replay Systems**

### **Common Web Components (Both Extensions):**
- **Streamlit frontend**: Interactive parameter control and visualization
- **Flask replay backend**: RESTful API for game state management
- **JavaScript visualization**: Real-time board state rendering
- **WebSocket support**: Live algorithm/model execution updates

### **PyGame Replay Features:**
- **Algorithm/model step-through**: Manual or automatic replay
- **Performance metrics overlay**: Real-time statistics display
- **Decision visualization**: Show algorithm reasoning or model confidence
- **Comparison mode**: Side-by-side algorithm/model comparison

### **Web Replay Features:**
- **Browser-based**: No local installation required
- **Responsive design**: Works on desktop and mobile
- **Share functionality**: URL-based replay sharing
- **Export capabilities**: Save replays as videos or images

## 📊 **Dataset Generation System**

### **Dataset Storage Structure:**
```
ROOT/logs/extensions/datasets/
├── grid-size-8/
│   ├── tabular_bfs_data.csv
│   ├── tabular_astar_data.csv
│   ├── sequential_mixed_data.npz
│   └── graph_mixed_data.parquet
├── grid-size-10/
│   ├── tabular_bfs_data.csv
│   ├── tabular_astar_data.csv
│   ├── tabular_mixed_data.csv
│   └── sequential_mixed_data.npz
└── grid-size-12/
    └── [similar structure]
```

### **Data Formats Supported:**
- **CSV**: Tabular data for XGBoost, LightGBM, simple neural networks
- **NPZ**: NumPy arrays for sequential/temporal models (LSTM, GRU)
- **Parquet**: Efficient storage for large datasets with complex structures
- **JSON**: Human-readable format for debugging and analysis

### **Data Structures Supported:**
- **Tabular**: Flattened board state + engineered features (126 features)
- **Sequential**: Time-series data for RNN/LSTM models
- **Graph**: Node/edge representations for Graph Neural Networks
- **Image**: Board state as images for CNN models

## 🚀 **Evolution Summary**

### **v0.02 → v0.03 Progression:**

**Heuristics:**
- ✅ **CLI only** → **Streamlit web application**
- ✅ **No replay** → **PyGame + Flask web replay**
- ✅ **Basic logging** → **CSV dataset generation**
- ✅ **Console output** → **Interactive web visualization**

**Supervised Learning:**
- ✅ **CLI training only** → **Interactive training interface**
- ✅ **No visualization** → **Real-time training progress plots**
- ✅ **Basic evaluation** → **Comprehensive model comparison**
- ✅ **Static training** → **Dynamic hyperparameter tuning**

### **v0.03 → v0.04 Preview:**
- **Heuristics only**: Numerical datasets → **Language-rich datasets for LLM fine-tuning**
- **Supervised**: v0.03 is sufficient (no v0.04 planned)

## 📋 **Implementation Checklist**

### **For Heuristics v0.03:**
- [ ] **Streamlit app.py** with algorithm tabs
- [ ] **Dataset generation** scripts and CLI
- [ ] **PyGame replay** system
- [ ] **Flask web replay** system
- [ ] **Agent folder copied** exactly from v0.02
- [ ] **Configuration renamed** to avoid conflicts

### **For Supervised Learning v0.03:**
- [ ] **Streamlit training interface** with live updates
- [ ] **Model comparison dashboard** with metrics
- [ ] **Dataset integration** from heuristics-v0.03
- [ ] **PyGame model visualization** 
- [ ] **Web-based model replay** system
- [ ] **All model types** from v0.02 integrated

### **Shared Infrastructure:**
- [ ] **Common dataset utilities** in extensions/common/
- [ ] **Consistent web styling** across both extensions
- [ ] **Cross-extension compatibility** for datasets
- [ ] **Performance benchmarking** between algorithms and models

## 🎯 **Success Criteria**

### **Web Interface Goals:**
- Interactive parameter adjustment with immediate feedback
- Real-time visualization of algorithm/model performance
- Seamless switching between algorithms/models
- Professional, responsive web design

### **Dataset Generation Goals:**
- Multiple data formats for different model types
- Configurable grid sizes and game parameters
- High-quality labeled training data
- Efficient storage and loading mechanisms

### **Replay System Goals:**
- Smooth visualization of game progression
- Clear display of algorithm reasoning or model decisions
- Export capabilities for analysis and presentation
- Cross-platform compatibility (desktop and web)

---

**Remember**: v0.03 is about **user experience** and **data production**. Create polished interfaces that make algorithms/models accessible and generate high-quality datasets for the ML ecosystem.
