# Data Format Decision Guide

> **Important Guidelines**: Both `heuristics-v0.03` and `heuristics-v0.04` are widely used depending on use cases and scenarios. For supervised learning and other general purposes, both versions can be used. For LLM fine-tuning, only `heuristics-v0.04` will be used. The CSV format is **NOT legacy** - it's actively used and valuable for supervised learning. All file operations must use UTF-8 encoding for cross-platform compatibility (SUPREME_RULE NO.7).

## 📊 **Format Selection Matrix**

| Format | Best For | Use Cases | Grid Size Support |
|--------|----------|-----------|-------------------|
| **CSV (16-feature)** | Tree models, Simple MLPs | XGBoost, LightGBM, Random Forest | ✅ Universal |
| **JSONL** | LLM Fine-tuning | Prompt-completion pairs | ✅ Universal |
| **NPZ (Sequential)** | RNN/LSTM, RL Experience | Temporal patterns, State sequences | ✅ Universal |
| **NPZ (2D Arrays)** | CNNs, Spatial Analysis | Image-like processing | ✅ Universal |
| **NPZ (Raw Arrays)** | Evolutionary Algorithms | Genetic operators, Population data | ✅ Universal |

## 🔧 **Format Specifications**

### **CSV (16-Feature Tabular)**
**Purpose**: Tree-based models and simple neural networks
**Source**: `heuristics-v0.03` and `heuristics-v0.04` (both widely used) - actively used, NOT legacy
```python
# Fixed 16 features, grid-size agnostic
features = [
    'head_x', 'head_y', 'apple_x', 'apple_y',  # Position
    'snake_length',                             # Game state
    'apple_dir_up', 'apple_dir_down', 'apple_dir_left', 'apple_dir_right',  # Direction
    'danger_straight', 'danger_left', 'danger_right',  # Collision risk
    'free_space_up', 'free_space_down', 'free_space_left', 'free_space_right'  # Free space
]
```

### **JSONL (Language-Rich)**
**Purpose**: LLM fine-tuning with natural language explanations
**Source**: `heuristics-v0.04` only - new capability for LLM fine-tuning
```json
{"prompt": "Snake at (5,5), apple at (8,8). What move?", "completion": "Move RIGHT because it shortens distance to apple while avoiding obstacles."}
```

### **NPZ (Sequential)**
**Purpose**: Time-series models and reinforcement learning
```python
# State sequences with temporal structure
data = {
    'states': np.array([state_sequence]),      # Shape: (timesteps, features)
    'actions': np.array([action_sequence]),    # Shape: (timesteps,)
    'rewards': np.array([reward_sequence])     # Shape: (timesteps,)
}
```

### **NPZ (2D Arrays)**
**Purpose**: Convolutional neural networks
```python
# Board as image with channels
board = np.zeros((grid_size, grid_size, channels))
# channels: [snake_body, snake_head, apple, walls]
```

### **NPZ (Raw Arrays) - Evolutionary Algorithms**
**Purpose**: Population-based optimization with genetic operators

```python
# Specialized evolutionary data format
evolutionary_data = {
    # Population Structure
    'population': np.array(shape=(population_size, individual_length)),
    'fitness_scores': np.array(shape=(population_size, num_objectives)),
    'generation_history': np.array(shape=(num_generations, population_size, individual_length)),
    
    # Genetic Operators Data
    'crossover_points': np.array(shape=(num_crossovers, 2)),  # Parent indices
    'mutation_mask': np.array(shape=(population_size, individual_length)),  # Boolean mask
    'selection_pressure': np.array(shape=(num_generations,)),  # Selection statistics
    
    # Fitness Landscape
    'fitness_landscape': np.array(shape=(grid_size, grid_size, num_objectives)),
    'pareto_front': np.array(shape=(pareto_size, num_objectives)),
    
    # Evolutionary Metadata
    'generation_metadata': {
        'best_fitness': np.array(shape=(num_generations,)),
        'average_fitness': np.array(shape=(num_generations,)),
        'diversity_metrics': np.array(shape=(num_generations,)),
        'convergence_rate': np.array(shape=(num_generations,))
    },
    
    # Game-Specific Evolutionary Data
    'game_performance': {
        'scores': np.array(shape=(population_size,)),
        'steps': np.array(shape=(population_size,)),
        'efficiency': np.array(shape=(population_size,)),
        'survival_rate': np.array(shape=(population_size,))
    }
}
```

**Why Evolutionary Algorithms Need This Special Format:**
- **Population-centric operations**: Direct genetic representation for batch operations
- **Multi-objective support**: Fitness vectors and Pareto front tracking
- **Genetic operator efficiency**: Crossover and mutation history tracking
- **Fitness landscape analysis**: Spatial representation of optimization landscape
- **Game-specific correlation**: Link genetic traits to game performance

## 🎯 **Extension-Specific Format Requirements**

### **Heuristics Extensions**
- **v0.01-v0.03**: CSV format for supervised learning
- **v0.04**: CSV format for supervised learning + JSONL format for LLM fine-tuning

### **Supervised Learning Extensions**
- **Tree Models**: CSV format from heuristics-v0.03 or heuristics-v0.04 (both widely used)
- **Neural Networks**: CSV (MLP), NPZ 2D (CNN), NPZ Sequential (RNN)
- **Graph Models**: NPZ with graph representations

### **Reinforcement Learning Extensions**
- **Experience Replay**: NPZ Sequential format
- **Policy Data**: NPZ with state-action pairs

### **Evolutionary Extensions**
- **Population Data**: NPZ Raw Arrays (specialized evolutionary format)
- **Fitness Data**: NPZ Raw Arrays (multi-objective fitness vectors)
- **Genetic History**: NPZ Raw Arrays (generation history and operator tracking)

## 🔄 **Format Selection Decision Tree**

```
What is your primary use case?
├─ LLM Fine-tuning → JSONL (from heuristics-v0.04)
├─ Tree-based models → CSV (16-feature from heuristics-v0.04)
├─ CNN/Spatial → NPZ (2D Arrays)
├─ RNN/Temporal → NPZ (Sequential)
├─ RL Experience → NPZ (Sequential)
└─ Evolutionary → NPZ (Raw Arrays - Specialized)
```

## 🚫 **Anti-Patterns to Avoid**

### **Wrong Format for Algorithm**
```python
# ❌ WRONG: CSV for CNNs
cnn_model.train(csv_data)  # Loses spatial structure

# ✅ CORRECT: 2D arrays for CNNs
cnn_model.train(spatial_arrays)  # Preserves spatial relationships

# ❌ WRONG: CSV for Evolutionary Algorithms
agent_ga.evolve(csv_population)  # Loses genetic structure

# ✅ CORRECT: Raw arrays for Evolutionary Algorithms
agent_ga.evolve(raw_population)  # Preserves genetic representation
```

## 🎯 **Important Guidelines: Version Selection Guidelines**

- **For supervised learning**: Use CSV from either heuristics-v0.03 or heuristics-v0.04 (both widely used)
- **For LLM fine-tuning**: Use JSONL from heuristics-v0.04 only
- **For research**: Use both formats from heuristics-v0.04
- **CSV is ACTIVE**: Not legacy - actively used for supervised learning
- **JSONL is ADDITIONAL**: New capability for LLM fine-tuning (heuristics-v0.04 only)
- **UTF-8 Encoding**: All file operations must use UTF-8 encoding (SUPREME_RULE NO.7)

## 🔄 **Cross-Extension Integration**
- **Heuristics v0.03**: Generates standardized CSV datasets for supervised learning
- **Heuristics v0.04**: Generates standardized CSV datasets for supervised learning + JSONL datasets for LLM fine-tuning
- **Supervised v0.01+**: Consumes CSV datasets from heuristics-v0.03 or heuristics-v0.04 for training all model types
- **Evaluation**: Consistent comparison framework across all algorithm types

## ✅ **Success Indicators**

### **Working Implementation Examples**
- **Heuristics v0.04**: Successfully generates both CSV and JSONL datasets
- **Dataset Generation**: Automatic incremental updates after each game
- **File Handling**: Proper UTF-8 encoding and file management
- **Agent Integration**: Agents provide comprehensive data formatting
- **Cross-Format Support**: Multiple data formats generated efficiently

### **Quality Standards**
- **Data Accuracy**: Generated data accurately represents game state
- **Format Compliance**: Strict adherence to format specifications
- **Performance**: Efficient data generation without blocking game execution
- **Error Handling**: Robust error handling for data generation failures

---

**This data format decision guide ensures optimal format selection for different algorithm types while maintaining cross-extension compatibility and following forward-looking architecture principles.**
