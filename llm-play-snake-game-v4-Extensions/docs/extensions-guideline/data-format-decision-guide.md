# Data Format Decision Guide for Snake Game AI

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for data format decisions and supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`).

> **See also:** `csv-schema-1.md`, `csv-schema-2.md`, `final-decision-10.md`, `heuristics-as-foundation.md`, `evolutionary.md`.

# Data Format Decision Guide

> **Authoritative Reference**: This document serves as a **GOOD_RULES** authoritative reference for data format decisions and is the single source of truth for all data format decisions across the Snake Game AI project.

> **Important Guidelines**: Both `heuristics-v0.03` and `heuristics-v0.04` are widely used depending on use cases and scenarios. For supervised learning and other general purposes, both versions can be used. For LLM fine-tuning, only `heuristics-v0.04` will be used. The CSV format is **NOT legacy** - it's actively used and valuable for supervised learning.

## 🎯 **Core Philosophy: Format Follows Function**

Data formats are chosen based on **algorithm requirements**, not convenience. Each format serves specific use cases and enables optimal performance for particular model types, strictly following SUPREME_RULES from `final-decision-10.md`.

### **Guidelines Alignment**
- **SUPREME_RULES from `final-decision-10.md` Guideline 1**: Enforces reading all GOOD_RULES before making data format architectural changes to ensure comprehensive understanding
- **SUPREME_RULES from `final-decision-10.md` Guideline 2**: Uses precise `final-decision-N.md` format consistently when referencing architectural decisions and data format patterns
- **SUPREME_RULES compliant logging**: Enables lightweight common utilities with OOP extensibility while maintaining data format patterns through inheritance rather than tight coupling

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

## 📁 **Standardized Storage Structure**

All formats follow the path structure from final-decision-1.md:
```
logs/extensions/datasets/grid-size-N/{extension}_v{version}_{timestamp}/
├── {algorithm}/
│   ├── processed_data/
│   │   ├── tabular_data.csv        # CSV format (ACTIVE, NOT legacy)
│   │   ├── reasoning_data.jsonl    # JSONL format (heuristics-v0.04 only)
│   │   ├── sequential_data.npz     # NPZ Sequential
│   │   ├── spatial_data.npz        # NPZ 2D Arrays
│   │   ├── raw_data.npz            # NPZ Raw Arrays (general)
│   │   └── evolutionary_data.npz   # NPZ Raw Arrays (evolutionary specific)
│   └── game_logs/                  # Original game data
```

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

### **Format Mixing Within Version**
```python
# ❌ WRONG: Multiple formats in same version
heuristics-v0.03 → CSV AND JSONL

# ✅ CORRECT: Single format per version purpose
heuristics-v0.03 → CSV only
heuristics-v0.04 → CSV + JSONL (definitive version)
```

### **Wrong Format for Algorithm**
```python
# ❌ WRONG: CSV for CNNs
cnn_model.train(csv_data)  # Loses spatial structure

# ✅ CORRECT: 2D arrays for CNNs
cnn_model.train(spatial_arrays)  # Preserves spatial relationships

# ❌ WRONG: CSV for Evolutionary Algorithms
ga_agent.evolve(csv_population)  # Loses genetic structure

# ✅ CORRECT: Raw arrays for Evolutionary Algorithms
ga_agent.evolve(raw_population)  # Preserves genetic representation
```


## 🔍 **Validation Requirements**

All extensions MUST validate format compliance:
```python
from extensions.common.validation import validate_dataset
from utils.factory_utils import DatasetFactory # TODO: maybe DatasetFactory should be in one of the files (or a new python file) in the extensions/common folder.

def generate_dataset():
    dataset_factory = DatasetFactory()
    loader = dataset_factory.create("CSV")  # CANONICAL create() method per SUPREME_RULES
    result = validate_dataset(output_path)
    if not result.is_valid:
        raise ValueError(f"Dataset validation failed: {result.message}")

def generate_evolutionary_dataset():
    evolutionary_data = create_evolutionary_dataset()
    result = validate_dataset(output_path)
    if not result.is_valid:
        raise ValueError(f"Evolutionary dataset validation failed: {result.message}")
```

## 📚 **Implementation Guidelines**

### **CSV Generation (heuristics-v0.03 and heuristics-v0.04)**
```python
def create_csv_dataset(game_states, grid_size):
    """Create 16-feature CSV dataset from heuristics-v0.03 or heuristics-v0.04"""
    features = extract_tabular_features(game_states, grid_size)
    return pd.DataFrame(features)
```

### **JSONL Generation (heuristics-v0.04 only)**
```python
def create_jsonl_dataset(game_states, explanations):
    """Create JSONL dataset for LLM fine-tuning from heuristics-v0.04"""
    with open(output_path, 'w') as f:
        for state, explanation in zip(game_states, explanations):
            json.dump({
                "prompt": format_state(state),
                "completion": explanation
            }, f)
            f.write('\n')
```

### **NPZ Generation**
```python
def create_npz_dataset(data_dict, output_path):
    """Create NPZ dataset with multiple arrays"""
    np.savez(output_path, **data_dict)
```

### **Evolutionary NPZ Generation**
```python
def create_evolutionary_dataset(population, fitness_scores, generation_history):
    """Create specialized evolutionary NPZ dataset"""
    evolutionary_data = {
        'population': population,
        'fitness_scores': fitness_scores,
        'generation_history': generation_history,
        'crossover_points': crossover_history,
        'mutation_mask': mutation_history,
        'selection_pressure': selection_history,
        'fitness_landscape': compute_fitness_landscape(),
        'pareto_front': compute_pareto_front(),
        'generation_metadata': {
            'best_fitness': best_fitness_history,
            'average_fitness': avg_fitness_history,
            'diversity_metrics': diversity_history,
            'convergence_rate': convergence_history
        },
        'game_performance': {
            'scores': game_scores,
            'steps': game_steps,
            'efficiency': game_efficiency,
            'survival_rate': survival_rates
        }
    }
    
    np.savez(output_path, **evolutionary_data)
```

## 🎯 **Important Guidelines: Version Selection Guidelines**

- **For supervised learning**: Use CSV from either heuristics-v0.03 or heuristics-v0.04 (both widely used)
- **For LLM fine-tuning**: Use JSONL from heuristics-v0.04 only
- **For research**: Use both formats from heuristics-v0.04
- **CSV is ACTIVE**: Not legacy - actively used for supervised learning
- **JSONL is ADDITIONAL**: New capability for LLM fine-tuning (heuristics-v0.04 only)

## 🔄 **Cross-Extension Integration**
- **Heuristics v0.03**: Generates standardized CSV datasets for supervised learning
- **Heuristics v0.04**: Generates standardized CSV datasets for supervised learning + JSONL datasets for LLM fine-tuning
- **Supervised v0.02+**: Consumes CSV datasets from heuristics-v0.03 or heuristics-v0.04 for training all model types
- **Evaluation**: Consistent comparison framework across all algorithm types

---

**This guide ensures consistent, optimal data format selection across all Snake Game AI extensions while maintaining interoperability and performance. Both heuristics-v0.03 and heuristics-v0.04 are widely used depending on use cases.**

## 🔗 **See Also**

- **`csv-schema-1.md`**: Core CSV schema documentation
- **`csv-schema-2.md`**: CSV schema utilities and implementation
- **`evolutionary.md`**: Evolutionary algorithm data representation
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`heuristics-as-foundation.md`**: Heuristics as data foundation