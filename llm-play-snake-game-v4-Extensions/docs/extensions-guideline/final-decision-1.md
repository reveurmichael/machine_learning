# Final Decision: Logs/Extensions Directory Structure

## 🚫 **CRITICAL: NO singleton_utils.py in extensions/common/**

**STOP! READ THIS FIRST**: This project **explicitly FORBIDS**:
- ❌ **singleton_utils.py in extensions/common/utils/**
- ❌ **Any wrapper around ROOT/utils/singleton_utils.py**
- ✅ **USE ROOT/utils/singleton_utils.py** directly when truly needed (it's already generic)

## 🎯 **Executive Summary**

This document establishes the **definitive directory structure** for organizing datasets and models in the `./logs/extensions/` folder across all Snake Game AI tasks (1-5). The structure reflects the **multi-directional data ecosystem** where all tasks can both **consume and generate** high-quality datasets and models.

## 🧠 **Key Architectural Insights**

### **1. Multi-Directional Data Flow**
Unlike traditional linear pipelines, our ecosystem recognizes that:
- **All tasks generate datasets** during training/evaluation
- **Better models create better datasets** (positive feedback loop)
- **Cross-task pollination** improves overall system performance
- **Training produces both models AND datasets simultaneously**

### **2. Task Performance Hierarchy**
Expected performance progression (generally):
1. **Heuristics** (Task 1): Baseline, interpretable, deterministic
2. **Supervised** (Task 2): Better than heuristics, fast inference
3. **Reinforcement Learning** (Task 3): **Potentially optimal**, learned through exploration
4. **LLM Fine-tuned** (Task 4): High performance + natural language reasoning
5. **LLM Distilled** (Task 5): Efficient while maintaining quality

### **3. Dataset Quality Features by Source**

| Task Type | Performance | Interpretability | Speed | Special Features |
|-----------|-------------|------------------|-------|------------------|
| **Heuristics** | Baseline | Highest | Fast | Algorithm traces, search paths |
| **Supervised** | Good | Medium | Very Fast | Confidence scores, feature importance |
| **Reinforcement** | Potentially Optimal | Low | Fast | Q-values, policy distributions, exploration data |
| **LLM Fine-tuned** | High | Highest | Slow | Natural language explanations, step-by-step reasoning |
| **LLM Distilled** | Good | High | Medium | Compressed reasoning, efficiency optimized |

## 📁 **Final Directory Structure**

### **Datasets Organization**

```
logs/extensions/datasets/
└── grid-size-N/
    ├── heuristics_v0.03_{timestamp}/          # Task 1 → Task 2 (Traditional ML)
    │   ├── bfs/
    │   │   ├── game_logs/                     # Original game_N.json, summary.json
    │   │   └── processed_data/
    │   │       ├── tabular_data.csv           # For supervised learning
    │   │       ├── sequential_data.npz        # For RNN/LSTM
    │   │       └── metadata.json
    │   └── astar/ [same structure]
    │
    ├── heuristics_v0.04_{timestamp}/          # Task 1 → Task 4 (LLM Fine-tuning)
    │   ├── bfs/
    │   │   ├── game_logs/                     # Original game_N.json, summary.json
    │   │   └── processed_data/
    │   │       ├── tabular_data.csv           # Legacy format
    │   │       ├── reasoning_data.jsonl       # 🔥 For LLM fine-tuning
    │   │       └── metadata.json
    │   └── astar/ [same structure]
    │
    ├── supervised_v0.02_{timestamp}/          # Task 2 → Others (Improved datasets)
    │   ├── mlp_generated/
    │   │   ├── game_logs/                     # MLP-generated games
    │   │   └── processed_data/
    │   │       ├── tabular_data.csv           # Higher quality features
    │   │       ├── sequential_data.npz        # Better temporal patterns
    │   │       ├── confidence_scores.csv      # Model confidence per move
    │   │       └── metadata.json
    │   ├── xgboost_generated/
    │   └── ensemble_generated/                # Best supervised performance
    │
    ├── reinforcement_v0.02_{timestamp}/       # Task 3 → Others (Optimal datasets)
    │   ├── dqn_generated/
    │   │   ├── game_logs/                     # DQN-generated games
    │   │   ├── experience_replay/
    │   │   │   ├── transitions.npz            # High-quality state transitions
    │   │   │   └── episode_data.json
    │   │   └── processed_data/
    │   │       ├── tabular_data.csv           # Optimal policy features
    │   │       ├── q_values.npz               # Q-value annotations
    │   │       ├── action_probabilities.npz   # Policy distributions
    │   │       └── metadata.json
    │   ├── ppo_generated/
    │   └── curriculum_generated/              # Curriculum learning datasets
    │
    ├── llm_finetune_v0.02_{timestamp}/        # Task 4 → Others (Language-grounded)
    │   ├── lora_generated/
    │   │   ├── game_logs/                     # LLM-generated games
    │   │   └── processed_data/
    │   │       ├── reasoning_data.jsonl       # Rich explanations
    │   │       ├── tabular_data.csv           # Human-interpretable features
    │   │       ├── language_features.npz      # Embedding-based features
    │   │       └── metadata.json
    │   └── full_model_generated/
    │
    └── llm_distillation_v0.02_{timestamp}/    # Task 5 → Others (Efficient)
        ├── student_generated/
        │   ├── game_logs/                     # Distilled model games
        │   └── processed_data/
        │       ├── reasoning_data.jsonl       # Compressed explanations
        │       ├── efficiency_metrics.csv     # Speed/quality trade-offs
        │       └── metadata.json
        └── ensemble_generated/
```

### **Models Organization (Integrated with Dataset Generation)**

```
logs/extensions/models/
└── grid-size-N/
    ├── supervised_v0.02_{timestamp}/
    │   ├── mlp/
    │   │   ├── model_artifacts/
    │   │   │   ├── model.pth                      # Primary model output
    │   │   │   ├── model.onnx                     # Deployment format
    │   │   │   ├── config.json                    # Model configuration
    │   │   │   └── feature_importance.json        # Model interpretability
    │   │   ├── training_process/
    │   │   │   ├── training_history/
    │   │   │   │   ├── loss_curves.json
    │   │   │   │   ├── metrics_per_epoch.json
    │   │   │   │   └── checkpoints/
    │   │   │   └── generated_datasets/            # 🔥 Datasets created during training
    │   │   │       ├── validation_games/
    │   │   │       │   ├── game_N.json            # Games played during validation
    │   │   │       │   └── summary.json
    │   │   │       ├── evaluation_datasets/
    │   │   │       │   ├── tabular_data.csv       # Features from evaluation games
    │   │   │       │   ├── confidence_scores.csv  # Model confidence per move
    │   │   │       │   └── prediction_analysis.npz
    │   │   │       └── dataset_metadata.json
    │   │   └── deployment_ready/
    │   │       ├── optimized_model.onnx           # Production-ready
    │   │       └── inference_config.json
    │   └── xgboost/ [same structure]
    │
    ├── reinforcement_v0.02_{timestamp}/
    │   ├── dqn/
    │   │   ├── model_artifacts/
    │   │   │   ├── policy_network.pth             # Primary RL model
    │   │   │   ├── target_network.pth             # Target network
    │   │   │   ├── config.json                    # RL hyperparameters
    │   │   │   └── final_policy.onnx              # Deployment format
    │   │   ├── training_process/
    │   │   │   ├── training_history/
    │   │   │   │   ├── episode_rewards.json       # Training metrics
    │   │   │   │   ├── loss_curves.json           # Q-learning losses
    │   │   │   │   ├── exploration_stats.json     # Epsilon decay, etc.
    │   │   │   │   └── checkpoints/
    │   │   │   └── generated_datasets/            # 🔥 Experience + Evaluation data
    │   │   │       ├── experience_replay/
    │   │   │       │   ├── transitions.npz        # (s,a,r,s',done) tuples
    │   │   │       │   ├── episode_data.json      # Episode statistics
    │   │   │       │   └── exploration_heatmaps.npz
    │   │   │       ├── evaluation_games/
    │   │   │       │   ├── game_N.json            # Games from evaluation episodes
    │   │   │       │   ├── summary.json
    │   │   │       │   └── q_value_traces.npz     # Q-values at each step
    │   │   │       ├── policy_datasets/
    │   │   │       │   ├── tabular_data.csv       # State-action features
    │   │   │       │   ├── action_probabilities.npz
    │   │   │       │   └── value_estimates.npz
    │   │   │       └── dataset_metadata.json
    │   │   └── deployment_ready/
    │   │       ├── optimized_policy.onnx
    │   │       └── inference_config.json
    │   └── ppo/ [same structure]
    │
    ├── llm_finetune_v0.02_{timestamp}/
    │   ├── lora_adapters/
    │   │   ├── model_artifacts/
    │   │   │   ├── adapter_model.bin              # LoRA weights
    │   │   │   ├── adapter_config.json            # LoRA configuration
    │   │   │   ├── base_model_info.json           # Base model reference
    │   │   │   └── merged_model/                  # Optional merged weights
    │   │   │       ├── pytorch_model.bin
    │   │   │       └── config.json
    │   │   ├── training_process/
    │   │   │   ├── training_history/
    │   │   │   │   ├── training_loss.json         # Fine-tuning losses
    │   │   │   │   ├── validation_metrics.json    # Perplexity, BLEU, etc.
    │   │   │   │   ├── snake_performance.json     # Game performance during training
    │   │   │   │   └── checkpoints/
    │   │   │   └── generated_datasets/            # 🔥 LLM-generated reasoning data
    │   │   │       ├── training_games/
    │   │   │       │   ├── game_N.json            # Games during fine-tuning
    │   │   │       │   └── summary.json
    │   │   │       ├── reasoning_datasets/
    │   │   │       │   ├── reasoning_data.jsonl   # Rich explanations generated
    │   │   │       │   ├── prompt_completion_pairs.jsonl
    │   │   │       │   ├── quality_scores.json    # Reasoning quality metrics
    │   │   │       │   └── language_features.npz  # Embedding representations
    │   │   │       ├── evaluation_datasets/
    │   │   │       │   ├── zero_shot_games.json   # Games without further training
    │   │   │       │   ├── few_shot_games.json    # Games with examples
    │   │   │       │   └── reasoning_quality.json
    │   │   │       └── dataset_metadata.json
    │   │   └── deployment_ready/
    │   │       ├── optimized_adapter.bin          # Quantized/optimized
    │   │       └── inference_config.json
    │   └── full_finetune/ [same structure]
    │
    └── llm_distillation_v0.02_{timestamp}/
        ├── student_models/
        │   ├── model_artifacts/
        │   │   ├── distilled_model.bin            # Compressed student model
        │   │   ├── config.json                    # Student architecture
        │   │   ├── teacher_reference.json         # Teacher model info
        │   │   └── compression_stats.json         # Size/speed improvements
        │   ├── training_process/
        │   │   ├── training_history/
        │   │   │   ├── distillation_loss.json     # KL divergence, etc.
        │   │   │   ├── student_performance.json   # Student vs teacher metrics
        │   │   │   ├── compression_metrics.json   # Speed/memory improvements
        │   │   │   └── checkpoints/
        │   │   └── generated_datasets/            # 🔥 Distillation comparison data
        │   │       ├── comparison_games/
        │   │       │   ├── teacher_games.json     # Teacher-generated games
        │   │       │   ├── student_games.json     # Student-generated games
        │   │       │   └── comparison_analysis.json
        │   │       ├── efficiency_datasets/
        │   │       │   ├── reasoning_data.jsonl   # Compressed explanations
        │   │       │   ├── efficiency_metrics.csv # Speed/quality trade-offs
        │   │       │   └── knowledge_transfer.npz # What knowledge was preserved
        │   │       ├── ablation_datasets/
        │   │       │   ├── component_analysis.json # Which parts matter most
        │   │       │   └── performance_degradation.csv
        │   │       └── dataset_metadata.json
        │   └── deployment_ready/
        │       ├── production_model.bin           # Final optimized model
        │       ├── inference_config.json
        │       └── deployment_guide.md
        └── ensemble_models/ [same structure]
```

## 🔄 **Cross-Task Data Ecosystem**

### **Dataset Consumption Patterns**

```python
DATASET_CONSUMPTION = {
    'supervised_training': {
        'preferred_sources': ['reinforcement_generated', 'heuristics_v0.03', 'heuristics_v0.04', 'supervised_generated'],
        'reason': 'High-quality labeled data for training better models'
    },
    'rl_training': {
        'preferred_sources': ['heuristics_v0.03', 'heuristics_v0.04', 'supervised_generated'],
        'reason': 'Initial policy/value function, curriculum learning'
    },
    'llm_finetuning': {
        'preferred_sources': ['heuristics_v0.04', 'llm_finetune_generated'],
        'reason': 'Language-rich explanations for reasoning'
    },
    'llm_distillation': {
        'preferred_sources': ['llm_finetune_generated'],
        'reason': 'Teacher model outputs for student training'
    },
    'comparative_analysis': {
        'preferred_sources': ['all_sources'],
        'reason': 'Performance benchmarking across approaches'
    }
}
```

### **Training Session Outputs**

Each training session produces **dual outputs**:

1. **Primary Output**: The trained model with deployment artifacts
2. **Secondary Output**: High-quality datasets from evaluation/gameplay
3. **Process Artifacts**: Training logs, metrics, checkpoints

### **Dataset Quality Evolution**

```python
TRAINING_DATASET_EVOLUTION = {
    'early_training': {
        'model_quality': 'poor',
        'dataset_quality': 'low',
        'use_case': 'debugging, architecture validation'
    },
    'mid_training': {
        'model_quality': 'improving', 
        'dataset_quality': 'medium',
        'use_case': 'curriculum learning, intermediate benchmarks'
    },
    'final_model': {
        'model_quality': 'best',
        'dataset_quality': 'highest',
        'use_case': 'training next generation models, research datasets'
    }
}
```

## 🛠️ **Implementation Framework**

### **Path Manager Implementation**

```python
# extensions/common/task_aware_path_manager.py
class TaskAwarePathManager:
    """
    Centralized path management for task-aware directory structure
    
    Design Patterns:
    - Facade Pattern: Simplifies complex path management
    - Factory Pattern: Creates appropriate structures per task type
    - Strategy Pattern: Different path strategies for different tasks
    """
    
    TASK_DATA_TYPES = {
        'heuristics': ['tabular_data.csv', 'sequential_data.npz', 'reasoning_data.jsonl'],
        'supervised': ['model.pth', 'model.onnx', 'config.json', 'confidence_scores.csv'],
        'reinforcement': ['policy_network.pth', 'transitions.npz', 'q_values.npz'],
        'llm_finetune': ['adapter_model.bin', 'reasoning_data.jsonl', 'language_features.npz'],
        'llm_distillation': ['distilled_model.bin', 'efficiency_metrics.csv']
    }
    
    def __init__(self, extension_type: str, version: str, grid_size: int):
        self.extension_type = extension_type
        self.version = version
        self.grid_size = grid_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"{extension_type}_v{version}_{self.timestamp}"
    
    def get_dataset_structure(self, algorithm: str) -> Dict[str, Path]:
        """Create dataset directory structure"""
        base_path = Path("logs/extensions/datasets") / f"grid-size-{self.grid_size}" / self.session_name / algorithm
        
        return {
            'base': base_path,
            'game_logs': base_path / 'game_logs',
            'processed_data': base_path / 'processed_data'
        }
    
    def get_model_structure(self, model_name: str) -> Dict[str, Path]:
        """Create model directory structure with integrated dataset generation"""
        base_path = Path("logs/extensions/models") / f"grid-size-{self.grid_size}" / self.session_name / model_name
        
        return {
            'base': base_path,
            'model_artifacts': base_path / 'model_artifacts',
            'training_process': base_path / 'training_process',
            'training_history': base_path / 'training_process' / 'training_history',
            'generated_datasets': base_path / 'training_process' / 'generated_datasets',
            'deployment_ready': base_path / 'deployment_ready'
        }
```

### **Usage Examples**

```python
# Heuristics v0.04 - generating language-rich datasets
path_manager = TaskAwarePathManager("heuristics", "0.04", grid_size=10)
paths = path_manager.get_dataset_structure("bfs")

# Save reasoning data for LLM fine-tuning
reasoning_path = paths['processed_data'] / 'reasoning_data.jsonl'
with open(reasoning_path, 'w') as f:
    for step in game_steps:
        f.write(json.dumps({
            "prompt": f"Snake head at {step.head}, apple at {step.apple}. Plan move:",
            "completion": f"Move {step.action} because {step.reasoning}"
        }) + '\n')

# Reinforcement Learning - saving experience + evaluation data
rl_manager = TaskAwarePathManager("reinforcement", "0.02", grid_size=10)
model_paths = rl_manager.get_model_structure("dqn")

# Save model
torch.save(policy_net.state_dict(), model_paths['model_artifacts'] / 'policy_network.pth')

# Save generated datasets during training
np.savez(model_paths['generated_datasets'] / 'experience_replay' / 'transitions.npz',
         states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones)
```

## 📊 **Data Format Standards**

### **Grid-Size Agnostic CSV Schema**

To enable models to generalize across different grid sizes, all tabular datasets use **normalized features** that remain consistent regardless of board dimensions:

```python
# Standardized 16-feature schema for all grid sizes
GRID_SIZE_AGNOSTIC_FEATURES = [
    # Metadata columns (2)
    'game_id',                # Unique game session identifier
    'step_in_game',          # Step number within the game
    
    # Normalized position features (4)
    'head_x_normalized',     # head_x / grid_size (0.0 to 1.0)
    'head_y_normalized',     # head_y / grid_size (0.0 to 1.0)
    'apple_x_normalized',    # apple_x / grid_size (0.0 to 1.0)
    'apple_y_normalized',    # apple_y / grid_size (0.0 to 1.0)
    
    # Game state (1)
    'snake_length',          # Current snake length (absolute)
    
    # Relative direction features (4) - grid-size independent
    'apple_dir_up',          # 1 if apple is above snake head, else 0
    'apple_dir_down',        # 1 if apple is below snake head, else 0
    'apple_dir_left',        # 1 if apple is left of snake head, else 0
    'apple_dir_right',       # 1 if apple is right of snake head, else 0
    
    # Immediate danger features (3) - grid-size independent
    'danger_straight',       # 1 if collision ahead, else 0
    'danger_left',           # 1 if collision to left, else 0
    'danger_right',          # 1 if collision to right, else 0
    
    # Proportional free space features (4)
    'free_space_up_ratio',   # free_space_up / grid_size (0.0 to 1.0)
    'free_space_down_ratio', # free_space_down / grid_size (0.0 to 1.0)
    'free_space_left_ratio', # free_space_left / grid_size (0.0 to 1.0)
    'free_space_right_ratio',# free_space_right / grid_size (0.0 to 1.0)
    
    # Target column (1)
    'target_move'            # Next move (UP, DOWN, LEFT, RIGHT)
]

# Total: 19 columns (2 metadata + 16 features + 1 target)
```

### **Normalization Benefits**

| Feature Type | Original (Grid-Dependent) | Normalized (Grid-Agnostic) | Benefit |
|--------------|--------------------------|----------------------------|---------|
| **Positions** | `head_x=5` (on 10×10) | `head_x_normalized=0.5` | Model learns relative positions |
| **Free Space** | `free_space_up=3` (on 10×10) | `free_space_up_ratio=0.3` | Proportional space awareness |
| **Directions** | `apple_dir_up=1` | `apple_dir_up=1` | Already grid-agnostic |
| **Dangers** | `danger_straight=1` | `danger_straight=1` | Already grid-agnostic |

### **Cross-Grid Training Examples**

```python
# Example: Training on mixed grid sizes
training_data = [
    # 8×8 grid example
    {'head_x_normalized': 0.625, 'apple_x_normalized': 0.25, 'free_space_up_ratio': 0.375, 'target_move': 'LEFT'},
    
    # 10×10 grid example  
    {'head_x_normalized': 0.6, 'apple_x_normalized': 0.2, 'free_space_up_ratio': 0.3, 'target_move': 'LEFT'},
    
    # 16×16 grid example
    {'head_x_normalized': 0.5625, 'apple_x_normalized': 0.1875, 'free_space_up_ratio': 0.3125, 'target_move': 'LEFT'}
]

# Model sees consistent feature ranges regardless of original grid size
# All normalized features range from 0.0 to 1.0
# Model can generalize from small grids to large grids and vice versa
```

### **Task-Specific Formats**

| Task Type | Primary Data Format | Secondary Formats | Special Features |
|-----------|-------------------|------------------|------------------|
| **Heuristics** | CSV (tabular) | NPZ (sequential), JSONL (reasoning) | Algorithm traces, search paths, **grid-size normalized** |
| **Supervised** | CSV (features) | ONNX (deployment), NPZ (predictions) | Confidence scores, feature importance, **cross-grid compatibility** |
| **Reinforcement** | NPZ (experience) | JSON (episodes), CSV (metrics) | Q-values, policy distributions, **normalized state representations** |
| **LLM Fine-tuning** | JSONL (reasoning) | NPZ (embeddings), JSON (metadata) | Prompt-completion pairs, quality scores |
| **LLM Distillation** | BIN (models) | JSON (comparisons), CSV (efficiency) | Compression analysis, knowledge transfer |

### **Metadata Standards**

Each dataset/model directory includes:
- **`metadata.json`**: Generation parameters, model info, performance metrics
- **`dataset_metadata.json`**: Data characteristics, quality scores, usage notes
- **Training logs**: Complete training history and metrics
- **Deployment configs**: Production-ready configuration files

## 🎯 **Benefits of This Structure**

### **1. Scalability**
- **Grid-size independent**: Works with any board size
- **Version agnostic**: New versions integrate seamlessly
- **Task extensible**: Easy to add new task types

### **2. Research Enablement**
- **Complete provenance**: Track data lineage across tasks
- **Performance comparison**: Easy benchmarking across approaches
- **Ablation studies**: Organized comparison data

### **3. Production Readiness**
- **Deployment artifacts**: ONNX models, configs, optimization stats
- **Performance monitoring**: Generated datasets for validation
- **Efficiency tracking**: Speed/memory/accuracy trade-offs

### **4. Educational Value**
- **Clear progression**: See how models improve over time
- **Interpretable outputs**: Reasoning explanations, feature importance
- **Complete examples**: Full training → deployment pipeline

## 🚀 **Implementation Timeline**

### **Phase 1: Core Infrastructure**
1. Implement `TaskAwarePathManager`
2. Update all existing extensions to use new structure
3. Create validation scripts for directory compliance

### **Phase 2: Cross-Task Integration**
1. Implement dataset consumption utilities
2. Create model evaluation frameworks
3. Build comparative analysis tools

### **Phase 3: Advanced Features**
1. Automated dataset quality assessment
2. Model performance tracking
3. Deployment optimization pipelines

## 📋 **Compliance Requirements**

### **All Extensions Must**
- ✅ Use `TaskAwarePathManager` for all path operations
- ✅ Generate both models and datasets during training
- ✅ Include deployment-ready artifacts
- ✅ Follow task-specific data format standards
- ✅ Include comprehensive metadata
- ✅ Support grid-size parameterization

### **Validation Checklist**
- [ ] Directory structure follows specification
- [ ] All required metadata files present
- [ ] Data formats match task requirements
- [ ] Deployment artifacts generated
- [ ] Cross-task compatibility maintained

---

**This structure establishes a comprehensive, scalable foundation for the multi-task Snake Game AI ecosystem, enabling advanced research while maintaining production readiness.**
