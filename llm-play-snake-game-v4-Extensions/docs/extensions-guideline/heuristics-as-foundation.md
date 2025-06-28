# Heuristics as Foundation for ML Ecosystem

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ and extension guidelines and follows **KEEP_THOSE_MARKDOWN_FILES_SIMPLE_RULES** guidelines with a target length of 300-500 lines.

## 🎯 **Core Philosophy: Heuristics Drive the ML Ecosystem**

Heuristic algorithms serve as the cornerstone of the entire machine learning pipeline, providing high-quality labeled data that powers all downstream learning approaches. This follows the multi-directional data ecosystem established in the GOOD_RULES.

### **SUPREME_RULES Alignment**
- **SUPREME_RULE NO.1**: Follows all established GOOD_RULES patterns for foundational heuristic implementations
- **SUPREME_RULE NO.2**: Uses precise `final-decision-N.md` format consistently throughout heuristic architecture references
- **SUPREME_RULE NO.3**: Uses lightweight, OOP-based common utilities with simple logging (print() statements) rather than complex *.log file mechanisms

### **Design Philosophy**
- **Ground Truth Generation**: Deterministic algorithms create perfect training labels
- **Curriculum Foundation**: Systematic progression from simple to complex strategies
- **Language Bridge**: v0.04 converts algorithmic reasoning into natural language
- **Performance Baseline**: Establishes benchmarks for all learning approaches

## 🔄 **Multi-Directional Data Flow**

### **Data Lineage Architecture**
Following Final Decision 1 patterns:

```
Heuristics (Foundation)
├── v0.03 → CSV Datasets ──────────→ Supervised Learning (Task-2)
│                                   ├── Neural Networks (MLP, CNN, LSTM)
│                                   ├── Tree Models (XGBoost, LightGBM)
│                                   └── Graph Models (GCN, GraphSAGE)
│
├── v0.03 → NPZ Datasets ──────────→ Reinforcement Learning (Task-3)
│                                   ├── DQN, PPO, A3C training
│                                   └── Experience replay initialization
│
└── v0.04 → JSONL Datasets ────────→ LLM Fine-tuning (Task-4)
                                    ├── Supervised fine-tuning
                                    └── Language-grounded reasoning
```

### **Cross-Extension Data Consumption**
```python
# Extensions consume heuristic datasets following standardized paths
from extensions.common.path_utils import get_dataset_path

# Supervised learning consumes CSV data
supervised_dataset = get_dataset_path(
    extension_type="heuristics",
    version="0.03",
    grid_size=10,
    algorithm="bfs",
    timestamp="20250625_143022"
) / "processed_data" / "tabular_data.csv"

# LLM fine-tuning consumes JSONL data  
llm_dataset = get_dataset_path(
    extension_type="heuristics",
    version="0.04",
    grid_size=10,
    algorithm="astar",
    timestamp="20250625_143022"
) / "processed_data" / "reasoning_data.jsonl"
```

## 🏗️ **Heuristic Algorithm Hierarchy**

### **Progressive Algorithm Complexity**
Following the extension evolution patterns:

```python
# v0.01: Foundation algorithm
class BFSAgent(BaseAgent):
    """Breadth-first search - guaranteed shortest path"""
    
# v0.02: Algorithm variations
class BFSSafeGreedyAgent(BFSAgent):
    """BFS with safety heuristics and greedy optimization"""
    
class AStarAgent(BaseAgent):
    """A* pathfinding with Manhattan distance heuristic"""
    
class HamiltonianAgent(BaseAgent):
    """Hamiltonian path - theoretical optimum strategy"""
```

### **Algorithm Characteristics**
| Algorithm | Optimality | Speed | Dataset Quality | Use Case |
|-----------|------------|-------|-----------------|----------|
| **BFS** | Optimal | Slow | High-quality paths | Baseline training |
| **A*** | Optimal | Fast | Efficient paths | Production training |
| **Hamiltonian** | Perfect | Deterministic | Perfect labels | Curriculum learning |
| **Greedy BFS** | Good | Very Fast | Practical patterns | Real-world simulation |

## 📊 **Dataset Generation Standards**

### **Grid-Size Agnostic Design**
Following Final Decision 2 configuration patterns:

```python
from config.game_constants import VALID_MOVES, DIRECTIONS
from extensions.common.config.dataset_formats import HEURISTIC_FEATURES

class HeuristicDatasetGenerator:
    """Generates standardized datasets from heuristic gameplay"""
    
    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self.feature_extractor = TabularFeatureExtractor()
        
    def generate_csv_dataset(self, algorithm: str, num_games: int = 1000):
        """Generate CSV dataset with 16 standardized features"""
        dataset_rows = []
        
        for game_id in range(num_games):
            game_data = self.run_game_with_algorithm(algorithm)
            
            for step, state in enumerate(game_data.states):
                # Extract 16 grid-size agnostic features
                features = self.feature_extractor.extract_features(state, self.grid_size)
                
                csv_row = {
                    'game_id': game_id,
                    'step_in_game': step,
                    **features,  # 16 standardized features
                    'target_move': game_data.moves[step]
                }
                dataset_rows.append(csv_row)
                
        return pd.DataFrame(dataset_rows)
```

### **Language-Rich Dataset Generation (v0.04)**
```python
class HeuristicReasoningGenerator:
    """Generates JSONL datasets with natural language explanations"""
    
    def generate_jsonl_dataset(self, algorithm: str, num_games: int = 1000):
        """Generate JSONL dataset with reasoning explanations"""
        jsonl_entries = []
        
        agent = self.create_reasoning_agent(algorithm)
        
        for game_id in range(num_games):
            for state, move, reasoning in agent.play_with_explanations():
                entry = {
                    "prompt": self.format_state_prompt(state),
                    "completion": f"Move: {move}. Reasoning: {reasoning}"
                }
                jsonl_entries.append(entry)
                
        return jsonl_entries
```

## 🎯 **Quality Assurance Standards**

### **Data Quality Metrics**
- **Path Optimality**: Percentage of shortest paths found
- **Game Completion**: Success rate reaching maximum score
- **Coverage**: Variety of game states encountered
- **Consistency**: Reproducible behavior across runs

### **Validation Requirements**
Following Final Decision 2 validation patterns:
```python
from extensions.common.validation.dataset_validator import validate_heuristic_dataset

# Automatic validation during dataset generation
validation_result = validate_heuristic_dataset(
    dataset_path=generated_dataset_path,
    expected_algorithms=["bfs", "astar", "hamiltonian"],
    grid_size=grid_size,
    min_games_per_algorithm=1000
)

if not validation_result.is_valid:
    raise DatasetValidationError(validation_result.errors)
```

## 🚀 **Integration with Learning Extensions**

### **Supervised Learning Dependencies**
All supervised learning extensions acknowledge their dependency on heuristic data:

```python
"""
Supervised Learning Extension v0.02

Data Source: Heuristic algorithms from extensions/heuristics-v0.03+
Training Data: CSV datasets with 16 engineered features
Model Types: Neural networks, tree models, graph models

This extension depends entirely on high-quality datasets generated
by heuristic algorithms for training and evaluation.
"""
```

### **Cross-Extension Data Contracts**
- **Feature Consistency**: Same 16 features across all extensions
- **Format Standardization**: CSV for tabular, NPZ for sequential, JSONL for language
- **Grid-Size Independence**: Works across all supported board sizes
- **Version Compatibility**: Clear data lineage tracking

## 🔗 **GOOD_RULES Integration**

This document integrates with the following authoritative references from the **GOOD_RULES** system:

### **Core Architecture Integration**
- **`agents.md`**: Follows BaseAgent interface and factory patterns for all heuristic implementations
- **`config.md`**: Uses authorized configuration hierarchies for heuristic algorithm parameters
- **`core.md`**: Inherits from base classes and follows established inheritance patterns

### **Extension Development Standards**
- **`extensions-v0.02.md`** through **`extensions-v0.04.md`**: Follows version progression guidelines for heuristic evolution
- **`standalone.md`**: Maintains standalone principle (extension + common = self-contained)
- **`single-source-of-truth.md`**: Provides authoritative ground truth data for all other extensions
- **`supervised.md`**: Supplies foundational datasets for supervised learning approaches

### **Data and Path Management**
- **`data-format-decision-guide.md`**: Generates all major data formats (CSV, NPZ, JSONL) as the primary data source
- **`csv-schema-1.md`** and **`csv-schema-2.md`**: Implements 16-feature tabular schema as the definitive source
- **`unified-path-management-guide.md`**: Uses centralized path utilities from extensions/common/
- **`datasets-folder.md`**: Establishes the foundational tier in the data ecosystem hierarchy

### **Cross-Extension Integration**
- **`reinforcement-learning.md`**: Provides initial datasets and baselines for RL training
- **`fine-tuning.md`**: Supplies language-rich datasets (v0.04 JSONL) for LLM fine-tuning
- **`tree-models.md`**: Generates optimal CSV datasets for tree-based supervised learning
- **`stable-baseline.md`**: Provides experience initialization data for RL frameworks

### **Implementation Quality**
- **`documentation-as-first-class-citizen.md`**: Maintains rich docstrings and algorithmic explanations
- **`elegance.md`**: Follows code quality and educational value standards
- **`naming_conventions.md`**: Uses consistent naming across all heuristic implementations

## 📝 **Simple Logging Examples (SUPREME_RULE NO.3)**

All code examples in this document follow **SUPREME_RULE NO.3** by using simple print() statements rather than complex logging mechanisms:

```python
# ✅ CORRECT: Simple logging as per SUPREME_RULE NO.3
def generate_foundation_datasets(self, algorithms, num_games):
    print(f"[HeuristicFoundation] Generating datasets for {len(algorithms)} algorithms")
    
    for algorithm in algorithms:
        print(f"[HeuristicFoundation] Processing {algorithm} algorithm...")
        
        # Generate CSV for supervised learning
        csv_dataset = self.generate_csv_dataset(algorithm, num_games)
        print(f"[HeuristicFoundation] Generated {len(csv_dataset)} CSV samples for {algorithm}")
        
        # Generate JSONL for LLM fine-tuning (v0.04 only)
        if self.version == "0.04":
            jsonl_dataset = self.generate_jsonl_dataset(algorithm, num_games)
            print(f"[HeuristicFoundation] Generated {len(jsonl_dataset)} JSONL samples for {algorithm}")
        
        print(f"[HeuristicFoundation] {algorithm} dataset generation completed")

# ✅ CORRECT: Educational progress tracking
def validate_foundation_quality(self, dataset_path):
    print(f"[FoundationValidator] Validating dataset quality at {dataset_path}")
    
    # Quality metrics
    optimality_score = self._check_path_optimality()
    coverage_score = self._check_state_coverage()
    consistency_score = self._check_algorithmic_consistency()
    
    print(f"[FoundationValidator] Optimality: {optimality_score:.3f}")
    print(f"[FoundationValidator] Coverage: {coverage_score:.3f}")
    print(f"[FoundationValidator] Consistency: {consistency_score:.3f}")
    
    overall_quality = (optimality_score + coverage_score + consistency_score) / 3
    print(f"[FoundationValidator] Overall foundation quality: {overall_quality:.3f}")
```

---

**Heuristics provide the essential foundation for the entire machine learning ecosystem, generating high-quality labeled data that enables sophisticated learning across neural networks, reinforcement learning, and language models while maintaining full compliance with established GOOD_RULES standards and serving as the authoritative ground truth source.**
