# Extensions Common Utilities

This package provides shared utilities for all Snake Game AI extensions, following the principle that **each extension + common folder = standalone unit**.

## 🎯 Philosophy

- **Single Source of Truth**: Common utilities prevent code duplication across extensions
- **Extension Independence**: Extensions only share code through this common package  
- **Educational Value**: Clear separation between extension-specific and shared code
- **Grid-Size Agnostic**: All utilities work with any supported grid size (8-20)
- **Format Flexibility**: Support for multiple data formats (CSV, JSONL, NPZ)

### **SUPREME_RULE NO.3 & NO.4 Implementation**

This package follows two critical supreme rules:

#### **SUPREME_RULE NO.3**: Flexibility for Innovation
"We should be able to add new extensions easily and try out new ideas. Therefore, code in the 'extensions/common/' folder should NOT be too restrictive."

- **No Hard-Coded Restrictions**: Algorithm lists, version numbers, and extension types are not artificially limited
- **Encourages Experimentation**: Easy to add new algorithms, extension types, and versions
- **Educational Flexibility**: Supports creative exploration while maintaining architectural integrity

#### **SUPREME_RULE NO.4**: OOP Extensibility
"While things in the folder 'extensions/common/' are expected to be shared across all extensions, we expect exceptions to be made for certain extensions, as we have a very future-proof mindset. Therefore, whenever possible, make things in the 'extensions/common/' folder OOP, so that, if exceptions are to be made, they can extend those classes in the 'extensions/common/' folder, to adapt to the exceptions and some exceptional needs for those certain extensions."

- **Inheritance-Ready**: All major classes designed for extension through inheritance
- **Protected Extension Points**: Methods marked for selective customization by subclasses
- **Virtual Methods**: Complete behavior replacement when needed for specialized requirements
- **Composition Support**: Pluggable components for flexible customization

## 📁 Package Structure

```
extensions/common/
├── __init__.py                    # Main package exports
├── README.md                      # This file
├── 
├── config/                        # Configuration management
│   ├── __init__.py
│   ├── ml_constants.py           # ML hyperparameters and settings
│   ├── training_defaults.py      # Training configuration defaults
│   ├── dataset_formats.py        # Data format specifications
│   ├── path_constants.py         # Path templates and naming
│   ├── validation_rules.py       # Validation thresholds
│   └── model_registry.py         # Model type definitions
│
├── validation/                    # Comprehensive validation system
│   ├── __init__.py
│   ├── validation_types.py       # Common validation classes
│   ├── dataset_validator.py      # Dataset format validation
│   ├── model_validator.py        # Model output validation  
│   ├── path_validator.py         # Path structure validation
│   ├── config_validator.py       # Configuration access validation
│   └── extension_validator.py    # Extension compliance validation
│
├── path_utils.py                  # Path management utilities
├── csv_schema.py                  # Grid-agnostic CSV utilities
├── dataset_loader.py              # Standardized dataset loading
├── factory_utils.py               # Shared factory patterns
├── extension_utils.py             # Extension environment management
└── test_utils.py                  # Testing framework
```

## 🚀 Quick Start

### Basic Usage
```python
# Standard extension setup
from extensions.common import create_extension_environment

# Set up complete extension environment
project_root, ext_path, logger = create_extension_environment(
    extension_type="heuristics",
    version="0.03", 
    grid_size=10,
    algorithm="bfs"
)

# Use standardized dataset loading
from extensions.common import load_dataset_for_training

X_train, X_val, X_test, y_train, y_val, y_test, info = load_dataset_for_training(
    dataset_paths=["path/to/heuristics_v0.03_dataset.csv"],
    grid_size=10
)
```

### Path Management
```python
from extensions.common import ensure_project_root, get_dataset_path

# Ensure working in project root
project_root = ensure_project_root()

# Generate standardized paths
dataset_path = get_dataset_path(
    extension_type="heuristics",
    version="0.03",
    grid_size=10, 
    algorithm="bfs",
    timestamp="20240101_120000"
)
# Result: logs/extensions/datasets/grid-size-10/heuristics_v0.03_20240101_120000/
```

### CSV Schema (Grid-Size Agnostic)
```python
from extensions.common import generate_csv_schema, create_csv_row

# Works with any grid size
schema = generate_csv_schema(grid_size=16)  # 16x16 grid
row = create_csv_row(game_state, "UP", game_id=1, step=5, grid_size=16)
```

### Validation System
```python
from extensions.common import validate_dataset_format, validate_config_access

# Validate dataset format
result = validate_dataset_format(
    dataset_path="path/to/dataset.csv",
    extension_type="heuristics", 
    version="0.03",
    expected_format="csv"
)

# Validate configuration access (enforces LLM whitelist)
validate_config_access("heuristics", ["config.game_constants"])  # ✅ Allowed
validate_config_access("heuristics", ["config.llm_constants"])   # ❌ Forbidden
```

## 🏗️ Key Components

### 1. Configuration Management (`config/`)

Centralized configuration system with clear access control:

- **Universal Constants**: Used by all extensions (`game_constants`, `ui_constants`)
- **Extension Constants**: Shared across extensions (`ml_constants`, `training_defaults`)
- **LLM Constants**: Only for LLM-focused extensions (whitelist enforced)

```python
# ✅ All extensions can use
from extensions.common.config import DEFAULT_LEARNING_RATE, MIN_GRID_SIZE

# ✅ Only agentic-llms-*, llm-*, vision-language-model-* extensions
from config.llm_constants import AVAILABLE_PROVIDERS  # Enforced by validation
```

### 2. Validation System (`validation/`)

Comprehensive validation ensuring architectural compliance:

- **Dataset Validation**: Format compliance, schema validation
- **Model Validation**: Output format, performance thresholds
- **Path Validation**: Directory structure, naming conventions
- **Config Validation**: Access control, import restrictions
- **Extension Validation**: Standalone compliance, version requirements

### 3. Data Processing

Grid-size agnostic data processing with multiple format support:

**CSV Format (16-Feature Schema)**:
- ✅ Tree models (XGBoost, LightGBM, Random Forest)
- ✅ Simple neural networks (MLP)
- ✅ Traditional ML (SVM, Logistic Regression)

**JSONL Format**:
- ✅ LLM fine-tuning (prompt-completion pairs)
- ✅ Language-rich datasets

**NPZ Format**:
- ✅ Sequential data (LSTM, RNN)
- ✅ Spatial data (CNN)
- ✅ Raw arrays (Evolutionary algorithms)

### 4. Factory Patterns (`factory_utils.py`)

Standardized factory implementations for creating:
- Agents based on algorithm type
- Models based on architecture type  
- Validators based on data type
- Datasets based on format type

### 5. Extension Management (`extension_utils.py`)

Complete extension lifecycle management:
- Environment setup and validation
- Logging configuration
- Directory creation
- Extension discovery and analysis

## 📊 Data Format Decision Guide

| Algorithm Type | Recommended Format | Rationale |
|---------------|-------------------|-----------|
| **Heuristics** | CSV (16-feature) | Tabular data perfect for analysis |
| **Tree Models** | CSV (16-feature) | Optimal for XGBoost, LightGBM |
| **Neural Networks (MLP)** | CSV (16-feature) | Simple tabular input |
| **CNNs** | NPZ (2D arrays) | Spatial relationships preserved |
| **RNNs/LSTMs** | NPZ (sequential) | Temporal patterns captured |
| **Evolutionary** | NPZ (raw arrays) | Direct genetic manipulation |
| **LLM Fine-tuning** | JSONL | Language-rich prompt-completion pairs |

## 🧪 Testing

Run comprehensive tests for all utilities:

```bash
cd extensions/common
python test_utils.py
```

Or programmatically:
```python
from extensions.common import run_common_utilities_tests
summary = run_common_utilities_tests()
```

## 🔒 Architectural Guarantees

### Single Source of Truth
- Each constant/utility has exactly one authoritative location
- No duplication between extensions
- Clear import patterns prevent dependencies

### Extension Independence  
- Extensions only share code through common package
- No cross-extension imports allowed
- Each extension + common = standalone unit

### Grid-Size Agnostic
- All utilities work with any grid size (8-20)
- No hardcoded grid dimensions
- Scalable feature extraction

### Format Flexibility
- Multiple data formats supported
- Format selection based on algorithm requirements
- Consistent interfaces across formats

### Access Control
- LLM constants only accessible to LLM-focused extensions
- Configuration validation enforces architectural boundaries
- Clear separation of concerns

## 📋 Usage Patterns

### For Heuristics Extensions
```python
from extensions.common import create_extension_environment, generate_csv_schema
from extensions.common.config import MIN_GRID_SIZE, MAX_GRID_SIZE

# Standard setup
project_root, ext_path, logger = create_extension_environment(
    "heuristics", "0.03", 10, "bfs"
)

# Generate datasets
schema = generate_csv_schema(grid_size=10)
```

### For Supervised Learning Extensions
```python
from extensions.common import load_dataset_for_training, DatasetLoader
from extensions.common.config import DEFAULT_LEARNING_RATE, EARLY_STOPPING_PATIENCE

# Load training data
X_train, X_val, X_test, y_train, y_val, y_test, info = load_dataset_for_training(
    dataset_paths=["heuristics_v0.03_dataset.csv"],
    grid_size=10
)
```

### For LLM Extensions (Whitelisted)
```python
from extensions.common import create_extension_environment
from config.llm_constants import AVAILABLE_PROVIDERS  # ✅ Allowed for agentic-llms-*
from config.prompt_templates import SYSTEM_PROMPT     # ✅ Allowed for llm-*

# Standard setup with LLM access
project_root, ext_path, logger = create_extension_environment(
    "agentic-llms", "0.02", 10, "react"
)
```

### Extending Common Classes (SUPREME_RULE NO.4)

The common utilities are designed for extension when specialized behavior is needed:

#### Custom Feature Extractor
```python
from extensions.common.csv_schema import TabularFeatureExtractor

class RLFeatureExtractor(TabularFeatureExtractor):
    """Extension-specific feature extractor for RL agents."""
    
    def _initialize_extractor_settings(self):
        # SUPREME_RULE NO.4: Extension point for specialized settings
        self.include_reward_features = True
        self.temporal_window = 5
    
    def _extract_extension_specific_features(self, game_state):
        # SUPREME_RULE NO.4: Add RL-specific features
        if hasattr(game_state, 'reward_history'):
            return {
                "recent_reward": game_state.reward_history[-1],
                "reward_trend": self._calculate_reward_trend(game_state)
            }
        return {}
```

#### Custom Dataset Loader
```python
from extensions.common.dataset_loader import BaseDatasetLoader

class EvolutionaryDatasetLoader(BaseDatasetLoader):
    """Specialized loader for evolutionary algorithm data."""
    
    def _initialize_loader_specific_settings(self):
        # SUPREME_RULE NO.4: Extension point for specialized configurations
        self.population_validator = PopulationValidator()
        self.genetic_processor = GeneticDataProcessor()
    
    def _generate_extension_specific_metadata(self, data, file_path):
        # SUPREME_RULE NO.4: Add evolutionary-specific metadata
        return {
            "population_size": self._count_population(data),
            "generation_count": self._count_generations(data),
            "fitness_distribution": self._analyze_fitness(data)
        }
```

#### Custom Validator
```python
from extensions.common.validation.extension_validator import ExtensionValidator

class RLExtensionValidator(ExtensionValidator):
    """Specialized validator for RL extensions."""
    
    def _initialize_validator_settings(self):
        # SUPREME_RULE NO.4: Extension point for specialized validators
        self.rl_model_validator = RLModelValidator()
        self.environment_validator = EnvironmentValidator()
    
    def _validate_extension_specific(self, context):
        # SUPREME_RULE NO.4: Add RL-specific validation
        results = []
        
        # Validate RL model architecture
        if self._has_rl_models(context):
            results.append(self._validate_rl_models(context))
        
        # Validate environment configuration
        results.append(self._validate_environment_config(context))
        
        return results
```

## 🔗 Related Documentation

- **Configuration Architecture**: `docs/extensions-guideline/config.md`
- **Path Management**: `docs/extensions-guideline/unified-path-management-guide.md`
- **Data Formats**: `docs/extensions-guideline/data-format-decision-guide.md`
- **Validation Rules**: `docs/extensions-guideline/validation-types.md`
- **Factory Patterns**: `docs/extensions-guideline/unified-factory-pattern-guide.md`

---

**The common utilities package ensures consistent, scalable, and maintainable shared functionality while preserving the architectural principle that each extension + common folder forms a standalone unit.** 