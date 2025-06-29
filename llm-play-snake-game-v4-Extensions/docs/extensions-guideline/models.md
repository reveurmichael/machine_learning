# Model Architecture for Extensions

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for model architecture standards and supplements the _Final Decision Series_ and extension guidelines.

## 🎯 **Core Philosophy: Universal Model Foundation**

The model architecture demonstrates perfect base class design where generic model interfaces provide foundation functionality while extension-specific implementations add specialized behavior for different machine learning approaches.

### **Guidelines Alignment**
- **final-decision-10.md Guideline 1**: Enforces reading all GOOD_RULES before making model architectural changes to ensure comprehensive understanding
- **final-decision-10.md Guideline 2**: Uses precise `final-decision-N.md` format consistently when referencing architectural decisions and model patterns
- **simple logging**: Enables lightweight common utilities with OOP extensibility while maintaining model patterns through inheritance rather than tight coupling

### **Design Philosophy**
- **Universal Base Classes**: Generic model interfaces for all extensions
- **Framework Agnostic**: Support multiple ML frameworks with consistent patterns
- **Cross-Platform Compatibility**: Reliable model saving/loading across all platforms
- **Grid-Size Independence**: Models work across different board configurations

## 🏗️ **Model Storage Architecture**

### **Standardized Directory Structure**
Following final-decision-1.md:

```
logs/extensions/models/
└── grid-size-N/                          # Grid-size specific organization
    ├── supervised_v0.02_{timestamp}/      # Supervised learning models
    │   ├── mlp/
    │   │   ├── model_artifacts/           # Primary model outputs
    │   │   │   ├── model.pth             # PyTorch format
    │   │   │   ├── model.onnx            # Cross-platform format
    │   │   │   └── config.json           # Model configuration
    │   │   └── training_process/
    │   │       └── generated_datasets/    # Datasets created during training
    │   └── xgboost/ [same structure]
    │
    ├── reinforcement_v0.02_{timestamp}/   # RL agent models
    └── llm_finetune_v0.02_{timestamp}/    # Fine-tuned language models
```

### **Path Management Integration**
Following final-decision-6.md:

```python
from extensions.common.path_utils import get_model_path

# Grid-size agnostic model path generation
model_path = get_model_path(
    extension_type="supervised", 
    version="0.02",
    grid_size=grid_size,  # Any supported size
    algorithm="mlp",
    timestamp=timestamp
)
```

## 🔧 **Cross-Platform Model Format Standards**

### **PyTorch Models**
Following professional model persistence patterns:

```python
# ✅ Save state_dict with metadata (cross-platform)
torch.save({
    'model_state': model.state_dict(),
    'epoch': epoch,
    'optimizer_state': optimizer.state_dict(),
    'meta': {
        'torch_version': torch.__version__,
        'model_class': 'MLPAgent',
        'grid_size': grid_size,
        'timestamp': datetime.utcnow().isoformat(),
        'git_sha': get_git_sha()
    }
}, model_path / 'model.pth')

# ✅ Export to ONNX for framework-agnostic inference
torch.onnx.export(model, dummy_input, model_path / 'model.onnx',
                  input_names=['input'], output_names=['output'],
                  opset_version=11)
```

### **Tree-Based Models**
Using stable, human-readable formats:

```python
# ✅ XGBoost - JSON format (version-stable)
bst.get_booster().save_model(model_path / 'model.json')

# ✅ LightGBM - Text format (human-readable)
gbm.save_model(model_path / 'model.txt')
```

### **Reinforcement Learning Models**
Supporting multiple RL frameworks:

```python
# ✅ Save RL agent with full context
rl_checkpoint = {
    'policy_state': agent.policy.state_dict(),
    'optimizer_state': agent.optimizer.state_dict(),
    'replay_buffer': agent.replay_buffer.get_state(),
    'training_metrics': agent.get_training_metrics(),
    'hyperparameters': agent.config
}
torch.save(rl_checkpoint, model_path / 'agent.pth')
```

## 🏗️ **Model Architecture (SUPREME_RULES Compliant)**

### **Factory Pattern Implementation (CANONICAL create() METHOD)**
**CRITICAL REQUIREMENT**: All Model factories MUST use the canonical `create()` method exactly as specified in `final-decision-10.md` SUPREME_RULES:

```python
class ModelFactory:
    """
    Factory Pattern for AI models following final-decision-10.md SUPREME_RULES
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for AI model creation
    Educational Value: Shows how SUPREME_RULES apply to advanced AI systems -
    canonical patterns work regardless of AI complexity.
    
    Reference: final-decision-10.md SUPREME_RULES for canonical method naming
    """
    
    _registry = {
        "MLP": MLPModel,
        "CNN": CNNModel,
        "LSTM": LSTMModel,
        "XGBOOST": XGBoostModel,
        "RANDOM_FOREST": RandomForestModel,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs):  # CANONICAL create() method - SUPREME_RULES
        """Create model using canonical create() method following final-decision-10.md"""
        model_class = cls._registry.get(model_type.upper())
        if not model_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
        print(f"[ModelFactory] Creating model: {model_type}")  # Simple logging - SUPREME_RULES
        return model_class(**kwargs)

# ❌ FORBIDDEN: Non-canonical method names (violates SUPREME_RULES)
class ModelFactory:
    def create_model(self, model_type: str):  # FORBIDDEN - not canonical
        """This violates SUPREME_RULES: factory methods must be named create()"""
        pass
    
    def build_neural_network(self, model_type: str):  # FORBIDDEN - not canonical
        """This violates SUPREME_RULES: factory methods must be named create()"""
        pass
    
    def make_ai_model(self, model_type: str):  # FORBIDDEN - not canonical
        """This violates SUPREME_RULES: factory methods must be named create()"""
        pass
```

## 🧠 **Design Patterns for Model Management**

### **Template Method Pattern**
Base model classes define consistent loading/saving workflow:

```python
class BaseModel:
    """Template method pattern for model persistence"""
    
    def save_model(self, model_path: Path) -> None:
        """Template method defining model saving workflow"""
        # Step 1: Validate model state
        self.validate_model_state()
        
        # Step 2: Prepare metadata (generic)
        metadata = self.prepare_metadata()
        
        # Step 3: Save model weights (hook for extensions)
        self.save_weights(model_path)
        
        # Step 4: Save configuration (generic)
        self.save_configuration(model_path, metadata)
        
        # Step 5: Export to standard formats (hook for extensions)
        self.export_standard_formats(model_path)
        
    # Hook methods for model-specific implementation
    def save_weights(self, model_path: Path) -> None:
        """Override for model-specific weight saving"""
        raise NotImplementedError
        
    def export_standard_formats(self, model_path: Path) -> None:
        """Override for format-specific exports (ONNX, etc.)"""
        pass  # Optional for some model types
```

### **Strategy Pattern for Format Support**
```python
class ModelFormatStrategy:
    """Strategy interface for different model formats"""
    
    def save(self, model, path: Path) -> None:
        """Save model in specific format"""
        raise NotImplementedError
        
    def load(self, path: Path):
        """Load model from specific format"""
        raise NotImplementedError
        
class PyTorchFormatStrategy(ModelFormatStrategy):
    """PyTorch model format implementation"""
    
    def save(self, model, path: Path) -> None:
        torch.save(model.state_dict(), path / 'model.pth')
        
class ONNXFormatStrategy(ModelFormatStrategy):
    """ONNX export format implementation"""
    
    def save(self, model, path: Path) -> None:
        torch.onnx.export(model, dummy_input, path / 'model.onnx')
```

## 🚀 **Grid-Size Independence**

### **Scalable Model Architecture**
Models designed to work across different grid sizes:

```python
class UniversalMLPAgent(BaseModel):
    """Grid-size agnostic MLP model"""
    
    def __init__(self, grid_size: int = 10):
        # Universal feature extraction (16 features regardless of grid size)
        self.feature_extractor = TabularFeatureExtractor()
        self.input_size = 16  # Fixed feature count
        
        # Scalable network architecture
        self.network = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # UP, DOWN, LEFT, RIGHT
        )
        
    def extract_features(self, game_state: Dict[str, Any]) -> torch.Tensor:
        """Extract grid-size independent features"""
        return self.feature_extractor.extract_features(game_state, self.grid_size)
```

## 🎯 **Extension Integration Benefits**

### **Supervised Learning Extensions**
- Consistent model interfaces across all ML frameworks
- Standardized saving/loading for reproducible experiments
- Cross-format compatibility for deployment flexibility

### **Reinforcement Learning Extensions**
- Complete agent state persistence including replay buffers
- Training progress tracking and resumption capabilities
- Hyperparameter configuration management

### **LLM Fine-tuning Extensions**
- Adapter-based model storage for efficient fine-tuning
- Version control for different model variants
- Integration with existing model loading infrastructure

---

**The model architecture provides a robust, scalable foundation for machine learning across all extension types while maintaining consistency with the established patterns from the final-decision series.**

## grid_size

The grid_size should not be fixed to 10, because models trained by machine learning/DL/RL will be stored in ./logs/extensions/models/grid-size-N/{extension_type}_v{version}_{timestamp}/{model_name}/ following the structure defined in final-decision-1.md.

Datasets generated by the heuristics/ML/DL will be stored in ./logs/extensions/datasets/grid-size-N/{extension_type}_v{version}_{timestamp}/{algorithm_name}/processed_data/ following the structure defined in final-decision-1.md.

## 📋 **SUPREME_RULES Implementation Checklist for Models**

### **Mandatory Requirements**
- [ ] **Canonical Method**: All factories use `create()` method exactly (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses print() statements only for all model operations (final-decision-10.md compliance)
- [ ] **GOOD_RULES Reference**: References `final-decision-10.md` in all model documentation
- [ ] **Pattern Consistency**: Follows canonical patterns across all model implementations

### **Model-Specific Standards**
- [ ] **Model Creation**: Canonical factory patterns for all model types
- [ ] **Model Loading**: Canonical factory patterns for all loading operations
- [ ] **Model Saving**: Canonical patterns for all saving operations
- [ ] **Format Support**: Simple logging for all format conversion operations

### **Educational Integration**
- [ ] **Clear Examples**: Simple examples using canonical `create()` method for model systems
- [ ] **Pattern Explanation**: Clear explanation of canonical patterns in model context
- [ ] **Best Practices**: Demonstration of SUPREME_RULES in model management systems
- [ ] **Learning Value**: Easy to understand canonical patterns regardless of model complexity

---

**Model architecture ensures consistent, educational, and maintainable AI model management across all Snake Game AI extensions while maintaining strict compliance with `final-decision-10.md` SUPREME_RULES.**

## 🔗 **See Also**

- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`agents.md`**: Authoritative reference for agent implementation with canonical patterns
- **`core.md`**: Base class architecture following canonical principles
- **`factory-design-pattern.md`**: Canonical factory implementation for all systems

## format of the model files
 
To ensure your trained models remain **cross-platform** (Windows, macOS, Linux) and **time-proof** (readable years down the road), follow these guidelines for **saving** and **loading** in each framework, and consider exporting to **standardized interchange formats**.

---

## 1. PyTorch

### 🔒 Save the **state\_dict**, not the full object

* **Why?**  Only saves parameter tensors, not class definitions or code.
* **Format:** PyTorch's native binary format (Pickle under the hood), but tied to `state_dict`.

```python
# Saving
import torch

model = MyModel(...)
# After training…
torch.save({
    'model_state': model.state_dict(),
    'epoch': epoch,
    'optimizer_state': optimizer.state_dict(),
    'meta': {
        'torch_version': torch.__version__,
        'model_class': 'MyModel',
        'timestamp': datetime.utcnow().isoformat(),
    }
}, 'model_pt_v1.pth')
```

```python
# Loading
import torch
from my_models import MyModel  # ensure code is available

checkpoint = torch.load('model_pt_v1.pth', map_location='cpu')
model = MyModel(**checkpoint['meta'].get('init_args', {}))
model.load_state_dict(checkpoint['model_state'])
model.eval()
```

#### 📦  Export to **ONNX** for framework-agnostic inference

```python
# Export
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'model.onnx',
                  input_names=['input'], output_names=['output'],
                  opset_version=11)
```

---

## 2. XGBoost

### 🔧 Use **JSON** or **UBJ** formats rather than binary Pickle

* **Why?**  JSON and UBJ are text-based, version-safe, and human-readable.

```python
# Training and saving
import xgboost as xgb

bst = xgb.XGBClassifier(**params)
bst.fit(X_train, y_train)

# JSON
bst.get_booster().save_model('model_xgb_v1.json')

# Or binary UBJ
bst.get_booster().save_model('model_xgb_v1.ubj')
```

```python
# Loading
import xgboost as xgb

bst = xgb.XGBClassifier()
bst.load_model('model_xgb_v1.json')
```

---

## 3. LightGBM

### 💾 Save as **JSON** or **text** model files

* LightGBM's text format is also human-readable, version-stable.

```python
# Training and saving
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)
gbm = lgb.train(params, lgb_train)
gbm.save_model('model_lgb_v1.txt')    # default text format
```

```python
# Loading
import lightgbm as lgb

gbm = lgb.Booster(model_file='model_lgb_v1.txt')
```
