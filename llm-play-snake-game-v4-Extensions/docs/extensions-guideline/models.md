# Models for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines model standards.

> **See also:** `supervised.md`, `reinforcement-learning.md`, SUPREME_RULES from `final-decision-10.md`, `data-format-decision-guide.md`.

## 🎯 **Core Philosophy: Model Architecture Standards**

Models for Snake Game AI provide **standardized model architectures** across different learning approaches. This system follows SUPREME_RULES from `final-decision-10.md` and uses canonical `create()` methods throughout.

### **Educational Value**
- **Model Architecture**: Understanding different model architectures and their applications
- **Learning Approaches**: Learning how different models work for different tasks
- **Performance Comparison**: Comparing model performance with canonical factory methods
- **Canonical Patterns**: All implementations use canonical `create()` method per SUPREME_RULES

## 🏗️ **Model Factory (CANONICAL)**

### **Model Factory (SUPREME_RULES Compliant)**
```python
from utils.factory_utils import SimpleFactory

class ModelFactory:
    """
    Factory Pattern for models following SUPREME_RULES from final-decision-10.md
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for model systems
    Educational Value: Shows how SUPREME_RULES apply to model architecture
    """
    
    _registry = {
        "MLP": MLPModel,
        "CNN": CNNModel,
        "TRANSFORMER": TransformerModel,
        "XGBOOST": XGBoostModel,
        "RANDOM_FOREST": RandomForestModel,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs):  # CANONICAL create() method per SUPREME_RULES
        """Create model using canonical create() method following SUPREME_RULES from final-decision-10.md"""
        model_class = cls._registry.get(model_type.upper())
        if not model_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
        print(f"[ModelFactory] Creating model: {model_type}")  # SUPREME_RULES compliant logging
        return model_class(**kwargs)
```

### **MLP Model Implementation**
```python
class MLPModel:
    """
    Multi-Layer Perceptron model following SUPREME_RULES.
    
    Design Pattern: Strategy Pattern
    Purpose: Implements MLP for Snake game decision making
    Educational Value: Shows MLP implementation with canonical patterns
    """
    
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int = 4):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.model = self._build_model()
        print(f"[MLPModel] Initialized with {len(hidden_sizes)} hidden layers")  # SUPREME_RULES compliant logging
    
    def _build_model(self):
        """Build MLP model architecture"""
        layers = []
        prev_size = self.input_size
        
        for hidden_size in self.hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, self.output_size))
        return nn.Sequential(*layers)
    
    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 100) -> dict:
        """Train MLP model"""
        print(f"[MLPModel] Starting training with {len(X_train)} samples")  # SUPREME_RULES compliant logging
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"[MLPModel] Epoch {epoch}, Loss: {loss.item():.4f}")  # SUPREME_RULES compliant logging
        
        print(f"[MLPModel] Training completed")  # SUPREME_RULES compliant logging
        return {'final_loss': loss.item(), 'epochs': epochs}
```

## 📊 **Simple Logging for Model Operations**

All model operations must use simple print statements as mandated by SUPREME_RULES from `final-decision-10.md`:

```python
# ✅ CORRECT: Simple logging for models (SUPREME_RULES compliance)
def train_model_pipeline(model_type: str, X_train: torch.Tensor, y_train: torch.Tensor):
    print(f"[ModelRunner] Starting {model_type} training")  # SUPREME_RULES compliant logging
    
    model = ModelFactory.create(model_type, input_size=X_train.shape[1])  # CANONICAL create() method per SUPREME_RULES
    results = model.train(X_train, y_train)
    
    print(f"[ModelRunner] Model training completed with loss: {results['final_loss']:.4f}")  # SUPREME_RULES compliant logging
    return results
```

## 🎯 **Model Types**

### **CNN Model**
```python
class CNNModel:
    """
    Convolutional Neural Network model following SUPREME_RULES.
    
    Design Pattern: Template Method Pattern
    Purpose: Implements CNN for spatial game state processing
    Educational Value: Shows CNN implementation with canonical patterns
    """
    
    def __init__(self, grid_size: int = 10, num_channels: int = 3):
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.model = self._build_cnn()
        print(f"[CNNModel] Initialized for {grid_size}x{grid_size} grid")  # SUPREME_RULES compliant logging
    
    def _build_cnn(self):
        """Build CNN model architecture"""
        return nn.Sequential(
            nn.Conv2d(self.num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (self.grid_size // 4) ** 2, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
    
    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 100) -> dict:
        """Train CNN model"""
        print(f"[CNNModel] Starting CNN training with {len(X_train)} samples")  # SUPREME_RULES compliant logging
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"[CNNModel] Epoch {epoch}, Loss: {loss.item():.4f}")  # SUPREME_RULES compliant logging
        
        print(f"[CNNModel] CNN training completed")  # SUPREME_RULES compliant logging
        return {'final_loss': loss.item(), 'epochs': epochs}
```

### **Transformer Model**
```python
class TransformerModel:
    """
    Transformer model following SUPREME_RULES.
    
    Design Pattern: Strategy Pattern
    Purpose: Implements transformer for sequence-based game state processing
    Educational Value: Shows transformer implementation with canonical patterns
    """
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8):
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.model = self._build_transformer()
        print(f"[TransformerModel] Initialized with d_model={d_model}, nhead={nhead}")  # SUPREME_RULES compliant logging
    
    def _build_transformer(self):
        """Build transformer model architecture"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=512,
            dropout=0.1
        )
        
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        return nn.Sequential(
            nn.Linear(self.input_size, self.d_model),
            transformer_encoder,
            nn.Linear(self.d_model, 4)
        )
    
    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 100) -> dict:
        """Train transformer model"""
        print(f"[TransformerModel] Starting transformer training with {len(X_train)} samples")  # SUPREME_RULES compliant logging
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"[TransformerModel] Epoch {epoch}, Loss: {loss.item():.4f}")  # SUPREME_RULES compliant logging
        
        print(f"[TransformerModel] Transformer training completed")  # SUPREME_RULES compliant logging
        return {'final_loss': loss.item(), 'epochs': epochs}
```

## 🎓 **Educational Applications with Canonical Patterns**

### **Model Understanding**
- **Architecture Design**: Understanding different model architectures using canonical factory methods
- **Learning Approaches**: Learning how different models work for different tasks with simple logging
- **Performance Comparison**: Comparing model performance using canonical patterns
- **Model Selection**: Choosing appropriate models for specific tasks following SUPREME_RULES

### **Model Benefits**
- **Standardization**: Standardized model architectures that follow SUPREME_RULES
- **Performance**: High-performance models with canonical factory methods
- **Flexibility**: Flexible model selection for different tasks
- **Educational Value**: Clear examples of model implementation following SUPREME_RULES

## 📋 **SUPREME_RULES Implementation Checklist for Models**

### **Mandatory Requirements**
- [ ] **Canonical Method**: All factories use `create()` method exactly (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses print() statements only for all model operations (SUPREME_RULES compliance)
- [ ] **Model Integration**: Proper integration with training and evaluation systems
- [ ] **Pattern Consistency**: Follows canonical patterns across all model implementations

### **Model-Specific Standards**
- [ ] **Architecture Design**: Well-designed model architectures with canonical factory methods
- [ ] **Performance**: High-performance models for Snake game tasks
- [ ] **Flexibility**: Flexible model selection for different learning approaches
- [ ] **Documentation**: Clear model explanations following SUPREME_RULES

---

**Models provide standardized architectures for Snake Game AI while maintaining strict SUPREME_RULES from `final-decision-10.md` compliance and educational value.**

## 🔗 **See Also**

- **`supervised.md`**: Supervised learning standards
- **`reinforcement-learning.md`**: Reinforcement learning standards
- **SUPREME_RULES from `final-decision-10.md`**: Governance system and canonical standards
- **`data-format-decision-guide.md`**: Data format standards

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
