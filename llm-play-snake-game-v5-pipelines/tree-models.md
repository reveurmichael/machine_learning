# Tree Models for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines tree model patterns for extensions.

> **Guidelines Alignment:**
> - This document is governed by the guidelines in `final-decision-10.md`.
> - All agent factories must use the canonical method name `create()` (never `create_agent`, `create_model`, etc.).
> - All code must use simple print logging (simple logging).
> - Reference: `extensions/common/utils/factory_utils.py` for the canonical `SimpleFactory` implementation.

> **See also:** `agents.md`, `core.md`, `final-decision-10.md`, `factory-design-pattern.md`, `config.md`, `csv-schema-1.md`.

## 🎯 **Core Philosophy: Interpretable Decision Trees**

Tree-based models provide interpretable, efficient decision-making for Snake game AI. They excel at the 16-feature tabular data format and offer clear decision paths that can be analyzed and understood.

### **Guidelines Alignment**
- **final-decision-10.md Guideline 1**: Follows all established GOOD_RULES patterns
- **final-decision-10.md Guideline 2**: References `final-decision-N.md` format consistently  
- **simple logging**: Uses lightweight, OOP-based common utilities with simple logging (print() statements)

## 🏗️ **Extension Structure**

### **Directory Layout**
```
extensions/tree-models-v0.02/
├── __init__.py
├── agents/
│   ├── __init__.py               # Agent factory
│   ├── agent_xgboost.py          # XGBoost implementation
│   ├── agent_lightgbm.py         # LightGBM implementation
│   ├── agent_randomforest.py     # Random Forest implementation
│   └── agent_decisiontree.py     # Decision Tree implementation
├── models/
│   ├── __init__.py
│   ├── xgboost_model.py          # XGBoost model wrapper
│   ├── lightgbm_model.py         # LightGBM model wrapper
│   └── sklearn_wrappers.py       # Scikit-learn wrappers
├── training/
│   ├── __init__.py
│   ├── trainer.py                # Training pipeline
│   └── hyperparameter_tuner.py   # Hyperparameter optimization
├── game_logic.py                 # Tree model game logic
├── game_manager.py               # Tree model manager
└── main.py                       # CLI interface
```

## 🔧 **Implementation Patterns**

### **Tree Model Factory**
```python
class TreeModelFactory:
    """
    Simple factory for tree-based models
    
    Design Pattern: Factory Pattern
    - Simple dictionary-based registry
    - Canonical create() method
    - Easy extension for new tree models
    """
    
    _registry = {
        "XGBOOST": XGBoostAgent,
        "LIGHTGBM": LightGBMAgent,
        "RANDOMFOREST": RandomForestAgent,
        "DECISIONTREE": DecisionTreeAgent,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs):
        """Create tree model by type (canonical: create())"""
        model_class = cls._registry.get(model_type.upper())
        if not model_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model: {model_type}. Available: {available}")
        print(f"[TreeModelFactory] Creating: {model_type}")  # simple logging
        return model_class(**kwargs)
```

### **XGBoost Agent Implementation**
```python
class XGBoostAgent(BaseAgent):
    """
    XGBoost-based Snake game agent
    
    Design Pattern: Strategy Pattern
    - Encapsulates XGBoost-specific decision logic
    - Uses 16-feature tabular data format
    - Provides interpretable decision paths
    """
    
    def __init__(self, name: str, grid_size: int):
        super().__init__(name, grid_size)
        self.model = None
        self.feature_names = [
            'head_x', 'head_y', 'apple_x', 'apple_y',
            'snake_length', 'apple_dir_up', 'apple_dir_down',
            'apple_dir_left', 'apple_dir_right', 'danger_straight',
            'danger_left', 'danger_right', 'free_space_up',
            'free_space_down', 'free_space_left', 'free_space_right'
        ]
        print(f"[XGBoostAgent] Initialized with {len(self.feature_names)} features")  # simple logging
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan move using XGBoost model"""
        if self.model is None:
            print("[XGBoostAgent] Model not trained, using random move")  # simple logging
            return random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        
        features = self._extract_features(game_state)
        prediction = self.model.predict([features])[0]
        print(f"[XGBoostAgent] Predicted move: {prediction}")  # simple logging
        return prediction
```

## 📊 **Data Format Integration**

### **CSV Schema Compatibility**
Tree models use the standardized 16-feature CSV schema from `csv-schema-1.md`:

```python
from extensions.common.csv_schema_utils import create_csv_row

# Create training data using canonical schema
csv_row = create_csv_row(
    game_state=current_state,
    target_move="RIGHT",
    game_id=1,
    step_in_game=5,
    grid_size=grid_size
)
```

### **Dataset Loading**
```python
from extensions.common.dataset_utils import load_dataset_for_training

# Load training data from heuristics-v0.04
X_train, X_val, X_test, y_train, y_val, y_test, info = load_dataset_for_training(
    dataset_paths=["path/to/heuristics_v0.04_dataset.csv"],
    grid_size=grid_size
)
```

## 🚀 **Advanced Features**

### **Feature Importance Analysis**
```python
def analyze_feature_importance(self):
    """Analyze feature importance for interpretability"""
    if hasattr(self.model, 'feature_importances_'):
        importance = self.model.feature_importances_
        for feature, imp in zip(self.feature_names, importance):
            print(f"[FeatureImportance] {feature}: {imp:.4f}")  # simple logging
```

### **Decision Path Visualization**
```python
def explain_decision(self, game_state: Dict[str, Any]):
    """Explain the decision-making process"""
    features = self._extract_features(game_state)
    print(f"[DecisionExplanation] Features: {dict(zip(self.feature_names, features))}")  # simple logging
    
    if hasattr(self.model, 'predict_proba'):
        probabilities = self.model.predict_proba([features])[0]
        moves = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        for move, prob in zip(moves, probabilities):
            print(f"[DecisionExplanation] {move}: {prob:.3f}")  # simple logging
```

## 🎓 **Educational Applications**

### **Interpretability**
- **Decision Paths**: Understand how decisions are made
- **Feature Importance**: Identify which features matter most
- **Model Transparency**: Clear, explainable decision-making
- **Debugging**: Easy to identify and fix decision problems

### **Performance Analysis**
- **Training Speed**: Fast training compared to neural networks
- **Prediction Speed**: Efficient inference for real-time play
- **Memory Efficiency**: Low memory requirements
- **Scalability**: Handle large datasets efficiently

## 🔗 **Integration with Other Extensions**

### **With Heuristics**
- Use heuristic algorithms for feature engineering
- Compare tree model decisions with algorithmic approaches
- Create hybrid heuristic-tree systems

### **With Supervised Learning**
- Use tree models as baseline for neural networks
- Compare interpretability vs. performance trade-offs
- Create ensemble methods combining different approaches

### **With Reinforcement Learning**
- Use tree models for reward function approximation
- Combine tree models with RL exploration strategies
- Create interpretable RL agents

## 🔗 **See Also**

- **`agents.md`**: Authoritative reference for agent implementation standards
- **`core.md`**: Base class architecture for all agents
- **`final-decision-10.md`**: final-decision-10.md governance system

---

**Tree models provide interpretable, efficient decision-making for Snake Game AI, offering clear decision paths and excellent performance on tabular data while maintaining educational value and technical consistency.**

