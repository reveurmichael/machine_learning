# LoRA (Low-Rank Adaptation) for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision.md`) and defines LoRA standards.

> **See also:** `fine-tuning.md`, `llm-distillation.md`, SUPREME_RULES from `final-decision.md`, `data-format-decision-guide.md`.

## 🎯 **Core Philosophy: Efficient Fine-tuning**

LoRA (Low-Rank Adaptation) enables **efficient fine-tuning** of large language models for Snake Game AI by updating only a small number of parameters. This approach follows SUPREME_RULES from `final-decision.md` and uses canonical `create()` methods throughout.

### **Educational Value**
- **Parameter Efficiency**: Understanding how to efficiently adapt large models
- **Memory Optimization**: Learning to reduce memory requirements during training
- **Rapid Adaptation**: Quickly adapting models to new tasks with canonical factory methods
- **Canonical Patterns**: All implementations use canonical `create()` method per SUPREME_RULES

## 🏗️ **LoRA Factory (CANONICAL)**

### **LoRA Factory (SUPREME_RULES Compliant)**
```python
from utils.factory_utils import SimpleFactory

class LoRAFactory:
    """
    Factory Pattern for LoRA adaptation following SUPREME_RULES from final-decision.md
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for efficient fine-tuning systems
    Educational Value: Shows how SUPREME_RULES apply to parameter-efficient training
    """
    
    _registry = {
        "STANDARD_LORA": StandardLoRA,
        "ADAPTIVE_LORA": AdaptiveLoRA,
        "HIERARCHICAL_LORA": HierarchicalLoRA,
        "QUANTIZED_LORA": QuantizedLoRA,
    }
    
    @classmethod
    def create(cls, lora_type: str, **kwargs):  # CANONICAL create() method per SUPREME_RULES
        """Create LoRA system using canonical create() method following SUPREME_RULES from final-decision.md"""
        lora_class = cls._registry.get(lora_type.upper())
        if not lora_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown LoRA type: {lora_type}. Available: {available}")
        print_info(f"[LoRAFactory] Creating LoRA: {lora_type}")  # SUPREME_RULES compliant logging
        return lora_class(**kwargs)
```

### **Standard LoRA Implementation**
```python
class StandardLoRA:
    """
    Standard LoRA implementation following SUPREME_RULES.
    
    Design Pattern: Strategy Pattern
    Purpose: Implements standard low-rank adaptation
    Educational Value: Shows LoRA implementation with canonical patterns
    """
    
    def __init__(self, base_model: str, rank: int = 16, alpha: float = 32.0):
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.lora_layers = {}
        print_info(f"[StandardLoRA] Initialized with rank {rank}, alpha {alpha}")  # SUPREME_RULES compliant logging
    
    def adapt_model(self, target_task: str) -> dict:
        """Adapt model using LoRA for target task"""
        print_info(f"[StandardLoRA] Starting adaptation for {target_task}")  # SUPREME_RULES compliant logging
        
        # Initialize LoRA layers
        self._initialize_lora_layers()
        print_info(f"[StandardLoRA] Initialized {len(self.lora_layers)} LoRA layers")  # SUPREME_RULES compliant logging
        
        # Train LoRA parameters
        training_results = self._train_lora_parameters(target_task)
        print_success(f"[StandardLoRA] LoRA training completed")  # SUPREME_RULES compliant logging
        
        # Evaluate adaptation
        evaluation_results = self._evaluate_adaptation(target_task)
        print_success(f"[StandardLoRA] Adaptation evaluation completed")  # SUPREME_RULES compliant logging
        
        return {
            'task': target_task,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'parameter_count': self._count_lora_parameters()
        }
    
    def _initialize_lora_layers(self):
        """Initialize LoRA layers for attention modules"""
        for name, module in self.base_model.named_modules():
            if 'attention' in name and hasattr(module, 'weight'):
                # Create LoRA layers for attention weights
                lora_layer = LoRALayer(
                    in_features=module.weight.shape[1],
                    out_features=module.weight.shape[0],
                    rank=self.rank,
                    alpha=self.alpha
                )
                self.lora_layers[name] = lora_layer
```

## 📊 **Simple Logging for LoRA Operations**

All LoRA operations must use simple print statements as mandated by SUPREME_RULES from `final-decision.md`:

```python
# ✅ CORRECT: Simple logging for LoRA (SUPREME_RULES compliance)
def run_lora_adaptation(lora_type: str, base_model: str, target_task: str):
            print_info(f"[LoRARunner] Starting {lora_type} adaptation for {target_task}")  # SUPREME_RULES compliant logging
    
    lora_system = LoRAFactory.create(lora_type, base_model=base_model)  # CANONICAL create() method per SUPREME_RULES
    results = lora_system.adapt_model(target_task)
    
    print_success(f"[LoRARunner] LoRA adaptation completed with {results['parameter_count']} parameters")  # SUPREME_RULES compliant logging
    return results
```

## 🎯 **LoRA Techniques**

### **Adaptive LoRA**
```python
class AdaptiveLoRA:
    """
    Adaptive LoRA following SUPREME_RULES.
    
    Design Pattern: Template Method Pattern
    Purpose: Implements adaptive rank selection for LoRA
    Educational Value: Shows adaptive LoRA with canonical patterns
    """
    
    def __init__(self, base_model: str, min_rank: int = 4, max_rank: int = 64):
        self.base_model = base_model
        self.min_rank = min_rank
        self.max_rank = max_rank
        print_info(f"[AdaptiveLoRA] Initialized with rank range {min_rank}-{max_rank}")  # SUPREME_RULES compliant logging
    
    def adapt_model(self, target_task: str) -> dict:
        """Adapt model using adaptive LoRA"""
        print_info(f"[AdaptiveLoRA] Starting adaptive adaptation for {target_task}")  # SUPREME_RULES compliant logging
        
        # Determine optimal rank
        optimal_rank = self._determine_optimal_rank(target_task)
        print_info(f"[AdaptiveLoRA] Optimal rank determined: {optimal_rank}")  # SUPREME_RULES compliant logging
        
        # Create LoRA with optimal rank
        lora_layer = StandardLoRA(self.base_model, rank=optimal_rank)
        results = lora_layer.adapt_model(target_task)
        
        return {
            'task': target_task,
            'optimal_rank': optimal_rank,
            'results': results
        }
    
    def _determine_optimal_rank(self, target_task: str) -> int:
        """Determine optimal rank for the target task"""
        # Simple heuristic: more complex tasks need higher rank
        task_complexity = self._assess_task_complexity(target_task)
        
        if task_complexity == 'simple':
            return self.min_rank
        elif task_complexity == 'medium':
            return (self.min_rank + self.max_rank) // 2
        else:
            return self.max_rank
```

### **Hierarchical LoRA**
```python
class HierarchicalLoRA:
    """
    Hierarchical LoRA following SUPREME_RULES.
    
    Design Pattern: Builder Pattern
    Purpose: Implements hierarchical LoRA adaptation
    Educational Value: Shows hierarchical LoRA with canonical patterns
    """
    
    def __init__(self, base_model: str, hierarchy_levels: int = 3):
        self.base_model = base_model
        self.hierarchy_levels = hierarchy_levels
        self.hierarchical_layers = {}
        print_info(f"[HierarchicalLoRA] Initialized with {hierarchy_levels} levels")  # SUPREME_RULES compliant logging
    
    def adapt_model(self, target_task: str) -> dict:
        """Adapt model using hierarchical LoRA"""
        print_info(f"[HierarchicalLoRA] Starting hierarchical adaptation for {target_task}")  # SUPREME_RULES compliant logging
        
        adaptation_results = []
        
        for level in range(self.hierarchy_levels):
            # Create LoRA for this hierarchy level
            level_lora = self._create_level_lora(level, target_task)
            
            # Train level-specific parameters
            level_results = self._train_level_parameters(level_lora, level, target_task)
            adaptation_results.append(level_results)
            
            print_info(f"[HierarchicalLoRA] Completed level {level + 1}")  # SUPREME_RULES compliant logging
        
        return {
            'task': target_task,
            'hierarchy_levels': self.hierarchy_levels,
            'level_results': adaptation_results
        }
    
    def _create_level_lora(self, level: int, target_task: str) -> dict:
        """Create LoRA for specific hierarchy level"""
        # Different ranks for different levels
        rank = 8 * (2 ** level)  # 8, 16, 32 for 3 levels
        
        return {
            'level': level,
            'rank': rank,
            'layers': self._get_level_layers(level)
        }
```

## 🎓 **Educational Applications with Canonical Patterns**

### **LoRA Understanding**
- **Parameter Efficiency**: Understanding how to efficiently adapt large models using canonical factory methods
- **Memory Optimization**: Learning to reduce memory requirements during training with simple logging
- **Rapid Adaptation**: Quickly adapting models to new tasks using canonical patterns
- **Rank Selection**: Understanding how to choose appropriate LoRA ranks following SUPREME_RULES

### **LoRA Benefits**
- **Efficiency**: Parameter-efficient fine-tuning that follows SUPREME_RULES
- **Memory**: Reduced memory requirements during training
- **Speed**: Faster adaptation to new tasks with canonical factory methods
- **Educational Value**: Clear examples of efficient fine-tuning following SUPREME_RULES

## 📋 **SUPREME_RULES Implementation Checklist for LoRA**

### **Mandatory Requirements**
- [ ] **Canonical Method**: All factories use `create()` method exactly (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses utils/print_utils.py functions only for all LoRA operations (SUPREME_RULES compliance)
- [ ] **Model Integration**: Proper integration with base models
- [ ] **Pattern Consistency**: Follows canonical patterns across all LoRA implementations

### **LoRA-Specific Standards**
- [ ] **Parameter Efficiency**: Significant reduction in trainable parameters with canonical factory methods
- [ ] **Memory Optimization**: Reduced memory requirements during training
- [ ] **Performance**: Maintained model performance with efficient adaptation
- [ ] **Documentation**: Clear LoRA explanations following SUPREME_RULES

---

**LoRA enables efficient fine-tuning of large language models for Snake Game AI while maintaining strict SUPREME_RULES from `final-decision.md` compliance and educational value.**

## 🔗 **See Also**

- **`fine-tuning.md`**: Fine-tuning standards
- **`llm-distillation.md`**: LLM distillation standards
- **SUPREME_RULES from `final-decision.md`**: Governance system and canonical standards
- **`data-format-decision-guide.md`**: Data format standards