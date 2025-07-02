# Fine-Tuning LLMs for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines fine-tuning standards for LLM extensions.

> **See also:** `heuristics-as-foundation.md`, `llm-distillation.md`, `supervised.md`, SUPREME_RULES from `final-decision-10.md`, `data-format-decision-guide.md`.

## 🎯 **Core Philosophy: LLM Fine-Tuning with Heuristic Foundation**

Fine-tuning large language models for Snake game AI leverages **high-quality heuristic datasets** to create intelligent reasoning agents. This approach combines the algorithmic precision of heuristics with the natural language understanding of LLMs, strictly following SUPREME_RULES from `final-decision-10.md`.

### **Educational Value**
- **Transfer Learning**: Understanding how to adapt pre-trained models
- **Reasoning Enhancement**: Teaching LLMs structured problem-solving
- **Data Quality**: Learning the importance of high-quality training data
- **Canonical Patterns**: All implementations use canonical `create()` method per SUPREME_RULES

## 🏗️ **Fine-Tuning Architecture (CANONICAL PATTERNS)**

### **LLM Factory (SUPREME_RULES Compliant)**
```python
from utils.factory_utils import SimpleFactory

class LLMFineTuningFactory:
    """
    Factory Pattern for LLM fine-tuning following SUPREME_RULES from final-decision-10.md
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for LLM fine-tuning systems
    Educational Value: Shows how SUPREME_RULES apply to advanced LLM operations
    """
    
    _registry = {
        "OPENAI_GPT": OpenAIFineTuner,
        "HUGGINGFACE": HuggingFaceFineTuner,
        "LOCAL_MODEL": LocalFineTuner,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs):  # CANONICAL create() method per SUPREME_RULES
        """Create LLM fine-tuner using canonical create() method following SUPREME_RULES from final-decision-10.md"""
        tuner_class = cls._registry.get(model_type.upper())
        if not tuner_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
        print(f"[LLMFineTuningFactory] Creating tuner: {model_type}")  # SUPREME_RULES compliant logging
        return tuner_class(**kwargs)
```

### **Data Preparation Pipeline**
```python
class FineTuningDataProcessor:
    """
    Processes heuristic datasets for LLM fine-tuning following SUPREME_RULES.
    
    Design Pattern: Pipeline Pattern
    Purpose: Transform heuristic data into LLM training format
    Educational Value: Shows data transformation with canonical patterns
    """
    
    def __init__(self):
        print("[FineTuningDataProcessor] Initialized")  # SUPREME_RULES compliant logging
    
    def prepare_training_data(self, heuristic_datasets: list) -> list:
        """Convert heuristic datasets to fine-tuning format"""
        training_data = []
        
        for dataset_path in heuristic_datasets:
            print(f"[FineTuningDataProcessor] Processing: {dataset_path}")  # SUPREME_RULES compliant logging
            dataset_data = self._load_heuristic_data(dataset_path)
            llm_data = self._convert_to_llm_format(dataset_data)
            training_data.extend(llm_data)
        
        print(f"[FineTuningDataProcessor] Generated {len(training_data)} training examples")  # SUPREME_RULES compliant logging
        return training_data
```

## 📊 **Simple Logging for Fine-Tuning Operations**

All fine-tuning operations must use simple print statements as mandated by SUPREME_RULES from `final-decision-10.md`:

```python
# ✅ CORRECT: Simple logging for fine-tuning (SUPREME_RULES compliance)
def fine_tune_model(model_config: dict, training_data: list):
    print(f"[FineTuner] Starting fine-tuning with {len(training_data)} examples")  # SUPREME_RULES compliant logging
    
    # Training process
    for epoch in range(model_config['epochs']):
        loss = train_epoch(training_data)
        print(f"[FineTuner] Epoch {epoch}: loss={loss:.4f}")  # SUPREME_RULES compliant logging
    
    print(f"[FineTuner] Fine-tuning completed")  # SUPREME_RULES compliant logging
    return fine_tuned_model
```

## 🎓 **Educational Applications with Canonical Patterns**

### **LLM Training Understanding**
- **Transfer Learning**: Adapting pre-trained models using canonical factory patterns
- **Data Quality**: Understanding importance of high-quality heuristic data
- **Evaluation**: Measuring LLM performance with simple logging throughout
- **Reasoning**: Teaching structured problem-solving following SUPREME_RULES

### **Heuristics to LLM Pipeline**
- **Data Source**: Use heuristic-v0.04 datasets for JSONL format (SUPREME_RULES requirement)
- **Preprocessing**: Convert structured decisions to natural language explanations
- **Training**: Fine-tune models using canonical patterns
- **Evaluation**: Test reasoning capabilities with simple logging

## 📋 **SUPREME_RULES Implementation Checklist for Fine-Tuning**

### **Mandatory Requirements**
- [ ] **Canonical Method**: All factories use `create()` method exactly (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses print() statements only for all fine-tuning operations (SUPREME_RULES compliance)
- [ ] **GOOD_RULES Reference**: References SUPREME_RULES from `final-decision-10.md` in all documentation
- [ ] **Data Source**: Uses heuristics-v0.04 JSONL datasets (SUPREME_RULES compliant format)

---

**Fine-tuning LLMs for Snake Game AI demonstrates how canonical patterns and simple logging provide consistent foundations for advanced AI training while maintaining strict SUPREME_RULES from `final-decision-10.md` compliance.**

## 🔗 **See Also**

- **`heuristics-as-foundation.md`**: Foundation datasets for fine-tuning
- **`llm-distillation.md`**: Alternative LLM training approaches
- **`supervised.md`**: Traditional ML approaches following canonical patterns
- **SUPREME_RULES from `final-decision-10.md`**: Governance system and canonical standards
- **`data-format-decision-guide.md`**: Authoritative data format selection
