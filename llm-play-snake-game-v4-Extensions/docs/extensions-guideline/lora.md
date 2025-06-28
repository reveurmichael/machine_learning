# LoRA Fine-Tuning Architecture for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ and extension guidelines and follows **KEEP_THOSE_MARKDOWN_FILES_SIMPLE_RULES** guidelines with a target length of 300-500 lines.

## 🎯 **Core Philosophy: Efficient LLM Adaptation**

Low-Rank Adaptation (LoRA) enables efficient fine-tuning of large language models by training only a small fraction of parameters while maintaining the power of the full model. This follows the established patterns from the Final Decision series for model management and training pipelines.

### **SUPREME_RULES Alignment**
- **SUPREME_RULE NO.1**: Follows all established GOOD_RULES patterns for LLM fine-tuning implementations  
- **SUPREME_RULE NO.2**: Uses precise `final-decision-N.md` format consistently throughout LoRA references
- **SUPREME_RULE NO.3**: Uses lightweight, OOP-based common utilities with simple logging (print() statements) rather than complex *.log file mechanisms

### **Design Philosophy**
- **Parameter Efficiency**: Fine-tune 0.1-1% of original model parameters
- **Memory Optimization**: Reduced GPU requirements during training
- **Task Specialization**: Adapt pre-trained models to Snake game reasoning
- **Modular Architecture**: Pluggable adapters following factory patterns

## 🏗️ **LoRA Integration Architecture**

### **Extension Structure**
Following Final Decision 5 directory patterns:

```
extensions/llm-finetune-v0.03/
├── adapters/                      # LoRA adapter implementations
│   ├── __init__.py               # Adapter factory
│   ├── lora_adapter.py           # Core LoRA implementation
│   ├── adapter_config.py         # Configuration management
│   └── adapter_manager.py        # Adapter lifecycle management
├── training/                      # Training pipeline
│   ├── lora_trainer.py           # LoRA-specific training logic
│   ├── data_collator.py          # Game data collation
│   └── metrics.py                # Training metrics
├── models/                        # Model management
│   ├── model_factory.py          # Model creation with LoRA
│   └── checkpoint_manager.py     # Training checkpoints
└── config/                        # Configuration
    └── lora_presets.py           # Common LoRA configurations

# Note: Following SUPREME_RULE NO.3, training parameters are defined locally:
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32  
DEFAULT_EPOCHS = 100
```

### **Factory Pattern Integration**
Following Final Decision 7-8 factory patterns:

```python
class LoRAAdapterFactory:
    """Factory for creating LoRA adapters"""
    
    _adapter_registry = {
        "lora": StandardLoRAAdapter,
        "qlora": QLoRAAdapter,
        "adalora": AdaLoRAAdapter,
    }
    
    @classmethod
    def create_adapter(cls, adapter_type: str, base_model: str, config: LoRAConfig):
        """Create LoRA adapter by type"""
        adapter_class = cls._adapter_registry.get(adapter_type)
        if not adapter_class:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        return adapter_class(base_model=base_model, config=config)
        
    @classmethod
    def load_adapter(cls, model_path: Path, adapter_path: Path):
        """Load pre-trained LoRA adapter"""
        from peft import PeftModel
        return PeftModel.from_pretrained(model_path, adapter_path)
```

## 🔧 **LoRA Implementation Patterns**

### **Configuration Management**
Following Final Decision 2 configuration standards:

```python
# SUPREME_RULE NO.3: Define constants locally instead of importing from common
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05

class LoRAConfig:
    """LoRA adapter configuration following GOOD_RULES patterns"""
    
    def __init__(self, 
                 rank: int = DEFAULT_LORA_RANK,
                 alpha: int = DEFAULT_LORA_ALPHA,
                 dropout: float = DEFAULT_LORA_DROPOUT,
                 target_modules: List[str] = None):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        
    @classmethod
    def get_preset(cls, preset_name: str) -> "LoRAConfig":
        """Get predefined LoRA configuration"""
        presets = {
            "lightweight": cls(rank=8, alpha=16, dropout=0.1),
            "balanced": cls(rank=16, alpha=32, dropout=0.05),
            "high_capacity": cls(rank=32, alpha=64, dropout=0.02)
        }
        return presets.get(preset_name, cls())
```

### **Training Pipeline Integration**
```python
class LoRATrainer:
    """LoRA training pipeline following template method pattern"""
    
    def __init__(self, model, tokenizer, config: LoRAConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.setup_trainer()
        
    def train(self, train_dataset, eval_dataset) -> TrainingResult:
        """Template method for LoRA training workflow"""
        # Step 1: Prepare model with LoRA adapter
        self.prepare_model_for_training()
        
        # Step 2: Setup data collation
        data_collator = self.create_data_collator()
        
        # Step 3: Configure training arguments
        training_args = self.get_training_arguments()
        
        # Step 4: Create trainer
        trainer = self.create_trainer(data_collator, training_args)
        
        # Step 5: Execute training
        result = trainer.train()
        
        # Step 6: Save adapter
        self.save_adapter()
        
        return result
        
    def prepare_model_for_training(self):
        """Apply LoRA adapter to base model"""
        from peft import get_peft_model, LoraConfig, TaskType
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.rank,
            lora_alpha=self.config.alpha,
            lora_dropout=self.config.dropout,
            target_modules=self.config.target_modules
        )
        
        self.model = get_peft_model(self.model, peft_config)
```

## 📊 **Dataset Integration**

### **JSONL Dataset Processing**
Following Final Decision 1 dataset structure:

```python
class SnakeGameLoRADataset:
    """Dataset class for LoRA training on Snake game data"""
    
    def __init__(self, dataset_path: Path, tokenizer, max_length: int = 512):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_jsonl_data()
        
    def load_jsonl_data(self) -> List[Dict[str, str]]:
        """Load JSONL dataset from heuristics v0.04"""
        data = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                # Validate JSONL format
                if 'prompt' in entry and 'completion' in entry:
                    data.append(entry)
        return data
        
    def __getitem__(self, idx):
        """Get tokenized training example"""
        item = self.data[idx]
        
        # Format for instruction tuning
        input_text = f"### Instruction:\n{item['prompt']}\n\n### Response:\n{item['completion']}"
        
        # Tokenize
        tokenized = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': tokenized['input_ids'].squeeze()
        }
```

## 🚀 **Model Storage and Loading**

### **Path Management Integration**
Following Final Decision 6:

```python
from extensions.common.path_utils import get_model_path

class LoRAModelManager:
    """Manages LoRA model storage and loading"""
    
    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        
    def save_lora_adapter(self, model, adapter_name: str, timestamp: str):
        """Save LoRA adapter following standardized paths"""
        model_path = get_model_path(
            extension_type="llm_finetune",
            version="0.03",
            grid_size=self.grid_size,
            algorithm=adapter_name,
            timestamp=timestamp
        )
        
        adapter_path = model_path / "lora_adapters" / "model_artifacts"
        adapter_path.mkdir(parents=True, exist_ok=True)
        
        # Save adapter weights only (much smaller than full model)
        model.save_pretrained(adapter_path)
        
        # Save metadata
        metadata = {
            "adapter_type": "lora",
            "base_model": model.base_model.config.name_or_path,
            "grid_size": self.grid_size,
            "training_timestamp": timestamp,
            "peft_config": model.peft_config
        }
        
        with open(adapter_path / "adapter_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def load_lora_adapter(self, adapter_path: Path, base_model_name: str):
        """Load LoRA adapter for inference"""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        # Apply LoRA adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        return model
```

## 🎯 **Integration with Extensions**

### **LLM Fine-tuning Extension Integration**
```python
class LLMFineTuneManager:
    """Manages LLM fine-tuning with LoRA support"""
    
    def __init__(self, base_model: str = "microsoft/DialoGPT-medium"):
        self.base_model = base_model
        self.adapter_factory = LoRAAdapterFactory()
        self.model_manager = LoRAModelManager()
        
    def train_on_heuristic_data(self, 
                               dataset_path: Path, 
                               adapter_config: LoRAConfig,
                               training_config: Dict[str, Any]) -> TrainingResult:
        """Train LoRA adapter on heuristic reasoning data"""
        
        # Create adapter
        adapter = self.adapter_factory.create_adapter(
            adapter_type="lora",
            base_model=self.base_model,
            config=adapter_config
        )
        
        # Load dataset
        dataset = SnakeGameLoRADataset(dataset_path, adapter.tokenizer)
        
        # Train adapter
        trainer = LoRATrainer(adapter.model, adapter.tokenizer, adapter_config)
        result = trainer.train(dataset, dataset)  # Use same dataset for eval in demo
        
        # Save adapter
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_manager.save_lora_adapter(
            adapter.model, 
            adapter_name="snake_reasoning",
            timestamp=timestamp
        )
        
        return result
```

## 🔗 **GOOD_RULES Integration**

This document integrates with the following authoritative references from the **GOOD_RULES** system:

### **Core Architecture Integration**
- **`agents.md`**: Follows BaseAgent interface and factory patterns for all LoRA fine-tuned implementations
- **`config.md`**: Uses authorized LLM configuration hierarchies for LoRA training parameters
- **`core.md`**: Inherits from base classes and follows established inheritance patterns

### **Extension Development Standards**
- **`extensions-v0.02.md`** through **`extensions-v0.04.md`**: Follows version progression guidelines  
- **`standalone.md`**: Maintains standalone principle (extension + common = self-contained)
- **`single-source-of-truth.md`**: Avoids duplication, uses centralized utilities
- **`fine-tuning.md`**: Follows fine-tuning extension standards and patterns

### **Data and Path Management**
- **`data-format-decision-guide.md`**: Uses JSONL format specifically designed for LLM fine-tuning
- **`datasets-folder.md`**: Uses datasets from heuristics-v0.04 for language-rich training data
- **`unified-path-management-guide.md`**: Uses centralized path utilities from extensions/common/
- **`models.md`**: Follows model versioning and adapter storage standards

### **UI and Interaction Standards**
- **`app.md`** and **`dashboard.md`**: Integrates with Streamlit architecture for LoRA training interfaces
- **`unified-streamlit-architecture-guide.md`**: Follows OOP Streamlit patterns for fine-tuning monitoring

### **Implementation Quality**
- **`documentation-as-first-class-citizen.md`**: Maintains rich docstrings and design pattern documentation
- **`elegance.md`**: Follows code quality and educational value standards
- **`naming_conventions.md`**: Uses consistent naming across all LoRA implementations

## 📝 **Simple Logging Examples (SUPREME_RULE NO.3)**

All code examples in this document follow **SUPREME_RULE NO.3** by using simple print() statements rather than complex logging mechanisms:

```python
# ✅ CORRECT: Simple logging as per SUPREME_RULE NO.3
def train_lora_adapter(self, train_dataset, eval_dataset):
    print(f"[LoRATrainer] Starting LoRA fine-tuning...")
    print(f"[LoRATrainer] Training samples: {len(train_dataset)}")
    print(f"[LoRATrainer] LoRA rank: {self.config.rank}, alpha: {self.config.alpha}")
    
    # Apply LoRA to model
    self.prepare_model_for_training()
    print(f"[LoRATrainer] LoRA adapter applied to model")
    
    # Train with progress tracking
    result = self.trainer.train()
    print(f"[LoRATrainer] Training completed - Final loss: {result.training_loss:.4f}")
    
    # Save adapter
    self.save_adapter()
    print(f"[LoRATrainer] LoRA adapter saved successfully")

# ✅ CORRECT: Educational progress tracking  
def load_jsonl_dataset(self, dataset_path):
    print(f"[DataLoader] Loading JSONL dataset from {dataset_path}")
    
    data = []
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print(f"[DataLoader] Processed {i} samples...")
            data.append(json.loads(line.strip()))
    
    print(f"[DataLoader] Loaded {len(data)} training examples")
    return data
```

---

**LoRA fine-tuning provides parameter-efficient adaptation of large language models for Snake Game AI, enabling specialized reasoning capabilities while maintaining computational efficiency and full compliance with established GOOD_RULES standards.**