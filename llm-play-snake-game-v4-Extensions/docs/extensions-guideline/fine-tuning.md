# LLM Fine-Tuning for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and follows established architectural patterns.

## 🎯 **Core Philosophy: Specialized LLM Training**

LLM fine-tuning enables the creation of specialized language models that excel at specific tasks like Snake game playing. By training on game-specific data with reasoning patterns, these models develop deep understanding of game mechanics and strategic thinking.

### **Design Philosophy**
- **Task Specialization**: Models trained specifically for Snake game excellence
- **Reasoning Enhancement**: Training on explanation-rich datasets
- **Efficient Training**: Parameter-efficient fine-tuning methods
- **Educational Value**: Demonstrating domain adaptation techniques

## 🧠 **Fine-Tuning Approaches**

### **Full Fine-Tuning**
- Complete model parameter updates
- Maximum customization for Snake game domain
- Requires significant computational resources
- Best performance but highest cost

### **Parameter-Efficient Fine-Tuning (PEFT)**
- **LoRA (Low-Rank Adaptation)**: Efficient parameter updates through low-rank matrices
- **Adapters**: Small neural networks inserted into existing layers
- **Prefix Tuning**: Learn continuous prompts for task guidance
- **P-Tuning v2**: Learnable prompt tokens across all layers

### **Instruction Tuning**
- Training on instruction-following data for Snake game
- Enhanced ability to follow game rules and strategies
- Better generalization to unseen game scenarios
- Improved reasoning and explanation capabilities

## 🏗️ **Extension Structure**

### **Directory Layout**
```
extensions/llm-finetune-v0.02/
├── __init__.py
├── agents/
│   ├── __init__.py               # Agent factory
│   ├── agent_full_ft.py          # Full fine-tuning agent
│   ├── agent_lora.py             # LoRA fine-tuned agent
│   ├── agent_adapter.py          # Adapter-based agent
│   └── agent_prefix.py           # Prefix-tuned agent
├── training/
│   ├── __init__.py
│   ├── full_finetune.py          # Full fine-tuning pipeline
│   ├── lora_trainer.py           # LoRA training implementation
│   ├── adapter_trainer.py        # Adapter training pipeline
│   └── instruction_tuner.py      # Instruction tuning methods
├── data/
│   ├── __init__.py
│   ├── dataset_processor.py      # Process JSONL datasets from heuristics-v0.04
│   ├── prompt_formatter.py       # Format training prompts
│   └── validation_sets.py        # Validation data preparation
├── evaluation/
│   ├── __init__.py
│   ├── performance_evaluator.py  # Game performance metrics
│   ├── reasoning_evaluator.py    # Reasoning quality assessment
│   └── comparison_utils.py       # Compare with base models
├── models/
│   ├── __init__.py
│   ├── model_loader.py           # Load and save fine-tuned models
│   ├── lora_config.py            # LoRA configuration utilities
│   └── inference_utils.py        # Optimized inference helpers
├── game_logic.py                 # Fine-tuned LLM game logic
├── game_manager.py               # Multi-model manager
└── main.py                       # Training and evaluation CLI
```

## 🔧 **Training Pipeline Implementation**

### **LoRA Fine-Tuning**
```python
class LoRATrainer:
    """
    LoRA (Low-Rank Adaptation) Fine-Tuning for Snake Game LLMs
    
    Design Pattern: Strategy Pattern
    - Implements parameter-efficient fine-tuning strategy
    - Maintains base model while adding task-specific adaptations
    - Enables efficient training and deployment
    
    Educational Value:
    Demonstrates how to specialize large models for specific domains
    without full parameter updates, making fine-tuning accessible.
    """
    
    def __init__(self, base_model: str, lora_config: Dict[str, Any]):
        self.base_model_name = base_model
        self.lora_config = lora_config
        self.model = None
        self.tokenizer = None
    
    def setup_model(self):
        """Initialize model with LoRA configuration"""
        from peft import get_peft_model, LoraConfig, TaskType
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config.get("rank", 16),
            lora_alpha=self.lora_config.get("alpha", 32),
            lora_dropout=self.lora_config.get("dropout", 0.1),
            target_modules=self.lora_config.get("target_modules", ["q_proj", "v_proj"])
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
    
    def train(self, dataset_path: str, output_dir: str):
        """Train model using LoRA fine-tuning"""
        # Load and process dataset from heuristics-v0.04
        train_dataset = self._load_jsonl_dataset(dataset_path)
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=2,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=self._get_data_collator(),
        )
        
        # Train model
        trainer.train()
        trainer.save_model()
```

### **Dataset Processing**
```python
class SnakeGameDatasetProcessor:
    """
    Process JSONL datasets from heuristics-v0.04 for LLM fine-tuning
    
    Design Pattern: Builder Pattern
    - Constructs training datasets step-by-step
    - Handles different prompt formats and augmentations
    - Ensures data quality and consistency
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 2048
    
    def process_jsonl_dataset(self, jsonl_path: str) -> Dataset:
        """Process JSONL dataset from heuristics-v0.04"""
        with open(jsonl_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        # Format training examples
        formatted_examples = []
        for item in data:
            # Create instruction-following format
            prompt = self._format_instruction_prompt(item)
            completion = item['completion']
            
            # Tokenize
            full_text = f"{prompt}{completion}<|endoftext|>"
            tokens = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            
            formatted_examples.append({
                'input_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask'],
                'labels': tokens['input_ids'].copy()  # For causal LM
            })
        
        return Dataset.from_list(formatted_examples)
    
    def _format_instruction_prompt(self, data_item: Dict) -> str:
        """Format game state as instruction-following prompt"""
        return f"""Below is an instruction that describes a Snake game situation, paired with input that provides game state context. Write a response that appropriately completes the request.

### Instruction:
Analyze the Snake game state and determine the best move. Provide your reasoning.

### Input:
{data_item['prompt']}

### Response:
"""
```

## 🚀 **Advanced Fine-Tuning Techniques**

### **Multi-Objective Training**
- **Performance Optimization**: Train for high game scores
- **Reasoning Quality**: Maintain explanation capabilities
- **Safety Constraints**: Avoid invalid moves and crashes
- **Efficiency Targets**: Balance performance with inference speed

### **Curriculum Learning**
- **Progressive Difficulty**: Start with simple scenarios, increase complexity
- **Strategy Scaffolding**: Build from basic moves to advanced strategies
- **Error Recovery**: Emphasize learning from failure scenarios
- **Multi-Grid Training**: Train across different board sizes

### **Reinforcement Learning from Human Feedback (RLHF)**
- **Preference Learning**: Learn from human strategy preferences
- **Reward Modeling**: Train reward models for game quality
- **Policy Optimization**: Use PPO to optimize against learned rewards
- **Constitutional Training**: Embed game rules as constitutional principles

## 🎓 **Evaluation and Analysis**

### **Performance Metrics**
```python
class FineTuningEvaluator:
    """
    Comprehensive evaluation of fine-tuned Snake game LLMs
    
    Metrics:
    - Game Performance: Score, survival time, efficiency
    - Reasoning Quality: Coherence, accuracy, helpfulness
    - Model Efficiency: Inference speed, memory usage
    - Generalization: Performance on unseen scenarios
    """
    
    def evaluate_model(self, model_path: str, test_games: int = 100):
        """Comprehensive model evaluation"""
        results = {
            'game_performance': self._evaluate_game_performance(model_path, test_games),
            'reasoning_quality': self._evaluate_reasoning_quality(model_path),
            'efficiency_metrics': self._evaluate_efficiency(model_path),
            'generalization': self._evaluate_generalization(model_path)
        }
        return results
    
    def _evaluate_game_performance(self, model_path: str, num_games: int):
        """Evaluate actual game playing performance"""
        agent = self._load_finetuned_agent(model_path)
        
        scores = []
        survival_times = []
        
        for _ in range(num_games):
            game_result = self._run_single_game(agent)
            scores.append(game_result['score'])
            survival_times.append(game_result['steps'])
        
        return {
            'average_score': np.mean(scores),
            'score_std': np.std(scores),
            'average_survival': np.mean(survival_times),
            'max_score': max(scores),
            'success_rate': len([s for s in scores if s > 10]) / len(scores)
        }
```

### **Comparative Analysis**
- **vs. Base Models**: Compare performance against untuned models
- **vs. Heuristics**: Benchmark against algorithmic approaches
- **vs. Supervised Models**: Compare with traditional ML approaches
- **PEFT Method Comparison**: Compare LoRA, adapters, prefix tuning

## 📊 **Training Configuration Examples**

### **LoRA Configuration**
```python
LORA_CONFIGS = {
    'lightweight': {
        'rank': 8,
        'alpha': 16,
        'dropout': 0.1,
        'target_modules': ['q_proj', 'v_proj']
    },
    'standard': {
        'rank': 16,
        'alpha': 32,
        'dropout': 0.1,
        'target_modules': ['q_proj', 'v_proj', 'k_proj', 'out_proj']
    },
    'comprehensive': {
        'rank': 32,
        'alpha': 64,
        'dropout': 0.05,
        'target_modules': ['q_proj', 'v_proj', 'k_proj', 'out_proj', 'gate_proj', 'up_proj', 'down_proj']
    }
}
```

### **Training Commands**
```bash
# LoRA fine-tuning
python main.py --mode train --method lora \
  --base-model microsoft/DialoGPT-medium \
  --dataset ../../logs/extensions/datasets/grid-size-10/heuristics_v0.04_*/*/reasoning_data.jsonl \
  --output-dir ./models/snake-lora-v1

# Full fine-tuning (requires more resources)
python main.py --mode train --method full \
  --base-model microsoft/DialoGPT-small \
  --dataset ../../logs/extensions/datasets/grid-size-10/heuristics_v0.04_*/*/reasoning_data.jsonl \
  --output-dir ./models/snake-full-v1

# Evaluation
python main.py --mode evaluate \
  --model-path ./models/snake-lora-v1 \
  --test-games 100 \
  --grid-size 10
```

## 🔗 **Integration with Extension Ecosystem**

### **Data Sources**
- **Primary**: JSONL datasets from `heuristics-v0.04` (language-rich explanations)
- **Augmentation**: Synthetic data generation for edge cases
- **Validation**: Human-annotated preference datasets
- **Testing**: Standardized benchmarks across grid sizes

### **Model Deployment**
- **Inference Optimization**: Quantization, pruning for efficient deployment
- **API Integration**: Compatible with existing LLM provider interfaces
- **Batch Processing**: Efficient multi-game evaluation capabilities
- **Real-time Deployment**: Low-latency inference for interactive play

### **Continuous Improvement**
- **Feedback Loops**: Learn from deployment performance
- **Data Flywheel**: Use model performance to generate better training data
- **A/B Testing**: Compare model versions in live environments
- **Transfer Learning**: Apply learned representations to new game variants

## 🔮 **Future Research Directions**

### **Advanced Architectures**
- **Mixture of Experts**: Specialized expert networks for different game phases
- **Retrieval-Augmented Generation**: Access to strategy databases during inference
- **Multi-Modal Models**: Integration of visual game understanding
- **Hierarchical Planning**: Models that plan at multiple time scales

### **Training Innovations**
- **Meta-Learning**: Quick adaptation to new game variants or rules
- **Few-Shot Learning**: Rapid learning from minimal examples
- **Continual Learning**: Learning new strategies without forgetting old ones
- **Self-Supervised Learning**: Learning game understanding from gameplay alone

---

**LLM fine-tuning for Snake Game AI demonstrates how large language models can be specialized for specific domains, combining the reasoning capabilities of foundation models with task-specific expertise. This approach bridges the gap between general AI capabilities and specialized game-playing performance.**
