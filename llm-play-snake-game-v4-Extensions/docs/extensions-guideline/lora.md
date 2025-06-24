# LoRa Fine-Tuning for Snake Game AI LLM Models

This document explains the implementation and usage of Low-Rank Adaptation (LoRa) fine-tuning techniques for optimizing Large Language Models in the Snake Game AI project.

## 🎯 **LoRa Overview in Snake Game Context**

LoRa (Low-Rank Adaptation) enables efficient fine-tuning of large language models by:
- **Parameter Efficiency**: Fine-tune only 0.1-1% of original model parameters
- **Memory Efficiency**: Reduced GPU memory requirements during training
- **Task Specialization**: Adapt pre-trained models to Snake game-specific reasoning
- **Quick Iteration**: Fast training cycles for experimentation
- **Model Preservation**: Keep original model weights intact

### **Why LoRa for Snake Game AI?**
- **Cost-effective**: Fine-tune large models without massive compute resources
- **Domain Adaptation**: Adapt general language models to spatial reasoning and game strategy
- **Multiple Variants**: Create specialized models for different grid sizes or game modes
- **Ablation Studies**: Easy comparison between different fine-tuning approaches

## 🏗️ **LoRa Architecture in Extensions**

### **Extension Structure for LLM Fine-Tuning**
```
extensions/llm-finetune-v0.03/
├── adapters/               # LoRa adapter implementations
│   ├── lora_adapter.py    # Core LoRa adaptation logic
│   ├── adapter_config.py  # Configuration management
│   └── adapter_factory.py # Factory for different adapter types
├── training/              # Training pipeline
│   ├── lora_trainer.py    # LoRa-specific training logic
│   ├── data_collator.py   # Custom data collation for game data
│   └── metrics.py         # Training and evaluation metrics
├── models/                # Model management
│   ├── model_manager.py   # Model loading and adapter application
│   └── checkpoint_manager.py # Checkpoint and versioning
└── config/               # Configuration files
    ├── lora_config.yaml  # LoRa hyperparameters
    └── training_config.yaml # Training configuration
```

### **LoRa Adapter Implementation**
```python
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

class SnakeGameLoraAdapter:
    """
    LoRa adapter specifically designed for Snake Game AI fine-tuning
    
    Design Patterns Used:
    - Adapter Pattern: Adapts pre-trained models to game-specific tasks
    - Strategy Pattern: Different LoRa configurations for different model sizes
    - Factory Pattern: Creates appropriate adapters based on model type
    - Template Method Pattern: Common fine-tuning workflow with customizable steps
    
    Features:
    - Dynamic rank adjustment based on model size
    - Game-specific target modules selection
    - Gradient checkpointing for memory efficiency
    - Mixed precision training support
    """
    
    def __init__(self, model_name: str, lora_config: dict):
        self.model_name = model_name
        self.lora_config = self._create_lora_config(lora_config)
        self.base_model = None
        self.peft_model = None
        self.tokenizer = None
    
    def _create_lora_config(self, config: dict) -> LoraConfig:
        """
        Create LoRa configuration optimized for Snake Game AI
        
        Configuration Strategy:
        - Higher rank for attention layers (spatial reasoning)
        - Lower rank for MLP layers (efficiency)
        - Task-specific alpha values for learning rate scaling
        """
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.get('rank', 16),  # Low-rank dimension
            lora_alpha=config.get('alpha', 32),  # Scaling parameter
            lora_dropout=config.get('dropout', 0.1),  # Regularization
            target_modules=self._get_target_modules(config),
            bias=config.get('bias', 'none'),
            inference_mode=False
        )
    
    def _get_target_modules(self, config: dict) -> list:
        """
        Select target modules based on model architecture and game requirements
        
        Strategy for Snake Game:
        - Focus on attention mechanisms for spatial understanding
        - Include query, key, value projections for better spatial reasoning
        - Optionally include MLP layers for decision-making logic
        """
        model_type = config.get('model_type', 'llama')
        
        if model_type in ['llama', 'llama2', 'vicuna']:
            return ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        elif model_type in ['gpt2', 'gpt-neo']:
            return ['c_attn', 'c_proj']
        elif model_type == 'bloom':
            return ['query_key_value', 'dense']
        else:
            # Default attention targets
            return ['q_proj', 'k_proj', 'v_proj']
    
    def load_base_model(self):
        """Load base model with optimizations for fine-tuning"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with memory optimizations
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # Mixed precision
            device_map="auto",          # Automatic device placement
            trust_remote_code=True,
            load_in_8bit=True          # 8-bit quantization for memory
        )
        
        # Apply LoRa adapter
        self.peft_model = get_peft_model(self.base_model, self.lora_config)
        
        return self.peft_model
    
    def print_trainable_parameters(self):
        """Print information about trainable parameters"""
        self.peft_model.print_trainable_parameters()
        
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
```

### **Snake Game-Specific Training Pipeline**
```python
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import json

class SnakeGameDataset(Dataset):
    """
    Dataset class for Snake Game AI training with LoRa
    
    Handles:
    - JSONL game trajectory data
    - Prompt template formatting
    - Input/output tokenization
    - Dynamic sequence length handling
    """
    
    def __init__(self, jsonl_file: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_jsonl_data(jsonl_file)
        self.prompt_template = self._create_prompt_template()
    
    def _load_jsonl_data(self, file_path: str) -> list:
        """Load and preprocess JSONL trajectory data"""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                trajectory = json.loads(line)
                # Convert trajectory to training examples
                examples = self._trajectory_to_examples(trajectory)
                data.extend(examples)
        return data
    
    def _trajectory_to_examples(self, trajectory: dict) -> list:
        """
        Convert game trajectory to training examples
        
        Each example consists of:
        - Game state description
        - Agent's reasoning/thought process
        - Final move decision
        """
        examples = []
        moves = trajectory['moves']
        
        for i, move_data in enumerate(moves):
            # Create context from game state
            context = self._format_game_state(move_data['game_state'])
            
            # Extract agent reasoning and decision
            reasoning = move_data.get('llm_response', '')
            decision = move_data['move']
            
            # Create prompt-response pair
            prompt = self.prompt_template.format(
                game_state=context,
                move_history=self._format_move_history(moves[:i])
            )
            
            response = f"{reasoning}\n\nMove: {decision}"
            
            examples.append({
                'prompt': prompt,
                'response': response,
                'full_text': f"{prompt}{response}"
            })
        
        return examples
    
    def _create_prompt_template(self) -> str:
        """Create prompt template optimized for Snake Game AI"""
        return """You are a Snake Game AI. Analyze the current game state and decide the best move.

Game State:
{game_state}

Previous Moves:
{move_history}

Think step by step about:
1. Current snake position and direction
2. Apple location and path to reach it
3. Potential collisions and obstacles
4. Optimal strategy for this situation

Your reasoning and move:"""
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize the full text
        encoding = self.tokenizer(
            item['full_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = encoding['input_ids'].clone()
        
        # Mask prompt tokens in labels (only learn from response)
        prompt_encoding = self.tokenizer(
            item['prompt'],
            max_length=self.max_length,
            truncation=True
        )
        prompt_length = len(prompt_encoding['input_ids'])
        labels[:, :prompt_length] = -100  # Ignore prompt in loss calculation
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

class LoRaTrainer:
    """
    Specialized trainer for LoRa fine-tuning of Snake Game AI models
    
    Features:
    - Memory-efficient training strategies
    - Custom learning rate scheduling
    - Game-specific evaluation metrics
    - Checkpoint management and model versioning
    """
    
    def __init__(self, model, tokenizer, config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.training_args = self._create_training_args()
    
    def _create_training_args(self) -> TrainingArguments:
        """Create training arguments optimized for LoRa fine-tuning"""
        return TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config.get('epochs', 3),
            per_device_train_batch_size=self.config.get('batch_size', 4),
            per_device_eval_batch_size=self.config.get('eval_batch_size', 4),
            gradient_accumulation_steps=self.config.get('gradient_accumulation', 4),
            warmup_steps=self.config.get('warmup_steps', 100),
            learning_rate=self.config.get('learning_rate', 5e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            logging_steps=self.config.get('logging_steps', 10),
            evaluation_strategy="steps",
            eval_steps=self.config.get('eval_steps', 100),
            save_steps=self.config.get('save_steps', 500),
            save_total_limit=self.config.get('save_total_limit', 3),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard for simplicity
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=self.config.get('fp16', True),  # Mixed precision training
            gradient_checkpointing=True,  # Memory optimization
        )
    
    def train(self, train_dataset, eval_dataset=None):
        """Execute LoRa fine-tuning with game-specific optimizations"""
        
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self._create_data_collator(),
            compute_metrics=self._compute_metrics if eval_dataset else None
        )
        
        # Start training
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        
        return trainer
    
    def _create_data_collator(self):
        """Create custom data collator for Snake Game data"""
        from transformers import DataCollatorForLanguageModeling
        
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal language modeling
            pad_to_multiple_of=8  # Optimization for tensor cores
        )
    
    def _compute_metrics(self, eval_pred):
        """Compute game-specific evaluation metrics"""
        import numpy as np
        
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Extract move predictions
        move_accuracy = self._calculate_move_accuracy(decoded_preds, decoded_labels)
        
        return {
            'move_accuracy': move_accuracy,
            'perplexity': np.exp(eval_pred.predictions.mean())
        }
    
    def _calculate_move_accuracy(self, predictions: list, references: list) -> float:
        """Calculate accuracy of move predictions"""
        correct = 0
        total = 0
        
        for pred, ref in zip(predictions, references):
            pred_move = self._extract_move_from_text(pred)
            ref_move = self._extract_move_from_text(ref)
            
            if pred_move and ref_move:
                total += 1
                if pred_move == ref_move:
                    correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _extract_move_from_text(self, text: str) -> str:
        """Extract move direction from generated text"""
        import re
        
        # Look for move patterns in text
        move_pattern = r'Move:\s*(UP|DOWN|LEFT|RIGHT)'
        match = re.search(move_pattern, text.upper())
        
        return match.group(1) if match else None
```

## 🔧 **LoRa Configuration and Optimization**

### **Model-Specific LoRa Configurations**
```python
class LoRaConfigManager:
    """
    Manages LoRa configurations for different model sizes and types
    
    Provides optimized configurations based on:
    - Model size (7B, 13B, 30B+)
    - Hardware constraints (GPU memory)
    - Training objectives (speed vs. quality)
    """
    
    CONFIGS = {
        'llama_7b_efficient': {
            'rank': 8,
            'alpha': 16,
            'dropout': 0.1,
            'target_modules': ['q_proj', 'v_proj'],
            'batch_size': 8,
            'gradient_accumulation': 2
        },
        'llama_7b_quality': {
            'rank': 16,
            'alpha': 32,
            'dropout': 0.05,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'batch_size': 4,
            'gradient_accumulation': 4
        },
        'llama_13b_efficient': {
            'rank': 8,
            'alpha': 16,
            'dropout': 0.1,
            'target_modules': ['q_proj', 'v_proj'],
            'batch_size': 4,
            'gradient_accumulation': 4
        },
        'large_model_memory_optimized': {
            'rank': 4,
            'alpha': 8,
            'dropout': 0.1,
            'target_modules': ['q_proj'],
            'batch_size': 2,
            'gradient_accumulation': 8,
            'load_in_8bit': True,
            'gradient_checkpointing': True
        }
    }
    
    @classmethod
    def get_config(cls, model_name: str, optimization_target: str = 'balanced') -> dict:
        """Get optimized LoRa configuration"""
        
        # Determine model size
        if '7b' in model_name.lower():
            base_config = 'llama_7b_efficient' if optimization_target == 'speed' else 'llama_7b_quality'
        elif '13b' in model_name.lower():
            base_config = 'llama_13b_efficient'
        else:
            base_config = 'large_model_memory_optimized'
        
        return cls.CONFIGS[base_config].copy()
```

### **Training Optimization Strategies**
```python
class LoRaOptimizer:
    """
    Optimization strategies for LoRa fine-tuning
    
    Implements various techniques to improve training efficiency
    and model performance for Snake Game AI applications
    """
    
    @staticmethod
    def setup_gradient_checkpointing(model):
        """Enable gradient checkpointing for memory efficiency"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    @staticmethod
    def setup_mixed_precision(trainer_args):
        """Configure mixed precision training"""
        trainer_args.fp16 = True
        trainer_args.dataloader_pin_memory = False
        trainer_args.group_by_length = True
    
    @staticmethod
    def optimize_for_memory(model, config):
        """Apply memory optimizations"""
        # Enable 8-bit quantization if configured
        if config.get('load_in_8bit', False):
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            return quantization_config
        
        return None
    
    @staticmethod
    def create_learning_rate_scheduler(optimizer, num_training_steps: int):
        """Create custom learning rate scheduler for LoRa"""
        from transformers import get_linear_schedule_with_warmup
        
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )
```

## 📊 **LoRa Evaluation and Analysis**

### **Model Performance Analysis**
```python
class LoRaEvaluator:
    """
    Comprehensive evaluation framework for LoRa-fine-tuned models
    
    Evaluates models on:
    - Game performance metrics
    - Inference speed and efficiency
    - Memory usage and resource consumption
    - Adaptation quality compared to base model
    """
    
    def __init__(self, base_model, lora_model, test_dataset):
        self.base_model = base_model
        self.lora_model = lora_model
        self.test_dataset = test_dataset
    
    def evaluate_game_performance(self) -> dict:
        """Evaluate model performance on actual Snake game tasks"""
        
        # Test on different grid sizes
        results = {}
        for grid_size in [8, 10, 12, 16]:
            game_results = self._run_game_evaluation(grid_size)
            results[f'grid_{grid_size}'] = game_results
        
        return results
    
    def evaluate_inference_efficiency(self) -> dict:
        """Measure inference speed and resource usage"""
        
        import time
        import psutil
        
        # Measure inference time
        start_time = time.time()
        self._run_inference_benchmark()
        inference_time = time.time() - start_time
        
        # Measure memory usage
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            'inference_time_per_sample': inference_time / len(self.test_dataset),
            'memory_usage_mb': memory_usage,
            'tokens_per_second': self._calculate_token_throughput()
        }
    
    def compare_with_base_model(self) -> dict:
        """Compare LoRa model performance with base model"""
        
        base_metrics = self._evaluate_model(self.base_model)
        lora_metrics = self._evaluate_model(self.lora_model)
        
        return {
            'base_model': base_metrics,
            'lora_model': lora_metrics,
            'improvement': {
                'game_score': lora_metrics['avg_score'] - base_metrics['avg_score'],
                'success_rate': lora_metrics['success_rate'] - base_metrics['success_rate'],
                'efficiency_gain': lora_metrics['inference_speed'] / base_metrics['inference_speed']
            }
        }
```

## 🚀 **Usage Examples and Best Practices**

### **Complete LoRa Fine-Tuning Pipeline**
```python
def main():
    """Complete example of LoRa fine-tuning for Snake Game AI"""
    
    # Configuration
    model_name = "microsoft/DialoGPT-medium"  # Or any compatible model
    lora_config = LoRaConfigManager.get_config(model_name, "balanced")
    
    # Initialize LoRa adapter
    adapter = SnakeGameLoraAdapter(model_name, lora_config)
    model = adapter.load_base_model()
    
    print("Model loaded with LoRa adapter:")
    adapter.print_trainable_parameters()
    
    # Prepare dataset
    train_dataset = SnakeGameDataset(
        "logs/extensions/datasets/grid-size-10/heuristics_v0.04_trajectories.jsonl",
        adapter.tokenizer
    )
    
    eval_dataset = SnakeGameDataset(
        "logs/extensions/datasets/grid-size-10/heuristics_v0.04_eval_trajectories.jsonl",
        adapter.tokenizer
    )
    
    # Setup trainer
    training_config = {
        'output_dir': './logs/extensions/models/grid-size-10/llm_lora_v1',
        'epochs': 3,
        'batch_size': 4,
        'learning_rate': 5e-5,
        'eval_steps': 100,
        'save_steps': 500
    }
    
    trainer = LoRaTrainer(model, adapter.tokenizer, training_config)
    
    # Train the model
    print("Starting LoRa fine-tuning...")
    result = trainer.train(train_dataset, eval_dataset)
    
    print("Training completed!")
    print(f"Final evaluation loss: {result.log_history[-1]['eval_loss']}")
    
    # Save adapter only (much smaller than full model)
    model.save_pretrained("./models/snake_game_lora_adapter")
    
    return model, adapter.tokenizer

if __name__ == "__main__":
    fine_tuned_model, tokenizer = main()
```

### **Best Practices for LoRa in Snake Game AI**

1. **Data Quality**: Use high-quality trajectory data from successful games
2. **Hyperparameter Tuning**: Start with conservative settings and adjust based on results
3. **Evaluation**: Always compare against base model and human performance
4. **Memory Management**: Use gradient checkpointing and mixed precision for large models
5. **Incremental Training**: Train on progressively larger datasets and more complex scenarios

---

**LoRa fine-tuning enables efficient adaptation of large language models to Snake Game AI tasks, providing a cost-effective path to specialized high-performance game agents.**