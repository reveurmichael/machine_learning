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

### **Extension Structure for LLM Fine-Tuning** # TODO: this is not the final structure. Maybe it's good, maybe not. Up to you to adopt it or not.
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
        "logs/extensions/datasets/grid-size-10/heuristics_v0.04_20250625_143022/bfs/processed_data/reasoning_data.jsonl",
        adapter.tokenizer
    )
    
    eval_dataset = SnakeGameDataset(
        "logs/extensions/datasets/grid-size-10/heuristics_v0.04_20250625_143022/bfs/processed_data/reasoning_data.jsonl",
        adapter.tokenizer
    )
    
    # Setup trainer
    training_config = {
        'output_dir': './logs/extensions/models/grid-size-10/llm_finetune_v0.02_20250625_143022/lora_adapters/',
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
    model.save_pretrained("./logs/extensions/models/grid-size-10/llm_finetune_v0.02_20250625_143022/lora_adapters/model_artifacts/")
    
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