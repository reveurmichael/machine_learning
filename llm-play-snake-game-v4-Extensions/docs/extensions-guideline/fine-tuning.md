# Fine-tuning for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines fine-tuning patterns for extensions.

> **See also:** `agents.md`, `core.md`, `final-decision-10.md`, `factory-design-pattern.md`, `config.md`, `lora.md`.

# Fine-tuning for Snake Game AI

## 🎯 **Core Philosophy: Specialized Model Adaptation**

Fine-tuning enables large language models to specialize in Snake game reasoning by adapting pre-trained knowledge to game-specific patterns. This approach leverages the rich language capabilities of LLMs while focusing on game-playing expertise.

## 🏗️ **Extension Structure**

### **Directory Layout**
```
extensions/fine-tuning-v0.02/
├── __init__.py
├── agents/
│   ├── __init__.py               # Agent factory
│   ├── agent_finetuned.py        # Fine-tuned agent
│   └── agent_instruction.py      # Instruction-tuned agent
├── training/
│   ├── __init__.py
│   ├── trainer.py                # Training pipeline
│   ├── data_processor.py         # Data preprocessing
│   └── metrics.py                # Training metrics
├── models/
│   ├── __init__.py
│   ├── model_manager.py          # Model management
│   └── checkpoint_handler.py     # Checkpoint utilities
├── game_logic.py                 # Fine-tuning game logic
├── game_manager.py               # Fine-tuning manager
└── main.py                       # CLI interface
```

## 🔧 **Implementation Patterns**

### **Fine-tuning Agent Factory**
```python
class FineTuningAgentFactory:
    """
    Simple factory for fine-tuned agents
    
    Design Pattern: Factory Pattern
    - Simple dictionary-based registry
    - Canonical create() method
    - Easy extension for new fine-tuning approaches
    """
    
    _registry = {
        "FINETUNED": FineTunedAgent,
        "INSTRUCTION": InstructionTunedAgent,
        "LORA": LoRATunedAgent,
    }
    
    @classmethod
    def create(cls, agent_type: str, **kwargs):
        """Create fine-tuned agent by type (canonical: create())"""
        agent_class = cls._registry.get(agent_type.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown agent: {agent_type}. Available: {available}")
        print(f"[FineTuningAgentFactory] Creating: {agent_type}")  # Simple logging
        return agent_class(**kwargs)
```

### **Fine-tuned Agent Implementation**
```python
class FineTunedAgent(BaseAgent):
    """
    Fine-tuned LLM agent for Snake game
    
    Design Pattern: Adapter Pattern
    - Wraps fine-tuned model with game interface
    - Maintains consistent agent interface
    - Provides specialized game reasoning
    """
    
    def __init__(self, name: str, grid_size: int):
        super().__init__(name, grid_size)
        self.model = None
        self.tokenizer = None
        print(f"[FineTunedAgent] Initialized fine-tuned agent: {name}")
    
    def load_model(self, model_path: str):
        """Load fine-tuned model"""
        self.model = self._load_finetuned_model(model_path)
        self.tokenizer = self._load_tokenizer(model_path)
        print(f"[FineTunedAgent] Loaded model from: {model_path}")
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan move using fine-tuned model"""
        if self.model is None:
            print("[FineTunedAgent] Model not loaded, using random move")
            return random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        
        prompt = self._format_game_state(game_state)
        response = self._generate_response(prompt)
        move = self._parse_response(response)
        print(f"[FineTunedAgent] Generated move: {move}")
        return move
```

## 📊 **Training Integration**

### **Fine-tuning Pipeline**
```python
class FineTuningTrainer:
    """
    Fine-tuning pipeline for Snake game AI
    
    Design Pattern: Template Method Pattern
    - Standardized training workflow
    - Customizable training parameters
    - Efficient model adaptation
    """
    
    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.trainer = None
        print(f"[FineTuningTrainer] Initialized for {base_model_name}")
    
    def train(self, training_data, training_config: Dict[str, Any]):
        """Train fine-tuned model on Snake game data"""
        print(f"[FineTuningTrainer] Starting fine-tuning with {len(training_data)} samples")
        
        # Setup training configuration
        training_config.update({
            'learning_rate': training_config.get('learning_rate', 5e-5),
            'batch_size': training_config.get('batch_size', 4),
            'epochs': training_config.get('epochs', 3)
        })
        
        # Train model
        self.trainer = self._setup_trainer(training_config)
        self.trainer.train(training_data)
        
        print(f"[FineTuningTrainer] Fine-tuning completed successfully")
```

## 🚀 **Advanced Features**

### **Instruction Tuning**
```python
class InstructionTunedAgent(FineTunedAgent):
    """
    Instruction-tuned agent for better reasoning
    
    Design Pattern: Decorator Pattern
    - Adds instruction-following capabilities
    - Improves reasoning and explanation
    - Maintains base fine-tuning functionality
    """
    
    def __init__(self, name: str, grid_size: int):
        super().__init__(name, grid_size)
        self.instruction_template = self._load_instruction_template()
        print(f"[InstructionTunedAgent] Initialized instruction-tuned agent")
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan move with instruction following"""
        instruction = self._format_instruction(game_state)
        response = self._generate_instruction_response(instruction)
        move = self._parse_instruction_response(response)
        print(f"[InstructionTunedAgent] Instruction-based move: {move}")
        return move
```

### **LoRA Fine-tuning Integration**
```python
class LoRATunedAgent(FineTunedAgent):
    """
    LoRA fine-tuned agent for efficiency
    
    Design Pattern: Strategy Pattern
    - Uses LoRA for efficient fine-tuning
    - Maintains performance with fewer parameters
    - Enables rapid adaptation
    """
    
    def __init__(self, name: str, grid_size: int):
        super().__init__(name, grid_size)
        self.lora_config = self._load_lora_config()
        print(f"[LoRATunedAgent] Initialized LoRA-tuned agent")
    
    def setup_lora(self, r: int = 16, alpha: int = 32):
        """Setup LoRA adaptation layers"""
        self.lora_adapter = self._create_lora_adapter(r, alpha)
        print(f"[LoRATunedAgent] LoRA setup complete: r={r}, alpha={alpha}")
```

## 📈 **Data Pipeline Integration**

### **Training Data Preparation**
```python
class FineTuningDataProcessor:
    """
    Process training data for fine-tuning
    
    Design Pattern: Pipeline Pattern
    - Sequential data processing steps
    - Configurable preprocessing options
    - Quality validation and filtering
    """
    
    def __init__(self):
        self.processors = []
        print("[FineTuningDataProcessor] Initialized")
    
    def add_processor(self, processor):
        """Add data processing step"""
        self.processors.append(processor)
        print(f"[FineTuningDataProcessor] Added processor: {processor.__class__.__name__}")
    
    def process(self, raw_data):
        """Process data through pipeline"""
        processed_data = raw_data
        for processor in self.processors:
            processed_data = processor.process(processed_data)
        print(f"[FineTuningDataProcessor] Processed {len(processed_data)} samples")
        return processed_data
```

## 🎯 **Integration with Other Extensions**

### **Heuristics Integration**
- **Training Data**: Use heuristics-v0.04 JSONL datasets for fine-tuning
- **Baseline Comparison**: Compare fine-tuned performance with heuristic algorithms
- **Hybrid Approaches**: Combine fine-tuned reasoning with heuristic pathfinding

### **Supervised Learning Integration**
- **Feature Engineering**: Use supervised learning insights for prompt engineering
- **Performance Validation**: Validate fine-tuned models against supervised baselines
- **Ensemble Methods**: Combine fine-tuned and supervised approaches

### **Reinforcement Learning Integration**
- **Reward Shaping**: Use RL insights to improve fine-tuning objectives
- **Policy Distillation**: Distill RL policies into fine-tuned models
- **Multi-Modal Learning**: Combine RL exploration with fine-tuned reasoning

## 📊 **Performance Monitoring**

### **Training Metrics**
- **Loss Tracking**: Monitor training and validation loss
- **Game Performance**: Track game-specific metrics (score, survival time)
- **Reasoning Quality**: Evaluate explanation quality and consistency
- **Computational Efficiency**: Monitor training time and resource usage

### **Evaluation Framework**
```python
class FineTuningEvaluator:
    """
    Comprehensive evaluation of fine-tuned models
    
    Design Pattern: Strategy Pattern
    - Multiple evaluation strategies
    - Configurable evaluation metrics
    - Comparative analysis capabilities
    """
    
    def __init__(self):
        self.metrics = {}
        print("[FineTuningEvaluator] Initialized")
    
    def evaluate_model(self, model, test_data):
        """Evaluate fine-tuned model performance"""
        results = {}
        for metric_name, metric_func in self.metrics.items():
            results[metric_name] = metric_func(model, test_data)
        print(f"[FineTuningEvaluator] Evaluation completed: {results}")
        return results
```

---

**Fine-tuning represents a powerful approach to adapting large language models for specialized Snake game reasoning, combining the flexibility of LLMs with the precision of game-specific training. This extension enables sophisticated reasoning capabilities while maintaining the educational value of the project.**
