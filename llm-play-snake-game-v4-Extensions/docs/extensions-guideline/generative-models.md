# Generative Models for Snake Game AI Extensions

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines generative model patterns for extensions.

> **See also:** `agents.md`, `core.md`, `config.md`, `final-decision-10.md`, `factory-design-pattern.md`.

## 🎯 **Core Philosophy: Generative AI Integration**

Generative models in the Snake Game AI project enable **creative and adaptive decision-making** through advanced AI techniques like language models, diffusion models, and generative adversarial networks. These models can reason, plan, and generate novel strategies, strictly following `final-decision-10.md` SUPREME_RULES.

### **Educational Value**
- **Generative AI**: Understanding modern generative model capabilities
- **Creative Problem Solving**: Learning how AI can generate novel solutions
- **Adaptive Systems**: Experience AI that adapts to new situations
- **Advanced Reasoning**: See how generative models reason and plan

## 🏗️ **Factory Pattern: Canonical Method is create()**

All generative model factories must use the canonical method name `create()` for instantiation, not `create_generative_agent()` or any other variant. This ensures consistency and aligns with the KISS principle.

### **Generative Model Factory Implementation**
```python
class GenerativeModelFactory:
    """
    Factory for generative model agents following SUPREME_RULES.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create generative model agents with canonical patterns
    Educational Value: Shows how canonical factory patterns work with generative AI
    """
    
    _registry = {
        "LLM": LLMAgent,
        "DIFFUSION": DiffusionAgent,
        "GAN": GANAgent,
        "TRANSFORMER": TransformerAgent,
        "VISION_LANGUAGE": VisionLanguageAgent,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs):  # CANONICAL create() method
        """Create generative model agent using canonical create() method (SUPREME_RULES compliance)"""
        agent_class = cls._registry.get(model_type.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
        print(f"[GenerativeModelFactory] Creating agent: {model_type}")  # Simple logging
        return agent_class(**kwargs)

# ❌ FORBIDDEN: Non-canonical method names (violates SUPREME_RULES)
class GenerativeModelFactory:
    def create_generative_agent(self, model_type: str):  # FORBIDDEN - not canonical
        pass
    
    def build_generative_model(self, model_type: str):  # FORBIDDEN - not canonical
        pass
    
    def make_llm_agent(self, model_type: str):  # FORBIDDEN - not canonical
        pass
```

## 🧠 **Generative Model Architecture Patterns**

### **Language Model Agent**
```python
class LLMAgent(BaseAgent):
    """
    Language Model agent for generative decision making.
    
    Design Pattern: Strategy Pattern
    Purpose: Uses language models for reasoning and planning
    Educational Value: Shows how LLMs can reason about game strategies
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("LLM", config)
        self.model = None
        self.prompt_template = self.config.get('prompt_template', DEFAULT_PROMPT)
        self.load_model()
        print(f"[LLMAgent] Initialized LLM agent")  # Simple logging
    
    def load_model(self):
        """Load language model"""
        model_name = self.config.get('model_name', 'gpt-3.5-turbo')
        # Model loading implementation
        print(f"[LLMAgent] Loaded model: {model_name}")  # Simple logging
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan move using language model reasoning"""
        # Format game state for LLM
        prompt = self._format_prompt(game_state)
        
        # Get LLM response
        response = self._get_llm_response(prompt)
        
        # Parse response to extract move
        move = self._parse_move_from_response(response)
        
        print(f"[LLMAgent] Generated move: {move}")  # Simple logging
        return move
    
    def _format_prompt(self, game_state: Dict[str, Any]) -> str:
        """Format game state into LLM prompt"""
        # Prompt formatting implementation
        return f"{self.prompt_template}\n\nGame State: {game_state}"
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from language model"""
        # LLM API call implementation
        return "Move RIGHT to approach the apple"
    
    def _parse_move_from_response(self, response: str) -> str:
        """Parse move direction from LLM response"""
        # Response parsing implementation
        if "RIGHT" in response.upper():
            return "RIGHT"
        elif "LEFT" in response.upper():
            return "LEFT"
        elif "UP" in response.upper():
            return "UP"
        elif "DOWN" in response.upper():
            return "DOWN"
        else:
            return "UP"  # Default fallback
```

### **Diffusion Model Agent**
```python
class DiffusionAgent(BaseAgent):
    """
    Diffusion model agent for creative strategy generation.
    
    Design Pattern: Strategy Pattern
    Purpose: Uses diffusion models for creative decision making
    Educational Value: Shows how diffusion models can generate strategies
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("DIFFUSION", config)
        self.model = None
        self.noise_schedule = self.config.get('noise_schedule', 'linear')
        self.load_model()
        print(f"[DiffusionAgent] Initialized diffusion agent")  # Simple logging
    
    def load_model(self):
        """Load diffusion model"""
        model_path = self.config.get('model_path')
        if model_path and os.path.exists(model_path):
            self.model = torch.load(model_path)
            self.model.eval()
            print(f"[DiffusionAgent] Loaded diffusion model from {model_path}")  # Simple logging
        else:
            print(f"[DiffusionAgent] No diffusion model found at {model_path}")  # Simple logging
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan move using diffusion model"""
        # Convert game state to latent representation
        latent = self._state_to_latent(game_state)
        
        # Generate strategy using diffusion
        strategy = self._generate_strategy(latent)
        
        # Convert strategy to move
        move = self._strategy_to_move(strategy, game_state)
        
        print(f"[DiffusionAgent] Generated move: {move}")  # Simple logging
        return move
    
    def _state_to_latent(self, game_state: Dict[str, Any]) -> torch.Tensor:
        """Convert game state to latent representation"""
        # Latent encoding implementation
        return torch.randn(1, 64)  # Placeholder
    
    def _generate_strategy(self, latent: torch.Tensor) -> torch.Tensor:
        """Generate strategy using diffusion model"""
        # Diffusion generation implementation
        return torch.randn(1, 16)  # Placeholder
    
    def _strategy_to_move(self, strategy: torch.Tensor, game_state: Dict[str, Any]) -> str:
        """Convert strategy to move direction"""
        # Strategy to move conversion
        move_probs = torch.softmax(strategy, dim=1)
        move_idx = torch.argmax(move_probs).item()
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        return directions[move_idx]
```

### **Vision-Language Model Agent**
```python
class VisionLanguageAgent(BaseAgent):
    """
    Vision-Language Model agent for multimodal reasoning.
    
    Design Pattern: Strategy Pattern
    Purpose: Uses vision-language models for game state understanding
    Educational Value: Shows how multimodal AI can understand game visuals
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("VISION_LANGUAGE", config)
        self.model = None
        self.image_processor = None
        self.load_model()
        print(f"[VisionLanguageAgent] Initialized vision-language agent")  # Simple logging
    
    def load_model(self):
        """Load vision-language model"""
        model_name = self.config.get('model_name', 'llava-v1.5-7b')
        # Model loading implementation
        print(f"[VisionLanguageAgent] Loaded model: {model_name}")  # Simple logging
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan move using vision-language reasoning"""
        # Generate game visualization
        image = self._generate_game_image(game_state)
        
        # Create multimodal prompt
        prompt = self._create_multimodal_prompt(game_state)
        
        # Get vision-language response
        response = self._get_vlm_response(image, prompt)
        
        # Parse move from response
        move = self._parse_move_from_response(response)
        
        print(f"[VisionLanguageAgent] Generated move: {move}")  # Simple logging
        return move
    
    def _generate_game_image(self, game_state: Dict[str, Any]) -> PIL.Image:
        """Generate visual representation of game state"""
        # Image generation implementation
        return PIL.Image.new('RGB', (100, 100), color='white')  # Placeholder
    
    def _create_multimodal_prompt(self, game_state: Dict[str, Any]) -> str:
        """Create prompt for vision-language model"""
        return f"Look at this Snake game state and tell me what move to make next. Current score: {game_state.get('score', 0)}"
    
    def _get_vlm_response(self, image: PIL.Image, prompt: str) -> str:
        """Get response from vision-language model"""
        # VLM API call implementation
        return "Move RIGHT to get the apple"
    
    def _parse_move_from_response(self, response: str) -> str:
        """Parse move direction from VLM response"""
        # Response parsing implementation
        if "RIGHT" in response.upper():
            return "RIGHT"
        elif "LEFT" in response.upper():
            return "LEFT"
        elif "UP" in response.upper():
            return "UP"
        elif "DOWN" in response.upper():
            return "DOWN"
        else:
            return "UP"  # Default fallback
```

## 📊 **Training and Fine-tuning Standards**

### **Generative Model Training Pipeline**
```python
class GenerativeTrainingPipeline:
    """
    Training pipeline for generative models.
    
    Design Pattern: Template Method Pattern
    Purpose: Provides consistent training workflow for generative models
    Educational Value: Shows how to train generative models for game AI
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.trainer = None
        print(f"[GenerativeTrainingPipeline] Initialized training pipeline")  # Simple logging
    
    def prepare_data(self):
        """Prepare training data for generative model"""
        # Data preparation implementation
        print(f"[GenerativeTrainingPipeline] Data preparation complete")  # Simple logging
    
    def train_model(self):
        """Train the generative model"""
        if self.model is None:
            self.model = self._build_model()
        
        # Training loop implementation
        print(f"[GenerativeTrainingPipeline] Training complete")  # Simple logging
    
    def save_model(self, model_path: str):
        """Save trained generative model"""
        torch.save(self.model, model_path)
        print(f"[GenerativeTrainingPipeline] Model saved to {model_path}")  # Simple logging
    
    def _build_model(self):
        """Build generative model architecture"""
        model_type = self.config.get('model_type', 'LLM')
        
        if model_type == 'LLM':
            return self._build_llm_model()
        elif model_type == 'DIFFUSION':
            return self._build_diffusion_model()
        elif model_type == 'VISION_LANGUAGE':
            return self._build_vlm_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _build_llm_model(self):
        """Build language model"""
        # LLM architecture implementation
        print(f"[GenerativeTrainingPipeline] Built LLM model")  # Simple logging
        return None  # Placeholder
    
    def _build_diffusion_model(self):
        """Build diffusion model"""
        # Diffusion architecture implementation
        print(f"[GenerativeTrainingPipeline] Built diffusion model")  # Simple logging
        return None  # Placeholder
    
    def _build_vlm_model(self):
        """Build vision-language model"""
        # VLM architecture implementation
        print(f"[GenerativeTrainingPipeline] Built VLM model")  # Simple logging
        return None  # Placeholder
```

## 📋 **Implementation Checklist**

### **Required Components**
- [ ] **Factory Pattern**: Uses canonical `create()` method
- [ ] **Model Architecture**: Implements appropriate generative model type
- [ ] **Prompt Engineering**: Effective prompt design for generative models
- [ ] **Response Parsing**: Robust parsing of generative model outputs
- [ ] **Training Pipeline**: Standardized training workflow
- [ ] **Simple Logging**: Uses print() statements for debugging

### **Quality Standards**
- [ ] **Model Performance**: Meets performance benchmarks
- [ ] **Response Quality**: Generates coherent and useful responses
- [ ] **Error Handling**: Graceful handling of model failures
- [ ] **Documentation**: Clear documentation of model capabilities

### **Integration Requirements**
- [ ] **Game State Compatibility**: Works with standard game state format
- [ ] **Factory Integration**: Compatible with agent factory patterns
- [ ] **Configuration**: Supports standard configuration system
- [ ] **Evaluation**: Integrates with evaluation framework

---

**Generative model standards ensure consistent, high-quality AI implementations across all Snake Game AI extensions. By following these standards, developers can create robust, educational, and creative AI agents that integrate seamlessly with the overall framework.**

## 🔗 **See Also**

- **`agents.md`**: Agent implementation standards
- **`core.md`**: Base class architecture and inheritance patterns
- **`config.md`**: Configuration management
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`factory-design-pattern.md`**: Factory pattern implementation
