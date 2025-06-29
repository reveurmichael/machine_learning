# Generative Models for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines generative model patterns for extensions.

> **See also:** `agents.md`, `core.md`, `config.md`, `final-decision-10.md`, `factory-design-pattern.md`.

## 🎯 **Core Philosophy: AI-Driven Content Generation**

Generative models in the Snake Game AI context focus on creating new game content, strategies, and experiences through AI generation. These models demonstrate how machine learning can be used not just for optimization but for novel content creation and discovery.

### **Design Philosophy**
- **Content Creation**: Automated generation of game scenarios and levels
- **Simple Logging**: All components use print() statements only (per `final-decision-10.md`)
- **Canonical Patterns**: Factory methods use `create()` (never `create_model()`)
- **Educational Value**: Demonstrate AI-driven creative processes

## 🧠 **Generative Model Categories**

### **Level Generation Models**
```python
# Create level generator using canonical factory method
level_generator = GenerativeModelFactory.create("VAE_LEVEL", grid_size=10)  # Canonical create()
new_levels = level_generator.generate_level(num_samples=5)
print(f"[LevelGeneration] Created {len(new_levels)} new game levels")  # Simple logging
```

### **Gameplay Data Synthesis**
```python
# Create trajectory generator using canonical factory method
trajectory_generator = GenerativeModelFactory.create("TRANSFORMER_TRAJECTORY")  # Canonical create()
synthetic_data = trajectory_generator.generate_trajectory(initial_state)
print(f"[DataSynthesis] Generated synthetic trajectory")  # Simple logging
```

### **Strategy Generation**
```python
# Create strategy generator using canonical factory method
strategy_generator = GenerativeModelFactory.create("GAN_STRATEGY")  # Canonical create()
novel_strategies = strategy_generator.generate_strategy(num_strategies=3)
print(f"[StrategyGeneration] Created {len(novel_strategies)} novel strategies")  # Simple logging
```

## 🎯 **Factory Pattern Implementation (CANONICAL create() METHOD)**
**CRITICAL REQUIREMENT**: All generative model factories MUST use the canonical `create()` method exactly as specified in `final-decision-10.md` SUPREME_RULES:

```python
class GenerativeModelFactory:
    """
    Factory Pattern for generative model agents following final-decision-10.md SUPREME_RULES
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for generative AI agents
    Educational Value: Shows how SUPREME_RULES apply to advanced AI systems -
    canonical patterns work regardless of AI complexity.
    
    Reference: final-decision-10.md SUPREME_RULES for canonical method naming
    """
    
    _registry = {
        "GPT": GPTAgent,
        "CLAUDE": ClaudeAgent,  
        "LLAMA": LlamaAgent,
        "GEMINI": GeminiAgent,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs):  # CANONICAL create() method - SUPREME_RULES
        """Create generative model agent using canonical create() method following final-decision-10.md"""
        agent_class = cls._registry.get(model_type.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model: {model_type}. Available: {available}")
        print(f"[GenerativeModelFactory] Creating agent: {model_type}")  # Simple logging - SUPREME_RULES
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

## 🔧 **Core Implementation Components (SUPREME_RULES Compliant)**

### **Base Generative Agent**
```python
class BaseGenerativeAgent(BaseAgent):
    """
    Base class for generative model agents following final-decision-10.md SUPREME_RULES.
    
    Design Pattern: Template Method Pattern (Canonical Implementation)
    Educational Value: Inherits from BaseAgent to maintain consistency
    while adding generative capabilities using canonical factory patterns.
    
    Reference: final-decision-10.md for canonical agent architecture
    """
    
    def __init__(self, name: str, grid_size: int, 
                 model_config: dict = None):
        super().__init__(name, grid_size)
        
        self.model_config = model_config or {}
        self.generation_history = []
        
        print(f"[{name}] Generative Agent initialized")  # Simple logging - SUPREME_RULES
    
    def plan_move(self, game_state: dict) -> str:
        """Plan move using generative model with simple logging throughout"""
        print(f"[{self.name}] Starting generative analysis")  # Simple logging
        
        # Generate move using canonical patterns
        move = self._generate_move(game_state)
        
        # Store generation history
        self.generation_history.append({
            'state': game_state,
            'move': move,
            'timestamp': self._get_timestamp()
        })
        
        print(f"[{self.name}] Generated move: {move}")  # Simple logging
        return move
    
    def _generate_move(self, game_state: dict) -> str:
        """Generate move using generative model (override in subclasses)"""
        raise NotImplementedError("Subclasses must implement move generation")
```

### **GPT Agent Implementation**
```python
class GPTAgent(BaseGenerativeAgent):
    """
    GPT-based generative agent following canonical patterns.
    
    Educational Value: Shows how canonical factory patterns scale
    to complex AI systems while maintaining simple logging.
    """
    
    def __init__(self, name: str, grid_size: int, **kwargs):
        super().__init__(name, grid_size, **kwargs)
        self.client = self._initialize_openai_client()
        print(f"[{name}] GPT client initialized")  # Simple logging
    
    def _generate_move(self, game_state: dict) -> str:
        """Generate move using GPT with simple logging"""
        print(f"[{self.name}] Querying GPT for move generation")  # Simple logging
        
        try:
            # Format game state for GPT
            prompt = self._format_state_for_gpt(game_state)
            
            # Query GPT
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            
            move = self._extract_move_from_response(response.choices[0].message.content)
            print(f"[{self.name}] GPT response processed")  # Simple logging
            return move
            
        except Exception as e:
            print(f"[{self.name}] GPT error: {e}")  # Simple logging
            return "UP"  # Default fallback
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client"""
        pass  # Implementation details
    
    def _format_state_for_gpt(self, game_state: dict) -> str:
        """Format game state for GPT input"""
        pass  # Implementation details
    
    def _extract_move_from_response(self, response: str) -> str:
        """Extract move from GPT response"""
        pass  # Implementation details
```

### **Claude Agent Implementation**
```python
class ClaudeAgent(BaseGenerativeAgent):
    """
    Claude-based generative agent following canonical patterns.
    
    Educational Value: Demonstrates how canonical patterns enable
    consistent implementation across different generative providers.
    """
    
    def __init__(self, name: str, grid_size: int, **kwargs):
        super().__init__(name, grid_size, **kwargs)
        self.client = self._initialize_anthropic_client()
        print(f"[{name}] Claude client initialized")  # Simple logging
    
    def _generate_move(self, game_state: dict) -> str:
        """Generate move using Claude with simple logging"""
        print(f"[{self.name}] Querying Claude for move generation")  # Simple logging
        
        try:
            # Format game state for Claude
            prompt = self._format_state_for_claude(game_state)
            
            # Query Claude
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": prompt}]
            )
            
            move = self._extract_move_from_response(response.content[0].text)
            print(f"[{self.name}] Claude response processed")  # Simple logging
            return move
            
        except Exception as e:
            print(f"[{self.name}] Claude error: {e}")  # Simple logging
            return "UP"  # Default fallback
    
    def _initialize_anthropic_client(self):
        """Initialize Anthropic client"""
        pass  # Implementation details
    
    def _format_state_for_claude(self, game_state: dict) -> str:
        """Format game state for Claude input"""
        pass  # Implementation details
    
    def _extract_move_from_response(self, response: str) -> str:
        """Extract move from Claude response"""
        pass  # Implementation details
```

## 📊 **Simple Logging Standards for Generative Operations**

### **Required Logging Pattern (SUPREME_RULES)**
All generative operations MUST use simple print statements as established in `final-decision-10.md`:

```python
# ✅ CORRECT: Simple logging for generative operations (SUPREME_RULES compliance)
def process_generative_request(prompt: str, model_type: str):
    print(f"[GenerativeProcessor] Starting generation with {model_type}")  # Simple logging - REQUIRED
    
    # Generation phase
    response = generate_content(prompt, model_type)
    print(f"[GenerativeProcessor] Generation completed")  # Simple logging
    
    # Processing phase
    result = process_response(response)
    print(f"[GenerativeProcessor] Response processed")  # Simple logging
    
    print(f"[GenerativeProcessor] Generative cycle completed")  # Simple logging
    return result

# ❌ FORBIDDEN: Complex logging frameworks (violates SUPREME_RULES)
# import logging
# logger = logging.getLogger(__name__)

# def process_generative_request(prompt: str, model_type: str):
#     logger.info(f"Starting generative processing")  # FORBIDDEN - complex logging
#     # This violates final-decision-10.md SUPREME_RULES
```

## 🎓 **Educational Applications with Canonical Patterns**

### **AI Creativity Understanding**
- **Content Generation**: Clear examples of AI-driven content creation using canonical patterns
- **Model Integration**: See how canonical `create()` method works with complex generative systems
- **Provider Abstraction**: Understand how canonical patterns enable consistent interfaces across different AI providers
- **Creative Processes**: Experience AI-driven creative decision-making following SUPREME_RULES compliance

### **Pattern Consistency Across AI Complexity**
- **Factory Patterns**: All generative components use canonical `create()` method consistently
- **Simple Logging**: Print statements provide clear visibility into generative operations
- **Educational Value**: Canonical patterns work identically across simple and complex AI
- **SUPREME_RULES**: Advanced generative systems follow same standards as basic heuristics

## 📋 **SUPREME_RULES Implementation Checklist for Generative Models**

### **Mandatory Requirements**
- [ ] **Canonical Method**: All factories use `create()` method exactly (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses print() statements only for all generative operations (final-decision-10.md compliance)
- [ ] **GOOD_RULES Reference**: References `final-decision-10.md` in all generative documentation
- [ ] **Pattern Consistency**: Follows canonical patterns across all generative implementations

### **Generative-Specific Standards**
- [ ] **Model Integration**: Canonical factory patterns for all generative model components
- [ ] **Provider Abstraction**: Canonical factory patterns for all AI provider systems
- [ ] **Content Generation**: Canonical patterns for all content creation systems
- [ ] **Creative Processes**: Simple logging for all generative decision-making operations

### **Educational Integration**
- [ ] **Clear Examples**: Simple examples using canonical `create()` method for generative systems
- [ ] **Pattern Explanation**: Clear explanation of canonical patterns in generative AI context
- [ ] **Best Practices**: Demonstration of SUPREME_RULES in advanced generative systems
- [ ] **Learning Value**: Easy to understand canonical patterns regardless of generative complexity

---

**Generative models represent the cutting edge of AI creativity while maintaining strict compliance with `final-decision-10.md` SUPREME_RULES, proving that canonical patterns and simple logging provide consistent foundations across all AI complexity levels.**

## 🔗 **See Also**

- **`agents.md`**: Authoritative reference for agent implementation with canonical patterns
- **`core.md`**: Base class architecture following canonical principles  
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`factory-design-pattern.md`**: Canonical factory implementation for all systems
