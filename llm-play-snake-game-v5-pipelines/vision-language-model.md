# Vision-Language Models for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines vision-language model patterns for extensions.

# Vision-Language Models for Snake Game AI

> **Guidelines Alignment:**
> - This document is governed by the guidelines in SUPREME_RULES from `final-decision-10.md`.
> - All agent factories must use the canonical method name `create()` (never `create_agent`, `create_model`, etc.).
> - All code must use simple print logging (simple logging).
> - Reference: `extensions/common/utils/factory_utils.py` for the canonical `SimpleFactory` implementation.

> **See also:** `agents.md`, `core.md`, `final-decision-10.md`, `factory-design-pattern.md`, `config.md`.

## 🎯 **Core Philosophy: Multimodal AI + SUPREME_RULES Compliance**

Vision-Language Models (VLMs) represent a cutting-edge approach that combines visual understanding with natural language processing for Snake Game AI. **This extension strictly follows the SUPREME_RULES** established in SUPREME_RULES from `final-decision-10.md`, particularly the **canonical `create()` method patterns and simple logging requirements**.

### **Guidelines Alignment**
- **final-decision-10.md Guideline 1**: Follows all established GOOD_RULES patterns for multimodal AI integration
- **final-decision-10.md Guideline 2**: Uses precise `final-decision-N.md` format consistently throughout VLM implementations
- **simple logging**: Lightweight, OOP-based common utilities with simple logging (print() statements only)

### **Educational Value**
- **Multimodal Understanding**: Learn how AI interprets visual and textual data simultaneously using canonical patterns
- **Vision Processing**: Understand visual reasoning following SUPREME_RULES compliance
- **Advanced AI**: Experience cutting-edge VLM capabilities with simple logging throughout
- **Pattern Recognition**: Canonical `create()` method enables consistent learning across vision extensions

## 🏗️ **VLM Extension Architecture (SUPREME_RULES Compliant)**

### **Factory Pattern Implementation (CANONICAL create() METHOD)**
**CRITICAL REQUIREMENT**: All VLM factories MUST use the canonical `create()` method exactly as specified in SUPREME_RULES from `final-decision-10.md`:

```python
class VLMAgentFactory:
    """
    Factory Pattern for VLM agents following SUPREME_RULES from `final-decision-10.md`.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for multimodal AI agents
    Educational Value: Shows how SUPREME_RULES apply to advanced AI systems -
    canonical patterns work regardless of AI complexity.
    
    Reference: SUPREME_RULES from `final-decision-10.md` for canonical method naming
    """
    
    _registry = {
        "GPT4_VISION": GPT4VisionAgent,
        "CLAUDE_VISION": ClaudeVisionAgent,  
        "LLAVA": LLaVAAgent,
        "GEMINI_VISION": GeminiVisionAgent,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs):  # CANONICAL create() method - SUPREME_RULES
        """Create VLM agent using canonical create() method following SUPREME_RULES from `final-decision-10.md`"""
        agent_class = cls._registry.get(model_type.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown VLM model: {model_type}. Available: {available}")
        print(f"[VLMAgentFactory] Creating VLM agent: {model_type}")  # Simple logging - SUPREME_RULES
        return agent_class(**kwargs)

# ❌ FORBIDDEN: Non-canonical method names (violates SUPREME_RULES)
class VLMAgentFactory:
    def create_vlm_agent(self, model_type: str):  # FORBIDDEN - not canonical
        pass
    
    def build_vision_model(self, model_type: str):  # FORBIDDEN - not canonical
        pass
    
    def make_multimodal_agent(self, model_type: str):  # FORBIDDEN - not canonical
        pass
```

### **Visual Renderer Factory (CANONICAL PATTERN)**
```python
class VisionRendererFactory:
    """
    Factory for visual rendering strategies following SUPREME_RULES.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Educational Value: Shows how canonical create() method enables
    consistent visual processing across different VLM providers.
    
    Reference: SUPREME_RULES from `final-decision-10.md` for canonical factory standards
    """
    
    _registry = {
        "HIGH_CONTRAST": HighContrastRenderer,
        "DETAILED": DetailedGridRenderer,
        "MINIMALIST": MinimalistRenderer,
        "ANNOTATED": AnnotatedRenderer,
    }
    
    @classmethod
    def create(cls, renderer_type: str, **kwargs):  # CANONICAL create() method
        """Create visual renderer using canonical create() method (SUPREME_RULES compliance)"""
        renderer_class = cls._registry.get(renderer_type.upper())
        if not renderer_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown renderer: {renderer_type}. Available: {available}")
        print(f"[VisionRendererFactory] Creating renderer: {renderer_type}")  # Simple logging
        return renderer_class(**kwargs)
```

### **Prompt Strategy Factory (CANONICAL PATTERN)**
```python
class VLMPromptFactory:
    """
    Factory for VLM prompt strategies following SUPREME_RULES.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Educational Value: Demonstrates canonical create() method for
    multimodal prompt engineering across different VLM architectures.
    
    Reference: SUPREME_RULES from `final-decision-10.md` for factory implementation
    """
    
    _registry = {
        "CHAIN_OF_THOUGHT": CoTPromptStrategy,
        "REACT": ReActPromptStrategy,
        "DIRECT": DirectPromptStrategy,
        "GUIDED": GuidedAnalysisPromptStrategy,
    }
    
    @classmethod
    def create(cls, strategy_type: str, **kwargs):  # CANONICAL create() method
        """Create prompt strategy using canonical create() method (SUPREME_RULES compliance)"""
        strategy_class = cls._registry.get(strategy_type.upper())
        if not strategy_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown prompt strategy: {strategy_type}. Available: {available}")
        print(f"[VLMPromptFactory] Creating prompt strategy: {strategy_type}")  # Simple logging
        return strategy_class(**kwargs)
```

## 🎮 **Visual State Representation (Simple Logging)**

### **Game State Renderer (SUPREME_RULES Compliant)**
```python
class GameStateRenderer:
    """
    Convert game states to VLM-compatible visual formats following SUPREME_RULES.
    
    Design Pattern: Strategy Pattern (Canonical Implementation)
    Purpose: Multiple rendering strategies using canonical factory patterns
    Educational Value: Shows how simple logging and canonical patterns
    work together in complex visual processing pipelines.
    
    Reference: SUPREME_RULES from `final-decision-10.md` for simple logging standards
    """
    
    def __init__(self, grid_size: int = 10, renderer_type: str = "HIGH_CONTRAST"):
        self.grid_size = grid_size
        self.renderer = VisionRendererFactory.create(renderer_type, grid_size=grid_size)  # Canonical
        print(f"[GameStateRenderer] Initialized for {grid_size}x{grid_size} grid")  # Simple logging
    
    def render_state_for_vlm(self, game_state: dict) -> bytes:
        """Create high-quality visual representation with simple logging"""
        print(f"[GameStateRenderer] Starting render for step {game_state.get('step', 0)}")  # Simple logging
        
        # Simple visualization optimized for VLM analysis
        image_bytes = self.renderer.create_visualization(game_state)
        
        print(f"[GameStateRenderer] Generated visualization: {len(image_bytes)} bytes")  # Simple logging
        return image_bytes
    
    def create_multiframe_sequence(self, game_states: list) -> bytes:
        """Create multi-frame visualization for temporal analysis"""
        print(f"[GameStateRenderer] Creating sequence: {len(game_states)} frames")  # Simple logging
        
        sequence_data = self.renderer.create_sequence(game_states)
        
        print(f"[GameStateRenderer] Sequence created: {len(sequence_data)} bytes")  # Simple logging
        return sequence_data
```

## 🧠 **VLM Agent Implementation (CANONICAL PATTERNS)**

### **Base VLM Agent (SUPREME_RULES Compliant)**
```python
class BaseVLMAgent(BaseAgent):
    """
    Base class for VLM agents following SUPREME_RULES from `final-decision-10.md`.
    
    Design Pattern: Template Method Pattern (Canonical Implementation)
    Educational Value: Inherits from BaseAgent to maintain consistency
    while adding multimodal capabilities using canonical factory patterns.
    
    Reference: final-decision-10.md for canonical agent architecture
    """
    
    def __init__(self, name: str, grid_size: int, 
                 renderer_type: str = "HIGH_CONTRAST",
                 prompt_strategy: str = "CHAIN_OF_THOUGHT"):
        super().__init__(name, grid_size)
        
        # Use canonical factory patterns
        self.renderer = VisionRendererFactory.create(renderer_type, grid_size=grid_size)  # Canonical
        self.prompt_strategy = VLMPromptFactory.create(prompt_strategy)  # Canonical
        
        self.visual_history = []
        self.analysis_history = []
        
        print(f"[{name}] VLM Agent initialized with {renderer_type} renderer")  # Simple logging
    
    def plan_move(self, game_state: dict) -> str:
        """Plan move using VLM analysis with simple logging throughout"""
        print(f"[{self.name}] Starting VLM analysis")  # Simple logging
        
        # Create visual representation
        visual_data = self.renderer.render_state_for_vlm(game_state)
        print(f"[{self.name}] Visual representation created")  # Simple logging
        
        # Analyze with VLM
        analysis = self._analyze_with_vlm(visual_data, game_state)
        self.analysis_history.append(analysis)
        print(f"[{self.name}] VLM analysis completed")  # Simple logging
        
        # Extract move decision
        move = self._extract_move_from_analysis(analysis)
        
        print(f"[{self.name}] VLM decided: {move}")  # Simple logging
        return move
    
    def _analyze_with_vlm(self, visual_data: bytes, game_state: dict) -> dict:
        """Analyze visual data with VLM (override in subclasses)"""
        raise NotImplementedError("Subclasses must implement VLM analysis")
    
    def _extract_move_from_analysis(self, analysis: dict) -> str:
        """Extract move from VLM analysis"""
        pass  # Implementation details
```

### **GPT-4 Vision Agent Implementation**
```python
class GPT4VisionAgent(BaseVLMAgent):
    """
    GPT-4 Vision agent following canonical patterns.
    
    Educational Value: Shows how canonical factory patterns scale
    to complex multimodal AI systems while maintaining simple logging.
    """
    
    def __init__(self, name: str, grid_size: int, **kwargs):
        super().__init__(name, grid_size, **kwargs)
        self.client = self._initialize_openai_client()
        print(f"[{name}] GPT-4 Vision client initialized")  # Simple logging
    
    def _analyze_with_vlm(self, visual_data: bytes, game_state: dict) -> dict:
        """Analyze with GPT-4 Vision using simple logging"""
        print(f"[{self.name}] Querying GPT-4 Vision")  # Simple logging
        
        try:
            # Create multimodal prompt
            prompt = self.prompt_strategy.create_prompt(game_state)
            
            # Query GPT-4 Vision
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{visual_data}"}}
                    ]}
                ]
            )
            
            analysis = self._parse_vlm_response(response.choices[0].message.content)
            print(f"[{self.name}] GPT-4 Vision response processed")  # Simple logging
            return analysis
            
        except Exception as e:
            print(f"[{self.name}] GPT-4 Vision error: {e}")  # Simple logging
            return {"move": "UP", "reasoning": "Error fallback"}  # Default fallback
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client"""
        pass  # Implementation details
    
    def _parse_vlm_response(self, response: str) -> dict:
        """Parse VLM response into structured analysis"""
        pass  # Implementation details
```

### **Claude Vision Agent Implementation**
```python
class ClaudeVisionAgent(BaseVLMAgent):
    """
    Claude Vision agent following canonical patterns.
    
    Educational Value: Demonstrates how canonical patterns enable
    consistent implementation across different VLM providers.
    """
    
    def __init__(self, name: str, grid_size: int, **kwargs):
        super().__init__(name, grid_size, **kwargs)
        self.client = self._initialize_anthropic_client()
        print(f"[{name}] Claude Vision client initialized")  # Simple logging
    
    def _analyze_with_vlm(self, visual_data: bytes, game_state: dict) -> dict:
        """Analyze with Claude Vision using simple logging"""
        print(f"[{self.name}] Querying Claude Vision")  # Simple logging
        
        try:
            # Create multimodal prompt
            prompt = self.prompt_strategy.create_prompt(game_state)
            
            # Query Claude Vision
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": visual_data}}
                    ]}
                ]
            )
            
            analysis = self._parse_vlm_response(response.content[0].text)
            print(f"[{self.name}] Claude Vision response processed")  # Simple logging
            return analysis
            
        except Exception as e:
            print(f"[{self.name}] Claude Vision error: {e}")  # Simple logging
            return {"move": "UP", "reasoning": "Error fallback"}  # Default fallback
    
    def _initialize_anthropic_client(self):
        """Initialize Anthropic client"""
        pass  # Implementation details
    
    def _parse_vlm_response(self, response: str) -> dict:
        """Parse VLM response into structured analysis"""
        pass  # Implementation details
```

## 📊 **Simple Logging Standards for VLM Operations**

### **Required Logging Pattern (SUPREME_RULES)**
All VLM operations MUST use simple print statements as established in `final-decision-10.md`:

```python
# ✅ CORRECT: Simple logging for VLM operations (SUPREME_RULES compliance)
def process_vlm_request(visual_data: bytes, prompt: str, model_type: str):
    print(f"[VLMProcessor] Starting VLM analysis with {model_type}")  # Simple logging - REQUIRED
    
    # Visual processing phase
    processed_image = preprocess_visual_data(visual_data)
    print(f"[VLMProcessor] Visual preprocessing completed")  # Simple logging
    
    # VLM analysis phase
    analysis = query_vlm(processed_image, prompt, model_type)
    print(f"[VLMProcessor] VLM analysis completed")  # Simple logging
    
    # Result extraction phase
    result = extract_result(analysis)
    print(f"[VLMProcessor] Result extraction completed")  # Simple logging
    
    print(f"[VLMProcessor] VLM cycle completed")  # Simple logging
    return result

# ❌ FORBIDDEN: Complex logging frameworks (violates SUPREME_RULES)
# import logging
# logger = logging.getLogger(__name__)

# def process_vlm_request(visual_data: bytes, prompt: str, model_type: str):
#     logger.info(f"Starting VLM processing")  # FORBIDDEN - complex logging
#     # This violates final-decision-10.md SUPREME_RULES
```

## 🎓 **Educational Applications with Canonical Patterns**

### **Multimodal AI Understanding**
- **Visual Processing**: Clear examples of visual data processing using canonical patterns
- **VLM Integration**: See how canonical `create()` method works with complex multimodal systems
- **Provider Abstraction**: Understand how canonical patterns enable consistent interfaces across different VLM providers
- **Visual Reasoning**: Experience AI visual decision-making following SUPREME_RULES compliance

### **Pattern Consistency Across AI Complexity**
- **Factory Patterns**: All VLM components use canonical `create()` method consistently
- **Simple Logging**: Print statements provide clear visibility into multimodal operations
- **Educational Value**: Canonical patterns work identically across simple and complex AI
- **SUPREME_RULES**: Advanced multimodal systems follow same standards as basic heuristics

## 📋 **SUPREME_RULES Implementation Checklist for VLMs**

### **Mandatory Requirements**
- [ ] **Canonical Method**: All factories use `create()` method exactly (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses print() statements only for all VLM operations (final-decision-10.md compliance)
- [ ] **GOOD_RULES Reference**: References `final-decision-10.md` in all VLM documentation
- [ ] **Pattern Consistency**: Follows canonical patterns across all VLM implementations

### **VLM-Specific Standards**
- [ ] **Visual Processing**: Canonical factory patterns for all visual rendering components
- [ ] **VLM Integration**: Canonical factory patterns for all VLM provider systems
- [ ] **Multimodal Analysis**: Canonical patterns for all visual-textual analysis systems
- [ ] **Visual Reasoning**: Simple logging for all visual decision-making operations

### **Educational Integration**
- [ ] **Clear Examples**: Simple examples using canonical `create()` method for VLM systems
- [ ] **Pattern Explanation**: Clear explanation of canonical patterns in multimodal AI context
- [ ] **Best Practices**: Demonstration of SUPREME_RULES in advanced VLM systems
- [ ] **Learning Value**: Easy to understand canonical patterns regardless of multimodal complexity

---

**Vision-Language Models represent the cutting edge of multimodal AI while maintaining strict compliance with `final-decision-10.md` SUPREME_RULES, proving that canonical patterns and simple logging provide consistent foundations across all AI complexity levels.**

## 🔗 **See Also**

- **`agents.md`**: Authoritative reference for agent implementation with canonical patterns
- **`core.md`**: Base class architecture following canonical principles  
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`factory-design-pattern.md`**: Canonical factory implementation for all systems