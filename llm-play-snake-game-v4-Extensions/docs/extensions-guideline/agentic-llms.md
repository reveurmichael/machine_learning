# Agentic LLMs for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines agentic LLM patterns for extensions.

> **See also:** `agents.md`, `core.md`, `config.md`, `final-decision-10.md`, `factory-design-pattern.md`.

## 🎯 **Core Philosophy: LLM-Powered Autonomous Agents + SUPREME_RULES**

Agentic LLMs represent the next evolution of AI systems that can reason, plan, and act autonomously in complex environments. **This extension strictly follows the SUPREME_RULES** established in `final-decision-10.md`, particularly the **canonical `create()` method patterns and simple logging requirements** for all agentic behaviors.

### **Guidelines Alignment**
- **final-decision-10.md Guideline 1**: Follows all established GOOD_RULES patterns for autonomous agent architectures
- **final-decision-10.md Guideline 2**: Uses precise `final-decision-N.md` format consistently throughout agentic implementations
- **simple logging**: Lightweight, OOP-based common utilities with simple logging (print() statements only)

### **Educational Value**
- **Autonomous Reasoning**: Learn how LLMs reason and plan using canonical patterns
- **Multi-Step Planning**: Understand complex decision-making with simple logging throughout
- **Self-Reflection**: Experience AI self-improvement following SUPREME_RULES compliance
- **Tool Integration**: See canonical patterns in LLM-tool integration systems

## 🧠 **Agentic Architecture Components (CANONICAL PATTERNS)**

### **Agentic LLM Factory (SUPREME_RULES Compliant)**
**CRITICAL REQUIREMENT**: All agentic LLM factories MUST use the canonical `create()` method exactly as specified in `final-decision-10.md` SUPREME_RULES:

```python
class AgenticLLMFactory:
    """
    Factory Pattern for Agentic LLM agents following final-decision-10.md SUPREME_RULES
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for autonomous AI agents
    Educational Value: Shows how SUPREME_RULES apply to advanced AI systems -
    canonical patterns work regardless of AI complexity.
    
    Reference: final-decision-10.md SUPREME_RULES for canonical method naming
    """
    
    _registry = {
        "AUTOGPT": AutoGPTAgent,
        "BABYAGI": BabyAGIAgent,
        "LANGCHAIN": LangChainAgent,
        "CUSTOM": CustomAgenticAgent,
    }
    
    @classmethod
    def create(cls, agent_type: str, **kwargs):  # CANONICAL create() method - SUPREME_RULES
        """Create Agentic LLM agent using canonical create() method following final-decision-10.md"""
        agent_class = cls._registry.get(agent_type.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown Agentic LLM: {agent_type}. Available: {available}")
        print(f"[AgenticLLMFactory] Creating agent: {agent_type}")  # Simple logging - SUPREME_RULES
        return agent_class(**kwargs)

# ❌ FORBIDDEN: Non-canonical method names (violates SUPREME_RULES)
class AgenticLLMFactory:
    def create_agentic_agent(self, agent_type: str):  # FORBIDDEN - not canonical
        pass
    
    def build_autonomous_agent(self, agent_type: str):  # FORBIDDEN - not canonical
        pass
    
    def make_agentic_llm(self, agent_type: str):  # FORBIDDEN - not canonical
        pass
```

### **Tool Integration Factory (CANONICAL PATTERN)**
```python
class AgenticToolFactory:
    """
    Factory for agentic tool integration following SUPREME_RULES.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Educational Value: Shows how canonical create() method enables
    consistent tool integration across different agentic architectures.
    
    Reference: final-decision-10.md for canonical factory standards
    """
    
    _registry = {
        "PATHFINDING": PathfindingTool,
        "ANALYSIS": GameStateAnalyzer,
        "MEMORY": MemoryManager,
        "PLANNING": PlanningTool,
    }
    
    @classmethod
    def create(cls, tool_type: str, **kwargs):  # CANONICAL create() method
        """Create agentic tool using canonical create() method (SUPREME_RULES compliance)"""
        tool_class = cls._registry.get(tool_type.upper())
        if not tool_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown tool: {tool_type}. Available: {available}")
        print(f"[AgenticToolFactory] Creating tool: {tool_type}")  # Simple logging
        return tool_class(**kwargs)
```

### **Prompt Strategy Factory (CANONICAL PATTERN)**
```python
class AgenticPromptFactory:
    """
    Factory for agentic prompt strategies following SUPREME_RULES.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Educational Value: Demonstrates canonical create() method for
    complex reasoning prompt strategies across agentic architectures.
    
    Reference: final-decision-10.md SUPREME_RULES for factory implementation
    """
    
    _registry = {
        "REACT_PROMPTS": ReActPromptStrategy,
        "COT_PROMPTS": ChainOfThoughtPromptStrategy,
        "TOOL_PROMPTS": ToolUsePromptStrategy,
        "PLANNING_PROMPTS": PlanningPromptStrategy,
    }
    
    @classmethod
    def create(cls, strategy_type: str, **kwargs):  # CANONICAL create() method
        """Create prompt strategy using canonical create() method (SUPREME_RULES compliance)"""
        strategy_class = cls._registry.get(strategy_type.upper())
        if not strategy_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown prompt strategy: {strategy_type}. Available: {available}")
        print(f"[AgenticPromptFactory] Creating prompt strategy: {strategy_type}")  # Simple logging
        return strategy_class(**kwargs)
```

## 🔧 **Implementation Patterns (SUPREME_RULES Compliant)**

### **ReAct Agent Implementation (CANONICAL PATTERNS)**
```python
class ReActAgent(BaseAgent):
    """
    ReAct (Reasoning + Acting) Agent for Snake Game following SUPREME_RULES.
    
    Design Pattern: Strategy Pattern (Canonical Implementation)
    Purpose: Alternates between reasoning and acting phases using canonical patterns
    Educational Value: Shows how canonical factory patterns work with
    autonomous reasoning systems while maintaining simple logging.
    
    Reference: final-decision-10.md for canonical agent architecture
    """
    
    def __init__(self, name: str, grid_size: int, 
                 prompt_strategy: str = "REACT_PROMPTS"):
        super().__init__(name, grid_size)
        
        # Use canonical factory patterns
        self.prompt_manager = AgenticPromptFactory.create(prompt_strategy)  # Canonical
        self.reasoning_history = []
        self.decision_history = []
        
        print(f"[{name}] ReAct Agent initialized with {prompt_strategy}")  # Simple logging
    
    def plan_move(self, game_state: dict) -> str:
        """Plan move using ReAct reasoning pattern with simple logging throughout"""
        print(f"[{self.name}] Starting ReAct reasoning cycle")  # Simple logging
        
        # Observation (using simple logging)
        observation = self._format_observation(game_state)
        print(f"[{self.name}] Observation formatted")  # Simple logging
        
        # Thought (Reasoning with simple logging)
        thought = self._generate_thought(observation)
        self.reasoning_history.append(thought)
        print(f"[{self.name}] Reasoning completed")  # Simple logging
        
        # Action (using canonical patterns)
        action = self._decide_action(thought, observation)
        self.decision_history.append({
            'observation': observation,
            'thought': thought,
            'action': action
        })
        
        print(f"[{self.name}] ReAct decided: {action}")  # Simple logging
        return action
    
    def _format_observation(self, game_state: dict) -> str:
        """Format game state observation with simple logging"""
        print(f"[{self.name}] Formatting observation")  # Simple logging
        
        observation = f"""
        Snake Head: {game_state.get('head_position')}
        Apple Position: {game_state.get('apple_position')}
        Snake Length: {game_state.get('snake_length')}
        Current Score: {game_state.get('score')}
        """
        
        print(f"[{self.name}] Observation ready")  # Simple logging
        return observation
    
    def _generate_thought(self, observation: str) -> str:
        """Generate reasoning thought with simple logging"""
        print(f"[{self.name}] Generating reasoning thought")  # Simple logging
        
        # Use canonical prompt manager
        prompt = self.prompt_manager.create_reasoning_prompt(observation)
        thought = self._query_llm(prompt)
        
        print(f"[{self.name}] Thought generated")  # Simple logging
        return thought
    
    def _decide_action(self, thought: str, observation: str) -> str:
        """Decide action based on reasoning with simple logging"""
        print(f"[{self.name}] Making action decision")  # Simple logging
        
        # Use canonical prompt manager for action decision
        action_prompt = self.prompt_manager.create_action_prompt(thought, observation)
        action_response = self._query_llm(action_prompt)
        
        # Extract move from response
        action = self._extract_move(action_response)
        print(f"[{self.name}] Action decided: {action}")  # Simple logging
        return action
```

### **Tool-Using Agent Implementation (CANONICAL PATTERNS)**
```python
class ToolUsingAgent(BaseAgent):
    """
    Agent that uses external tools to augment LLM reasoning following SUPREME_RULES.
    
    Design Pattern: Decorator Pattern (Canonical Implementation)
    Purpose: Wraps LLM reasoning with tool capabilities using canonical patterns
    Educational Value: Shows how canonical factory patterns enable
    consistent tool integration across different AI systems.
    
    Reference: final-decision-10.md for canonical tool integration
    """
    
    def __init__(self, name: str, grid_size: int, 
                 tool_types: list = None,
                 prompt_strategy: str = "TOOL_PROMPTS"):
        super().__init__(name, grid_size)
        
        # Use canonical factory patterns for tools
        self.tools = {}
        tool_types = tool_types or ["PATHFINDING", "ANALYSIS", "MEMORY"]
        
        for tool_type in tool_types:
            tool = AgenticToolFactory.create(tool_type, grid_size=grid_size)  # Canonical
            self.tools[tool_type.lower()] = tool
            print(f"[{name}] Initialized tool: {tool_type}")  # Simple logging
        
        # Use canonical prompt factory
        self.prompt_manager = AgenticPromptFactory.create(prompt_strategy)  # Canonical
        
        print(f"[{name}] Tool-Using Agent initialized")  # Simple logging
    
    def plan_move(self, game_state: dict) -> str:
        """Plan move using tools when needed with simple logging throughout"""
        print(f"[{self.name}] Starting tool-augmented planning")  # Simple logging
        
        # Assess tool needs using simple logic
        tool_assessments = self._assess_tool_needs(game_state)
        print(f"[{self.name}] Tool assessment completed")  # Simple logging
        
        # Use tools as needed with canonical patterns
        tool_results = {}
        for tool_name, is_needed in tool_assessments.items():
            if is_needed and tool_name in self.tools:
                print(f"[{self.name}] Using tool: {tool_name}")  # Simple logging
                result = self.tools[tool_name].analyze(game_state)
                tool_results[tool_name] = result
                print(f"[{self.name}] Tool {tool_name} completed")  # Simple logging
        
        # Generate decision with tool context
        decision = self._generate_move_with_context(game_state, tool_results)
        print(f"[{self.name}] Tool-assisted decision: {decision}")  # Simple logging
        return decision
    
    def _assess_tool_needs(self, game_state: dict) -> dict:
        """Assess which tools are needed with simple logging"""
        print(f"[{self.name}] Assessing tool requirements")  # Simple logging
        
        assessments = {
            "pathfinding": self._needs_pathfinding(game_state),
            "analysis": self._needs_analysis(game_state),
            "memory": self._needs_memory_lookup(game_state),
        }
        
        print(f"[{self.name}] Tool needs: {assessments}")  # Simple logging
        return assessments
    
    def _generate_move_with_context(self, game_state: dict, tool_results: dict) -> str:
        """Generate move decision with tool context using canonical patterns"""
        print(f"[{self.name}] Generating move with tool context")  # Simple logging
        
        # Use canonical prompt manager with tool context
        context_prompt = self.prompt_manager.create_tool_context_prompt(game_state, tool_results)
        response = self._query_llm(context_prompt)
        
        # Extract move using simple validation
        move = self._extract_move(response)
        print(f"[{self.name}] Context-based move: {move}")  # Simple logging
        return move
```

### **Memory-Enhanced Agent (CANONICAL PATTERNS)**
```python
class MemoryEnhancedAgent(BaseAgent):
    """
    Agent with persistent memory following SUPREME_RULES.
    
    Design Pattern: Memento Pattern (Canonical Implementation)
    Purpose: Maintains persistent memory using canonical factory patterns
    Educational Value: Shows how canonical patterns work with
    memory-enhanced autonomous systems while maintaining simple logging.
    
    Reference: final-decision-10.md for canonical memory integration
    """
    
    def __init__(self, name: str, grid_size: int,
                 memory_type: str = "MEMORY",
                 prompt_strategy: str = "COT_PROMPTS"):
        super().__init__(name, grid_size)
        
        # Use canonical factory patterns
        self.memory_manager = AgenticToolFactory.create(memory_type, grid_size=grid_size)  # Canonical
        self.prompt_manager = AgenticPromptFactory.create(prompt_strategy)  # Canonical
        
        self.game_history = []
        self.strategy_patterns = []
        
        print(f"[{name}] Memory-Enhanced Agent initialized")  # Simple logging
    
    def plan_move(self, game_state: dict) -> str:
        """Plan move using memory and learning with simple logging throughout"""
        print(f"[{self.name}] Starting memory-enhanced planning")  # Simple logging
        
        # Retrieve relevant memories using canonical patterns
        relevant_memories = self.memory_manager.retrieve_relevant(game_state)
        print(f"[{self.name}] Retrieved {len(relevant_memories)} memories")  # Simple logging
        
        # Analyze current situation with memory context
        analysis = self._analyze_with_memory(game_state, relevant_memories)
        print(f"[{self.name}] Memory analysis completed")  # Simple logging
        
        # Make decision using canonical prompt patterns
        move = self._decide_with_memory_context(game_state, analysis)
        
        # Store experience for future learning
        self._store_experience(game_state, move, analysis)
        print(f"[{self.name}] Experience stored for learning")  # Simple logging
        
        print(f"[{self.name}] Memory-enhanced decision: {move}")  # Simple logging
        return move
    
    def _analyze_with_memory(self, game_state: dict, memories: list) -> dict:
        """Analyze situation with memory context using simple logging"""
        print(f"[{self.name}] Analyzing with memory context")  # Simple logging
        
        # Use canonical prompt manager for memory analysis
        memory_prompt = self.prompt_manager.create_memory_analysis_prompt(game_state, memories)
        analysis_response = self._query_llm(memory_prompt)
        
        # Parse analysis response
        analysis = self._parse_analysis_response(analysis_response)
        print(f"[{self.name}] Memory analysis parsed")  # Simple logging
        return analysis
    
    def _store_experience(self, game_state: dict, move: str, analysis: dict) -> None:
        """Store experience for future learning with simple logging"""
        print(f"[{self.name}] Storing experience")  # Simple logging
        
        experience = {
            'game_state': game_state,
            'move_taken': move,
            'analysis': analysis,
            'timestamp': self._get_timestamp()
        }
        
        # Use canonical memory manager for storage
        self.memory_manager.store_experience(experience)
        self.game_history.append(experience)
        
        print(f"[{self.name}] Experience stored successfully")  # Simple logging
```

## 📊 **Simple Logging Standards for Agentic Operations**

### **Required Logging Pattern (SUPREME_RULES)**
All agentic operations MUST use simple print statements as established in `final-decision-10.md`:

```python
# ✅ CORRECT: Simple logging for agentic operations (SUPREME_RULES compliance)
def process_agentic_reasoning(observation: str, tools: dict):
    print(f"[AgenticProcessor] Starting reasoning cycle")  # Simple logging - REQUIRED
    
    # Reasoning phase
    thought = generate_thought(observation)
    print(f"[AgenticProcessor] Reasoning completed")  # Simple logging
    
    # Tool usage phase
    for tool_name, tool in tools.items():
        result = tool.analyze(observation)
        print(f"[AgenticProcessor] Tool {tool_name} completed")  # Simple logging
    
    print(f"[AgenticProcessor] Agentic cycle completed")  # Simple logging
    return action

# ❌ FORBIDDEN: Complex logging frameworks (violates SUPREME_RULES)
# import logging
# logger = logging.getLogger(__name__)

# def process_agentic_reasoning(observation: str, tools: dict):
#     logger.info(f"Starting agentic reasoning")  # FORBIDDEN - complex logging
#     # This violates final-decision-10.md SUPREME_RULES
```

## 🎓 **Educational Applications with Canonical Patterns**

### **Autonomous AI Understanding**
- **Reasoning Transparency**: Clear step-by-step decision processes using canonical patterns
- **Tool Integration**: See how canonical `create()` method works with complex tool systems
- **Memory Systems**: Understand learning and memory with simple logging throughout
- **Planning Systems**: Experience multi-step planning following SUPREME_RULES compliance

### **Pattern Consistency Across AI Complexity**
- **Factory Patterns**: All agentic components use canonical `create()` method consistently
- **Simple Logging**: Print statements provide clear visibility into autonomous operations
- **Educational Value**: Canonical patterns work identically across simple and complex AI
- **SUPREME_RULES**: Advanced autonomous systems follow same standards as basic heuristics

## 📋 **SUPREME_RULES Implementation Checklist for Agentic LLMs**

### **Mandatory Requirements**
- [ ] **Canonical Method**: All factories use `create()` method exactly (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses print() statements only for all agentic operations (final-decision-10.md compliance)
- [ ] **GOOD_RULES Reference**: References `final-decision-10.md` in all agentic documentation
- [ ] **Pattern Consistency**: Follows canonical patterns across all agentic implementations

### **Agentic-Specific Standards**
- [ ] **Reasoning Systems**: Canonical factory patterns for all reasoning components
- [ ] **Tool Integration**: Canonical factory patterns for all tool systems
- [ ] **Memory Systems**: Canonical patterns for all memory and learning systems
- [ ] **Planning Systems**: Simple logging for all multi-step planning operations

### **Educational Integration**
- [ ] **Clear Examples**: Simple examples using canonical `create()` method for agentic systems
- [ ] **Pattern Explanation**: Clear explanation of canonical patterns in autonomous AI context
- [ ] **Best Practices**: Demonstration of SUPREME_RULES in advanced agentic systems
- [ ] **Learning Value**: Easy to understand canonical patterns regardless of autonomy complexity

---

**Agentic LLMs represent the pinnacle of autonomous AI systems while maintaining strict compliance with `final-decision-10.md` SUPREME_RULES, proving that canonical patterns and simple logging provide consistent foundations across all AI complexity levels.**

## 🔗 **See Also**

- **`agents.md`**: Authoritative reference for agent implementation with canonical patterns
- **`core.md`**: Base class architecture following canonical principles  
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`factory-design-pattern.md`**: Canonical factory implementation for all systems