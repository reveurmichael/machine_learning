# LLM with Chain-of-Thought Reasoning for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines LLM with Chain-of-Thought reasoning patterns.

> **See also:** `agents.md`, `core.md`, `config.md`, `final-decision-10.md`, `factory-design-pattern.md`.

## 🎯 **Core Philosophy: Explicit Step-by-Step Reasoning**

Chain-of-Thought (CoT) reasoning represents a breakthrough in LLM capabilities, enabling models to perform complex reasoning tasks by explicitly working through problems step-by-step. In the Snake Game AI context, CoT enables transparent, interpretable decision-making processes that can be analyzed, debugged, and improved.

### **Design Philosophy**
- **Transparent Reasoning**: Make every step of the decision process explicit and traceable
- **Simple Logging**: All components use print() statements only (per `final-decision-10.md`)
- **Canonical Patterns**: Factory methods use `create()` (never `create_agent()`)
- **Educational Value**: Demonstrate human-like problem-solving approaches with clear examples

## 🎯 **Factory Pattern Implementation (CANONICAL create() METHOD)**
**CRITICAL REQUIREMENT**: All CoT LLM factories MUST use the canonical `create()` method exactly as specified in `final-decision-10.md SUPREME_RULES:

```python
class CoTLLMFactory:
    """
    Factory Pattern for Chain-of-Thought LLM agents following final-decision-10.md SUPREME_RULES
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for reasoning-based LLM agents
    Educational Value: Shows how SUPREME_RULES apply to advanced reasoning systems -
    canonical patterns work regardless of reasoning complexity.
    
    Reference: final-decision-10.md SUPREME_RULES for canonical method naming
    """
    
    _registry = {
        "STEP_BY_STEP": StepByStepAgent,
        "REASONING_TREE": ReasoningTreeAgent,
        "MULTI_STEP": MultiStepAgent,
        "VERBAL_REASONING": VerbalReasoningAgent,
    }
    
    @classmethod
    def create(cls, agent_type: str, **kwargs):  # CANONICAL create() method - SUPREME_RULES
        """Create CoT LLM agent using canonical create() method following final-decision-10.md"""
        agent_class = cls._registry.get(agent_type.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {available}")
        print(f"[CoTLLMFactory] Creating agent: {agent_type}")  # Simple logging - SUPREME_RULES
        return agent_class(**kwargs)

# ❌ FORBIDDEN: Non-canonical method names (violates SUPREME_RULES)
class CoTLLMFactory:
    def create_cot_agent(self, agent_type: str):  # FORBIDDEN - not canonical
        pass
    
    def build_reasoning_agent(self, agent_type: str):  # FORBIDDEN - not canonical
        pass
    
    def make_thinking_agent(self, agent_type: str):  # FORBIDDEN - not canonical
        pass
```

## 🧠 **Chain-of-Thought Architecture**

### **Core Reasoning Components**
```python
# Create CoT agent using canonical factory method
cot_agent = CoTAgentFactory.create("COT_BASIC", name="ReasoningSnake")  # Canonical create()
print(f"[CoTArchitecture] Initialized reasoning agent")  # Simple logging

# Reasoning chain structure follows consistent pattern:
reasoning_steps = [
    "1. Problem Analysis: Assess current game state",
    "2. Goal Identification: Determine immediate objectives", 
    "3. Option Generation: Consider available moves",
    "4. Evaluation: Assess each option against criteria",
    "5. Decision: Select best move with justification"
]
print(f"[CoTArchitecture] Reasoning chain has {len(reasoning_steps)} steps")  # Simple logging
```

### **Advanced CoT Techniques**
- **Self-Consistency**: Generate multiple reasoning chains and choose most consistent
- **Tree of Thoughts**: Explore multiple reasoning branches systematically
- **Iterative Refinement**: Improve reasoning through multiple passes
- **Meta-Reasoning**: Reason about the reasoning process itself

## 🔧 **Implementation Patterns**

### **Core Implementation Components (SUPREME_RULES Compliant)**

### **Base CoT Agent**
```python
class BaseCoTAgent(BaseAgent):
    """
    Base class for Chain-of-Thought LLM agents following final-decision-10.md SUPREME_RULES.
    
    Design Pattern: Template Method Pattern (Canonical Implementation)
    Educational Value: Inherits from BaseAgent to maintain consistency
    while adding reasoning capabilities using canonical factory patterns.
    
    Reference: final-decision-10.md for canonical agent architecture
    """
    
    def __init__(self, name: str, grid_size: int, 
                 reasoning_strategy: str = "STEP_BY_STEP"):
        super().__init__(name, grid_size)
        
        self.reasoning_strategy = reasoning_strategy
        self.reasoning_history = []
        self.thought_process = []
        
        print(f"[{name}] CoT Agent initialized with {reasoning_strategy}")  # Simple logging - SUPREME_RULES
    
    def plan_move(self, game_state: dict) -> str:
        """Plan move using Chain-of-Thought reasoning with simple logging throughout"""
        print(f"[{self.name}] Starting CoT reasoning process")  # Simple logging
        
        # Generate reasoning steps using canonical patterns
        reasoning_steps = self._generate_reasoning_steps(game_state)
        self.reasoning_history.append(reasoning_steps)
        
        # Extract final decision
        move = self._extract_decision_from_reasoning(reasoning_steps)
        
        print(f"[{self.name}] CoT decided: {move}")  # Simple logging
        return move
    
    def _generate_reasoning_steps(self, game_state: dict) -> list:
        """Generate reasoning steps (override in subclasses)"""
        raise NotImplementedError("Subclasses must implement reasoning generation")
    
    def _extract_decision_from_reasoning(self, reasoning_steps: list) -> str:
        """Extract final decision from reasoning steps with simple logging"""
        print(f"[{self.name}] Extracting decision from {len(reasoning_steps)} reasoning steps")  # Simple logging
        
        # Simple decision extraction logic
        for step in reversed(reasoning_steps):
            if "move" in step.lower() or "direction" in step.lower():
                # Extract move from reasoning step
                move = self._parse_move_from_text(step)
                if move:
                    print(f"[{self.name}] Decision extracted: {move}")  # Simple logging
                    return move
        
        print(f"[{self.name}] No clear decision found, defaulting to UP")  # Simple logging
        return "UP"
```

### **Step-by-Step Agent Implementation**
```python
class StepByStepAgent(BaseCoTAgent):
    """
    Step-by-step reasoning agent following canonical patterns.
    
    Educational Value: Shows how canonical factory patterns scale
    to complex reasoning systems while maintaining simple logging.
    """
    
    def __init__(self, name: str, grid_size: int, **kwargs):
        super().__init__(name, grid_size, reasoning_strategy="STEP_BY_STEP", **kwargs)
        self.step_counter = 0
        print(f"[{name}] Step-by-Step reasoning agent ready")  # Simple logging
    
    def _generate_reasoning_steps(self, game_state: dict) -> list:
        """Generate step-by-step reasoning with simple logging"""
        print(f"[{self.name}] Generating step-by-step reasoning")  # Simple logging
        
        self.step_counter += 1
        reasoning_steps = []
        
        # Step 1: Analyze current position
        position_analysis = self._analyze_position(game_state)
        reasoning_steps.append(f"Step {self.step_counter}.1: {position_analysis}")
        
        # Step 2: Identify available moves
        available_moves = self._identify_available_moves(game_state)
        reasoning_steps.append(f"Step {self.step_counter}.2: {available_moves}")
        
        # Step 3: Evaluate each move
        move_evaluation = self._evaluate_moves(game_state, available_moves)
        reasoning_steps.append(f"Step {self.step_counter}.3: {move_evaluation}")
        
        # Step 4: Select best move
        best_move = self._select_best_move(move_evaluation)
        reasoning_steps.append(f"Step {self.step_counter}.4: Best move is {best_move}")
        
        print(f"[{self.name}] Generated {len(reasoning_steps)} reasoning steps")  # Simple logging
        return reasoning_steps
```

### **Reasoning Tree Agent Implementation**
```python
class ReasoningTreeAgent(BaseCoTAgent):
    """
    Tree-based reasoning agent following canonical patterns.
    
    Educational Value: Demonstrates how canonical patterns enable
    consistent implementation across different reasoning architectures.
    """
    
    def __init__(self, name: str, grid_size: int, **kwargs):
        super().__init__(name, grid_size, reasoning_strategy="REASONING_TREE", **kwargs)
        self.tree_depth = 3
        print(f"[{name}] Reasoning Tree agent initialized with depth {self.tree_depth}")  # Simple logging
    
    def _generate_reasoning_steps(self, game_state: dict) -> list:
        """Generate tree-based reasoning with simple logging"""
        print(f"[{self.name}] Building reasoning tree")  # Simple logging
        
        reasoning_tree = self._build_reasoning_tree(game_state, depth=self.tree_depth)
        reasoning_steps = self._flatten_tree_to_steps(reasoning_tree)
        
        print(f"[{self.name}] Tree built with {len(reasoning_steps)} reasoning steps")  # Simple logging
        return reasoning_steps
    
    def _build_reasoning_tree(self, game_state: dict, depth: int) -> dict:
        """Build reasoning tree structure with simple logging"""
        print(f"[{self.name}] Building tree at depth {depth}")  # Simple logging
        
        if depth == 0:
            return {"type": "leaf", "analysis": "Reached maximum depth"}
        
        # Build tree structure
        tree = {
            "type": "node",
            "analysis": self._analyze_current_state(game_state),
            "children": []
        }
        
        # Add child nodes for each possible move
        for move in ["UP", "DOWN", "LEFT", "RIGHT"]:
            child_state = self._simulate_move(game_state, move)
            child_tree = self._build_reasoning_tree(child_state, depth - 1)
            tree["children"].append({"move": move, "tree": child_tree})
        
        return tree
```

## 🚀 **Advanced Capabilities**

### **Multi-Agent CoT Integration**
```python
# Create multiple CoT agents using canonical factory method
basic_agent = CoTAgentFactory.create("COT_BASIC")  # Canonical create()
selfcheck_agent = CoTAgentFactory.create("COT_SELFCHECK")  # Canonical create()
tree_agent = CoTAgentFactory.create("COT_TREE")  # Canonical create()

print(f"[CoTIntegration] Created {3} different CoT agents")  # Simple logging

# Compare reasoning approaches
results = {}
for agent_name, agent in [("basic", basic_agent), ("selfcheck", selfcheck_agent), ("tree", tree_agent)]:
    decision = agent.plan_move(game_state)
    results[agent_name] = decision
    print(f"[CoTIntegration] {agent_name} decided: {decision}")  # Simple logging
```

### **Reasoning Quality Assessment**
```python
class CoTReasoningEvaluator:
    """
    Evaluator for assessing quality of Chain-of-Thought reasoning
    
    Simple logging: All evaluation uses print() statements only
    """
    
    def evaluate_reasoning_quality(self, reasoning_chain: str) -> float:
        """Assess overall quality of reasoning chain"""
        coherence_score = self._assess_coherence(reasoning_chain)
        completeness_score = self._assess_completeness(reasoning_chain)
        logic_score = self._assess_logic(reasoning_chain)
        
        overall_score = (coherence_score + completeness_score + logic_score) / 3
        print(f"[CoTEvaluator] Reasoning quality score: {overall_score}")  # Simple logging
        return overall_score
    
    def analyze_reasoning_patterns(self, reasoning_history: list):
        """Analyze patterns in reasoning chains over time"""
        print(f"[CoTEvaluator] Analyzing {len(reasoning_history)} reasoning chains")  # Simple logging
        patterns = self._extract_patterns(reasoning_history)
        print(f"[CoTEvaluator] Found {len(patterns)} reasoning patterns")  # Simple logging
        return patterns
```

## 📊 **Integration with Extensions**

### **With Heuristics**
```python
# Compare CoT reasoning with heuristic algorithms
heuristic_agent = HeuristicFactory.create("BFS")  # Canonical create()
cot_agent = CoTAgentFactory.create("COT_BASIC")  # Canonical create()

heuristic_move = heuristic_agent.plan_move(game_state)
cot_move = cot_agent.plan_move(game_state)

print(f"[Integration] Heuristic: {heuristic_move}, CoT: {cot_move}")  # Simple logging
```

### **With Supervised Learning**
```python
# Train models to predict reasoning quality
quality_predictor = SupervisedFactory.create("REASONING_QUALITY_PREDICTOR")  # Canonical create()
predicted_quality = quality_predictor.predict(reasoning_chain)
print(f"[Integration] Predicted reasoning quality: {predicted_quality}")  # Simple logging
```

## 🎓 **Educational Value**

### **Learning Objectives**
- **Transparent AI**: Understanding how AI systems can show their reasoning process
- **Problem Decomposition**: Learning to break complex problems into manageable steps
- **Reasoning Quality**: Assessing the quality and validity of reasoning chains
- **Simple Logging**: All examples demonstrate print()-based logging patterns

### **Research Applications**
- **Reasoning Analysis**: Study how different CoT approaches affect performance
- **Error Detection**: Identify common reasoning failures and their causes
- **Reasoning Transfer**: Apply successful reasoning patterns to new problems

## 📊 **Performance Monitoring**

### **Reasoning Metrics**
```python
class CoTMetrics:
    """
    Metrics collection for Chain-of-Thought reasoning performance
    
    Simple logging: All metrics use print() statements only
    """
    
    def track_reasoning_time(self, agent, game_state):
        """Track time spent on reasoning process"""
        start_time = time.time()
        decision = agent.plan_move(game_state)
        reasoning_time = time.time() - start_time
        print(f"[CoTMetrics] Reasoning completed in {reasoning_time:.2f} seconds")  # Simple logging
        return reasoning_time, decision
    
    def measure_reasoning_depth(self, reasoning_chain: str) -> int:
        """Measure depth of reasoning chain"""
        steps = reasoning_chain.count("Step")
        print(f"[CoTMetrics] Reasoning depth: {steps} steps")  # Simple logging
        return steps
```

## 📋 **SUPREME_RULES Implementation Checklist for Chain-of-Thought LLMs**

### **Mandatory Requirements**
- [ ] **Canonical Method**: All factories use `create()` method exactly (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses print() statements only for all CoT operations (final-decision-10.md compliance)
- [ ] **GOOD_RULES Reference**: References `final-decision-10.md` in all CoT documentation
- [ ] **Pattern Consistency**: Follows canonical patterns across all CoT implementations

### **CoT-Specific Standards**
- [ ] **Reasoning Systems**: Canonical factory patterns for all reasoning components
- [ ] **Step Generation**: Canonical factory patterns for all reasoning step types
- [ ] **Decision Extraction**: Canonical patterns for all decision extraction systems
- [ ] **Verification Systems**: Simple logging for all reasoning verification operations

### **Educational Integration**
- [ ] **Clear Examples**: Simple examples using canonical `create()` method for CoT systems
- [ ] **Pattern Explanation**: Clear explanation of canonical patterns in reasoning context
- [ ] **Best Practices**: Demonstration of SUPREME_RULES in advanced reasoning systems
- [ ] **Learning Value**: Easy to understand canonical patterns regardless of reasoning complexity

---

**Chain-of-Thought LLMs represent advanced reasoning capabilities while maintaining strict compliance with `final-decision-10.md` SUPREME_RULES, proving that canonical patterns and simple logging provide consistent foundations across all AI complexity levels.**

## 🔗 **See Also**

- **`agents.md`**: Authoritative reference for agent implementation with canonical patterns
- **`core.md`**: Base class architecture following canonical principles
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`factory-design-pattern.md`**: Canonical factory implementation for all systems
