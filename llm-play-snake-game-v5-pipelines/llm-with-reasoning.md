# LLM with Reasoning for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines LLM with reasoning standards.

> **See also:** `llm-with-cot.md`, `agentic-llms.md`, SUPREME_RULES from `final-decision-10.md`, `standalone.md`.

## 🎯 **Core Philosophy: Intelligent Reasoning**

LLM with reasoning enables **intelligent decision-making** for Snake Game AI through advanced reasoning capabilities. This approach follows SUPREME_RULES from `final-decision-10.md` and uses canonical `create()` methods throughout.

### **Educational Value**
- **Intelligent Reasoning**: Understanding how to implement advanced reasoning systems
- **Decision Making**: Learning to make intelligent decisions with clear logic
- **Problem Solving**: Solving complex game situations through reasoning
- **Canonical Patterns**: All implementations use canonical `create()` method per SUPREME_RULES

## 🏗️ **Reasoning Factory (CANONICAL)**

### **Reasoning Factory (SUPREME_RULES Compliant)**
```python
from utils.factory_utils import SimpleFactory

class ReasoningFactory:
    """
    Factory Pattern for LLM reasoning following SUPREME_RULES from final-decision-10.md
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for reasoning systems
    Educational Value: Shows how SUPREME_RULES apply to intelligent reasoning
    """
    
    _registry = {
        "LOGICAL": LogicalReasoning,
        "SPATIAL": SpatialReasoning,
        "STRATEGIC": StrategicReasoning,
        "ADAPTIVE": AdaptiveReasoning,
    }
    
    @classmethod
    def create(cls, reasoning_type: str, **kwargs):  # CANONICAL create() method per SUPREME_RULES
        """Create reasoning system using canonical create() method following SUPREME_RULES from final-decision-10.md"""
        reasoning_class = cls._registry.get(reasoning_type.upper())
        if not reasoning_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown reasoning type: {reasoning_type}. Available: {available}")
        print(f"[ReasoningFactory] Creating reasoning: {reasoning_type}")  # SUPREME_RULES compliant logging
        return reasoning_class(**kwargs)
```

### **Logical Reasoning Implementation**
```python
class LogicalReasoning:
    """
    Logical reasoning following SUPREME_RULES.
    
    Design Pattern: Strategy Pattern
    Purpose: Implements logical reasoning for game decisions
    Educational Value: Shows logical reasoning with canonical patterns
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logical_rules = self._initialize_logical_rules()
        print(f"[LogicalReasoning] Initialized with {len(self.logical_rules)} rules")  # SUPREME_RULES compliant logging
    
    def reason_about_move(self, game_state: dict) -> dict:
        """Apply logical reasoning to determine the best move"""
        print(f"[LogicalReasoning] Starting logical reasoning")  # SUPREME_RULES compliant logging
        
        # Apply logical rules
        rule_results = []
        for rule_name, rule_func in self.logical_rules.items():
            result = rule_func(game_state)
            rule_results.append({
                'rule': rule_name,
                'result': result,
                'confidence': self._calculate_rule_confidence(result)
            })
        
        # Synthesize results
        final_decision = self._synthesize_logical_results(rule_results)
        print(f"[LogicalReasoning] Logical reasoning completed: {final_decision}")  # SUPREME_RULES compliant logging
        
        return {
            'reasoning_type': 'logical',
            'rule_results': rule_results,
            'final_move': final_decision,
            'confidence': self._calculate_overall_confidence(rule_results)
        }
    
    def _initialize_logical_rules(self) -> dict:
        """Initialize logical reasoning rules"""
        return {
            'avoid_collision': self._rule_avoid_collision,
            'seek_apple': self._rule_seek_apple,
            'maintain_space': self._rule_maintain_space,
            'follow_path': self._rule_follow_path,
        }
    
    def _rule_avoid_collision(self, game_state: dict) -> dict:
        """Logical rule: avoid collision with walls and self"""
        head = game_state['snake_positions'][0]
        grid_size = game_state['grid_size']
        
        # Check wall collision
        if head[0] < 0 or head[0] >= grid_size or head[1] < 0 or head[1] >= grid_size:
            return {'valid': False, 'reason': 'wall_collision'}
        
        # Check self collision
        if head in game_state['snake_positions'][1:]:
            return {'valid': False, 'reason': 'self_collision'}
        
        return {'valid': True, 'reason': 'no_collision'}
```

## 📊 **Simple Logging for Reasoning Operations**

All reasoning operations must use simple print statements as mandated by SUPREME_RULES from `final-decision-10.md`:

```python
# ✅ CORRECT: Simple logging for reasoning (SUPREME_RULES compliance)
def run_reasoning_pipeline(reasoning_type: str, game_state: dict):
    print(f"[ReasoningRunner] Starting {reasoning_type} reasoning")  # SUPREME_RULES compliant logging
    
    reasoning_system = ReasoningFactory.create(reasoning_type)  # CANONICAL create() method per SUPREME_RULES
    result = reasoning_system.reason_about_move(game_state)
    
    print(f"[ReasoningRunner] Reasoning completed with confidence: {result['confidence']:.3f}")  # SUPREME_RULES compliant logging
    return result
```

## 🎯 **Reasoning Types**

### **Spatial Reasoning**
```python
class SpatialReasoning:
    """
    Spatial reasoning following SUPREME_RULES.
    
    Design Pattern: Template Method Pattern
    Purpose: Implements spatial reasoning for game navigation
    Educational Value: Shows spatial reasoning with canonical patterns
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.spatial_analysis = SpatialAnalysis()
        print(f"[SpatialReasoning] Initialized spatial reasoning")  # SUPREME_RULES compliant logging
    
    def reason_about_move(self, game_state: dict) -> dict:
        """Apply spatial reasoning to determine the best move"""
        print(f"[SpatialReasoning] Starting spatial reasoning")  # SUPREME_RULES compliant logging
        
        # Analyze spatial relationships
        spatial_analysis = self.spatial_analysis.analyze_spatial_relationships(game_state)
        
        # Calculate distances and directions
        distance_analysis = self._analyze_distances(game_state)
        
        # Determine optimal spatial path
        spatial_path = self._determine_spatial_path(spatial_analysis, distance_analysis)
        
        print(f"[SpatialReasoning] Spatial reasoning completed")  # SUPREME_RULES compliant logging
        
        return {
            'reasoning_type': 'spatial',
            'spatial_analysis': spatial_analysis,
            'distance_analysis': distance_analysis,
            'spatial_path': spatial_path,
            'final_move': spatial_path['next_move'],
            'confidence': spatial_path['confidence']
        }
    
    def _analyze_distances(self, game_state: dict) -> dict:
        """Analyze distances between game elements"""
        head = game_state['snake_positions'][0]
        apple = game_state['apple_position']
        
        # Calculate Manhattan distance to apple
        apple_distance = abs(head[0] - apple[0]) + abs(head[1] - apple[1])
        
        # Calculate distances to walls
        wall_distances = {
            'up': head[1],
            'down': game_state['grid_size'] - 1 - head[1],
            'left': head[0],
            'right': game_state['grid_size'] - 1 - head[0]
        }
        
        return {
            'apple_distance': apple_distance,
            'wall_distances': wall_distances
        }
```

### **Strategic Reasoning**
```python
class StrategicReasoning:
    """
    Strategic reasoning following SUPREME_RULES.
    
    Design Pattern: Strategy Pattern
    Purpose: Implements strategic reasoning for long-term planning
    Educational Value: Shows strategic reasoning with canonical patterns
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.strategy_planner = StrategyPlanner()
        print(f"[StrategicReasoning] Initialized strategic reasoning")  # SUPREME_RULES compliant logging
    
    def reason_about_move(self, game_state: dict) -> dict:
        """Apply strategic reasoning to determine the best move"""
        print(f"[StrategicReasoning] Starting strategic reasoning")  # SUPREME_RULES compliant logging
        
        # Analyze current strategy
        current_strategy = self._analyze_current_strategy(game_state)
        
        # Plan long-term strategy
        long_term_strategy = self.strategy_planner.plan_long_term_strategy(game_state)
        
        # Determine immediate tactical move
        tactical_move = self._determine_tactical_move(game_state, current_strategy, long_term_strategy)
        
        print(f"[StrategicReasoning] Strategic reasoning completed")  # SUPREME_RULES compliant logging
        
        return {
            'reasoning_type': 'strategic',
            'current_strategy': current_strategy,
            'long_term_strategy': long_term_strategy,
            'tactical_move': tactical_move,
            'final_move': tactical_move['move'],
            'confidence': tactical_move['confidence']
        }
    
    def _analyze_current_strategy(self, game_state: dict) -> dict:
        """Analyze the current strategic situation"""
        snake_length = len(game_state['snake_positions'])
        available_space = self._calculate_available_space(game_state)
        apple_position = game_state['apple_position']
        
        # Determine strategic phase
        if snake_length < 5:
            strategy_phase = 'growth'
        elif available_space < 0.3:
            strategy_phase = 'survival'
        else:
            strategy_phase = 'optimization'
        
        return {
            'phase': strategy_phase,
            'snake_length': snake_length,
            'available_space': available_space,
            'apple_position': apple_position
        }
```

## 🎓 **Educational Applications with Canonical Patterns**

### **Reasoning Understanding**
- **Intelligent Reasoning**: Understanding how to implement advanced reasoning systems using canonical factory methods
- **Decision Making**: Learning to make intelligent decisions with clear logic and simple logging
- **Problem Solving**: Solving complex game situations through reasoning using canonical patterns
- **Strategy Development**: Developing strategic thinking following SUPREME_RULES

### **Reasoning Benefits**
- **Intelligence**: Advanced reasoning capabilities that follow SUPREME_RULES
- **Adaptability**: Can adapt reasoning to different game situations
- **Transparency**: Clear reasoning process with canonical factory methods
- **Educational Value**: Clear examples of intelligent reasoning following SUPREME_RULES

## 📋 **SUPREME_RULES Implementation Checklist for Reasoning**

### **Mandatory Requirements**
- [ ] **Canonical Method**: All factories use `create()` method exactly (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses print() statements only for all reasoning operations (SUPREME_RULES compliance)
- [ ] **Reasoning Integration**: Proper integration with LLM systems
- [ ] **Pattern Consistency**: Follows canonical patterns across all reasoning implementations

### **Reasoning-Specific Standards**
- [ ] **Intelligent Reasoning**: Advanced reasoning capabilities with canonical factory methods
- [ ] **Decision Making**: Clear, logical decision-making process
- [ ] **Strategy Development**: Long-term strategic thinking
- [ ] **Documentation**: Clear reasoning explanations following SUPREME_RULES

---

**LLM with reasoning enables intelligent decision-making for Snake Game AI while maintaining strict SUPREME_RULES from `final-decision-10.md` compliance and educational value.**

## 🔗 **See Also**

- **`llm-with-cot.md`**: Chain-of-Thought reasoning standards
- **`agentic-llms.md`**: Agentic LLM standards
- **SUPREME_RULES from `final-decision-10.md`**: Governance system and canonical standards
- **`standalone.md`**: Standalone principle implementation
