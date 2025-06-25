# Extension Evolution Rules

> **Authoritative Reference**: This document defines the **explicit rules** for extension evolution across versions. It replaces all ambiguous language about "copy exactly", "enhancements allowed", etc.

## 🎯 **Core Evolution Philosophy**

Extension evolution follows **algorithmic stability** principles:
- **Core algorithms remain unchanged** once established in v0.02
- **Interfaces can be extended** but not modified
- **New functionality can be added** without breaking existing code
- **Version progression** serves specific purposes (CLI → Web → Data Generation)

## 📋 **Version Evolution Rules Matrix**

| Transition | Core Algorithms | Factory Registry | Interfaces | New Features | Breaking Changes |
|------------|----------------|------------------|------------|--------------|------------------|
| **v0.01 → v0.02** | ✅ Add new algorithms | ✅ Create factory | ✅ Define interfaces | ✅ Multiple algorithms | ✅ Allowed |
| **v0.02 → v0.03** | 🔒 **Copy exactly** | 🔒 **Stable** | ➕ Can extend | ✅ Web interface | ❌ **Forbidden** |
| **v0.03 → v0.04*** | 🔒 **Copy exactly** | 🔒 **Stable** | ➕ Can extend | ✅ JSONL generation | ❌ **Forbidden** |

***v0.04 only exists for heuristics extensions**

## 🚫 **Explicitly Forbidden Changes (v0.02+)**

### **Breaking Changes - NEVER ALLOWED**
```python
# ❌ FORBIDDEN: Modifying core algorithm logic
class BFSAgent(BaseAgent):
    def plan_move(self, game_state):
        # Changing the BFS algorithm implementation
        return modified_bfs_logic(game_state)  # FORBIDDEN!

# ❌ FORBIDDEN: Changing factory registration names
class AgentFactory:
    _registry = {
        "BFS_NEW": BFSAgent,  # FORBIDDEN! Was "BFS" in v0.02
    }

# ❌ FORBIDDEN: Removing existing agent files
# agents/agent_bfs.py → DELETED  # FORBIDDEN!

# ❌ FORBIDDEN: Changing public method signatures
class BFSAgent(BaseAgent):
    def plan_move(self, game_state, new_param):  # FORBIDDEN! Added parameter
        pass
```

### **Interface Violations - NEVER ALLOWED**
```python
# ❌ FORBIDDEN: Breaking BaseAgent interface
class BFSAgent(BaseAgent):
    # Missing required method - FORBIDDEN!
    # def plan_move(self, game_state): pass
    
# ❌ FORBIDDEN: Changing return types
class BFSAgent(BaseAgent):
    def plan_move(self, game_state) -> List[str]:  # Was str, now List[str] - FORBIDDEN!
        return ["UP", "RIGHT"]  # Breaking change!
```

## ✅ **Explicitly Allowed Changes (v0.02+)**

### **Extensions and Enhancements - PERMITTED**
```python
# ✅ ALLOWED: Adding new agent variants
# agents/agent_bfs_enhanced.py - NEW FILE
class BFSEnhancedAgent(BFSAgent):  # Inherits from stable BFS
    def plan_move(self, game_state):
        base_move = super().plan_move(game_state)  # Uses stable BFS
        return self.add_safety_check(base_move)    # Adds enhancement

# ✅ ALLOWED: Adding monitoring utilities
# agents/web_monitoring_utils.py - NEW FILE
class AgentMonitor:
    def track_performance(self, agent, game_state):
        # Web interface monitoring - NEW functionality
        pass

# ✅ ALLOWED: Adding new methods (not modifying existing)
class BFSAgent(BaseAgent):
    def plan_move(self, game_state):  # 🔒 Unchanged
        return self._bfs_implementation(game_state)
    
    def get_performance_metrics(self):  # ✅ NEW method
        return self.performance_data  # Added for web interface
```

### **Factory Extensions - PERMITTED**
```python
# ✅ ALLOWED: Adding new registrations (keeping existing)
class AgentFactory:
    _registry = {
        "BFS": BFSAgent,              # 🔒 Unchanged
        "ASTAR": AStarAgent,          # 🔒 Unchanged
        "BFS_ENHANCED": BFSEnhancedAgent,  # ✅ NEW addition
    }
```

## 🔧 **Version-Specific Evolution Guidelines**

### **v0.02 → v0.03: Web Interface Integration**
**Purpose**: Add web interface without breaking CLI functionality

**Required Stability**:
```python
# 🔒 MUST remain identical
agents/agent_bfs.py           # Core algorithm unchanged
agents/agent_astar.py         # Core algorithm unchanged
agents/__init__.py            # Factory registry unchanged
```

**Allowed Additions**:
```python
# ✅ NEW: Web interface components
dashboard/tab_main.py         # Web UI
scripts/replay_web.py         # Web replay
agents/agent_bfs_web_optimized.py  # Web-specific optimizations
```

### **v0.03 → v0.04: Language Generation (Heuristics Only)**
**Purpose**: Add JSONL generation for LLM fine-tuning

**Required Stability**:
```python
# 🔒 MUST remain identical to v0.03
agents/                       # ALL agent files unchanged
dashboard/                    # Web interface unchanged
scripts/main.py              # CLI unchanged
```

**Allowed Additions**:
```python
# ✅ NEW: Language generation only
scripts/generate_jsonl_dataset.py   # JSONL generation
agents/reasoning_mixin.py           # Add reasoning capability
```

## 🏗️ **Architectural Patterns for Safe Evolution**

### **Decorator Pattern for Enhancements**
```python
# ✅ SAFE: Enhance without modifying
class WebOptimizedBFS(BFSAgent):
    """Web-optimized wrapper for BFS with monitoring"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = WebMonitor()
    
    def plan_move(self, game_state):
        move = super().plan_move(game_state)  # 🔒 Uses stable BFS
        self.monitor.log_decision(move)       # ✅ Adds monitoring
        return move
```

### **Composition for New Features**
```python
# ✅ SAFE: Add features via composition
class ReasoningCapableBFS:
    def __init__(self):
        self.bfs_agent = BFSAgent()      # 🔒 Uses stable BFS
        self.reasoner = ReasoningEngine()  # ✅ NEW component
    
    def plan_move_with_reasoning(self, game_state):
        move = self.bfs_agent.plan_move(game_state)  # 🔒 Stable
        reasoning = self.reasoner.explain(move)      # ✅ NEW feature
        return move, reasoning
```

## 📊 **Compliance Validation**

### **Automated Checks (Required)**
```python
# extensions/common/validation/evolution_validator.py

def validate_evolution_compliance(v_old: str, v_new: str):
    """Validate that extension evolution follows rules"""
    
    # Check core files unchanged
    for agent_file in get_core_agent_files(v_old):
        assert files_identical(
            f"{v_old}/{agent_file}", 
            f"{v_new}/{agent_file}"
        ), f"Core agent {agent_file} was modified - FORBIDDEN!"
    
    # Check factory registry stability
    old_registry = get_factory_registry(v_old)
    new_registry = get_factory_registry(v_new)
    
    for name, agent_class in old_registry.items():
        assert name in new_registry, f"Registry entry {name} removed - FORBIDDEN!"
        assert new_registry[name] == agent_class, f"Registry {name} changed - FORBIDDEN!"
```

### **Manual Review Checklist**
- [ ] Are all v0.02 agent files byte-identical in v0.03?
- [ ] Do factory registrations maintain exact same names?
- [ ] Are all public method signatures unchanged?
- [ ] Do new features use composition/decoration patterns?
- [ ] Are breaking changes properly justified and documented?

## 🎯 **Benefits of Strict Evolution Rules**

### **Stability Guarantees**
- **Reproducibility**: v0.02 experiments remain valid
- **Reliability**: Core algorithms never break
- **Compatibility**: Extensions can reference stable interfaces

### **Educational Value**
- **Clear Progression**: Students see natural software evolution
- **Design Patterns**: Demonstrates safe extension techniques
- **Best Practices**: Shows how to add features without breaking existing code

---

**These explicit rules eliminate ambiguity and ensure stable, predictable extension evolution while enabling legitimate enhancements and new functionality.** 