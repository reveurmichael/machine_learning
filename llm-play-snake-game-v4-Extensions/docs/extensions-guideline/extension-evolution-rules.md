# Extension Evolution Rules

> **Authoritative Reference**: This document establishes the definitive rules for extension evolution from v0.01 through v0.04.

> **SUPREME_RULES**: Both `heuristics-v0.03` and `heuristics-v0.04` are widely used depending on use cases and scenarios. For supervised learning and other general purposes, both versions can be used. For LLM fine-tuning, only `heuristics-v0.04` will be used. The CSV format is **NOT legacy** - it's actively used and valuable for supervised learning.

## 🎯 **Core Evolution Philosophy**

Extensions evolve through **natural software progression** while maintaining **algorithmic stability** and **educational value**. Each version builds upon the previous while introducing new capabilities.

## 📋 **Version Evolution Matrix**

| Version | Purpose | Key Changes | Stability Rules |
|---------|---------|-------------|-----------------|
| **v0.01** | Proof of Concept | Single algorithm, minimal complexity | Basic structure |
| **v0.02** | Multi-Algorithm | Organized agents/, factory patterns | Core algorithms stable |
| **v0.03** | Web Interface | Streamlit app, dataset generation | Agents copied exactly |
| **v0.04** | Language Generation | JSONL datasets (heuristics only) | All v0.03 functionality preserved |

## 🔒 **Stability Rules by Version Transition**

### **v0.01 → v0.02: Allowed Breaking Changes**
```python
# ✅ ALLOWED: Major structural changes
- Add agents/ directory
- Implement factory patterns
- Add --algorithm command-line argument
- Reorganize file structure
- Add new algorithms

# ✅ ALLOWED: Interface changes
- Change main.py signature
- Add new configuration options
- Modify data structures
```

### **v0.02 → v0.03: Core Stability Required**
```python
# 🔒 REQUIRED: Copy agents/ exactly
agents/ directory must be identical to v0.02
- Same file names and class names
- Same factory registrations
- Same algorithm implementations
- Same method signatures

# ➕ ALLOWED: Add web enhancements
- Streamlit app.py
- Dashboard components
- Web-specific utilities
- Monitoring wrappers
- UI integration helpers

# ❌ FORBIDDEN: Core algorithm changes
- Modify existing agent logic
- Change factory registration names
- Break existing interfaces
- Remove agent files
```

### **v0.03 → v0.04: Maximum Stability (Heuristics Only)**
```python
# 🔒 REQUIRED: Copy v0.03 exactly
- All v0.03 functionality preserved
- Same agents/ directory structure (copied exactly from v0.03)
- Same web interface capabilities
- Same dataset generation (CSV)

# ➕ ALLOWED: Add JSONL capabilities
- JSONL dataset generation
- Language explanation features
- LLM fine-tuning utilities
- Enhanced reasoning output

# ❌ FORBIDDEN: Any breaking changes
- All v0.03 features must work unchanged
- No algorithm modifications
- No interface changes
```

## 🏗️ **Directory Structure Evolution**

### **v0.01 Template**
```
extensions/{algorithm}-v0.01/
├── __init__.py
├── main.py                    # Simple entry point
├── agent_{primary}.py         # Single algorithm
├── game_logic.py
├── game_manager.py
└── README.md
```

### **v0.02 Template**
```
extensions/{algorithm}-v0.02/
├── __init__.py
├── main.py                    # --algorithm argument
├── game_logic.py
├── game_manager.py
├── game_data.py               # NEW
├── agents/                    # NEW: Organized structure
│   ├── __init__.py           # Factory pattern
│   ├── agent_{type1}.py
│   ├── agent_{type2}.py
│   └── agent_{type3}.py
└── README.md
```

### **v0.03 Template**
```
extensions/{algorithm}-v0.03/
├── app.py                     # NEW: Streamlit app
├── dashboard/                 # NEW: UI components
├── scripts/                   # NEW: CLI tools
├── agents/                    # 🔒 Copied from v0.02
├── game_logic.py
├── game_manager.py
├── game_data.py
└── {algorithm}_config.py      # NEW
```

### **v0.04 Template (Heuristics Only)**
```
extensions/heuristics-v0.04/
├── app.py                     # Enhanced with JSONL
├── dashboard/                 # Enhanced with JSONL
├── scripts/                   # Enhanced with JSONL
├── agents/                    # 🔒 Copied from v0.03
├── game_logic.py              # Enhanced with JSONL
├── game_manager.py            # Enhanced with JSONL
├── game_data.py               # Enhanced with JSONL
└── heuristic_config.py
```

## 🎯 **Algorithm Stability Enforcement**

### **Required Stability Checks**
```python
# Validation script for v0.02 → v0.03 transition
def validate_agent_stability(v02_path, v03_path):
    """Ensure agents/ directory is copied exactly"""
    
    # Check file existence
    v02_agents = list_files(f"{v02_path}/agents/")
    v03_agents = list_files(f"{v03_path}/agents/")
    
    if v02_agents != v03_agents:
        raise ValidationError("Agent files must be identical")
    
    # Check factory registrations
    v02_factory = load_factory(f"{v02_path}/agents/__init__.py")
    v03_factory = load_factory(f"{v03_path}/agents/__init__.py")
    
    if v02_factory.registry != v03_factory.registry:
        raise ValidationError("Factory registrations must be identical")
```

### **Allowed Enhancements**
```python
# ✅ Allowed in v0.03: Web-specific enhancements
class BFSAgentWebOptimized(BFSAgent):
    """Web interface optimization wrapper"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.web_monitoring = WebMonitoring()
    
    def plan_move(self, game_state):
        # Original BFS logic unchanged
        result = super().plan_move(game_state)
        # Add web monitoring
        self.web_monitoring.record_decision(result)
        return result
```

## 📊 **Version Compatibility Matrix**

| Component | v0.01→v0.02 | v0.02→v0.03 | v0.03→v0.04 |
|-----------|-------------|-------------|-------------|
| **Core Algorithms** | ✅ Can change | 🔒 Copy exactly | 🔒 Copy exactly |
| **Factory Patterns** | ✅ Can add | 🔒 Stable | 🔒 Stable |
| **File Structure** | ✅ Can reorganize | ✅ Can add web | ✅ Can add JSONL |
| **CLI Interface** | ✅ Can change | ✅ Can enhance | ✅ Can enhance |
| **Data Formats** | ✅ Can change | ✅ Can add | ✅ Can add JSONL |

## 🚫 **Forbidden Patterns**

### **Breaking Algorithm Stability**
```python
# ❌ FORBIDDEN: Modify core algorithm in v0.03
class BFSAgent(BaseAgent):
    def plan_move(self, game_state):
        # ❌ Changed from v0.02 implementation
        return self.new_algorithm(game_state)  # BREAKING CHANGE
```

### **Removing Required Components**
```python
# ❌ FORBIDDEN: Remove agent files in v0.03
# agents/agent_bfs.py  # ❌ DELETED - BREAKING CHANGE
```

### **Changing Factory Registrations**
```python
# ❌ FORBIDDEN: Change registration names
_registry = {
    "BFS_NEW": BFSAgent,  # ❌ Changed from "BFS" - BREAKING CHANGE
}
```

## 🔍 **Compliance Validation**

### **Automated Checks**
```python
# Required validation for all version transitions
def validate_evolution_compliance(old_version, new_version):
    """Validate extension evolution compliance"""
    
    if new_version == "0.03":
        validate_agent_stability(old_version, new_version)
        validate_factory_stability(old_version, new_version)
    
    if new_version == "0.04":
        validate_v03_functionality_preserved(new_version)
        validate_jsonl_capabilities_added(new_version)
```

### **Manual Review Checklist**
- [ ] Core algorithms unchanged (v0.02→v0.03, v0.03→v0.04)
- [ ] Factory registrations stable
- [ ] Required functionality preserved
- [ ] New capabilities properly added
- [ ] Documentation updated
- [ ] Tests pass for all versions

---

**These evolution rules ensure consistent, stable extension development while enabling natural software progression and maintaining educational value.**

## 🎯 **SUPREME_RULES: Version Selection Guidelines**

- **For supervised learning**: Use CSV from either heuristics-v0.03 or heuristics-v0.04 (both widely used)
- **For LLM fine-tuning**: Use JSONL from heuristics-v0.04 only
- **For research**: Use both formats from heuristics-v0.04
- **CSV is ACTIVE**: Not legacy - actively used for supervised learning
- **JSONL is ADDITIONAL**: New capability for LLM fine-tuning (heuristics-v0.04 only)

**Both heuristics-v0.03 and heuristics-v0.04 are widely used depending on use cases and scenarios.** 