# Final Decision 0: Architecture Series Index & Meta-Guidelines

> **SUPREME AUTHORITY**: This document series represents the single source of truth for all architectural and implementation decisions across the Snake Game AI project.

## 🎯 **Purpose and Authority**

The **Final Decision Series (0-10)** establishes definitive architectural standards for the Snake Game AI project. These documents collectively form the authoritative reference that supersedes all other documentation when conflicts arise. Each decision is carefully designed to be self-contained while maintaining coherence with the entire series.

### **Authority Hierarchy**
1. **Final Decision Series** (highest authority)
2. Extension-specific guidelines (supplements Final Decisions)
3. Implementation documentation (follows Final Decisions)
4. Code comments and docstrings (implements Final Decisions)

## 📋 **Complete Document Map**

| Document | Authority Level | Theme | Key Architectural Decisions |
|----------|----------------|-------|----------------------------|
| **final-decision-0.md** | Meta | *Navigation & Meta-Guidelines* | Authority hierarchy, coherence rules, editing policy |
| **final-decision-1.md** | Core | **Directory Structure & Data Organization** | Grid-size hierarchy, multi-directional data ecosystem, logs/extensions structure |
| **final-decision-2.md** | Core | **Configuration & Validation Architecture** | Config separation, validation system, architectural standards |
| **final-decision-3.md** | Core | **Singleton Pattern Standards** | `SingletonABCMeta` usage, approved singletons, thread safety |
| **final-decision-4.md** | Core | **Agent Naming Conventions** | `agent_*.py` files, `*Agent` classes, naming validation |
| **final-decision-5.md** | Core | **Extension Directory Templates** | v0.01→v0.04 evolution, stability rules, breaking changes |
| **final-decision-6.md** | Core | **Path Management Standards** | Mandatory `path_utils.py`, cross-platform compatibility |
| **final-decision-7.md** | Advanced | **Factory Pattern Architecture** | Agent factories, design philosophy, extensibility |
| **final-decision-8.md** | Advanced | **Factory Implementation Details** | Layered architecture, registration patterns, error handling |
| **final-decision-9.md** | Advanced | **Streamlit OOP Architecture** | Base/Extension apps, dashboard patterns, UX standards |
| **final-decision-10.md** | Special | **GOOD_RULES System** | AI assistant guidelines, implementation rules |

## 🔄 **Cross-Document Coherence Requirements**

The following architectural principles are maintained consistently across **all** Final Decision documents:

### **Core Architecture Pillars**
1. **🗂️ Directory Structure**: Grid-size hierarchies (`logs/extensions/{datasets|models}/grid-size-N/...`) with multi-directional data flow (final-decision-1.md)
2. **⚙️ Configuration Management**: Universal constants in `ROOT/config/`, extension configs in `extensions/common/config/` (final-decision-2.md)
3. **🔒 Singleton Patterns**: Global managers using `SingletonABCMeta` for thread-safe state management (final-decision-3.md)
4. **🎯 Naming Standards**: Strict `agent_*.py` → `*Agent` class patterns across all extensions (final-decision-4.md)

### **Implementation Standards**
5. **📁 Extension Evolution**: v0.01→v0.04 progression with stability rules and breaking change controls (final-decision-5.md)
6. **🛣️ Path Management**: Mandatory use of `extensions/common/path_utils.py` for cross-platform reliability (final-decision-6.md)
7. **🏭 Factory Patterns**: Standardized agent creation with layered architecture and error handling (final-decision-7.md, final-decision-8.md)
8. **🌐 Streamlit Architecture**: OOP-based dashboard patterns with base/extension app hierarchy (final-decision-9.md)

### **Data and Integration Standards**
9. **📊 Schema Consistency**: Grid-size agnostic CSV schemas with 16 normalized features for universal compatibility
10. **🔗 Cross-Extension Integration**: Validation systems in `extensions/common/validation/` ensuring interoperability

## 📝 **Document Management Policy**

### **Editing Authority and Process**
- **Final Decision Modifications**: Only when new architectural decisions are finalized
- **Content Standards**: Rich explanations with motivation, trade-offs, and design patterns
- **Cross-References**: Link to related decisions rather than duplicating content
- **Consistency Validation**: All changes must maintain coherence across the series

### **Reference Format Standards**
When citing Final Decision documents, use the precise format:
- ✅ **Correct**: `final-decision-6.md`
- ❌ **Incorrect**: "Final Decision 6", "FD-6", "final-decision-N"

For GOOD_RULES references:
- ✅ **Correct**: `GOOD_RULES (corresponding-rule-markdown-file-name.md)`
- 📖 **Reference**: See `final-decision-10.md` for GOOD_RULES system details

### **Documentation Hierarchy Navigation**
Empty markdown files in `ROOT/docs/extensions-guideline/` should be ignored during navigation. Focus on substantive documentation with architectural content.

---

## 🏛️ **Architectural Authority Statement**

> **DEFINITIVE RULE**: The Final Decision Series (final-decision-0.md through final-decision-10.md) constitutes the supreme architectural authority for the Snake Game AI project. When conflicts arise between these documents and any other documentation, the Final Decision Series takes precedence. Only newer Final Decision documents with higher numbers can override previous decisions in the series.