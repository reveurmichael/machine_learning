# Agents Architecture and Implementation Guide

This document provides comprehensive guidelines for implementing and organizing AI agents across all Snake Game AI extensions, covering the progression from v0.01 to v0.04 versions.

## 🎯 **Agent Organization Overview**

The `agents/` folder structure represents the evolution from proof-of-concept implementations to sophisticated multi-algorithm systems:

### **Extension Version Progression:**
- **v0.01**: Single agent in extension root (proof-of-concept)  
- **v0.02**: Organized `agents/` package with multiple algorithms
- **v0.03**: Enhanced `agents/` with dashboard integration  
- **v0.04**: Advanced features with JSONL trajectory generation (heuristics only)

### **Directory Structure Rules:**

For extensions named `[Algorithm]-v0.0N`:

#### **v0.01 Structure (Proof-of-Concept)**
- Agents go directly in: `./extensions/[Algorithm]-v0.01/`
- Single-file agent implementations
- Basic game integration

#### **v0.02+ Structure (Multi-Algorithm)** 
- Agents go in: `./extensions/[Algorithm]-v0.0N/agents/`
- Where N ≥ 2 (v0.02, v0.03, and for heuristics: v0.04)
- Organized package structure with multiple agents
- Advanced features and integrations

### **Folder Structure Examples:**

#### **v0.01 Layout**
```
extensions/heuristics-v0.01/
├── agent_bfs.py           # BFS agent implementation
├── game_data.py          # Basic game data handling
├── game_logic.py         # Basic game logic
├── game_manager.py       # Simple game management
└── README.md            # Basic documentation
```

#### **v0.02+ Layout**
```
extensions/heuristics-v0.02/
├── __init__.py           # Extension configuration and factory
├── agents/               # Agent implementations package
│   ├── __init__.py      # Agent protocol and base classes
│   ├── agent_bfs.py     # BFS agent implementation
│   ├── agent_astar.py   # A* agent implementation
│   └── agent_hamiltonian.py # Hamiltonian agent
├── game_data.py         # Extended game data management
├── game_logic.py        # Enhanced game logic
├── game_manager.py      # Advanced game session management
├── scripts/             # CLI and automation scripts
└── README.md           # Comprehensive documentation
```

For v0.03, add the folder "dashboard"

## 🏗️ **Agent Implementation Standards**

### **Common Agent Protocol**
All agents across extensions must implement a consistent interface:

# TODO

It should extend BaseAgent of ROOT/core/game_agents.py

