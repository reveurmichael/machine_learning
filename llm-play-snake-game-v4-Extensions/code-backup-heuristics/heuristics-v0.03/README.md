# Heuristics v0.03 - Comprehensive Web Dashboard

**Natural Evolution from v0.02**: Complete web interface with game launching and replay capabilities

This v0.03 extension demonstrates the natural progression of software systems by building upon v0.02's multi-algorithm foundation with a comprehensive web-based interface. It shows how modern applications evolve from CLI-only tools to sophisticated web dashboards while maintaining all core functionality.

## 🚀 Evolution Path: v0.02 → v0.03

| Aspect | v0.02 (Multi-Algorithm CLI) | v0.03 (Web Dashboard) |
|--------|------------------------------|------------------------|
| **User Interface** | Command-line only | Comprehensive Streamlit web dashboard |
| **Entry Point** | `main.py` CLI script | `app.py` web application |
| **Code Organization** | Flat structure | Organized with `agents/` and `scripts/` folders |
| **Experiment Management** | Manual CLI execution | Web-based launch with parameter configuration |
| **Replay Capabilities** | External scripts only | Integrated PyGame + Web replay options |
| **Performance Analysis** | Console output only | Interactive charts and comparisons |
| **Experiment Overview** | No unified view | Comprehensive experiment dashboard |

## 🎯 Key Features Added in v0.03

### 1. **Comprehensive Web Dashboard**
- **Overview Tab**: Experiment statistics and performance comparison
- **Launch Tab**: Configure and launch any of the 7 algorithms via web interface
- **PyGame Replay Tab**: Launch PyGame replays with Task-0 infrastructure
- **Web Replay Tab**: Both Task-0 and heuristic-specific web replay options
- **Performance Analysis Tab**: Interactive charts and algorithm comparison

### 2. **Enhanced Code Organization**
```
heuristics-v0.03/
├── app.py                    # Main Streamlit dashboard (new entry point)
├── agents/                   # All algorithm agents (organized)
│   ├── agent_bfs.py
│   ├── agent_astar.py
│   └── ... (all 7 agents)
├── scripts/                  # CLI scripts (v0.02 main.py moved here)
│   └── main.py
├── replay_engine.py          # Heuristic-specific replay engine
├── replay_gui.py             # Web-based replay interface
└── game_*.py                 # Core game logic (inherited from v0.02)
```

### 3. **Extensive Task-0 Infrastructure Reuse**
- **File Management**: Uses `FileManager` for experiment discovery
- **Network Utilities**: Uses `ensure_free_port()` and `random_free_port()`
- **Replay Infrastructure**: Leverages `BaseReplayEngine` and replay scripts
- **Web Infrastructure**: Extends Task-0 web patterns and utilities
- **Session Management**: Follows Task-0 `session_utils.py` patterns

### 4. **Dual Replay System**
- **Task-0 Replay**: Universal replay using ROOT infrastructure
- **Heuristic Replay**: Algorithm-aware replay with performance metrics

## 🛠 How to Use

### Web Dashboard (Recommended)
```bash
cd extensions/heuristics-v0.03
streamlit run app.py
```

### CLI Mode (v0.02 compatibility)
```bash
cd extensions/heuristics-v0.03
python scripts/main.py --algorithm bfs --max-games 5
```

## 📊 Dashboard Tabs Overview

### 1. **📊 Overview Tab**
- Experiment statistics table
- Performance comparison charts
- Detailed experiment inspection
- Algorithm efficiency analysis

### 2. **🚀 Launch Games Tab**
- Algorithm selection with descriptions
- Configurable parameters (games, steps, grid size)
- Advanced options (verbose, debug mode)
- Background execution with progress tracking

### 3. **🎮 Replay (PyGame) Tab**
- Experiment and game selection
- Game performance metrics display
- One-click PyGame replay launch
- Raw JSON data inspection

### 4. **🌐 Replay (Web) Tab**
- Web server configuration
- Dual replay options:
  - **Task-0 Web Replay**: Universal replay infrastructure
  - **Heuristic Web Replay**: Algorithm-specific interface
- Live web replay with controls

### 5. **📈 Performance Analysis Tab**
- Multi-experiment comparison
- Interactive performance charts
- Algorithm efficiency metrics
- Statistical analysis tools

## 🧠 Algorithm Support

All 7 heuristic algorithms from v0.02 are fully supported:

| Algorithm | Description | Performance Characteristics |
|-----------|-------------|----------------------------|
| **BFS** | Breadth-First Search | Optimal paths, moderate performance |
| **BFS-Safe-Greedy** | BFS with safety validation | Best overall performer |
| **BFS-Hamiltonian** | BFS + Hamiltonian fallback | Hybrid approach |
| **DFS** | Depth-First Search | Educational/experimental |
| **A*** | A* Algorithm | Optimal with heuristics |
| **A*-Hamiltonian** | A* + Hamiltonian fallback | Advanced hybrid |
| **Hamiltonian** | Hamiltonian Cycle | Space-filling approach |

## 🔄 Replay Infrastructure

### Task-0 Replay (Universal)
- Uses ROOT `scripts/replay.py` and `scripts/replay_web.py`
- Compatible with all Task-0 tooling
- Standard replay interface

### Heuristic Replay (Enhanced)
- Algorithm-aware display
- Performance metrics during replay
- Pathfinding visualization
- Heuristic-specific insights

## 📈 Performance Metrics

The dashboard tracks comprehensive performance metrics:

- **Score Metrics**: Total score, average score, best score
- **Efficiency Metrics**: Score per step, score per round
- **Algorithm Metrics**: Success rates, pathfinding statistics
- **Comparison Metrics**: Cross-algorithm performance analysis

## 🎯 Design Patterns Demonstrated

### 1. **Facade Pattern**
- `HeuristicsApp` provides simplified interface to complex system
- Coordinates between launching, replay, and analysis components

### 2. **Adapter Pattern**
- `HeuristicReplayEngine` adapts Task-0 infrastructure for heuristics
- Maintains compatibility while adding algorithm-specific features

### 3. **Factory Pattern**
- Algorithm selection and replay engine creation
- Consistent interface for different algorithm types

### 4. **Template Method Pattern**
- Common replay structure with algorithm-specific implementations
- Consistent web interface patterns

## 🔗 Integration with Task-0

v0.03 demonstrates extensive reuse of Task-0 infrastructure:

### Base Classes Used
- `BaseReplayEngine` - Core replay functionality
- `FileManager` - Experiment discovery and management
- `BaseGameManager`, `BaseGameLogic`, `BaseGameData` - Game mechanics

### Utilities Leveraged
- `utils/network_utils.py` - Port management
- `utils/web_utils.py` - Web state building
- `utils/session_utils.py` - Session management patterns
- `config/*` - Universal constants and configuration

### Scripts Reused
- `scripts/replay.py` - PyGame replay
- `scripts/replay_web.py` - Web replay
- Dashboard patterns from `dashboard/` folder

## 🌟 What This Demonstrates

### Software Evolution
- **v0.01**: Proof of concept (single algorithm)
- **v0.02**: Multi-algorithm CLI system
- **v0.03**: Comprehensive web dashboard

### Architecture Maturity
- Clean separation of concerns
- Extensive code reuse
- Modern web interface
- Backward compatibility

### Educational Value
- Design pattern implementation
- Infrastructure reuse
- Natural software progression
- Best practices demonstration

## 🚀 Future Extensions

v0.03 provides a solid foundation for further evolution:

- **Algorithm Comparison**: Side-by-side algorithm visualization
- **Real-time Monitoring**: Live game progress tracking
- **Advanced Analytics**: Machine learning performance analysis
- **Multi-user Support**: Shared experiment environments

---

*v0.03 represents the culmination of natural software evolution - from simple proof of concept to sophisticated web application, while maintaining clean architecture and extensive code reuse.* 