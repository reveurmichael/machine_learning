# MVC Architecture & Game Runner Guidelines for Extensions

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and provides comprehensive guidance on implementing both web interfaces and game runners for extensions (Tasks 1-5).

> **See also:** `final-decision-10.md`, `flask.md`, `scripts.md`, `core.md`.

## 🎯 **Core Philosophy: Minimal Web + Simple Game Runners**

Extensions must implement **two complementary interfaces**:
1. **Web Interface**: Flask-based web applications using the minimal KISS architecture
2. **Game Runner**: Simple programmatic interface for testing and automation

### **Design Philosophy**
- **KISS Principles**: Keep web interfaces simple and focused
- **Copy-Paste Templates**: Extensions can copy Task-0 web scripts exactly
- **Dual Interface**: Same functionality available via web and programmatic APIs
- **Extensible Foundation**: Simple patterns that scale to complex algorithms

## 🌐 **Web Interface Architecture: Minimal Flask Apps**

Task-0 provides three **perfect foundation scripts** that demonstrate minimal Flask integration:

### **Foundation Templates (Copy-Paste Ready)**
```
ROOT/scripts/
├── human_play_web.py     # Human player web interface foundation
├── main_web.py           # LLM player web interface foundation  
└── replay_web.py         # Replay web interface foundation
```

### **Extension Pattern: Copy → Replace → Extend**

**Step 1: Copy Foundation Script**
```bash
# Task-1 Example: Copy human_play_web.py for heuristic algorithms
cp scripts/human_play_web.py extensions/heuristics-v0.03/scripts/heuristic_web.py
```

**Step 2: Replace Core Components**
```python
# Original (Task-0): Human input
class HumanWebApp(HumanGameApp):
    def __init__(self, grid_size: int = 10):
        super().__init__(grid_size=grid_size)

# Extension (Task-1): Heuristic algorithms
class HeuristicWebApp(HeuristicGameApp):
    def __init__(self, algorithm: str = "BFS", grid_size: int = 10):
        super().__init__(algorithm=algorithm, grid_size=grid_size)
        self.pathfinder = PathfindingFactory.create(algorithm)
```

**Step 3: Extend with Algorithm-Specific Features**
```python
# Add heuristic-specific application info
def get_application_info(self) -> dict:
    return {
        "name": "Task-1 Heuristic Player",
        "task_name": "task1",
        "algorithm": self.algorithm,
        "pathfinding_type": self.pathfinder.get_type(),
        "features": [
            "Optimal pathfinding",
            "Algorithm comparison",
            "Performance metrics",
            "Path visualization"
        ]
    }
```

## 🏗️ **Minimal Flask App Structure**

### **SimpleFlaskApp Base Class (ROOT/web/game_flask_app.py)**
```python
class SimpleFlaskApp:
    """
    Minimal Flask application following KISS principles.
    
    Educational Value: Shows how to create simple, extensible web apps
    Extension Pattern: All tasks inherit from this foundation
    """
    
    def __init__(self, name: str = "SnakeGame", port: Optional[int] = None):
        self.name = name
        self.port = port or self._get_random_port()
        self.app = Flask(__name__)
        self.configure_app()
        self.setup_routes()
    
    def run(self, host: str = "127.0.0.1", port: Optional[int] = None, debug: bool = False):
        """Run Flask application with proper port handling."""
        actual_port = port if port is not None else self.port
        self.app.run(host=host, port=actual_port, debug=debug)
```

### **Task-Specific Specializations**
```python
# Task-0: Human Game App
class HumanGameApp(SimpleFlaskApp):
    def __init__(self, grid_size: int = 10):
        super().__init__("Human Snake Game")
        self.grid_size = grid_size

# Task-1: Heuristic Game App (Extension Example)
class HeuristicGameApp(SimpleFlaskApp):
    def __init__(self, algorithm: str = "BFS", grid_size: int = 10):
        super().__init__("Heuristic Snake Game")
        self.algorithm = algorithm
        self.grid_size = grid_size
        self.pathfinder = PathfindingFactory.create(algorithm)

# Task-2: RL Game App (Extension Example)  
class RLGameApp(SimpleFlaskApp):
    def __init__(self, agent_type: str = "DQN", grid_size: int = 10):
        super().__init__("RL Snake Game")
        self.agent_type = agent_type
        self.grid_size = grid_size
        self.rl_agent = RLAgentFactory.create(agent_type)
```

## 🎮 **Game Runner Implementation**

Each extension must also provide a **programmatic game runner** for testing and automation.

### **BaseGameRunner Interface**
```python
# Location: core/game_runner.py
class BaseGameRunner:
    """
    Base class for programmatic game execution.
    
    Educational Value: Shows how to create testable, automatable interfaces
    Extension Pattern: All tasks implement this interface
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = []
    
    def run_single_game(self) -> Dict[str, Any]:
        """Run a single game and return results."""
        raise NotImplementedError("Subclasses must implement run_single_game")
    
    def run_multiple_games(self, num_games: int) -> List[Dict[str, Any]]:
        """Run multiple games and return aggregated results."""
        results = []
        for i in range(num_games):
            result = self.run_single_game()
            result['game_number'] = i + 1
            results.append(result)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical summary of all runs."""
        if not self.results:
            return {}
        
        scores = [r.get('score', 0) for r in self.results]
        return {
            'total_games': len(self.results),
            'average_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores)
        }
```

### **Extension Game Runner Examples**

**Task-1: Heuristic Game Runner**
```python
# Location: extensions/heuristics-v0.03/game_runner.py
class HeuristicGameRunner(BaseGameRunner):
    """
    Programmatic interface for running heuristic algorithms.
    
    Educational Value: Shows how to wrap algorithms for automation
    Extension Pattern: Copy this pattern for any algorithm-based task
    """
    
    def __init__(self, algorithm: str, grid_size: int = 10, **config):
        super().__init__(config)
        self.algorithm = algorithm
        self.grid_size = grid_size
        self.pathfinder = PathfindingFactory.create(algorithm)
    
    def run_single_game(self) -> Dict[str, Any]:
        """Run single heuristic game."""
        game_logic = HeuristicGameLogic(
            algorithm=self.algorithm,
            grid_size=self.grid_size
        )
        
        result = game_logic.play_game()
        return {
            'algorithm': self.algorithm,
            'score': result.get('score', 0),
            'steps': result.get('steps', 0),
            'path_length': len(result.get('path', [])),
            'success': result.get('success', False)
        }

# Usage Example
runner = HeuristicGameRunner(algorithm="BFS", grid_size=10)
results = runner.run_multiple_games(100)
stats = runner.get_statistics()
```

**Task-2: RL Game Runner**
```python
# Location: extensions/reinforcement-v0.03/game_runner.py
class RLGameRunner(BaseGameRunner):
    """
    Programmatic interface for running RL agents.
    
    Educational Value: Shows how to wrap RL training/evaluation
    Extension Pattern: Copy this pattern for any ML-based task
    """
    
    def __init__(self, agent_type: str, model_path: str = None, **config):
        super().__init__(config)
        self.agent_type = agent_type
        self.model_path = model_path
        self.rl_agent = RLAgentFactory.create(agent_type)
        
        if model_path:
            self.rl_agent.load_model(model_path)
    
    def run_single_game(self) -> Dict[str, Any]:
        """Run single RL game."""
        env = SnakeEnvironment(grid_size=self.config.get('grid_size', 10))
        
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while not env.done:
            action = self.rl_agent.select_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        return {
            'agent_type': self.agent_type,
            'total_reward': total_reward,
            'steps': steps,
            'score': env.score,
            'success': env.score > 0
        }

# Usage Example
runner = RLGameRunner(agent_type="DQN", model_path="models/best_dqn.pth")
results = runner.run_multiple_games(50)
stats = runner.get_statistics()
```

## 📋 **Extension Implementation Checklist**

### **Web Interface Requirements**
- [ ] **Copy foundation script** from Task-0 (`human_play_web.py`, `main_web.py`, or `replay_web.py`)
- [ ] **Replace core components** with extension-specific algorithms/models
- [ ] **Extend application info** with task-specific metadata
- [ ] **Add custom routes** for algorithm-specific functionality (optional)
- [ ] **Maintain same argument structure** for consistency

### **Game Runner Requirements**
- [ ] **Inherit from BaseGameRunner** for consistent interface
- [ ] **Implement run_single_game()** with algorithm-specific logic
- [ ] **Return standardized results** with task-specific metrics
- [ ] **Support configuration** via constructor parameters
- [ ] **Provide usage examples** and documentation

### **Integration Requirements**
- [ ] **Web and runner consistency** - same algorithms/models available in both
- [ ] **Shared configuration** - same parameters work for web and programmatic interfaces
- [ ] **Result compatibility** - web interface can display runner results
- [ ] **Testing coverage** - both interfaces thoroughly tested

## 🎯 **Extension Examples by Task**

### **Task-1 (Heuristics) Implementation**
```python
# Web Interface: extensions/heuristics-v0.03/scripts/heuristic_web.py
class HeuristicWebApp(HeuristicGameApp):
    def __init__(self, algorithm: str = "BFS", grid_size: int = 10):
        super().__init__(algorithm=algorithm, grid_size=grid_size)

# Game Runner: extensions/heuristics-v0.03/game_runner.py
class HeuristicGameRunner(BaseGameRunner):
    def run_single_game(self) -> Dict[str, Any]:
        # Pathfinding algorithm execution
        pass

# Factory Integration
def create_heuristic_web_app(algorithm: str, **config) -> HeuristicWebApp:
    return HeuristicWebApp(algorithm=algorithm, **config)

def create_heuristic_runner(algorithm: str, **config) -> HeuristicGameRunner:
    return HeuristicGameRunner(algorithm=algorithm, **config)
```

### **Task-2 (Reinforcement Learning) Implementation**
```python
# Web Interface: extensions/reinforcement-v0.03/scripts/rl_web.py
class RLWebApp(RLGameApp):
    def __init__(self, agent_type: str = "DQN", model_path: str = None):
        super().__init__(agent_type=agent_type)
        self.model_path = model_path

# Game Runner: extensions/reinforcement-v0.03/game_runner.py
class RLGameRunner(BaseGameRunner):
    def run_single_game(self) -> Dict[str, Any]:
        # RL agent execution
        pass

# Factory Integration
def create_rl_web_app(agent_type: str, **config) -> RLWebApp:
    return RLWebApp(agent_type=agent_type, **config)

def create_rl_runner(agent_type: str, **config) -> RLGameRunner:
    return RLGameRunner(agent_type=agent_type, **config)
```

### **Task-3 (Supervised Learning) Implementation**
```python
# Web Interface: extensions/supervised-v0.03/scripts/ml_web.py
class MLWebApp(MLGameApp):
    def __init__(self, model_type: str = "XGBoost", model_path: str = None):
        super().__init__(model_type=model_type)
        self.model_path = model_path

# Game Runner: extensions/supervised-v0.03/game_runner.py
class MLGameRunner(BaseGameRunner):
    def run_single_game(self) -> Dict[str, Any]:
        # ML model execution
        pass
```

## 🚀 **Benefits of Dual Interface Architecture**

### **Development Benefits**
- **Consistent Patterns**: Same structure across all extensions
- **Easy Testing**: Programmatic interface enables automated testing
- **Flexible Usage**: Choose web interface for interaction, runner for automation
- **Copy-Paste Learning**: Clear templates for rapid extension development

### **Educational Benefits**
- **Pattern Recognition**: Students see consistent architectural patterns
- **Interface Design**: Learn how to create both interactive and programmatic interfaces
- **Separation of Concerns**: Web presentation separate from core algorithm logic
- **Scalability**: Patterns that work for simple algorithms and complex ML models

### **Operational Benefits**
- **Automation Ready**: Game runners enable batch processing and evaluation
- **Web Accessible**: Flask interfaces provide remote access and visualization
- **Performance Testing**: Easy to benchmark algorithms across multiple runs
- **Integration Friendly**: Both interfaces work with external tools and frameworks

---

**This dual interface architecture ensures that extensions are both user-friendly (web interface) and automation-friendly (game runner), following KISS principles while providing maximum flexibility for different use cases.** 