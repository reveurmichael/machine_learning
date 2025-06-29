# Minimal Web Backend for Snake Game AI

> **Truly minimal, KISS, DRY, and extensible web backend for Task-0 and all future extensions.**

## 🎯 **Philosophy: KISS + DRY + Extensible**

This web backend follows strict KISS (Keep It Simple, Stupid) principles while being DRY (Don't Repeat Yourself) and easily extensible for all future tasks. No Task-0 pollution, no over-engineering, just clean, simple Flask applications.

### **Core Benefits**
- ✅ **Minimal**: Only essential files (`game_flask_app.py`, `templates/`, `static/`)
- ✅ **KISS**: Simple Flask apps without complex MVC patterns
- ✅ **DRY**: Reusable base class and factory functions
- ✅ **Extensible**: Copy-paste templates for any algorithm/model
- ✅ **No Pollution**: No Task-0 specific code in base classes

## 📁 **Directory Structure**

```
web/
├── game_flask_app.py          # Minimal Flask applications
├── templates/
│   ├── base.html              # Universal template (works for all tasks)
│   ├── human_play.html        # Human mode template
│   ├── main.html              # LLM mode template
│   └── replay.html            # Replay mode template
└── static/
    ├── css/style.css          # Shared styles
    └── js/
        ├── common.js          # Shared JavaScript utilities
        ├── human_play.js      # Human mode JavaScript
        ├── main.js            # LLM mode JavaScript
        └── replay.js          # Replay mode JavaScript
```

## 🚀 **Quick Start**

### **Task-0 Usage**

```python
# Human player web interface
from web.game_flask_app import create_human_app
app = create_human_app(grid_size=10)
app.run()

# LLM player web interface  
from web.game_flask_app import create_llm_app
app = create_llm_app(provider="hunyuan", model="hunyuan-turbos-latest")
app.run()

# Replay web interface
from web.game_flask_app import create_replay_app
app = create_replay_app(log_dir="logs/session_20250101")
app.run()
```

### **Extension Usage (Copy-Paste Pattern)**

```python
# Step 1: Copy the pattern
from web.game_flask_app import SimpleFlaskApp

# Step 2: Create your specialized app
class HeuristicWebApp(SimpleFlaskApp):
    def __init__(self, algorithm="BFS", grid_size=10):
        super().__init__("Heuristic Snake Game")
        self.algorithm = algorithm
        self.grid_size = grid_size
    
    def get_game_data(self):
        return {
            'name': self.name,
            'mode': 'heuristic',
            'algorithm': self.algorithm,
            'grid_size': self.grid_size,
            'status': 'ready'
        }
    
    def handle_control(self, data):
        action = data.get('action', '')
        if action == 'start':
            # Start your algorithm
            return {'status': 'started'}
        return {'status': 'processed'}

# Step 3: Create factory function
def create_heuristic_app(algorithm="BFS", **config):
    return HeuristicWebApp(algorithm=algorithm, **config)
```

## 🎨 **Base Classes**

### **SimpleFlaskApp - Universal Foundation**

```python
class SimpleFlaskApp:
    """Minimal Flask application foundation for all tasks."""
    
    def __init__(self, name: str, port: Optional[int] = None):
        # Automatic port allocation, minimal configuration
    
    def get_game_data(self) -> Dict[str, Any]:
        # Override: Data for template rendering
    
    def get_api_state(self) -> Dict[str, Any]:
        # Override: API state for AJAX calls
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Override: Handle control requests
    
    def run(self, host="127.0.0.1", debug=False):
        # Start Flask server
```

### **Specialized Apps (Task-0)**

```python
class HumanGameApp(SimpleFlaskApp):
    """Human player web application."""
    # Handles keyboard input, reset functionality

class LLMGameApp(SimpleFlaskApp):  
    """LLM player web application."""
    # Handles LLM provider/model configuration

class ReplayGameApp(SimpleFlaskApp):
    """Replay viewer web application."""
    # Handles log directory and game navigation
```

## 🔧 **Extension Patterns**

### **Pattern 1: Algorithm Extensions (Heuristics, Tree Models)**

```python
class AlgorithmWebApp(SimpleFlaskApp):
    """Template for algorithm-based extensions."""
    
    def __init__(self, algorithm_name: str, **params):
        super().__init__(f"{algorithm_name} Snake Game")
        self.algorithm = algorithm_name
        self.params = params
        # Initialize your algorithm here
    
    def get_game_data(self):
        return {
            'name': self.name,
            'mode': 'algorithm',
            'algorithm': self.algorithm,
            'parameters': self.params,
            'status': 'ready'
        }
    
    def handle_control(self, data):
        action = data.get('action', '')
        if action == 'step':
            # Execute one algorithm step
            result = self.algorithm.step()
            return {'action': 'step', 'result': result}
        elif action == 'reset':
            # Reset algorithm state
            self.algorithm.reset()
            return {'action': 'reset', 'status': 'reset'}
        return {'error': 'Unknown action'}

# Usage for different algorithms
def create_bfs_app(**config):
    return AlgorithmWebApp("BFS", **config)

def create_astar_app(**config):
    return AlgorithmWebApp("A*", **config)
```

### **Pattern 2: ML Model Extensions (Supervised, RL)**

```python
class MLModelWebApp(SimpleFlaskApp):
    """Template for ML model extensions."""
    
    def __init__(self, model_type: str, **config):
        super().__init__(f"{model_type} ML Snake Game")
        self.model_type = model_type
        self.config = config
        # Initialize your model here
        self.model = None
        self.training_metrics = {}
    
    def get_game_data(self):
        return {
            'name': self.name,
            'mode': 'ml_model',
            'model_type': self.model_type,
            'training_metrics': self.training_metrics,
            'status': 'ready'
        }
    
    def handle_control(self, data):
        action = data.get('action', '')
        if action == 'train':
            # Start training
            self.start_training()
            return {'action': 'train', 'status': 'training'}
        elif action == 'predict':
            # Make prediction
            state = data.get('state', [])
            prediction = self.model.predict(state)
            return {'action': 'predict', 'prediction': prediction}
        return {'error': 'Unknown action'}

# Usage for different models
def create_xgboost_app(**config):
    return MLModelWebApp("XGBoost", **config)

def create_dqn_app(**config):
    return MLModelWebApp("DQN", **config)
```

### **Pattern 3: Multi-Agent Extensions (Comparison, Ensemble)**

```python
class MultiAgentWebApp(SimpleFlaskApp):
    """Template for multi-agent extensions."""
    
    def __init__(self, agents: list, **config):
        super().__init__("Multi-Agent Snake Game")
        self.agents = agents
        self.current_agent = 0
        self.results = {}
    
    def get_game_data(self):
        return {
            'name': self.name,
            'mode': 'multi_agent',
            'agents': [agent.name for agent in self.agents],
            'current_agent': self.current_agent,
            'results': self.results,
            'status': 'ready'
        }
    
    def handle_control(self, data):
        action = data.get('action', '')
        if action == 'switch_agent':
            # Switch to different agent
            agent_index = data.get('agent_index', 0)
            self.current_agent = agent_index
            return {'action': 'switch_agent', 'current_agent': agent_index}
        elif action == 'compare':
            # Run comparison between agents
            results = self.run_comparison()
            return {'action': 'compare', 'results': results}
        return {'error': 'Unknown action'}
```

## 📊 **Template Integration**

The `base.html` template is designed to work with all extensions:

```html
<!-- Template automatically shows/hides sections based on game_data -->
<div id="algorithm-section" style="display:none;">
    <!-- Shows for algorithm-based extensions -->
</div>

<div id="training-section" style="display:none;">
    <!-- Shows for ML model extensions -->
</div>

<div id="llm-section" style="display:none;">
    <!-- Shows for LLM-based extensions -->
</div>
```

Your `get_game_data()` method controls which sections are visible:

```python
def get_game_data(self):
    return {
        'mode': 'heuristic',  # Shows algorithm-section
        'algorithm': 'BFS',
        'status': 'ready'
    }
```

## 🎯 **Extension Checklist**

When creating a new extension:

- [ ] **Copy SimpleFlaskApp pattern**: Inherit from `SimpleFlaskApp`
- [ ] **Override three methods**: `get_game_data()`, `get_api_state()`, `handle_control()`
- [ ] **Create factory function**: `create_your_app(**config)`
- [ ] **Test with base template**: Ensure `base.html` displays correctly
- [ ] **Add to scripts**: Create `scripts/your_task_web.py` following the pattern

## 🚀 **Script Integration**

Create web scripts following this pattern:

```python
# scripts/your_task_web.py
from web.game_flask_app import SimpleFlaskApp

class YourTaskWebApp(SimpleFlaskApp):
    # Your implementation here
    pass

def main():
    # Parse arguments
    # Create app
    app = YourTaskWebApp(your_params)
    # Run app
    app.run(host=host, debug=debug)

if __name__ == "__main__":
    main()
```

## 🎉 **Benefits Summary**

### **For Task-0**
- ✅ **Clean separation**: Web logic separate from game logic
- ✅ **Simple integration**: Direct Flask integration without complexity
- ✅ **Minimal dependencies**: Only Flask and utilities

### **For Extensions**
- ✅ **Copy-paste ready**: Clear patterns to follow
- ✅ **No Task-0 pollution**: Base classes are truly generic
- ✅ **Flexible specialization**: Override only what you need
- ✅ **Consistent UI**: Same template works for all tasks

### **For Maintenance**
- ✅ **Single source**: All web logic in one file
- ✅ **Easy debugging**: Simple Flask apps are easy to debug
- ✅ **Clear patterns**: Consistent structure across all extensions

---

**This minimal web backend provides the perfect foundation for Task-0 and all future extensions while maintaining KISS principles and avoiding over-engineering.** 