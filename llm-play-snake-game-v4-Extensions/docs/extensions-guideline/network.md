# Network Architecture: Random Port Strategy

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines network architecture patterns for Task-0 and all extensions.

> **See also:** `final-decision-10.md`, `mvc.md`, `flask.md`, `core.md`.

## 🎯 **Core Philosophy: Dynamic Port Allocation**

The Snake Game AI project uses **dynamic random port allocation** for all Flask applications to ensure **conflict-free deployment**, **multi-instance support**, and **development flexibility**. This approach follows KISS principles while providing robust networking capabilities for Task-0 and all extensions (Task 1-5).

### **Educational Value**
- **Conflict Resolution**: Demonstrates how to handle port conflicts in multi-service environments
- **Scalability**: Shows patterns for running multiple instances simultaneously
- **Development Workflow**: Enables parallel development and testing
- **Production Readiness**: Provides deployment-friendly networking patterns

## 🏗️ **Random Port Implementation**

### **Core Network Utilities**
The project provides centralized network utilities in `utils/network_utils.py`:

```python
"""Network utilities for dynamic port allocation."""

import random
import socket
import os

def find_free_port(start: int = 8000, max_port: int = 65535) -> int:
    """
    Find the first available TCP port starting from specified port.
    
    Design Pattern: Strategy Pattern (Network Resource Management)
    Purpose: Provides conflict-free port allocation
    Educational Value: Demonstrates robust network resource management
    """
    for port in range(start, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                print(f"[NetworkUtils] Found free port: {port}")  # Simple logging - SUPREME_RULES
                return port
            except OSError:
                continue
    
    raise RuntimeError("No free port found in the specified range")

def random_free_port(min_port: int = 8000, max_port: int = 9000) -> int:
    """
    Return a random free port within specified range.
    
    Design Pattern: Factory Pattern (Port Generation)
    Purpose: Provides random port allocation for development
    Educational Value: Shows how to combine randomization with resource validation
    """
    for _ in range(1000):  # Try up to 1000 random ports
        candidate = random.randint(min_port, max_port)
        if is_port_free(candidate):
            print(f"[NetworkUtils] Random free port: {candidate}")  # Simple logging
            return candidate
    
    # Fallback to sequential search
    return find_free_port(min_port)

def is_port_free(port: int) -> bool:
    """Check if specified port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", port))
            return True
        except OSError:
            return False
```

### **Flask Integration Pattern**
```python
# Task-0 and Extension Flask Applications
from utils.network_utils import random_free_port, find_free_port

class BaseFlaskApp:
    """
    Base Flask application with dynamic port allocation.
    
    Design Pattern: Template Method Pattern (Flask Application Lifecycle)
    Purpose: Provides consistent Flask application setup across all tasks
    Educational Value: Demonstrates how to create extensible web applications
    """
    
    def __init__(self, name: str, default_port: int = None):
        self.app = Flask(name)
        self.port = default_port or random_free_port()
        print(f"[BaseFlaskApp] Initialized {name} on port {self.port}")  # Simple logging
    
    def run(self, host: str = "127.0.0.1", debug: bool = False):
        """Run Flask application with dynamic port allocation."""
        # Ensure port is still free before starting
        if not is_port_free(self.port):
            self.port = find_free_port(self.port + 1)
            print(f"[BaseFlaskApp] Port conflict resolved, using: {self.port}")  # Simple logging
        
        print(f"[BaseFlaskApp] Starting server on http://{host}:{self.port}")  # Simple logging
        self.app.run(host=host, port=self.port, debug=debug)
```

## 🎯 **Why Random Ports? Strategic Benefits**

### **1. Development Workflow Enhancement**
```python
# Multiple developers can run simultaneously
# Developer A: Task-0 on random port 8234
python scripts/main_web.py  # Automatically finds free port

# Developer B: Task-1 Heuristics on random port 8567
cd extensions/heuristics-v0.03
python scripts/replay_web.py  # Different random port, no conflicts

# Developer C: Task-2 RL on random port 8891
cd extensions/reinforcement-v0.03
python app.py  # Another random port, seamless parallel development
```

### **2. Multi-Instance Support**
```python
# Run multiple game instances for comparison
# Instance 1: BFS algorithm on port 8123
python scripts/main_web.py --algorithm BFS

# Instance 2: A* algorithm on port 8456  
python scripts/main_web.py --algorithm ASTAR

# Instance 3: Hamiltonian on port 8789
python scripts/main_web.py --algorithm HAMILTONIAN

# All running simultaneously for performance comparison
```

### **3. CI/CD and Testing Benefits**
```python
# Automated testing with parallel test suites
def test_concurrent_flask_apps():
    """Test multiple Flask apps running simultaneously."""
    apps = []
    
    # Start multiple Flask apps for testing
    for i in range(5):
        app = BaseFlaskApp(f"test_app_{i}")
        apps.append(app)
        # Each gets a different random port automatically
    
    # All apps can run in parallel without port conflicts
    assert len(set(app.port for app in apps)) == 5  # All different ports
```

### **4. Production Deployment Flexibility**
```python
# Docker container deployment
# Each container gets random port, mapped externally
docker run -p 0:8000 snake-game-task0  # Docker assigns random external port
docker run -p 0:8000 snake-game-task1  # Different random external port
docker run -p 0:8000 snake-game-task2  # Another random external port

# Kubernetes deployment with service discovery
# Random internal ports, consistent external service names
```

## 🔧 **Task-Specific Implementation Patterns**

### **Task-0 (LLM Snake Game)**
```python
# scripts/main_web.py
from utils.network_utils import random_free_port
from web.controllers.game_controllers import GamePlayController

def main():
    """Launch Task-0 web interface with random port."""
    port = random_free_port(8000, 8100)  # Task-0 range
    
    app = Flask(__name__)
    controller = GamePlayController()
    
    # Register routes
    app.add_url_rule('/', 'index', controller.index)
    app.add_url_rule('/game', 'game', controller.game, methods=['GET', 'POST'])
    
    print(f"[Task0] Starting LLM Snake Game on port {port}")  # Simple logging
    app.run(host="127.0.0.1", port=port, debug=False)
```

### **Task-1 (Heuristics) Extension**
```python
# extensions/heuristics-v0.03/scripts/replay_web.py
from utils.network_utils import random_free_port
from web.controllers.base_controller import BaseController

class HeuristicReplayController(BaseController):
    """Heuristic algorithm replay controller."""
    
    def __init__(self):
        super().__init__()
        self.port = random_free_port(8100, 8200)  # Task-1 range
        print(f"[HeuristicReplay] Port assigned: {self.port}")  # Simple logging

def main():
    """Launch heuristic replay web interface."""
    controller = HeuristicReplayController()
    
    app = Flask(__name__)
    app.add_url_rule('/', 'index', controller.replay_index)
    app.add_url_rule('/replay/<algorithm>', 'replay', controller.replay_algorithm)
    
    print(f"[Task1] Starting Heuristic Replay on port {controller.port}")  # Simple logging
    app.run(host="127.0.0.1", port=controller.port)
```

### **Task-2 (Reinforcement Learning) Extension**
```python
# extensions/reinforcement-v0.03/scripts/training_web.py
from utils.network_utils import random_free_port

def main():
    """Launch RL training monitoring web interface."""
    port = random_free_port(8200, 8300)  # Task-2 range
    
    app = Flask(__name__)
    
    @app.route('/training/<agent_type>')
    def monitor_training(agent_type):
        """Monitor RL agent training progress."""
        # Implementation here
        pass
    
    print(f"[Task2] Starting RL Training Monitor on port {port}")  # Simple logging
    app.run(host="127.0.0.1", port=port)
```

## 🎯 **Port Range Allocation Strategy**

### **Standardized Port Ranges**
```python
# Port allocation strategy across all tasks
PORT_RANGES = {
    "task0": (8000, 8099),      # LLM Snake Game
    "task1": (8100, 8199),      # Heuristics
    "task2": (8200, 8299),      # Reinforcement Learning  
    "task3": (8300, 8399),      # Supervised Learning
    "task4": (8400, 8499),      # Distillation
    "task5": (8500, 8599),      # Advanced Extensions
    "development": (9000, 9999), # Development/Testing
}

def get_task_port(task_name: str) -> int:
    """Get random port within task-specific range."""
    if task_name not in PORT_RANGES:
        return random_free_port()  # Fallback to general range
    
    min_port, max_port = PORT_RANGES[task_name]
    return random_free_port(min_port, max_port)
```

### **Environment Variable Support**
```python
# Support for explicit port specification when needed
def get_server_host_port(task_name: str) -> tuple[str, int]:
    """
    Get host and port for server startup.
    
    Supports environment variables for production deployment:
    - WS_HOST: Override default host (127.0.0.1)
    - WS_PORT: Override random port allocation
    - TASK_PORT_RANGE: Override task-specific port range
    """
    host = os.getenv("WS_HOST", "127.0.0.1")
    port_env = os.getenv("WS_PORT")
    
    if port_env and port_env.isdigit():
        port = int(port_env)
        print(f"[Network] Using environment port: {port}")  # Simple logging
    else:
        port = get_task_port(task_name)
        print(f"[Network] Using random port for {task_name}: {port}")  # Simple logging
    
    return host, port
```

## 🚀 **Benefits Summary**

### **Development Benefits**
- ✅ **Parallel Development**: Multiple developers can work simultaneously
- ✅ **No Port Conflicts**: Automatic conflict resolution
- ✅ **Easy Testing**: Parallel test execution without interference
- ✅ **Quick Iteration**: No manual port management required

### **Deployment Benefits**
- ✅ **Container Friendly**: Works seamlessly with Docker/Kubernetes
- ✅ **Load Balancer Compatible**: Easy to scale horizontally
- ✅ **CI/CD Integration**: Automated testing without port conflicts
- ✅ **Multi-Environment**: Same code works in dev/staging/production

### **Educational Benefits**
- ✅ **Network Programming**: Demonstrates socket programming concepts
- ✅ **Resource Management**: Shows how to handle shared resources
- ✅ **System Design**: Illustrates scalable architecture patterns
- ✅ **Best Practices**: Teaches production-ready networking patterns

## 📋 **Implementation Checklist**

### **For All Tasks and Extensions**
- [ ] **Use `utils/network_utils.py`** for port allocation
- [ ] **Implement random port selection** in Flask applications
- [ ] **Follow task-specific port ranges** for organization
- [ ] **Support environment variable overrides** for production
- [ ] **Include simple logging** for port allocation events

### **Flask Application Standards**
- [ ] **Inherit from `BaseFlaskApp`** for consistent behavior
- [ ] **Check port availability** before starting server
- [ ] **Handle port conflicts gracefully** with fallback allocation
- [ ] **Log server startup information** with host and port
- [ ] **Support both random and explicit port specification**

### **Documentation Requirements**
- [ ] **Document port ranges** for each task/extension
- [ ] **Provide usage examples** for different scenarios
- [ ] **Explain conflict resolution** mechanisms
- [ ] **Include deployment guidelines** for production use

---

**Random port allocation ensures conflict-free, scalable, and development-friendly networking across all tasks and extensions while maintaining simplicity and educational value.**

## 🔗 **Cross-References**

### **Related Documents**
- **`final-decision-10.md`**: SUPREME_RULES for canonical networking patterns
- **`mvc.md`**: MVC architecture integration with Flask
- **`flask.md`**: Flask integration patterns for extensions
- **`core.md`**: Core architecture and networking integration

### **Implementation Files**
- **`utils/network_utils.py`**: Core network utilities and port management
- **`web/controllers/base_controller.py`**: Base controller with networking support
- **`scripts/main_web.py`**: Task-0 web interface with random ports
- **Extensions**: Task 1-5 Flask applications following same patterns

### **Educational Resources**
- **Socket Programming**: Low-level networking concepts
- **Flask Deployment**: Production deployment patterns
- **Container Networking**: Docker and Kubernetes integration
- **System Design**: Scalable networking architecture patterns 