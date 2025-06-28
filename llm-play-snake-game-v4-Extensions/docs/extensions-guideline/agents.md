# Agent Implementation Standards for Snake Game AI Extensions

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for agent implementation standards and supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`).

> **See also:** `core.md`, `factory-design-pattern.md`, `final-decision-10.md`, `standalone.md`.

## 🎯 **Core Philosophy: Algorithmic Decision Making**

Agents in the Snake Game AI project represent the core decision-making components that determine how the snake moves through the game environment. Each agent implements a specific algorithm or approach to solving the pathfinding and decision-making challenges of the Snake game.

### **Educational Value**
- **Algorithm Implementation**: Clear examples of different algorithmic approaches
- **Design Pattern Demonstration**: Factory patterns, strategy patterns, and inheritance
- **Comparative Analysis**: Easy comparison between different agent types
- **Extensibility**: Framework for implementing new algorithms

## 🏗️ **Factory Pattern: Canonical Method is create()**

All agent factories must use the canonical method name `create()` for instantiation, not `create_agent()` or any other variant. This ensures consistency and aligns with the KISS principle. Factories should be simple, dictionary-based, and avoid over-engineering.

### Reference Implementation

A generic, educational `SimpleFactory` is provided in `extensions/common/utils/factory_utils.py`:

```python
from extensions.common.utils.factory_utils import SimpleFactory

class MyAgent:
    def __init__(self, name):
        self.name = name

factory = SimpleFactory()
factory.register("myagent", MyAgent)
agent = factory.create("myagent", name="TestAgent")  # CANONICAL create() method
print(agent.name)  # Output: TestAgent
```

### Example Agent Factory

```python
class HeuristicAgentFactory:
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
    }
    @classmethod
    def create(cls, algorithm: str, **kwargs):  # CANONICAL create() method
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        print(f"[HeuristicAgentFactory] Creating agent: {algorithm}")  # Simple logging
        return agent_class(**kwargs)
```

## 🏗️ **Base Agent Interface**

All agents must implement a standardized interface that enables consistent integration with the game framework:

```python
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

class BaseAgent(ABC):
    """
    Base interface for all agents across extensions.
    
    Design Pattern: Strategy Pattern
    - Defines common agent interface
    - Subclasses implement specific algorithms
    - Enables runtime algorithm selection
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.stats = {}
        print(f"[BaseAgent] Initialized {name} agent")  # Simple logging
    
    @abstractmethod
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """
        Plan the next move based on current game state.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Move direction ('UP', 'DOWN', 'LEFT', 'RIGHT')
        """
        pass
    
    def update_stats(self, game_state: Dict[str, Any], move: str, result: str):
        """Update agent statistics based on move result"""
        if 'moves_made' not in self.stats:
            self.stats['moves_made'] = 0
        self.stats['moves_made'] += 1
        
        if result == 'APPLE_EATEN':
            if 'apples_eaten' not in self.stats:
                self.stats['apples_eaten'] = 0
            self.stats['apples_eaten'] += 1
        elif result == 'GAME_OVER':
            if 'games_lost' not in self.stats:
                self.stats['games_lost'] = 0
            self.stats['games_lost'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current agent statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset agent statistics"""
        self.stats = {}
        print(f"[BaseAgent] Reset stats for {self.name}")  # Simple logging
```

## 🚀 **Agent Implementation Examples**

### **Heuristic Agent Implementation**

```python
class BFSAgent(BaseAgent):
    """
    Breadth-First Search agent for pathfinding.
    
    Design Pattern: Strategy Pattern
    - Implements BFS algorithm for pathfinding
    - Provides optimal path to apple
    - Handles collision avoidance
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("BFS", config)
        self.path = []
        print(f"[BFSAgent] Initialized BFS agent")  # Simple logging
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan next move using BFS pathfinding"""
        snake_positions = game_state['snake_positions']
        apple_position = game_state['apple_position']
        grid_size = game_state['grid_size']
        
        # Find path to apple using BFS
        path = self._bfs_pathfinding(snake_positions[0], apple_position, snake_positions, grid_size)
        
        if path:
            self.path = path
            next_pos = path[1] if len(path) > 1 else path[0]
            return self._get_direction(snake_positions[0], next_pos)
        else:
            # No path found, use fallback strategy
            return self._fallback_move(snake_positions, grid_size)
    
    def _bfs_pathfinding(self, start: Tuple[int, int], goal: Tuple[int, int], 
                        obstacles: List[Tuple[int, int]], grid_size: int) -> List[Tuple[int, int]]:
        """BFS pathfinding implementation"""
        # BFS implementation details...
        pass
    
    def _get_direction(self, current: Tuple[int, int], next_pos: Tuple[int, int]) -> str:
        """Convert position difference to direction"""
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]
        
        if dx == 1: return 'RIGHT'
        elif dx == -1: return 'LEFT'
        elif dy == 1: return 'DOWN'
        elif dy == -1: return 'UP'
        else: return 'UP'  # Default fallback
    
    def _fallback_move(self, snake_positions: List[Tuple[int, int]], grid_size: int) -> str:
        """Fallback movement when no path is found"""
        # Fallback strategy implementation...
        pass
```

### **Machine Learning Agent Implementation**

```python
class MLPAgent(BaseAgent):
    """
    Multi-Layer Perceptron agent for supervised learning.
    
    Design Pattern: Strategy Pattern
    - Implements neural network-based decision making
    - Loads pre-trained model
    - Provides probabilistic move selection
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("MLP", config)
        self.model = None
        self.load_model()
        print(f"[MLPAgent] Initialized MLP agent")  # Simple logging
    
    def load_model(self):
        """Load pre-trained neural network model"""
        model_path = self.config.get('model_path')
        if model_path and os.path.exists(model_path):
            self.model = torch.load(model_path)
            self.model.eval()
            print(f"[MLPAgent] Loaded model from {model_path}")  # Simple logging
        else:
            print(f"[MLPAgent] No model found at {model_path}")  # Simple logging
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan next move using neural network prediction"""
        # Convert game state to model input
        state_vector = self._state_to_vector(game_state)
        
        # Get model prediction
        with torch.no_grad():
            prediction = self.model(torch.FloatTensor(state_vector))
            move_probs = torch.softmax(prediction, dim=0)
            move_idx = torch.argmax(move_probs).item()
        
        # Convert to direction
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        return directions[move_idx]
    
    def _state_to_vector(self, game_state: Dict[str, Any]) -> List[float]:
        """Convert game state to neural network input vector"""
        # State vectorization implementation...
        pass
```

### **Reinforcement Learning Agent Implementation**

```python
class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent for reinforcement learning.
    
    Design Pattern: Strategy Pattern
    - Implements DQN algorithm for RL
    - Maintains experience replay buffer
    - Provides epsilon-greedy exploration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("DQN", config)
        self.q_network = None
        self.target_network = None
        self.replay_buffer = []
        self.epsilon = self.config.get('epsilon_start', 0.9)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        print(f"[DQNAgent] Initialized DQN agent")  # Simple logging
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan next move using DQN algorithm"""
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Random exploration
            directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            return random.choice(directions)
        else:
            # Exploitation using Q-network
            state_vector = self._state_to_vector(game_state)
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state_vector))
                move_idx = torch.argmax(q_values).item()
            
            directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            return directions[move_idx]
    
    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        print(f"[DQNAgent] Updated epsilon to {self.epsilon:.3f}")  # Simple logging
    
    def _state_to_vector(self, game_state: Dict[str, Any]) -> List[float]:
        """Convert game state to neural network input vector"""
        # State vectorization implementation...
        pass
```

## 📊 **Agent Performance Standards**

### **Performance Metrics**
All agents should track and report these standard metrics:
- **Success Rate**: Percentage of games completed successfully
- **Average Score**: Mean score across all games
- **Average Steps**: Mean number of steps per game
- **Efficiency**: Score per step ratio
- **Algorithm-Specific Metrics**: Custom metrics for each agent type

### **Benchmarking Framework**
```python
class AgentBenchmarker:
    """
    Standardized benchmarking for agent performance.
    
    Design Pattern: Template Method Pattern
    - Defines common benchmarking workflow
    - Supports different agent types
    - Provides standardized reporting
    """
    
    def __init__(self, grid_size: int = 10, num_games: int = 100):
        self.grid_size = grid_size
        self.num_games = num_games
        print(f"[AgentBenchmarker] Initialized benchmarker")  # Simple logging
    
    def benchmark_agent(self, agent: BaseAgent) -> Dict[str, Any]:
        """Run comprehensive benchmark for agent"""
        print(f"[AgentBenchmarker] Starting benchmark for {agent.name}")  # Simple logging
        
        results = {
            'agent_name': agent.name,
            'grid_size': self.grid_size,
            'num_games': self.num_games,
            'success_rate': 0.0,
            'average_score': 0.0,
            'average_steps': 0.0,
            'efficiency': 0.0
        }
        
        # Run benchmark games
        successful_games = 0
        total_score = 0
        total_steps = 0
        
        for game in range(self.num_games):
            game_result = self._run_single_game(agent)
            if game_result['success']:
                successful_games += 1
                total_score += game_result['score']
                total_steps += game_result['steps']
        
        # Calculate metrics
        results['success_rate'] = successful_games / self.num_games
        results['average_score'] = total_score / self.num_games
        results['average_steps'] = total_steps / self.num_games
        results['efficiency'] = results['average_score'] / max(results['average_steps'], 1)
        
        print(f"[AgentBenchmarker] Benchmark completed for {agent.name}")  # Simple logging
        return results
    
    def _run_single_game(self, agent: BaseAgent) -> Dict[str, Any]:
        """Run a single game for benchmarking"""
        # Single game execution implementation...
        pass
```

## 🎯 **Agent Integration Standards**

### **Game Manager Integration**
All agents must integrate seamlessly with the game manager:

```python
class GameManager:
    """Game manager that integrates with agents"""
    
    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.game_state = {}
        print(f"[GameManager] Initialized with {agent.name}")  # Simple logging
    
    def run_game(self) -> Dict[str, Any]:
        """Run a complete game with the agent"""
        self._initialize_game()
        
        while not self._is_game_over():
            # Get agent's move
            move = self.agent.plan_move(self.game_state)
            
            # Execute move
            result = self._execute_move(move)
            
            # Update agent stats
            self.agent.update_stats(self.game_state, move, result)
            
            # Update game state
            self._update_game_state(move, result)
        
        return self._get_game_result()
    
    def _initialize_game(self):
        """Initialize game state"""
        # Game initialization implementation...
        pass
    
    def _is_game_over(self) -> bool:
        """Check if game is over"""
        # Game over check implementation...
        pass
    
    def _execute_move(self, move: str) -> str:
        """Execute agent's move"""
        # Move execution implementation...
        pass
    
    def _update_game_state(self, move: str, result: str):
        """Update game state after move"""
        # State update implementation...
        pass
    
    def _get_game_result(self) -> Dict[str, Any]:
        """Get final game result"""
        # Result compilation implementation...
        pass
```

## 📋 **Implementation Checklist**

### **Required Components**
- [ ] **Base Agent Interface**: Implements `BaseAgent` abstract class
- [ ] **Factory Pattern**: Uses canonical `create()` method
- [ ] **Move Planning**: Implements `plan_move()` method
- [ ] **Statistics Tracking**: Implements `update_stats()` and `get_stats()`
- [ ] **Configuration**: Supports configurable parameters
- [ ] **Error Handling**: Graceful handling of edge cases
- [ ] **Documentation**: Clear docstrings and comments

### **Quality Standards**
- [ ] **Algorithm Correctness**: Implements algorithm accurately
- [ ] **Performance**: Meets performance benchmarks
- [ ] **Robustness**: Handles edge cases gracefully
- [ ] **Extensibility**: Easy to extend and modify
- [ ] **Educational Value**: Clear and understandable implementation

### **Integration Requirements**
- [ ] **Game Manager**: Seamless integration with game framework
- [ ] **Factory System**: Compatible with agent factory patterns
- [ ] **Configuration**: Supports standard configuration system
- [ ] **Logging**: Uses simple print statements for debugging
- [ ] **Testing**: Comprehensive test coverage

---

**Agent implementation standards ensure consistent, high-quality algorithmic decision-making across all Snake Game AI extensions. By following these standards, developers can create robust, educational, and performant agents that integrate seamlessly with the overall framework.**

## 🔗 **See Also**

- **`core.md`**: Base class architecture and inheritance patterns
- **`factory-design-pattern.md`**: Factory pattern implementation guide
- **`final-decision-10.md`**: final-decision-10.md governance system
- **`standalone.md`**: Standalone principle and extension independence

