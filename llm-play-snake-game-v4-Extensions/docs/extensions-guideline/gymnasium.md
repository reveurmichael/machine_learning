# Gymnasium Environment Integration for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines Gymnasium environment integration patterns.

> **See also:** `agents.md`, `core.md`, `config.md`, `final-decision-10.md`, `factory-design-pattern.md`.

## 🎯 **Core Philosophy: Standardized RL Interface**

Gymnasium provides a standardized API for reinforcement learning environments, enabling seamless integration with popular RL libraries while maintaining compatibility with the native Snake Game architecture.

### **Design Philosophy**
- **Dual Implementation Strategy**: Native + Gymnasium environments coexist
- **Library Compatibility**: Works with Stable-Baselines3, Ray RLlib, and other frameworks
- **Consistent Interface**: Unified API across all RL implementations
- **Educational Value**: Demonstrates standard RL environment patterns

## 🏗️ **Integration Architecture**

### **Environment Factory Pattern**
Following established factory patterns:

```python
class SnakeEnvironmentFactory:
    """Factory for creating Snake game environments"""
    
    _env_registry = {
        "native": NativeSnakeEnvironment,
        "gymnasium": GymnasiumSnakeEnvironment,
        "vectorized": VectorizedSnakeEnvironment,
    }
    
    @classmethod
    def create(cls, env_type: str, grid_size: int = 10, **kwargs):
        """Create environment by type"""
        env_class = cls._env_registry.get(env_type)
        if not env_class:
            raise ValueError(f"Unknown environment type: {env_type}")
        print(f"[SnakeEnvironmentFactory] Creating {env_type} environment")  # Simple logging
        return env_class(grid_size=grid_size, **kwargs)
```

### **Gymnasium Environment Implementation**
```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SnakeGameEnvironment(gym.Env):
    """Gymnasium-compatible Snake game environment"""
    
    def __init__(self, grid_size: int = 10):
        super().__init__()
        self.grid_size = grid_size
        print(f"[SnakeGameEnvironment] Initialized for {grid_size}x{grid_size} grid")  # Simple logging
        
        # Action space: UP, DOWN, LEFT, RIGHT
        self.action_space = spaces.Discrete(4)
        
        # Observation space: Grid-size agnostic features (16 features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )
        
        # Initialize game logic using existing architecture
        from core.game_logic import BaseGameLogic
        self.game_logic = BaseGameLogic(grid_size=grid_size)
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.game_logic.reset_game()
        observation = self._get_observation()
        info = self._get_info()
        return observation, info
        
    def step(self, action):
        """Execute one environment step"""
        direction = self._action_to_direction(action)
        
        # Execute move using existing game logic
        success = self.game_logic.move_snake(direction)
        
        observation = self._get_observation()
        reward = self._calculate_reward(success)
        terminated = self.game_logic.game_over
        truncated = False  # No time limits in basic version
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
```

## 🔧 **RL Algorithm Integration**

### **Supported Algorithms**
Following the extension architecture patterns:

| Algorithm | Native Implementation | Gymnasium Environment | Primary Framework |
|-----------|----------------------|----------------------|-------------------|
| **DQN** | ✅ Direct integration | ✅ Stable-Baselines3 | PyTorch |
| **PPO** | ✅ Direct integration | ✅ Stable-Baselines3 | PyTorch |
| **A3C** | ✅ Direct integration | ✅ Ray RLlib | PyTorch |
| **SAC** | ✅ Direct integration | ✅ Stable-Baselines3 | PyTorch |

### **Framework Integration Examples**
```python
# Stable-Baselines3 integration
from stable_baselines3 import DQN, PPO
from extensions.common.gymnasium_env import SnakeGameEnvironment

env = SnakeGameEnvironment(grid_size=10)
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Ray RLlib integration
import ray
from ray.rllib.algorithms.ppo import PPOConfig

config = PPOConfig().environment(SnakeGameEnvironment).build()
algo = config.build()
```

## 🎯 **Benefits of Dual Implementation**

### **Native Implementation Benefits**
- **Direct Control**: Full access to game state and custom reward functions
- **Performance**: No environment wrapper overhead
- **Custom Features**: Snake-specific optimizations and debugging tools
- **Educational Value**: Clear understanding of RL algorithm internals

### **Gymnasium Implementation Benefits**
- **Library Compatibility**: Works with existing RL frameworks
- **Standardized Interface**: Consistent API across different projects
- **Community Support**: Access to pre-built algorithms and tools
- **Benchmarking**: Easy comparison with other environments

### **Cross-Compatibility Strategy**
Both implementations share the same core components:
- Same feature extraction (16 grid-size agnostic features)
- Same reward calculation logic
- Same action space and observation space
- Same game rules and physics

## 🚀 **Integration with Extension Architecture**

### **Reinforcement Learning Extensions**
```python
# extensions/reinforcement-v0.02/environments/
class RLEnvironmentManager:
    """Manages different environment types for RL extensions"""
    
    def __init__(self, env_type: str = "native", grid_size: int = 10):
        self.env_type = env_type
        self.grid_size = grid_size
        
        # Create environment based on type
        if env_type == "gymnasium":
            self.env = SnakeGameEnvironment(grid_size=grid_size)
        else:
            self.env = NativeSnakeEnvironment(grid_size=grid_size)
        print(f"[RLEnvironmentManager] Created {env_type} environment")  # Simple logging
            
    def create_vectorized_env(self, n_envs: int = 4):
        """Create vectorized environment for parallel training"""
        if self.env_type == "gymnasium":
            from stable_baselines3.common.vec_env import DummyVecEnv
            return DummyVecEnv([lambda: SnakeGameEnvironment(self.grid_size) 
                               for _ in range(n_envs)])
```

### **Training Pipeline Integration**
```python
# Compatible with existing training scripts
def train_agent(algorithm: str, env_type: str = "gymnasium", **kwargs):
    """Train RL agent with specified environment type"""
    
    # Create environment
    env_factory = SnakeEnvironmentFactory()
    env = env_factory.create(env_type, **kwargs)
    
    # Create agent using existing factory pattern
    agent_factory = RLAgentFactory()
    agent = agent_factory.create(algorithm, env=env)
    
    # Train using existing training pipeline
    trainer = AgentTrainer(agent, env)
    return trainer.train()
```

## 🎓 **Educational Applications**

### **Environment Design Patterns**
- **Adapter Pattern**: Wraps native environment with Gymnasium interface
- **Factory Pattern**: Creates different environment types
- **Strategy Pattern**: Different environment implementations
- **Observer Pattern**: Environment state monitoring

### **Learning Objectives**
- **Standard RL Interfaces**: Understand Gymnasium API design
- **Environment Wrapping**: Learn to adapt custom environments
- **Library Integration**: Work with popular RL frameworks
- **Performance Comparison**: Compare native vs. standardized implementations

## 📊 **Implementation Guidelines**

### **Environment Registration**
```python
# Register custom environment with Gymnasium
from gymnasium.envs.registration import register

register(
    id='SnakeGame-v0',
    entry_point='extensions.common.gymnasium_env:SnakeGameEnvironment',
    max_episode_steps=1000,
)
```

### **Feature Extraction Standardization**
```python
def _get_observation(self):
    """Extract standardized 16-feature observation"""
    # Grid-size agnostic features for consistent interface
    features = [
        # Snake position features (4)
        self.snake_head_x / self.grid_size,
        self.snake_head_y / self.grid_size,
        self.apple_x / self.grid_size,
        self.apple_y / self.grid_size,
        
        # Direction features (4)
        self.direction_up,
        self.direction_down,
        self.direction_left,
        self.direction_right,
        
        # Game state features (4)
        self.snake_length / (self.grid_size * self.grid_size),
        self.distance_to_apple / (self.grid_size * 2),
        self.danger_ahead,
        self.danger_left,
        
        # Additional features (4)
        self.danger_right,
        self.danger_behind,
        self.game_over,
        self.score / 100.0,
    ]
    return np.array(features, dtype=np.float32)
```

### **Reward Function Consistency**
```python
def _calculate_reward(self, success: bool) -> float:
    """Calculate reward consistent across implementations"""
    reward = 0.0
    
    if self.game_logic.apple_eaten:
        reward += 10.0
    elif self.game_logic.game_over:
        reward -= 10.0
    elif success:
        reward += 0.1  # Small positive reward for successful moves
    
    return reward
```

## 🔗 **Integration with Other Extensions**

### **With Heuristics**
- Use heuristic algorithms to validate Gymnasium environment behavior
- Compare performance between native and Gymnasium implementations
- Generate training data using both environment types

### **With Supervised Learning**
- Train models on data from both environment types
- Use supervised learning to validate environment consistency
- Create environment comparison tools

### **With Reinforcement Learning**
- Enable seamless switching between environment types
- Provide consistent training pipelines across implementations
- Support algorithm comparison studies

---

**Gymnasium integration provides a standardized interface for reinforcement learning while maintaining compatibility with the native Snake Game architecture. This dual implementation strategy enables both educational clarity and practical library integration.**







