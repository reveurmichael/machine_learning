# Gymnasium Environment Integration for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ and extension guidelines. Gymnasium integration follows the same architectural patterns established in the GOOD_RULES.

## 🎯 **Core Philosophy: Standardized RL Interface**

Gymnasium provides a standardized API for reinforcement learning environments, enabling seamless integration with popular RL libraries while maintaining compatibility with the native Snake Game architecture.

### **Design Philosophy**
- **Dual Implementation Strategy**: Native + Gymnasium environments coexist
- **Library Compatibility**: Works with Stable-Baselines3, Ray RLlib, and other frameworks
- **Consistent Interface**: Unified API across all RL implementations
- **Educational Value**: Demonstrates standard RL environment patterns

## 🏗️ **Integration Architecture**

### **Environment Factory Pattern**
Following Final Decision 7-8 factory patterns:

```python
class SnakeEnvironmentFactory:
    """Factory for creating Snake game environments"""
    
    _env_registry = {
        "native": NativeSnakeEnvironment,
        "gymnasium": GymnasiumSnakeEnvironment,
        "vectorized": VectorizedSnakeEnvironment,
    }
    
    @classmethod
    def create_environment(cls, env_type: str, grid_size: int = 10, **kwargs):
        """Create environment by type"""
        env_class = cls._env_registry.get(env_type)
        if not env_class:
            raise ValueError(f"Unknown environment type: {env_type}")
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
    env = env_factory.create_environment(env_type, **kwargs)
    
    # Create agent using existing factory pattern
    agent_factory = RLAgentFactory()
    agent = agent_factory.create_agent(algorithm, env=env)
    
    # Train using existing training pipeline
    trainer = AgentTrainer(agent, env)
    return trainer.train()
```

---

**Gymnasium integration provides a bridge between the Snake Game AI architecture and the broader reinforcement learning ecosystem, enabling both custom implementations and standard library compatibility while maintaining the educational and architectural benefits of the native system.**







