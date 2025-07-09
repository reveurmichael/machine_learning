# Gymnasium Integration for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision.md`) and defines Gymnasium integration standards.

> **See also:** `reinforcement-learning.md`, `stable-baseline.md`, SUPREME_RULES from `final-decision.md`, `standalone.md`.

## 🎯 **Core Philosophy: Standard RL Environment**

Gymnasium integration provides **standard reinforcement learning environment** for Snake Game AI, enabling compatibility with popular RL libraries and algorithms. This integration follows the Gymnasium API standards while maintaining SUPREME_RULES from `final-decision.md` compliance.

### **Educational Value**
- **RL Standards**: Understanding Gymnasium API and RL environment design
- **Algorithm Compatibility**: Learning to work with popular RL libraries
- **Environment Design**: Creating proper RL environments with canonical patterns
- **Integration**: Seamless integration with RL frameworks following SUPREME_RULES

## 🏗️ **Gymnasium Environment Factory (CANONICAL)**

### **Environment Factory (SUPREME_RULES Compliant)**
```python
from utils.factory_utils import SimpleFactory
import gymnasium as gym

class SnakeGymnasiumFactory:
    """
    Factory Pattern for Gymnasium environments following SUPREME_RULES from final-decision.md
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for RL environments
    Educational Value: Shows how SUPREME_RULES apply to RL environment creation
    """
    
    _registry = {
        "SNAKE_V0": SnakeEnvV0,
        "SNAKE_V1": SnakeEnvV1,
        "SNAKE_DISCRETE": SnakeDiscreteEnv,
        "SNAKE_CONTINUOUS": SnakeContinuousEnv,
    }
    
    @classmethod
    def create(cls, env_type: str, **kwargs):  # CANONICAL create() method per SUPREME_RULES
        """Create Gymnasium environment using canonical create() method following SUPREME_RULES from final-decision.md"""
        env_class = cls._registry.get(env_type.upper())
        if not env_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown environment type: {env_type}. Available: {available}")
        print_info(f"[SnakeGymnasiumFactory] Creating environment: {env_type}")  # SUPREME_RULES compliant logging
        return env_class(**kwargs)
```

### **Basic Snake Environment Implementation**
```python
class SnakeEnvV0(gym.Env):
    """
    Basic Snake environment following SUPREME_RULES.
    
    Design Pattern: Template Method Pattern
    Purpose: Provides standard Gymnasium interface for Snake game
    Educational Value: Shows RL environment implementation with canonical patterns
    """
    
    def __init__(self, grid_size: int = 10):
        super().__init__()
        self.grid_size = grid_size
        self.action_space = gym.spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(grid_size, grid_size, 3), dtype=np.float32
        )
        self.game_state = None
        print_info(f"[SnakeEnvV0] Initialized {grid_size}x{grid_size} environment")  # SUPREME_RULES compliant logging
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.game_state = self._create_initial_state()
        observation = self._get_observation()
        info = self._get_info()
        print_info("[SnakeEnvV0] Environment reset")  # SUPREME_RULES compliant logging
        return observation, info
    
    def step(self, action):
        """Take action and return next state"""
        # Execute action
        reward, done, truncated = self._execute_action(action)
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        
        print_info(f"[SnakeEnvV0] Action {action} - Reward: {reward}, Done: {done}")  # SUPREME_RULES compliant logging
        return observation, reward, done, truncated, info
```

## 📊 **Simple Logging for Gymnasium Operations**

All Gymnasium operations must use simple print statements as mandated by SUPREME_RULES from `final-decision.md`:

```python
# ✅ CORRECT: Simple logging for Gymnasium (SUPREME_RULES compliance)
def train_with_gymnasium(env, agent, episodes: int = 1000):
            print_info(f"[GymnasiumTrainer] Starting training with {episodes} episodes")  # SUPREME_RULES compliant logging
    
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        
        while True:
            action = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        if episode % 100 == 0:
            print_info(f"[GymnasiumTrainer] Episode {episode}: reward={total_reward}")  # SUPREME_RULES compliant logging
    
    print_success("[GymnasiumTrainer] Training completed")  # SUPREME_RULES compliant logging
```

## 🎯 **Environment Variants**

### **Discrete Action Space**
```python
class SnakeDiscreteEnv(SnakeEnvV0):
    """
    Discrete action space environment following SUPREME_RULES.
    
    Design Pattern: Strategy Pattern
    Purpose: Provides discrete action space for traditional RL algorithms
    Educational Value: Shows discrete environment design with canonical patterns
    """
    
    def __init__(self, grid_size: int = 10):
        super().__init__(grid_size)
        self.action_space = gym.spaces.Discrete(4)  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_meanings = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        print_info("[SnakeDiscreteEnv] Initialized discrete environment")  # SUPREME_RULES compliant logging
    
    def _execute_action(self, action):
        """Execute discrete action"""
        action_name = self.action_meanings[action]
        print_info(f"[SnakeDiscreteEnv] Executing action: {action_name}")  # SUPREME_RULES compliant logging
        # Action execution logic here
        return reward, done, truncated
```

### **Continuous Action Space**
```python
class SnakeContinuousEnv(SnakeEnvV0):
    """
    Continuous action space environment following SUPREME_RULES.
    
    Design Pattern: Adapter Pattern
    Purpose: Provides continuous action space for advanced RL algorithms
    Educational Value: Shows continuous environment design with canonical patterns
    """
    
    def __init__(self, grid_size: int = 10):
        super().__init__(grid_size)
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )  # Continuous direction vector
        print_info("[SnakeContinuousEnv] Initialized continuous environment")  # SUPREME_RULES compliant logging
    
    def _execute_action(self, action):
        """Execute continuous action"""
        direction = self._continuous_to_discrete(action)
        print_info(f"[SnakeContinuousEnv] Continuous action {action} -> direction {direction}")  # SUPREME_RULES compliant logging
        # Action execution logic here
        return reward, done, truncated
```

## 🎓 **Educational Applications with Canonical Patterns**

### **RL Environment Understanding**
- **Gymnasium API**: Learning standard RL environment interface using canonical factory methods
- **Action Spaces**: Understanding discrete vs continuous action spaces with simple logging
- **Observation Spaces**: Designing proper state representations following SUPREME_RULES
- **Reward Design**: Creating effective reward functions with canonical patterns

### **Integration Benefits**
- **Algorithm Compatibility**: Works with popular RL libraries using canonical patterns
- **Standard Interface**: Follows Gymnasium standards for broad compatibility
- **Performance**: Optimized for RL training with simple logging throughout
- **Extensibility**: Easy to extend with new environment variants following SUPREME_RULES

## 📋 **SUPREME_RULES Implementation Checklist for Gymnasium**

### **Mandatory Requirements**
- [ ] **Canonical Method**: All factories use `create()` method exactly (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses utils/print_utils.py functions only for all Gymnasium operations (SUPREME_RULES compliance)
- [ ] **Gymnasium Compliance**: Follows Gymnasium API standards
- [ ] **Pattern Consistency**: Follows canonical patterns across all environment implementations

### **Environment-Specific Standards**
- [ ] **Action Space**: Properly defined action spaces (discrete/continuous)
- [ ] **Observation Space**: Correctly shaped observation spaces
- [ ] **Reset Method**: Proper environment reset with canonical patterns
- [ ] **Step Method**: Correct step implementation following SUPREME_RULES

---

**Gymnasium integration provides standard RL environment for Snake Game AI while maintaining strict SUPREME_RULES from `final-decision.md` compliance and educational value.**

## 🔗 **See Also**

- **`reinforcement-learning.md`**: RL algorithm standards
- **`stable-baseline.md`**: Stable Baselines integration
- **SUPREME_RULES from `final-decision.md`**: Governance system and canonical standards
- **`standalone.md`**: Standalone principle implementation







