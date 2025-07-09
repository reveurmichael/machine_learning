# Stable Baselines3 Integration for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`` → `final-decision.md`) and defines Stable Baselines3 integration patterns.

# Stable Baselines3 Integration for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`` → `final-decision.md`) and defines Stable Baselines3 integration patterns.

> **Guidelines Alignment:**
> - This document is governed by the guidelines in `final-decision.md`.
> - All agent factories must use the canonical method name `create()` (never `create_agent`, `create_model`, etc.).
> - All code must use simple print logging (simple logging).
> - Reference: `extensions/common/utils/factory_utils.py` for the canonical `SimpleFactory` implementation.

> **See also:** `agents.md`, `core.md`, `final-decision.md`, `factory-design-pattern.md`, `config.md`.

## 🎯 **Core Philosophy: Production-Ready RL Framework Integration**

Stable Baselines3 (SB3) provides state-of-the-art reinforcement learning algorithms with a standardized interface. In the Snake Game AI ecosystem, SB3 enables rapid prototyping and deployment of proven RL algorithms while maintaining compatibility with the native architecture.

### **Guidelines Alignment**
- **final-decision.md Guideline 1**: Follows all established GOOD_RULES patterns
- **final-decision.md Guideline 2**: References `final-decision-N.md` format consistently  
- **simple logging**: Uses lightweight, OOP-based common utilities with simple logging using only the print functions from `ROOT/utils/print_utils.py` (such as `print_info`, `print_warning`, `print_success`, `print_error`, `print_important`). Never use raw print().

### **Design Philosophy**
- **Framework Integration**: Seamless integration with existing extensions architecture
- **Algorithm Diversity**: Access to multiple state-of-the-art RL algorithms
- **Production Readiness**: Battle-tested implementations for real-world deployment
- **Educational Excellence**: Clean examples of modern RL best practices

## 🏗️ **SB3 Integration Architecture**

### **Extension Structure**
Following established directory patterns:

```
extensions/reinforcement-v0.02/
├── stable_baselines/                    # SB3-specific implementations
│   ├── __init__.py                     # SB3 factory exports
│   ├── sb3_agent_factory.py            # SB3 agent creation
│   ├── sb3_environment_wrapper.py      # Gymnasium compatibility layer
│   ├── sb3_training_manager.py         # Training pipeline management
│   └── sb3_evaluation.py               # Evaluation and metrics
├── agents/                              # Standard agent directory
│   ├── agent_dqn_sb3.py               # SB3 DQN implementation
│   ├── agent_ppo_sb3.py               # SB3 PPO implementation
│   ├── agent_a2c_sb3.py               # SB3 A2C implementation
│   └── agent_sac_sb3.py               # SB3 SAC implementation
└── config/                              # Configuration
    ├── sb3_hyperparameters.py         # Algorithm-specific configs
    └── sb3_training_configs.py        # Training pipeline configs
```

### **SB3 Agent Factory**
Following established factory patterns:

```python
class SB3AgentFactory:
    """
    Factory for creating Stable Baselines3 agents
    
    Design Pattern: Factory Pattern
    Purpose: Create SB3 agents without exposing instantiation complexity
    Educational Note: Demonstrates clean integration between frameworks
    """
    
    _algorithm_registry = {
        "DQN": ("stable_baselines3", "DQN"),
        "PPO": ("stable_baselines3", "PPO"),
        "A2C": ("stable_baselines3", "A2C"),
        "SAC": ("stable_baselines3", "SAC"),
        "TD3": ("stable_baselines3", "TD3"),
    }
    
    @classmethod
    def create(cls, algorithm: str, env, **kwargs) -> BaseRLModel:
        """Create SB3 agent with specified algorithm"""
        if algorithm not in cls._algorithm_registry:
            available = list(cls._algorithm_registry.keys())
            raise ValueError(f"Algorithm '{algorithm}' not available. Available: {available}")
        
        module_name, class_name = cls._algorithm_registry[algorithm]
        module = importlib.import_module(module_name)
        algorithm_class = getattr(module, class_name)
        
        # Apply default configurations
        config = cls._get_default_config(algorithm)
        config.update(kwargs)
        
        from utils.print_utils import print_info
        print_info(f"[SB3AgentFactory] Creating {algorithm} agent")  # simple logging
        return algorithm_class(env=env, **config)
```

## 🧠 **Algorithm Implementations**

### **Deep Q-Network (DQN) Agent**
```python
class DQNAgentSB3(BaseRLAgent):
    """
    Deep Q-Network implementation using Stable Baselines3
    
    Design Pattern: Adapter Pattern
    Purpose: Adapt SB3 DQN to our agent interface
    Educational Note: Shows how to integrate external frameworks cleanly
    """
    
    def __init__(self, name: str = "DQN_SB3", grid_size: int = 10, **kwargs):
        super().__init__(name, grid_size)
        self.hyperparameters = self._get_hyperparameters(**kwargs)
        self.model = None
        self.environment = None
        from utils.print_utils import print_info
        print_info(f"[{name}] SB3 DQN agent initialized")  # simple logging
        
    def initialize(self, environment):
        """Initialize DQN model with environment"""
        self.environment = environment
        
        # Create SB3 DQN model
        self.model = DQN(
            policy="MlpPolicy",
            env=environment,
            learning_rate=self.hyperparameters['learning_rate'],
            buffer_size=self.hyperparameters['buffer_size'],
            batch_size=self.hyperparameters['batch_size'],
            gamma=self.hyperparameters['gamma'],
            verbose=1
        )
        from utils.print_utils import print_info
        print_info(f"[{self.name}] DQN model initialized")  # simple logging
    
    def train(self, total_timesteps: int = 100000, **kwargs) -> TrainingResults:
        """Train DQN agent using SB3"""
        if not self.model:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        from utils.print_utils import print_info
        print_info(f"[{self.name}] Starting training for {total_timesteps} timesteps")  # simple logging
        
        # Train the model
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        
        # Evaluate performance
        evaluation_results = self._evaluate_model()
        
        from utils.print_utils import print_success
        print_success(f"[{self.name}] Training completed. Final reward: {evaluation_results['mean_reward']:.2f}")  # simple logging
        
        return TrainingResults(
            algorithm="DQN_SB3",
            total_timesteps=total_timesteps,
            final_reward=evaluation_results['mean_reward']
        )
    
    def select_action(self, observation) -> int:
        """Select action using trained DQN model"""
        if not self.model:
            raise RuntimeError("Model not trained. Call train() first.")
        
        action, _states = self.model.predict(observation, deterministic=True)
        return int(action)
```

### **Proximal Policy Optimization (PPO) Agent**
```python
class PPOAgentSB3(BaseRLAgent):
    """
    Proximal Policy Optimization implementation using Stable Baselines3
    
    Educational Note: PPO is often considered the most robust RL algorithm
    for a wide variety of tasks, making it an excellent default choice.
    """
    
    def __init__(self, name: str = "PPO_SB3", grid_size: int = 10, **kwargs):
        super().__init__(name, grid_size)
        self.hyperparameters = self._get_hyperparameters(**kwargs)
        self.model = None
        from utils.print_utils import print_info
        print_info(f"[{name}] SB3 PPO agent initialized")  # simple logging
    
    def initialize(self, environment):
        """Initialize PPO model with environment"""
        self.model = PPO(
            policy="MlpPolicy",
            env=environment,
            learning_rate=self.hyperparameters['learning_rate'],
            n_steps=self.hyperparameters['n_steps'],
            batch_size=self.hyperparameters['batch_size'],
            n_epochs=self.hyperparameters['n_epochs'],
            gamma=self.hyperparameters['gamma'],
            verbose=1
        )
        from utils.print_utils import print_info
        print_info(f"[{self.name}] PPO model initialized")  # simple logging
```

## 🚀 **Advanced Features**

### **Environment Wrapper**
```python
class SnakeGameEnvironment(gym.Env):
    """Gymnasium-compatible Snake game environment for SB3"""
    
    def __init__(self, grid_size: int = 10):
        super().__init__()
        self.grid_size = grid_size
        from utils.print_utils import print_info
        print_info(f"[SnakeGameEnvironment] Initialized for {grid_size}x{grid_size} grid")  # simple logging
        
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
        truncated = False
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
```

## 🎓 **Educational Applications**

### **RL Algorithm Comparison**
- **DQN**: Value-based learning with experience replay
- **PPO**: Policy gradient with trust region optimization
- **A2C**: Advantage actor-critic with parallel environments
- **SAC**: Soft actor-critic for continuous action spaces

### **Framework Integration**
- **Clean Architecture**: Maintain separation between frameworks
- **Standardized Interfaces**: Consistent API across different RL libraries
- **Performance Benchmarking**: Compare different algorithm implementations
- **Production Deployment**: Use battle-tested implementations

## 🔗 **Integration with Other Extensions**

### **With Heuristics**
- Use heuristic algorithms for reward shaping
- Compare algorithmic vs. learned approaches
- Create hybrid heuristic-RL systems

### **With Supervised Learning**
- Use supervised models for function approximation
- Combine supervised and reinforcement learning
- Create hybrid learning systems

### **With Evolutionary Algorithms**
- Use evolutionary algorithms for hyperparameter optimization
- Evolve RL agent architectures
- Create evolutionary-RL hybrid systems

## 🔗 **See Also**

- **`agents.md`**: Authoritative reference for agent implementation standards
- **`core.md`**: Base class architecture for all agents
- **`final-decision.md`**: final-decision.md governance system

---

**Stable Baselines3 integration provides access to state-of-the-art reinforcement learning algorithms while maintaining clean architecture and educational value, enabling rapid development of robust RL solutions for Snake Game AI.**
