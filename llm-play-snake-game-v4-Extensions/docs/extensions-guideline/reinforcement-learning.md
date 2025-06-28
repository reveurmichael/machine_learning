# Reinforcement Learning for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines reinforcement learning patterns for extensions.

> **Guidelines Alignment:**
> - This document is governed by the guidelines in `final-decision-10.md`.
> - All extension factories must use the canonical method name `create()` (never `create_agent`, `create_model`, etc.).
> - All code must use simple print logging (simple logging).
> - Reference: `extensions/common/utils/factory_utils.py` for the canonical `SimpleFactory` implementation.
> - This file follows KEEP_THOSE_MARKDOWN_FILES_SIMPLE_RULES (target 300-500 lines).

> **See also:** `agents.md`, `core.md`, `final-decision-10.md`, `factory-design-pattern.md`, `config.md`, `stable-baseline.md`.

# Reinforcement Learning for Snake Game AI

## 🎯 **Core Philosophy: Learning Through Experience + SUPREME_RULES**

Reinforcement learning enables agents to learn optimal Snake game strategies through trial and error. **This extension strictly follows the SUPREME_RULES** established in `final-decision-10.md`, particularly the **canonical `create()` method patterns and simple logging requirements** for all learning-based systems.

### **Guidelines Alignment**
- **final-decision-10.md Guideline 1**: Follows all established GOOD_RULES patterns for reinforcement learning architectures
- **final-decision-10.md Guideline 2**: Uses precise `final-decision-N.md` format consistently throughout RL implementations
- **simple logging**: Lightweight, OOP-based common utilities with simple logging (print() statements only)

### **Educational Value**
- **Trial-and-Error Learning**: Understand autonomous learning using canonical patterns
- **Policy Optimization**: Experience gradient-based learning with simple logging throughout
- **Value Function Learning**: Learn temporal difference methods following SUPREME_RULES compliance
- **Exploration vs Exploitation**: See canonical patterns in learning-exploration trade-offs

## 🏗️ **Extension Structure**

### **Directory Layout**
```
extensions/reinforcement-v0.02/
├── __init__.py
├── agents/
│   ├── __init__.py               # Agent factory
│   ├── agent_dqn.py              # Deep Q-Network
│   ├── agent_ppo.py              # Proximal Policy Optimization
│   ├── agent_a3c.py              # Asynchronous Advantage Actor-Critic
│   └── agent_ddpg.py             # Deep Deterministic Policy Gradient
├── environments/
│   ├── __init__.py
│   ├── snake_env.py              # Snake game environment
│   └── env_wrapper.py            # Environment wrapper
├── models/
│   ├── __init__.py
│   ├── neural_networks.py        # Neural network architectures
│   └── model_manager.py          # Model management
├── training/
│   ├── __init__.py
│   ├── trainer.py                # Training pipeline
│   └── experience_replay.py      # Experience replay buffer
├── game_logic.py                 # RL game logic
├── game_manager.py               # RL manager
└── main.py                       # CLI interface
```

## 🔧 **Implementation Patterns**

### **RL Agent Factory (SUPREME_RULES Compliant)**
**CRITICAL REQUIREMENT**: All RL factories MUST use the canonical `create()` method exactly as specified in `final-decision-10.md` SUPREME_RULES:

```python
class RLAgentFactory:
    """
    Factory Pattern for Reinforcement Learning agents following final-decision-10.md SUPREME_RULES
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for RL AI agents
    Educational Value: Shows how SUPREME_RULES apply to advanced AI systems -
    canonical patterns work regardless of AI complexity.
    
    Reference: final-decision-10.md SUPREME_RULES for canonical method naming
    """
    
    _registry = {
        "DQN": DQNAgent,
        "PPO": PPOAgent,
        "A2C": A2CAgent,
        "SAC": SACAgent,
        "TD3": TD3Agent,
    }
    
    @classmethod
    def create(cls, algorithm_type: str, **kwargs):  # CANONICAL create() method - SUPREME_RULES
        """Create RL agent using canonical create() method following final-decision-10.md"""
        agent_class = cls._registry.get(algorithm_type.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown RL algorithm: {algorithm_type}. Available: {available}")
        print(f"[RLAgentFactory] Creating agent: {algorithm_type}")  # Simple logging - SUPREME_RULES
        return agent_class(**kwargs)

# ❌ FORBIDDEN: Non-canonical method names (violates SUPREME_RULES)
class RLAgentFactory:
    def create_rl_agent(self, algorithm_type: str):  # FORBIDDEN - not canonical
        pass
    
    def build_reinforcement_agent(self, algorithm_type: str):  # FORBIDDEN - not canonical
        pass
    
    def make_rl_algorithm(self, algorithm_type: str):  # FORBIDDEN - not canonical
        pass
```

### **DQN Agent Implementation (CANONICAL PATTERNS)**
```python
class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent for Snake Game following SUPREME_RULES.
    
    Design Pattern: Strategy Pattern (Canonical Implementation)
    Purpose: Deep reinforcement learning using canonical patterns
    Educational Value: Shows how canonical factory patterns work with
    neural network learning while maintaining simple logging.
    
    Reference: final-decision-10.md for canonical agent architecture
    """
    
    def __init__(self, name: str, grid_size: int,
                 network_type: str = "MLP",
                 buffer_type: str = "EXPERIENCE_REPLAY"):
        super().__init__(name, grid_size)
        
        # Use canonical factory patterns for RL components
        self.q_network = NeuralNetworkFactory.create(network_type, input_dim=grid_size*grid_size)  # Canonical
        self.target_network = NeuralNetworkFactory.create(network_type, input_dim=grid_size*grid_size)  # Canonical
        self.replay_buffer = ReplayBufferFactory.create(buffer_type, capacity=10000)  # Canonical
        
        self.epsilon = 1.0
        self.episode_count = 0
        
        print(f"[{name}] DQN Agent initialized with {network_type} network")  # Simple logging
    
    def plan_move(self, game_state: dict) -> str:
        """Plan move using DQN policy with simple logging throughout"""
        print(f"[{self.name}] Starting DQN decision process")  # Simple logging
        
        state = self._preprocess_state(game_state)
        print(f"[{self.name}] State preprocessed")  # Simple logging
        
        # Epsilon-greedy action selection using simple logic
        if random.random() < self.epsilon:
            action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
            print(f"[{self.name}] Random action: {action} (epsilon: {self.epsilon:.3f})")  # Simple logging
        else:
            q_values = self.q_network.predict(state)
            action_idx = np.argmax(q_values)
            action = ['UP', 'DOWN', 'LEFT', 'RIGHT'][action_idx]
            print(f"[{self.name}] Greedy action: {action} (Q-value: {np.max(q_values):.3f})")  # Simple logging
        
        print(f"[{self.name}] DQN decided: {action}")  # Simple logging
        return action
    
    def train_step(self, batch_size: int = 32) -> None:
        """Execute one training step with simple logging"""
        print(f"[{self.name}] Starting DQN training step")  # Simple logging
        
        if len(self.replay_buffer) < batch_size:
            print(f"[{self.name}] Insufficient experience for training")  # Simple logging
            return
        
        # Sample batch using canonical buffer
        batch = self.replay_buffer.sample(batch_size)
        print(f"[{self.name}] Sampled batch of {batch_size} experiences")  # Simple logging
        
        # Compute targets and update network
        loss = self._update_q_network(batch)
        print(f"[{self.name}] Q-network updated, loss: {loss:.4f}")  # Simple logging
        
        # Update target network periodically
        if self.episode_count % 100 == 0:
            self._update_target_network()
            print(f"[{self.name}] Target network updated")  # Simple logging
    
    def update_epsilon(self, episode: int, total_episodes: int) -> None:
        """Update exploration rate with simple logging"""
        old_epsilon = self.epsilon
        self.epsilon = max(0.01, 1.0 - episode / (total_episodes * 0.8))
        self.episode_count = episode
        
        print(f"[{self.name}] Epsilon updated: {old_epsilon:.3f} → {self.epsilon:.3f}")  # Simple logging
    
    def store_experience(self, state, action, reward, next_state, done) -> None:
        """Store experience in replay buffer with simple logging"""
        self.replay_buffer.add(state, action, reward, next_state, done)
        print(f"[{self.name}] Experience stored, buffer size: {len(self.replay_buffer)}")  # Simple logging
```

## 📊 **Environment Integration**

### **Snake Environment**
```python
class SnakeEnvironment:
    """
    Snake game environment for RL training
    
    Design Pattern: Adapter Pattern
    - Adapts Snake game to RL environment interface
    - Provides standardized observation and reward
    - Enables easy integration with RL frameworks
    """
    
    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self.reset()
        print(f"[SnakeEnvironment] Initialized {grid_size}x{grid_size} environment")
    
    def reset(self):
        """Reset environment to initial state"""
        self.snake_positions = [(5, 5)]
        self.apple_position = self._generate_apple()
        self.direction = 'NONE'
        self.score = 0
        self.steps = 0
        self.done = False
        print("[SnakeEnvironment] Environment reset")
        return self._get_observation()
    
    def step(self, action: str):
        """Execute action and return (observation, reward, done, info)"""
        if self.done:
            return self._get_observation(), 0, True, {}
        
        # Execute action
        old_score = self.score
        self._execute_action(action)
        
        # Calculate reward
        reward = self._calculate_reward(old_score)
        
        # Check termination
        self.done = self._check_termination()
        
        print(f"[SnakeEnvironment] Action: {action}, Reward: {reward}, Score: {self.score}")
        return self._get_observation(), reward, self.done, {}
```

## 🚀 **Advanced Features**

### **Experience Replay**
```python
class ExperienceReplayBuffer:
    """
    Experience replay buffer for stable learning
    
    Design Pattern: Decorator Pattern
    - Adds replay functionality to base learning
    - Improves sample efficiency and stability
    - Enables batch learning from past experiences
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        print(f"[ExperienceReplayBuffer] Initialized with capacity: {capacity}")
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        print(f"[ExperienceReplayBuffer] Sampled {len(batch)} experiences")
        return batch
```

### **PPO Implementation**
```python
class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization agent
    
    Design Pattern: Strategy Pattern
    - Implements PPO algorithm for policy optimization
    - Uses clipped objective for stable updates
    - Maintains separate policy and value networks
    """
    
    def __init__(self, name: str, grid_size: int):
        super().__init__(name, grid_size)
        self.policy_network = self._build_policy_network()
        self.value_network = self._build_value_network()
        self.clip_ratio = 0.2
        print(f"[PPOAgent] Initialized PPO agent: {name}")
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan move using PPO policy"""
        state = self._preprocess_state(game_state)
        action_probs = self.policy_network.predict(state)
        action_idx = np.random.choice(4, p=action_probs)
        action = ['UP', 'DOWN', 'LEFT', 'RIGHT'][action_idx]
        print(f"[PPOAgent] Selected action: {action} (prob: {action_probs[action_idx]:.3f})")
        return action
```

## 📋 **Configuration and Usage**

### **RL Configuration**
```python
RL_CONFIG = {
    'dqn': {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 32
    },
    'ppo': {
        'learning_rate': 0.0003,
        'gamma': 0.99,
        'clip_ratio': 0.2,
        'batch_size': 64,
        'epochs_per_update': 10
    },
    'a3c': {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'entropy_beta': 0.01,
        'num_workers': 4
    }
}
```

### **Training Commands**
```bash
python main.py --agent DQN --episodes 1000 --epsilon-decay 0.995
python main.py --agent PPO --episodes 500 --batch-size 64
python main.py --agent A3C --episodes 2000 --workers 4
```

## 🔗 **Integration with Other Extensions**

### **With Heuristics Extensions**
- Use heuristic performance as baseline for RL training
- Compare learned vs. algorithmic approaches
- Analyze exploration vs. exploitation strategies

### **With Supervised Learning**
- RL complements supervised approaches
- Provides autonomous learning capabilities
- Enables continuous improvement through experience

## 🎓 **Educational Applications**

### **Learning Algorithms**
- **Q-Learning**: Value-based learning with function approximation
- **Policy Gradient**: Direct policy optimization
- **Actor-Critic**: Combined value and policy learning
- **Multi-Agent**: Cooperative and competitive learning

### **Performance Analysis**
- **Sample Efficiency**: Learning speed and data requirements
- **Exploration**: Balancing exploration and exploitation
- **Generalization**: Performance on unseen scenarios
- **Stability**: Training stability and convergence

---

**Reinforcement learning provides autonomous learning capabilities for Snake game AI, enabling agents to discover sophisticated strategies through experience while maintaining full compliance with established GOOD_RULES standards.**
