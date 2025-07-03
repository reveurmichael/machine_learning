# Reinforcement Learning Standards for Snake Game AI Extensions

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines reinforcement learning standards.

> **Guidelines Alignment:**
> - This document is governed by the guidelines in `final-decision-10.md`.
> - All extension factories must use the canonical method name `create()` (never `create_agent`, `create_model`, etc.).
> - All code must use simple print logging (simple logging).
> - Reference: `extensions/common/utils/factory_utils.py` for the canonical `SimpleFactory` implementation.
> - This file follows KEEP_THOSE_MARKDOWN_FILES_SIMPLE_RULES (target 300-500 lines).

> **See also:** `agents.md`, `core.md`, `final-decision-10.md`, `factory-design-pattern.md`, `config.md`, `stable-baseline.md`.

## 🎯 **Core Philosophy: Learning Through Experience**

Reinforcement learning in the Snake Game AI project enables **autonomous learning** through trial and error, where agents learn optimal strategies by interacting with the game environment and receiving rewards. This approach creates adaptive, intelligent agents that improve over time, strictly following `final-decision-10.md` SUPREME_RULES.

### **Educational Value**
- **Reinforcement Learning**: Understanding RL principles and algorithms
- **Environment Interaction**: Learning how agents interact with environments
- **Reward Design**: Understanding reward function design and optimization
- **Policy Learning**: Experience with different policy learning approaches

## 🏗️ **Factory Pattern: Canonical Method is create()**

All extension factories must use the canonical method name `create()` (never `create_agent`, `create_model`, etc.).

### **Reinforcement Learning Factory Implementation**
```python
class ReinforcementLearningFactory:
    """
    Factory for reinforcement learning agents following SUPREME_RULES.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create RL agents with canonical patterns
    Educational Value: Shows how canonical factory patterns work with RL systems
    """
    
    _registry = {
        "DQN": DQNAgent,
        "PPO": PPOAgent,
        "A3C": A3CAgent,
        "DDPG": DDPGAgent,
        "SAC": SACAgent,
    }
    
    @classmethod
    def create(cls, algorithm_type: str, **kwargs):  # CANONICAL create() method
        """Create RL agent using canonical create() method (SUPREME_RULES compliance)"""
        agent_class = cls._registry.get(algorithm_type.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown algorithm type: {algorithm_type}. Available: {available}")
        from utils.print_utils import print_info
        print_info(f"[ReinforcementLearningFactory] Creating agent: {algorithm_type}")  # Simple logging
        return agent_class(**kwargs)

# ❌ FORBIDDEN: Non-canonical method names (violates SUPREME_RULES)
class ReinforcementLearningFactory:
    def create_rl_agent(self, algorithm_type: str):  # FORBIDDEN - not canonical
        pass
    
    def build_reinforcement_agent(self, algorithm_type: str):  # FORBIDDEN - not canonical
        pass
    
    def make_rl_algorithm(self, algorithm_type: str):  # FORBIDDEN - not canonical
        pass
```

## 🧠 **Reinforcement Learning Architecture Patterns**

### **Deep Q-Network (DQN) Agent**
```python
class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent for reinforcement learning.
    
    Design Pattern: Strategy Pattern
    Purpose: Uses Q-learning with deep neural networks
    Educational Value: Shows how to implement DQN for game AI
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("DQN", config)
        self.q_network = None
        self.target_network = None
        self.replay_buffer = []
        self.epsilon = self.config.get('epsilon_start', 0.9)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self._build_networks()
        from utils.print_utils import print_info
        print_info(f"[DQNAgent] Initialized DQN agent")  # Simple logging
    
    def _build_networks(self):
        """Build Q-networks"""
        input_size = self.config.get('input_size', 16)
        hidden_size = self.config.get('hidden_size', 64)
        output_size = 4  # 4 directions
        
        self.q_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        self.target_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Copy weights from Q-network to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.get('learning_rate', 0.001))
        from utils.print_utils import print_info
        print_info(f"[DQNAgent] Built Q-networks")  # Simple logging
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan move using DQN algorithm"""
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Random exploration
            directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            move = random.choice(directions)
            from utils.print_utils import print_info
            print_info(f"[DQNAgent] Random exploration: {move}")  # Simple logging
        else:
            # Exploitation using Q-network
            state_vector = self._state_to_vector(game_state)
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state_vector))
                move_idx = torch.argmax(q_values).item()
            
            directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            move = directions[move_idx]
            from utils.print_utils import print_info
            print_info(f"[DQNAgent] Q-network exploitation: {move}")  # Simple logging
        
        return move
    
    def store_experience(self, state: Dict[str, Any], action: str, reward: float, 
                        next_state: Dict[str, Any], done: bool):
        """Store experience in replay buffer"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.replay_buffer.append(experience)
        
        # Limit buffer size
        if len(self.replay_buffer) > self.config.get('buffer_size', 10000):
            self.replay_buffer.pop(0)
    
    def train(self, batch_size: int = 32):
        """Train the Q-network"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor([self._state_to_vector(exp['state']) for exp in batch])
        actions = torch.LongTensor([self._action_to_index(exp['action']) for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_states = torch.FloatTensor([self._state_to_vector(exp['next_state']) for exp in batch])
        dones = torch.BoolTensor([exp['done'] for exp in batch])
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.get('gamma', 0.99) * next_q_values * ~dones)
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        from utils.print_utils import print_info
        print_info(f"[DQNAgent] Training loss: {loss.item():.4f}")  # Simple logging
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        from utils.print_utils import print_info
        print_info(f"[DQNAgent] Updated target network")  # Simple logging
    
    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        from utils.print_utils import print_info
        print_info(f"[DQNAgent] Updated epsilon to {self.epsilon:.3f}")  # Simple logging
    
    def _state_to_vector(self, game_state: Dict[str, Any]) -> List[float]:
        """Convert game state to vector"""
        # Similar to supervised learning feature extraction
        features = []
        head_pos = game_state['snake_positions'][0]
        apple_pos = game_state['apple_position']
        
        # Position features
        features.extend([head_pos[0], head_pos[1], apple_pos[0], apple_pos[1]])
        
        # Game state features
        features.append(len(game_state['snake_positions']))
        
        # Direction features
        features.extend(self._get_direction_features(head_pos, apple_pos))
        
        # Danger features
        features.extend(self._get_danger_features(game_state))
        
        # Free space features
        features.extend(self._get_free_space_features(game_state))
        
        return features
    
    def _action_to_index(self, action: str) -> int:
        """Convert action string to index"""
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        return actions.index(action)
    
    def _get_direction_features(self, head_pos: tuple, apple_pos: tuple) -> List[float]:
        """Get direction features"""
        dx = apple_pos[0] - head_pos[0]
        dy = apple_pos[1] - head_pos[1]
        
        return [
            1.0 if dy < 0 else 0.0,  # apple_dir_up
            1.0 if dy > 0 else 0.0,  # apple_dir_down
            1.0 if dx < 0 else 0.0,  # apple_dir_left
            1.0 if dx > 0 else 0.0,  # apple_dir_right
        ]
    
    def _get_danger_features(self, game_state: Dict[str, Any]) -> List[float]:
        """Get danger features"""
        head_pos = game_state['snake_positions'][0]
        grid_size = game_state['grid_size']
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        danger_features = []
        
        for dx, dy in directions:
            new_pos = (head_pos[0] + dx, head_pos[1] + dy)
            is_danger = (
                new_pos in game_state['snake_positions'] or
                not (0 <= new_pos[0] < grid_size and 0 <= new_pos[1] < grid_size)
            )
            danger_features.append(1.0 if is_danger else 0.0)
        
        return danger_features[:3]  # Only straight, left, right
    
    def _get_free_space_features(self, game_state: Dict[str, Any]) -> List[float]:
        """Get free space features"""
        head_pos = game_state['snake_positions'][0]
        grid_size = game_state['grid_size']
        snake_positions = set(game_state['snake_positions'])
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        free_space_features = []
        
        for dx, dy in directions:
            free_count = 0
            current_pos = head_pos
            
            for _ in range(grid_size):
                current_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if (0 <= current_pos[0] < grid_size and 
                    0 <= current_pos[1] < grid_size and 
                    current_pos not in snake_positions):
                    free_count += 1
                else:
                    break
            
            free_space_features.append(free_count)
        
        return free_space_features
```

### **Proximal Policy Optimization (PPO) Agent**
```python
class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization agent for reinforcement learning.
    
    Design Pattern: Strategy Pattern
    Purpose: Uses PPO algorithm for policy optimization
    Educational Value: Shows how to implement PPO for game AI
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("PPO", config)
        self.policy_network = None
        self.value_network = None
        self.optimizer = None
        self._build_networks()
        from utils.print_utils import print_info
        print_info(f"[PPOAgent] Initialized PPO agent")  # Simple logging
    
    def _build_networks(self):
        """Build policy and value networks"""
        input_size = self.config.get('input_size', 16)
        hidden_size = self.config.get('hidden_size', 64)
        
        # Policy network
        self.policy_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)  # 4 actions
        )
        
        # Value network
        self.value_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Value estimate
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            lr=self.config.get('learning_rate', 0.0003)
        )
        
        from utils.print_utils import print_info
        print_info(f"[PPOAgent] Built policy and value networks")  # Simple logging
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan move using PPO policy"""
        state_vector = self._state_to_vector(game_state)
        
        # Get action probabilities
        with torch.no_grad():
            action_logits = self.policy_network(torch.FloatTensor(state_vector))
            action_probs = torch.softmax(action_logits, dim=0)
            
            # Sample action
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample().item()
        
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        move = directions[action_idx]
        
        from utils.print_utils import print_info
        print_info(f"[PPOAgent] Selected move: {move}")  # Simple logging
        return move
    
    def get_action_log_prob(self, state: Dict[str, Any], action: str) -> float:
        """Get log probability of action"""
        state_vector = self._state_to_vector(state)
        action_logits = self.policy_network(torch.FloatTensor(state_vector))
        action_probs = torch.softmax(action_logits, dim=0)
        action_dist = torch.distributions.Categorical(action_probs)
        
        action_idx = self._action_to_index(action)
        return action_dist.log_prob(torch.tensor(action_idx)).item()
    
    def get_value(self, state: Dict[str, Any]) -> float:
        """Get value estimate for state"""
        state_vector = self._state_to_vector(state)
        with torch.no_grad():
            value = self.value_network(torch.FloatTensor(state_vector))
        return value.item()
    
    def _state_to_vector(self, game_state: Dict[str, Any]) -> List[float]:
        """Convert game state to vector"""
        # Similar to DQN agent
        features = []
        head_pos = game_state['snake_positions'][0]
        apple_pos = game_state['apple_position']
        
        features.extend([head_pos[0], head_pos[1], apple_pos[0], apple_pos[1]])
        features.append(len(game_state['snake_positions']))
        features.extend(self._get_direction_features(head_pos, apple_pos))
        features.extend(self._get_danger_features(game_state))
        features.extend(self._get_free_space_features(game_state))
        
        return features
    
    def _action_to_index(self, action: str) -> int:
        """Convert action string to index"""
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        return actions.index(action)
    
    def _get_direction_features(self, head_pos: tuple, apple_pos: tuple) -> List[float]:
        """Get direction features"""
        dx = apple_pos[0] - head_pos[0]
        dy = apple_pos[1] - head_pos[1]
        
        return [
            1.0 if dy < 0 else 0.0,  # apple_dir_up
            1.0 if dy > 0 else 0.0,  # apple_dir_down
            1.0 if dx < 0 else 0.0,  # apple_dir_left
            1.0 if dx > 0 else 0.0,  # apple_dir_right
        ]
    
    def _get_danger_features(self, game_state: Dict[str, Any]) -> List[float]:
        """Get danger features"""
        head_pos = game_state['snake_positions'][0]
        grid_size = game_state['grid_size']
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        danger_features = []
        
        for dx, dy in directions:
            new_pos = (head_pos[0] + dx, head_pos[1] + dy)
            is_danger = (
                new_pos in game_state['snake_positions'] or
                not (0 <= new_pos[0] < grid_size and 0 <= new_pos[1] < grid_size)
            )
            danger_features.append(1.0 if is_danger else 0.0)
        
        return danger_features[:3]  # Only straight, left, right
    
    def _get_free_space_features(self, game_state: Dict[str, Any]) -> List[float]:
        """Get free space features"""
        head_pos = game_state['snake_positions'][0]
        grid_size = game_state['grid_size']
        snake_positions = set(game_state['snake_positions'])
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        free_space_features = []
        
        for dx, dy in directions:
            free_count = 0
            current_pos = head_pos
            
            for _ in range(grid_size):
                current_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if (0 <= current_pos[0] < grid_size and 
                    0 <= current_pos[1] < grid_size and 
                    current_pos not in snake_positions):
                    free_count += 1
                else:
                    break
            
            free_space_features.append(free_count)
        
        return free_space_features
```

## 📊 **Training Pipeline Standards**

### **Reinforcement Learning Training Pipeline**
```python
class RLTrainingPipeline:
    """
    Training pipeline for reinforcement learning agents.
    
    Design Pattern: Template Method Pattern
    Purpose: Provides consistent training workflow
    Educational Value: Shows how to train different RL algorithms
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent = None
        self.environment = None
        from utils.print_utils import print_info
        print_info(f"[RLTrainingPipeline] Initialized training pipeline")  # Simple logging
    
    def setup_environment(self, grid_size: int = 10):
        """Setup training environment"""
        self.environment = SnakeGameEnvironment(grid_size)
        from utils.print_utils import print_info
        print_info(f"[RLTrainingPipeline] Environment setup complete")  # Simple logging
    
    def setup_agent(self, algorithm_type: str):
        """Setup RL agent"""
        self.agent = ReinforcementLearningFactory.create(algorithm_type, **self.config)
        from utils.print_utils import print_info
        print_info(f"[RLTrainingPipeline] Agent setup complete")  # Simple logging
    
    def train(self, num_episodes: int):
        """Train the agent"""
        from utils.print_utils import print_info
        print_info(f"[RLTrainingPipeline] Starting training for {num_episodes} episodes")  # Simple logging
        
        for episode in range(num_episodes):
            episode_reward = self._train_episode()
            
            if episode % 100 == 0:
                from utils.print_utils import print_info
                print_info(f"[RLTrainingPipeline] Episode {episode+1}/{num_episodes} - "
                          f"Reward: {episode_reward:.2f}")  # Simple logging
    
    def _train_episode(self) -> float:
        """Train for one episode"""
        state = self.environment.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            # Agent selects action
            action = self.agent.plan_move(state)
            
            # Environment step
            next_state, reward, done, _ = self.environment.step(action)
            
            # Store experience (for algorithms that use replay buffer)
            if hasattr(self.agent, 'store_experience'):
                self.agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent (for algorithms that train online)
            if hasattr(self.agent, 'train'):
                self.agent.train()
            
            state = next_state
            total_reward += reward
        
        return total_reward
    
    def save_agent(self, agent_path: str):
        """Save trained agent"""
        if hasattr(self.agent, 'save_model'):
            self.agent.save_model(agent_path)
        else:
            torch.save(self.agent.state_dict(), agent_path)
        print(f"[RLTrainingPipeline] Agent saved to {agent_path}")  # Simple logging
```

## 📋 **Implementation Checklist**

### **Required Components**
- [ ] **Factory Pattern**: Uses canonical `create()` method
- [ ] **Agent Architecture**: Implements appropriate RL algorithm
- [ ] **Environment Interface**: Proper environment interaction
- [ ] **Training Pipeline**: Standardized training workflow
- [ ] **Simple Logging**: Uses print() statements for debugging

### **Quality Standards**
- [ ] **Agent Performance**: Meets performance benchmarks
- [ ] **Learning Efficiency**: Efficient learning process
- [ ] **Exploration Strategy**: Appropriate exploration mechanisms
- [ ] **Documentation**: Clear documentation of agent capabilities

### **Integration Requirements**
- [ ] **Environment Compatibility**: Works with standard game environment
- [ ] **Factory Integration**: Compatible with agent factory patterns
- [ ] **Configuration**: Supports standard configuration system
- [ ] **Evaluation**: Integrates with evaluation framework

---

**Reinforcement learning standards ensure consistent, high-quality RL implementations across all Snake Game AI extensions. By following these standards, developers can create robust, educational, and adaptive agents that integrate seamlessly with the overall framework.**

## 🔗 **See Also**

- **`agents.md`**: Agent implementation standards
- **`core.md`**: Base class architecture and inheritance patterns
- **`config.md`**: Configuration management
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`factory-design-pattern.md`**: Factory pattern implementation
