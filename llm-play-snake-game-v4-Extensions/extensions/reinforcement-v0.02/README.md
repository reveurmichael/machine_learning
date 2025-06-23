# Reinforcement Learning v0.02 - Multi-Agent Framework

A comprehensive reinforcement learning framework for Snake game AI supporting multiple RL algorithms with standardized training, evaluation, and model management.

## 🎯 **Design Philosophy**

- **Multi-Agent Support**: DQN, PPO, A3C, SAC algorithms
- **Standardized Training**: Unified training pipeline across all agents
- **Grid Size Flexibility**: Support for arbitrary grid sizes (8x8 to 20x20)
- **Experience Management**: Efficient replay buffers and experience collection
- **No Backward Compatibility**: Fresh, future-proof codebase

## 🏗️ **Architecture Overview**

### **Design Patterns Used**

1. **Template Method Pattern**: Standardized training and evaluation pipelines
2. **Strategy Pattern**: Pluggable RL algorithm implementations
3. **Factory Pattern**: Agent creation and management
4. **Singleton Pattern**: Experience replay and utility management

### **Key Components**

```
reinforcement-v0.02/
├── scripts/
│   ├── train.py             # CLI training script
│   ├── evaluate.py          # CLI evaluation script
│   └── compare.py           # Model comparison script
├── agents/
│   ├── dqn_agent.py         # Deep Q-Network agent
│   ├── ppo_agent.py         # Proximal Policy Optimization
│   ├── a3c_agent.py         # Asynchronous Advantage Actor-Critic
│   └── sac_agent.py         # Soft Actor-Critic
├── models/
│   ├── networks/            # Neural network architectures
│   └── policies/            # Policy implementations
└── utils/
    ├── replay_buffer.py     # Experience replay utilities
    └── environment.py       # Environment wrappers
```

## 🚀 **Quick Start**

### **Training Agents**

```bash
# Train DQN agent
python scripts/train.py --agent DQN --grid-size 15 --max-episodes 1000

# Train PPO agent
python scripts/train.py --agent PPO --grid-size 10 --max-episodes 500

# Train A3C agent
python scripts/train.py --agent A3C --grid-size 12 --max-episodes 800

# Train SAC agent
python scripts/train.py --agent SAC --grid-size 15 --max-episodes 600
```

### **Evaluation**

```bash
# Evaluate trained agent
python scripts/evaluate.py --agent DQN --grid-size 15

# Compare multiple agents
python scripts/compare.py --agents DQN,PPO,SAC --grid-size 10
```

### **Advanced Training**

```bash
# Custom hyperparameters
python scripts/train.py --agent DQN \
    --grid-size 15 \
    --max-episodes 2000 \
    --learning-rate 0.0005 \
    --gamma 0.95 \
    --epsilon 0.05 \
    --hidden-size 512 \
    --memory-size 50000 \
    --batch-size 64
```

## 🧠 **Supported Agents**

### **Deep Q-Network (DQN)**

| Feature | Description | Default Value |
|---------|-------------|---------------|
| **Architecture** | Deep Q-Network with experience replay | - |
| **Exploration** | Epsilon-greedy strategy | ε = 0.1 |
| **Memory** | Experience replay buffer | 10,000 experiences |
| **Target Network** | Fixed target network updates | Every 100 steps |
| **Loss Function** | Mean Squared Error | - |

**Key Advantages:**
- Stable training with experience replay
- Handles discrete action spaces efficiently
- Good for exploration-heavy environments

### **Proximal Policy Optimization (PPO)**

| Feature | Description | Default Value |
|---------|-------------|---------------|
| **Architecture** | Actor-Critic with clipped objectives | - |
| **Exploration** | Stochastic policy sampling | - |
| **Memory** | On-policy experience collection | - |
| **Clipping** | PPO clipping parameter | ε = 0.2 |
| **Loss Function** | Clipped surrogate objective | - |

**Key Advantages:**
- Sample efficient training
- Stable policy updates
- Good for continuous and discrete actions

### **Asynchronous Advantage Actor-Critic (A3C)**

| Feature | Description | Default Value |
|---------|-------------|---------------|
| **Architecture** | Asynchronous actor-critic networks | - |
| **Exploration** | Entropy regularization | - |
| **Memory** | Distributed experience collection | - |
| **Parallelism** | Multi-threaded training | - |
| **Loss Function** | Policy gradient + value function | - |

**Key Advantages:**
- Parallel training across multiple environments
- Efficient exploration through entropy
- Good for distributed training

### **Soft Actor-Critic (SAC)**

| Feature | Description | Default Value |
|---------|-------------|---------------|
| **Architecture** | Actor-Critic with entropy maximization | - |
| **Exploration** | Maximum entropy reinforcement learning | - |
| **Memory** | Experience replay buffer | 10,000 experiences |
| **Temperature** | Entropy regularization coefficient | α = 0.2 |
| **Loss Function** | Soft Q-learning + policy gradient | - |

**Key Advantages:**
- Sample efficient and stable
- Automatic temperature tuning
- Good for continuous action spaces

## 📊 **Model Management**

### **Standardized Saving/Loading**

All agents follow standardized saving/loading patterns:

```python
# Save agent with metadata
agent.save_model("my_dqn_agent", export_onnx=True)

# Load agent
agent.load_model("my_dqn_agent")
```

### **Model Directory Structure**

```
logs/extensions/models/
├── grid-size-8/
│   ├── dqn/
│   │   ├── dqn_agent.pth
│   │   ├── dqn_agent.onnx
│   │   └── dqn_agent_metadata.json
│   ├── ppo/
│   │   ├── ppo_agent.pth
│   │   ├── ppo_agent.onnx
│   │   └── ppo_agent_metadata.json
│   └── sac/
│       ├── sac_agent.pth
│       ├── sac_agent.onnx
│       └── sac_agent_metadata.json
├── grid-size-10/
└── grid-size-15/
```

### **Agent Metadata**

Each saved agent includes rich metadata:

```json
{
  "agent_type": "DQN",
  "grid_size": 15,
  "input_size": 229,
  "hidden_size": 256,
  "learning_rate": 0.001,
  "gamma": 0.99,
  "epsilon": 0.1,
  "memory_size": 10000,
  "target_update": 100,
  "torch_version": "2.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "training_metrics": {
    "episodes_trained": 1000,
    "final_avg_reward": 15.5,
    "final_avg_length": 45.2
  }
}
```

## 🔧 **Configuration**

### **Grid Size Support**

The framework supports arbitrary grid sizes:

```python
# Different grid sizes
agent_8x8 = DQNAgent(grid_size=8)    # 8x8 grid
agent_10x10 = DQNAgent(grid_size=10) # 10x10 grid (default)
agent_15x15 = DQNAgent(grid_size=15) # 15x15 grid
agent_20x20 = DQNAgent(grid_size=20) # 20x20 grid
```

### **State Representation**

Automatic state representation based on grid size:

```python
# State size scales with grid size
grid_size_8 = 68 features   # 8x8 + 4 position features
grid_size_10 = 104 features # 10x10 + 4 position features  
grid_size_15 = 229 features # 15x15 + 4 position features
grid_size_20 = 404 features # 20x20 + 4 position features
```

## 📈 **Training & Evaluation**

### **Training Metrics**

All agents track comprehensive training metrics:

- **Episode Rewards**: Average reward per episode
- **Episode Lengths**: Average steps per episode
- **Loss Values**: Training loss curves
- **Exploration Rate**: Epsilon decay (for DQN)
- **Training Time**: Wall-clock training duration

### **Evaluation Metrics**

Standardized evaluation across all agents:

- **Game Performance**: Average score, win rate, steps
- **Policy Quality**: Action distribution analysis
- **Exploration**: State coverage and diversity
- **Inference Speed**: Action selection latency
- **Memory Usage**: Model and buffer memory footprint

### **Agent Comparison**

```python
# Compare multiple agents
results = compare_agents(["DQN", "PPO", "SAC"], grid_size=15)
```

## 🎮 **Environment Integration**

### **Game Environment**

Seamless integration with Snake game environment:

```python
# Environment setup
env = SnakeGameEnvironment(grid_size=15)

# Agent training
agent = DQNAgent(grid_size=15)
agent.train(env, max_episodes=1000)
```

### **Custom Environments**

Support for custom environment wrappers:

```python
# Custom environment wrapper
class CustomSnakeEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.game_logic = GameLogic(grid_size)
    
    def reset(self):
        # Reset environment
        pass
    
    def step(self, action):
        # Execute action and return (state, reward, done, info)
        pass
```

## 🔄 **Experience Management**

### **Replay Buffer**

Efficient experience replay for off-policy algorithms:

```python
# Replay buffer configuration
buffer = ReplayBuffer(capacity=10000)

# Experience storage
buffer.push(state, action, reward, next_state, done)

# Experience sampling
batch = buffer.sample(batch_size=32)
```

### **Experience Collection**

Optimized experience collection strategies:

- **Uniform Sampling**: Random experience selection
- **Prioritized Sampling**: Priority-based experience selection
- **Multi-Step Returns**: N-step return calculations
- **Experience Augmentation**: Data augmentation techniques

## 🛠️ **Development**

### **Adding New Agents**

1. **Create Agent Class**: Implement the agent interface
2. **Add Training Logic**: Implement training pipeline
3. **Add to CLI**: Update training script
4. **Add Evaluation**: Create evaluation methods
5. **Add Tests**: Create unit tests

### **Agent Interface**

```python
class BaseRLAgent:
    def __init__(self, grid_size: int, **kwargs):
        pass
    
    def get_action(self, state: np.ndarray) -> int:
        pass
    
    def train(self, env, max_episodes: int, **kwargs) -> Dict[str, Any]:
        pass
    
    def save_model(self, agent_name: str) -> str:
        pass
    
    def load_model(self, agent_path: str) -> None:
        pass
```

### **Network Architectures**

Customizable neural network architectures:

```python
# Custom network architecture
class CustomNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)
```

## 📚 **Documentation**

- **API Reference**: Complete agent and utility documentation
- **Tutorials**: Step-by-step training and evaluation guides
- **Examples**: Code examples for all agent types
- **Best Practices**: Recommended configurations and workflows

## 🤝 **Contributing**

1. **Fork the Repository**: Create your own fork
2. **Create Feature Branch**: Work on new features
3. **Follow Standards**: Adhere to coding standards and patterns
4. **Add Tests**: Include comprehensive tests
5. **Submit Pull Request**: Submit for review

## 📄 **License**

This extension follows the same license as the main project.

---

**Reinforcement Learning v0.02** - Modern, extensible, and future-proof reinforcement learning framework for Snake game AI. 