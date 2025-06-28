# Stable Baselines3 Integration for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and follows **KEEP_THOSE_MARKDOWN_FILES_SIMPLE_RULES** guidelines with a target length of 300-500 lines.

## 🎯 **Core Philosophy: Production-Ready RL Framework Integration**

Stable Baselines3 (SB3) provides state-of-the-art reinforcement learning algorithms with a standardized interface. In the Snake Game AI ecosystem, SB3 enables rapid prototyping and deployment of proven RL algorithms while maintaining compatibility with the native architecture.

### **SUPREME_RULES Alignment**
- **SUPREME_RULE NO.1**: Follows all established GOOD_RULES patterns for RL framework integration
- **SUPREME_RULE NO.2**: Uses precise `final-decision-N.md` format consistently throughout SB3 references
- **SUPREME_RULE NO.3**: Uses lightweight, OOP-based common utilities with simple logging (print() statements) rather than complex *.log file mechanisms

### **Design Philosophy**
- **Framework Integration**: Seamless integration with existing extensions architecture
- **Algorithm Diversity**: Access to multiple state-of-the-art RL algorithms
- **Production Readiness**: Battle-tested implementations for real-world deployment
- **Educational Excellence**: Clean examples of modern RL best practices

## 🏗️ **SB3 Integration Architecture**

### **Extension Structure**
Following Final Decision 5 directory patterns:

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
Following Final Decision 7-8 factory patterns:

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
    def create_agent(cls, algorithm: str, env, **kwargs) -> BaseRLModel:
        """
        Create SB3 agent with specified algorithm (Following SUPREME_RULE NO.3)
        
        Educational Note (SUPREME_RULE NO.3):
        We should be able to add new extensions easily and try out new ideas.
        This factory is designed to be flexible and extensible.
        """
        if algorithm not in cls._algorithm_registry:
            available = list(cls._algorithm_registry.keys())
            raise ValueError(f"Algorithm '{algorithm}' not available in SB3 registry. "
                           f"Available: {available}. You can register new algorithms "
                           f"by extending the _algorithm_registry.")
        
        module_name, class_name = cls._algorithm_registry[algorithm]
        module = importlib.import_module(module_name)
        algorithm_class = getattr(module, class_name)
        
        # Apply default configurations
        config = cls._get_default_config(algorithm)
        config.update(kwargs)
        
        return algorithm_class(env=env, **config)
    
    @classmethod
    def _get_default_config(cls, algorithm: str) -> Dict[str, Any]:
        """Get default hyperparameters for algorithm"""
        from extensions.common.config.sb3_hyperparameters import SB3_DEFAULTS
        return SB3_DEFAULTS.get(algorithm, {})
```

## 🧠 **Algorithm Implementations**

### **Deep Q-Network (DQN) Agent**
```python
# extensions/reinforcement-v0.02/agents/agent_dqn_sb3.py
from stable_baselines3 import DQN
from extensions.common.agents.base_rl_agent import BaseRLAgent

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
        
    def initialize(self, environment):
        """Initialize DQN model with environment"""
        self.environment = environment
        
        # Create SB3 DQN model
        self.model = DQN(
            policy="MlpPolicy",
            env=environment,
            learning_rate=self.hyperparameters['learning_rate'],
            buffer_size=self.hyperparameters['buffer_size'],
            learning_starts=self.hyperparameters['learning_starts'],
            batch_size=self.hyperparameters['batch_size'],
            tau=self.hyperparameters['tau'],
            gamma=self.hyperparameters['gamma'],
            train_freq=self.hyperparameters['train_freq'],
            gradient_steps=self.hyperparameters['gradient_steps'],
            target_update_interval=self.hyperparameters['target_update_interval'],
            exploration_fraction=self.hyperparameters['exploration_fraction'],
            exploration_initial_eps=self.hyperparameters['exploration_initial_eps'],
            exploration_final_eps=self.hyperparameters['exploration_final_eps'],
            max_grad_norm=self.hyperparameters['max_grad_norm'],
            tensorboard_log=f"./logs/tensorboard/{self.name}/",
            verbose=1
        )
    
    def train(self, total_timesteps: int = 100000, **kwargs) -> TrainingResults:
        """Train DQN agent using SB3"""
        if not self.model:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=self.hyperparameters.get('log_interval', 100),
            **kwargs
        )
        
        # Evaluate performance
        evaluation_results = self._evaluate_model()
        
        return TrainingResults(
            algorithm="DQN_SB3",
            total_timesteps=total_timesteps,
            final_reward=evaluation_results['mean_reward'],
            training_time=evaluation_results['training_time'],
            episodes_completed=evaluation_results['episodes_completed']
        )
    
    def select_action(self, observation) -> int:
        """Select action using trained DQN model"""
        if not self.model:
            raise RuntimeError("Model not trained. Call train() first.")
        
        action, _states = self.model.predict(observation, deterministic=True)
        return int(action)
    
    def _get_hyperparameters(self, **kwargs) -> Dict[str, Any]:
        """Get DQN hyperparameters with overrides"""
        from extensions.common.config.sb3_hyperparameters import DQN_DEFAULTS
        hyperparams = DQN_DEFAULTS.copy()
        hyperparams.update(kwargs)
        return hyperparams
```

### **Proximal Policy Optimization (PPO) Agent**
```python
# extensions/reinforcement-v0.02/agents/agent_ppo_sb3.py
from stable_baselines3 import PPO

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
        
    def initialize(self, environment):
        """Initialize PPO model with environment"""
        self.environment = environment
        
        self.model = PPO(
            policy="MlpPolicy",
            env=environment,
            learning_rate=self.hyperparameters['learning_rate'],
            n_steps=self.hyperparameters['n_steps'],
            batch_size=self.hyperparameters['batch_size'],
            n_epochs=self.hyperparameters['n_epochs'],
            gamma=self.hyperparameters['gamma'],
            gae_lambda=self.hyperparameters['gae_lambda'],
            clip_range=self.hyperparameters['clip_range'],
            clip_range_vf=self.hyperparameters['clip_range_vf'],
            normalize_advantage=self.hyperparameters['normalize_advantage'],
            ent_coef=self.hyperparameters['ent_coef'],
            vf_coef=self.hyperparameters['vf_coef'],
            max_grad_norm=self.hyperparameters['max_grad_norm'],
            tensorboard_log=f"./logs/tensorboard/{self.name}/",
            verbose=1
        )
```

## 🔧 **Training Pipeline Integration**

### **SB3 Training Manager**
```python
class SB3TrainingManager:
    """
    Manages training pipeline for Stable Baselines3 agents
    
    Design Pattern: Template Method Pattern
    Purpose: Define common training workflow with algorithm-specific hooks
    Educational Note: Shows professional training pipeline organization
    """
    
    def __init__(self, agent_type: str, environment_type: str = "gymnasium"):
        self.agent_type = agent_type
        self.environment_type = environment_type
        self.training_callbacks = []
        
    def train_agent(self, grid_size: int = 10, total_timesteps: int = 100000,
                   save_model: bool = True, **kwargs) -> TrainingResults:
        """Complete training pipeline for SB3 agents"""
        
        # Step 1: Environment setup
        environment = self._setup_environment(grid_size)
        
        # Step 2: Agent creation
        agent = self._create_agent(environment, **kwargs)
        
        # Step 3: Training configuration
        callbacks = self._setup_callbacks()
        
        # Step 4: Training execution
        training_start = time.time()
        agent.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=100
        )
        training_time = time.time() - training_start
        
        # Step 5: Model evaluation
        evaluation_results = self._evaluate_agent(agent, environment)
        
        # Step 6: Model persistence
        if save_model:
            model_path = self._save_model(agent, grid_size)
            evaluation_results['model_path'] = model_path
        
        return TrainingResults(
            algorithm=self.agent_type,
            grid_size=grid_size,
            total_timesteps=total_timesteps,
            training_time=training_time,
            **evaluation_results
        )
    
    def _setup_callbacks(self) -> List[BaseCallback]:
        """Setup training callbacks for monitoring and early stopping"""
        from stable_baselines3.common.callbacks import (
            EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
        )
        
        callbacks = []
        
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env=self._create_eval_environment(),
            n_eval_episodes=10,
            eval_freq=5000,
            log_path="./logs/evaluations/",
            best_model_save_path="./logs/best_models/"
        )
        callbacks.append(eval_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="./logs/checkpoints/",
            name_prefix=f"{self.agent_type}_model"
        )
        callbacks.append(checkpoint_callback)
        
        return callbacks
```

## 📊 **Integration with Dataset Generation**

### **SB3 Data Collection**
```python
class SB3DataCollector:
    """
    Collect training data from SB3 agents for analysis
    
    Purpose: Bridge between SB3 training and our data analysis pipeline
    Educational Note: Shows how to extract learning insights from RL training
    """
    
    def __init__(self, agent, environment):
        self.agent = agent
        self.environment = environment
        
    def collect_episode_data(self, n_episodes: int = 1000) -> List[Dict]:
        """Collect episode data from trained agent"""
        episode_data = []
        
        for episode in range(n_episodes):
            obs = self.environment.reset()
            episode_steps = []
            total_reward = 0
            
            while True:
                action = self.agent.select_action(obs)
                next_obs, reward, done, info = self.environment.step(action)
                
                step_data = {
                    'episode': episode,
                    'observation': obs.tolist(),
                    'action': action,
                    'reward': reward,
                    'next_observation': next_obs.tolist(),
                    'done': done,
                    'game_info': info
                }
                episode_steps.append(step_data)
                total_reward += reward
                
                if done:
                    break
                obs = next_obs
            
            episode_summary = {
                'episode': episode,
                'total_reward': total_reward,
                'steps': len(episode_steps),
                'episode_data': episode_steps
            }
            episode_data.append(episode_summary)
        
        return episode_data
    
    def save_dataset(self, data: List[Dict], grid_size: int, 
                    timestamp: str = None) -> Path:
        """Save collected data following Final Decision 1 structure"""
        from extensions.common.path_utils import get_dataset_path
        
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        dataset_path = get_dataset_path(
            extension_type="reinforcement",
            version="0.02",
            grid_size=grid_size,
            algorithm=f"{self.agent.name}_sb3",
            timestamp=timestamp
        )
        
        # Save as JSONL for compatibility with other extensions
        jsonl_path = dataset_path / "episodes.jsonl"
        with open(jsonl_path, 'w') as f:
            for episode in data:
                f.write(json.dumps(episode) + '\n')
        
        return dataset_path
```

## 🎯 **Model Storage and Loading**

### **SB3 Model Persistence**
Following Final Decision 1 model storage patterns:

```python
class SB3ModelManager:
    """
    Manage SB3 model storage and loading
    
    Design Pattern: Strategy Pattern
    Purpose: Handle different model formats and storage strategies
    """
    
    def save_model(self, agent, grid_size: int, algorithm: str,
                  timestamp: str = None) -> Path:
        """Save SB3 model following standardized paths"""
        from extensions.common.path_utils import get_model_path
        
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = get_model_path(
            extension_type="reinforcement",
            version="0.02",
            grid_size=grid_size,
            algorithm=f"{algorithm}_sb3",
            timestamp=timestamp
        )
        
        # Create directory structure
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save SB3 model
        agent.model.save(model_path / "model")
        
        # Save metadata
        metadata = {
            'algorithm': algorithm,
            'grid_size': grid_size,
            'timestamp': timestamp,
            'sb3_version': stable_baselines3.__version__,
            'pytorch_version': torch.__version__,
            'hyperparameters': agent.hyperparameters,
            'training_results': getattr(agent, 'training_results', {})
        }
        
        with open(model_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
    
    def load_model(self, model_path: Path, algorithm: str):
        """Load SB3 model from standardized path"""
        from stable_baselines3 import DQN, PPO, A2C, SAC
        
        algorithm_map = {
            'DQN': DQN,
            'PPO': PPO,
            'A2C': A2C,
            'SAC': SAC
        }
        
        if algorithm not in algorithm_map:
            available = list(algorithm_map.keys())
            raise ValueError(f"Algorithm '{algorithm}' not available. "
                           f"Available: {available}. Following SUPREME_RULE NO.3, "
                           f"you can extend algorithm_map to add new algorithms.")
        
        model = algorithm_map[algorithm].load(model_path / "model")
        
        # Load metadata
        with open(model_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
```

## 🔗 **GOOD_RULES Integration**

This document integrates with the following authoritative references from the **GOOD_RULES** system:

### **Core Architecture Integration**
- **`agents.md`**: Follows BaseAgent interface and factory patterns for all SB3 RL implementations
- **`config.md`**: Uses authorized configuration hierarchies for SB3 hyperparameters
- **`core.md`**: Inherits from base classes and follows established inheritance patterns

### **Extension Development Standards**
- **`extensions-v0.02.md`** through **`extensions-v0.04.md`**: Follows version progression guidelines
- **`standalone.md`**: Maintains standalone principle (extension + common = self-contained)
- **`single-source-of-truth.md`**: Avoids duplication, uses centralized utilities
- **`reinforcement-learning.md`**: Follows RL extension standards and established RL patterns

### **Data and Path Management**
- **`data-format-decision-guide.md`**: Uses NPZ Sequential format for RL experience replay
- **`unified-path-management-guide.md`**: Uses centralized path utilities from extensions/common/
- **`datasets-folder.md`**: Follows standard directory structure for RL model storage
- **`models.md`**: Follows model versioning and storage standards

### **UI and Interaction Standards**
- **`app.md`** and **`dashboard.md`**: Integrates with Streamlit architecture for SB3 training interfaces
- **`unified-streamlit-architecture-guide.md`**: Follows OOP Streamlit patterns for RL monitoring

### **Implementation Quality**
- **`documentation-as-first-class-citizen.md`**: Maintains rich docstrings and design pattern documentation
- **`elegance.md`**: Follows code quality and educational value standards
- **`naming_conventions.md`**: Uses consistent naming across all SB3 implementations

## 📝 **Simple Logging Examples (SUPREME_RULE NO.3)**

All code examples in this document follow **SUPREME_RULE NO.3** by using simple print() statements rather than complex logging mechanisms:

```python
# ✅ CORRECT: Simple logging as per SUPREME_RULE NO.3
def train_sb3_agent(self, total_timesteps: int):
    print(f"[{self.name}] Starting SB3 training for {total_timesteps} timesteps")
    
    self.model.learn(
        total_timesteps=total_timesteps,
        log_interval=100
    )
    
    print(f"[{self.name}] SB3 training completed successfully")
    
    # Evaluate performance with simple logging
    mean_reward = self._evaluate_model()
    print(f"[{self.name}] Final mean reward: {mean_reward:.2f}")

# ✅ CORRECT: Educational progress tracking
def initialize_sb3_model(self, algorithm: str, environment):
    print(f"[SB3Factory] Initializing {algorithm} model...")
    
    model = self._create_model(algorithm, environment)
    print(f"[SB3Factory] {algorithm} model created with {model.policy} policy")
    
    # Model configuration logging
    if hasattr(model, 'learning_rate'):
        print(f"[SB3Factory] Learning rate: {model.learning_rate}")
    
    return model
```

---

**Stable Baselines3 integration provides production-ready RL algorithms for Snake Game AI, enabling rapid development and deployment of state-of-the-art reinforcement learning approaches while maintaining full compliance with established GOOD_RULES standards and educational objectives.**
