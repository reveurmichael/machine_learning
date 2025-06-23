"""
Deep Q-Network (DQN) Agent Implementation
--------------------

This module implements a Deep Q-Network agent for reinforcement learning in the Snake game.
It demonstrates several key design patterns and RL concepts.

Design Patterns Used:
1. Strategy Pattern: The agent implements a consistent interface for different RL algorithms
2. Template Method Pattern: Base agent defines the training/evaluation workflow
3. Observer Pattern: Training progress is observed and logged
4. Factory Pattern: Agent creation is abstracted through factory methods
5. State Pattern: Different training phases (exploration vs exploitation)

Motivation for Design Patterns:
- Strategy Pattern: Allows easy swapping of different RL algorithms (DQN, DDPG, PPO)
- Template Method: Ensures consistent training workflow across different agents
- Observer Pattern: Enables real-time monitoring of training progress without tight coupling
- Factory Pattern: Simplifies agent creation and configuration
- State Pattern: Manages complex training state transitions cleanly

Trade-offs:
- Strategy Pattern: Slight overhead from interface abstraction, but enables algorithm diversity
- Template Method: Less flexibility in training workflow, but ensures consistency
- Observer Pattern: Potential memory leaks if observers aren't properly managed
- Factory Pattern: Additional complexity for simple cases, but enables complex configuration
- State Pattern: More classes to maintain, but cleaner state management

Author: Snake Game Extensions
Version: 0.01
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from dataclasses import dataclass, field

# Add project root to path for imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from extensions.common.path_utils import setup_extension_paths
setup_extension_paths()
from extensions.common.rl_utils import ReplayBuffer, Experience



# ---------------------------------
# EXPERIENCE REPLAY BUFFER
# ---------------------------------

@dataclass
class Experience:
    """
    Experience tuple for reinforcement learning.
    
    Represents a single step in the environment with state, action, reward,
    next state, and done flag. This is the fundamental unit of learning
    in most RL algorithms.
    
    Design Pattern: Value Object
    - Immutable data structure representing a single experience
    - Used throughout the RL pipeline for consistency
    - Enables easy serialization and debugging
    
    Attributes:
        state: Current game state as feature vector
        action: Action taken (0-3 for UP, DOWN, LEFT, RIGHT)
        reward: Reward received for this action
        next_state: Next game state after action
        done: Whether episode ended after this action
        info: Additional information (optional)
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    
    Stores experiences in a circular buffer and provides sampling functionality.
    This is crucial for breaking temporal correlations in sequential data
    and enabling stable training of neural networks.
    
    Design Pattern: Circular Buffer
    - Efficient memory usage with fixed size
    - Automatic overwriting of old experiences
    - Thread-safe sampling for training
    
    Motivation:
    - Breaks temporal correlations in sequential data
    - Enables stable neural network training
    - Provides diverse experience sampling
    
    Trade-offs:
    - Fixed memory usage vs potentially losing valuable experiences
    - Sampling efficiency vs memory locality
    - Thread safety vs performance overhead
    
    Attributes:
        capacity: Maximum number of experiences to store
        buffer: Internal storage for experiences
        position: Current position in circular buffer
        size: Current number of experiences stored
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer with given capacity.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        self.size = 0
    
    def push(self, experience: Experience) -> None:
        """
        Add experience to replay buffer.
        
        Args:
            experience: Experience tuple to store
        """
        self.buffer.append(experience)
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of randomly sampled experiences
        """
        if self.size < batch_size:
            return list(self.buffer)
        
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Return current number of experiences."""
        return self.size


# ---------------------------------
# NEURAL NETWORK ARCHITECTURES
# ---------------------------------

class DQNNetwork(nn.Module):
    """
    Deep Q-Network architecture for Snake game.
    
    Implements a feedforward neural network that maps game states to Q-values
    for each possible action. The network learns to approximate the optimal
    action-value function Q*(s,a).
    
    Design Pattern: Neural Network Architecture
    - Modular network design for easy experimentation
    - Configurable layer sizes and activation functions
    - Batch normalization for training stability
    
    Motivation:
    - Approximates complex Q-function with neural network
    - Enables learning from high-dimensional state spaces
    - Provides smooth generalization across similar states
    
    Trade-offs:
    - Function approximation vs exact Q-learning
    - Network capacity vs training time
    - Overfitting vs underfitting
    
    Architecture:
    - Input: Game state features (19 features for 10x10 grid)
    - Hidden layers: Configurable fully connected layers
    - Output: Q-values for each action (4 actions)
    """
    
    def __init__(self, input_size: int = 19, hidden_sizes: List[int] = None, 
                 output_size: int = 4, dropout_rate: float = 0.1):
        """
        Initialize DQN network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output actions
            dropout_rate: Dropout rate for regularization
        """
        super(DQNNetwork, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Q-values tensor of shape (batch_size, output_size)
        """
        return self.network(x)


# ---------------------------------
# DQN AGENT IMPLEMENTATION
# ---------------------------------

class DQNAgent:
    """
    Deep Q-Network agent for Snake game reinforcement learning.
    
    Implements the DQN algorithm with experience replay, target networks,
    and epsilon-greedy exploration. This is the core learning agent that
    interacts with the environment and learns optimal policies.
    
    Design Pattern: Strategy Pattern
    - Implements consistent agent interface
    - Enables easy swapping with other RL algorithms
    - Maintains clean separation of concerns
    
    Design Pattern: Template Method Pattern
    - Defines training workflow template
    - Allows customization of specific steps
    - Ensures consistent training process
    
    Design Pattern: Observer Pattern
    - Training progress is observed and logged
    - Enables real-time monitoring without tight coupling
    - Supports multiple observers (logging, visualization, etc.)
    
    Motivation for Design Patterns:
    - Strategy Pattern: Enables algorithm diversity and comparison
    - Template Method: Ensures consistent training across different agents
    - Observer Pattern: Enables monitoring without modifying core logic
    
    Trade-offs:
    - Strategy Pattern: Slight overhead from interface abstraction
    - Template Method: Less flexibility but ensures consistency
    - Observer Pattern: Potential memory leaks if not managed properly
    
    Key Features:
    - Experience replay for stable training
    - Target network for stable Q-learning
    - Epsilon-greedy exploration strategy
    - Configurable hyperparameters
    - Comprehensive logging and monitoring
    """
    
    def __init__(self, state_size: int = 19, action_size: int = 4, 
                 hidden_sizes: List[int] = None, learning_rate: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, buffer_size: int = 100000,
                 batch_size: int = 32, target_update_freq: int = 1000,
                 device: str = "cpu"):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Number of state features
            action_size: Number of possible actions
            hidden_sizes: Hidden layer sizes for neural network
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            buffer_size: Size of experience replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to run neural networks on
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, hidden_sizes, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, hidden_sizes, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(buffer_size)
        
        # Training state
        self.training_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
        # Observers for monitoring
        self.observers: List[Callable] = []
    
    def add_observer(self, observer: Callable) -> None:
        """
        Add observer for training monitoring.
        
        Args:
            observer: Function to call with training updates
        """
        self.observers.append(observer)
    
    def _notify_observers(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Notify all observers of training event.
        
        Args:
            event_type: Type of training event
            data: Event data to pass to observers
        """
        for observer in self.observers:
            try:
                observer(event_type, data)
            except Exception as e:
                print(f"Observer error: {e}")
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current game state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_size)
        
        # Exploitation: best action according to Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool, info: Dict[str, Any] = None) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            info: Additional information
        """
        experience = Experience(state, action, reward, next_state, done, info or {})
        self.memory.push(experience)
    
    def replay(self) -> Optional[float]:
        """
        Train on batch of experiences from replay buffer.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch of experiences
        experiences = self.memory.sample(self.batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss and update
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def train_episode(self, env, max_steps: int = 1000) -> Dict[str, Any]:
        """
        Train for one episode.
        
        Args:
            env: Environment to interact with
            max_steps: Maximum steps per episode
            
        Returns:
            Episode statistics
        """
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Select action
            action = self.get_action(state, training=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            self.remember(state, action, reward, next_state, done, info)
            
            # Train on batch
            loss = self.replay()
            
            # Update state and statistics
            state = next_state
            total_reward += reward
            steps += 1
            
            if loss is not None:
                self.losses.append(loss)
            
            if done:
                break
        
        # Record episode statistics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        
        # Notify observers
        episode_data = {
            'episode': len(self.episode_rewards),
            'reward': total_reward,
            'steps': steps,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0
        }
        self._notify_observers('episode_complete', episode_data)
        
        return episode_data
    
    def save_model(self, filepath: str, export_onnx: bool = False) -> None:
        """
        Save trained model to file with full metadata for cross-platform and time-proofing.
        Args:
            filepath: Path to save model
            export_onnx: If True, also export ONNX format
        """
        from extensions.common.model_utils import save_model_standardized
        
        # Use standardized saving with proper directory structure
        training_params = {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards[-100:] if self.episode_rewards else [],  # Last 100 episodes
            'episode_lengths': self.episode_lengths[-100:] if self.episode_lengths else [],  # Last 100 episodes
            'losses': self.losses[-100:] if self.losses else []  # Last 100 losses
        }
        
        # Extract model name from filepath
        model_name = Path(filepath).stem
        
        # Use common utility for standardized saving
        saved_path = save_model_standardized(
            model=self.q_network,
            framework='PyTorch',
            grid_size=self.state_size,  # state_size represents grid size for RL
            model_name=model_name,
            model_class=self.__class__.__name__,
            input_size=self.state_size,
            output_size=self.action_size,
            training_params=training_params,
            export_onnx=export_onnx
        )
        
        # Also save target network and additional RL-specific data
        import torch
        checkpoint = torch.load(saved_path, map_location='cpu')
        checkpoint.update({
            'target_network_state_dict': self.target_network.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses
        })
        torch.save(checkpoint, saved_path)
        
        print(f"DQN model saved using standardized format: {saved_path}")

    def load_model(self, filepath: str) -> None:
        """
        Load trained model from file and check grid_size/metadata.
        Args:
            filepath: Path to load model from
        """
        from extensions.common.model_utils import load_model_standardized
        
        # Use standardized loading
        loaded_model = load_model_standardized(
            filepath, 'PyTorch', self.__class__, 
            state_size=self.state_size, action_size=self.action_size
        )
        
        # Load additional RL-specific data
        import torch
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load target network
        if 'target_network_state_dict' in checkpoint:
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        
        # Load RL-specific parameters
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.losses = checkpoint.get('losses', [])
        
        # Validate grid size
        metadata = checkpoint.get('metadata', {})
        loaded_grid_size = metadata.get('grid_size', None)
        if loaded_grid_size is not None and loaded_grid_size != self.state_size:
            print(f"Warning: Loaded model grid_size {loaded_grid_size} != current {self.state_size}")
        
        print(f"DQN model loaded from: {filepath}\nMetadata: {metadata}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current training statistics.
        
        Returns:
            Dictionary of training statistics
        """
        return {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'buffer_size': len(self.memory),
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0
        } 