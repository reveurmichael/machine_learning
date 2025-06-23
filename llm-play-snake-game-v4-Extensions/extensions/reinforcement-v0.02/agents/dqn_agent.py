"""
DQN Agent for Reinforcement Learning v0.02
=========================================

Deep Q-Network agent using PyTorch for Snake game.
Implements experience replay, target networks, and epsilon-greedy exploration.

Design Pattern: Strategy Pattern
- Implements RL agent interface
- Configurable hyperparameters
- Standardized training and evaluation
- Experience replay buffer
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from typing import Dict, Any
import random

from core.game_agents import SnakeAgent
from extensions.common.path_utils import setup_extension_paths
from extensions.common.model_utils import save_model_standardized
from extensions.common.rl_utils import ReplayBuffer
setup_extension_paths()


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Snake game.
    
    Design Pattern: Template Method
    - Configurable architecture with grid size support
    - Standardized forward pass
    - Dropout for regularization
    """
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_actions: int = 4):
        """
        Initialize DQN network.
        
        Args:
            input_size: Number of input features (depends on grid size)
            hidden_size: Size of hidden layers
            num_actions: Number of possible actions (4 directions)
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        
        # DQN architecture
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, num_actions)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DQN.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Q-values tensor of shape (batch_size, num_actions)
        """
        return self.layers(x)


class DQNAgent(SnakeAgent):
    """
    Deep Q-Network agent for Snake game.
    
    Design Pattern: Strategy Pattern
    - Implements SnakeAgent interface
    - Configurable hyperparameters
    - Experience replay and target networks
    - Epsilon-greedy exploration
    """
    
    def __init__(self, grid_size: int = 10, hidden_size: int = 256, learning_rate: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 0.1, memory_size: int = 10000,
                 target_update: int = 100):
        """
        Initialize DQN agent.
        
        Args:
            grid_size: Size of the game grid
            hidden_size: Size of hidden layers in DQN
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Exploration rate
            memory_size: Size of replay buffer
            target_update: Frequency of target network updates
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory_size = memory_size
        self.target_update = target_update
        
        # Get input size from state representation
        self.input_size = self._get_state_size()
        
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(self.input_size, hidden_size, 4).to(self.device)
        self.target_network = DQNNetwork(self.input_size, hidden_size, 4).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize components
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(memory_size)
        
        # Training state
        self.is_trained = False
        self.episode_count = 0
        self.step_count = 0
        
        # Direction mapping
        self.direction_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    
    def _get_state_size(self) -> int:
        """Get the size of state representation."""
        # For now, use a simple state representation
        # In practice, this would be more sophisticated
        return self.grid_size * self.grid_size + 4  # Board + head position + apple position
    
    def _get_state_representation(self, game_logic) -> np.ndarray:
        """
        Get state representation for the current game state.
        
        Args:
            game_logic: Game logic instance containing current state
            
        Returns:
            State representation as numpy array
        """
        # Get current game state
        state = game_logic.get_state_snapshot()
        
        # Create board representation
        board = np.zeros((self.grid_size, self.grid_size))
        
        # Mark snake positions
        for pos in state["snake_positions"]:
            board[pos[1], pos[0]] = 1  # Snake body
        
        # Mark apple position
        apple_pos = state["apple_position"]
        board[apple_pos[1], apple_pos[0]] = 2  # Apple
        
        # Flatten board and add head/apple positions
        board_flat = board.flatten()
        head_pos = state["head_position"]
        apple_pos = state["apple_position"]
        
        state_rep = np.concatenate([
            board_flat,
            [head_pos[0], head_pos[1], apple_pos[0], apple_pos[1]]
        ])
        
        return state_rep.astype(np.float32)
    
    def get_move(self, game_logic) -> str:
        """
        Get the next move for the current game state.
        
        Args:
            game_logic: Game logic instance containing current state
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT)
        """
        if not self.is_trained:
            raise RuntimeError("Agent must be trained before making predictions")
        
        # Get state representation
        state = self._get_state_representation(game_logic)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                self.q_network.eval()
                q_values = self.q_network(state_tensor)
                action = torch.argmax(q_values, dim=1).item()
        
        # Convert to direction
        return self.direction_map[action]
    
    def train(self, max_episodes: int = 1000, max_steps: int = 1000, 
              batch_size: int = 32, save_frequency: int = 100) -> Dict[str, Any]:
        """
        Train the DQN agent.
        
        Args:
            max_episodes: Maximum number of training episodes
            max_steps: Maximum steps per episode
            batch_size: Batch size for training
            save_frequency: Save model every N episodes
            
        Returns:
            Dictionary containing training metrics
        """
        print("Training DQN agent...")
        print(f"Grid size: {self.grid_size}, Hidden size: {self.hidden_size}")
        print(f"Learning rate: {self.learning_rate}, Gamma: {self.gamma}")
        print(f"Device: {self.device}")
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        losses = []
        
        for episode in range(max_episodes):
            # Initialize episode
            episode_reward = 0
            episode_length = 0
            
            # Reset game (this would be done by the game environment)
            # For now, we'll simulate training with random data
            
            for step in range(max_steps):
                # Simulate environment step (placeholder)
                current_state = np.random.randn(self.input_size).astype(np.float32)
                action = random.randint(0, 3)
                reward = random.uniform(-1, 1)
                next_state = np.random.randn(self.input_size).astype(np.float32)
                done = random.random() < 0.1  # 10% chance of episode ending
                
                # Store experience
                self.replay_buffer.push(current_state, action, reward, next_state, done)
                
                # Train if enough samples
                if len(self.replay_buffer) >= batch_size:
                    loss = self._train_step(batch_size)
                    losses.append(loss)
                
                episode_reward += reward
                episode_length += 1
                self.step_count += 1
                
                # Update target network
                if self.step_count % self.target_update == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            self.episode_count += 1
            
            # Save model periodically
            if (episode + 1) % save_frequency == 0:
                self.save_model(f"dqn_checkpoint_episode_{episode + 1}")
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                avg_loss = np.mean(losses[-10:]) if losses else 0
                print(f"Episode [{episode+1}/{max_episodes}] - "
                      f"Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}, "
                      f"Avg Loss: {avg_loss:.4f}")
        
        # Mark as trained
        self.is_trained = True
        
        # Return training metrics
        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "losses": losses,
            "final_avg_reward": np.mean(episode_rewards[-100:]),
            "final_avg_length": np.mean(episode_lengths[-100:]),
            "episodes_trained": max_episodes
        }
    
    def _train_step(self, batch_size: int) -> float:
        """
        Perform one training step.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Training loss
        """
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + (self.gamma * next_q_values * ~dones_tensor)
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, agent_name: str) -> str:
        """
        Save the trained agent with metadata.
        
        Args:
            agent_name: Name for the saved agent
            
        Returns:
            Path to the saved model file
        """
        if not self.is_trained:
            raise RuntimeError("Agent must be trained before saving")
        
        # Prepare metadata
        metadata = {
            "agent_type": "DQN",
            "grid_size": self.grid_size,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "memory_size": self.memory_size,
            "target_update": self.target_update,
            "torch_version": torch.__version__,
            "device": str(self.device),
            "timestamp": datetime.utcnow().isoformat(),
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "model_architecture": "DQN",
            "framework": "pytorch"
        }
        
        # Save using standardized utility
        model_path = save_model_standardized(
            model_name=agent_name,
            model=self.q_network,
            metadata=metadata,
            framework="pytorch",
            grid_size=self.grid_size,
            export_onnx=True
        )
        
        print(f"DQN agent saved to: {model_path}")
        return model_path
    
    def load_model(self, agent_path: str) -> None:
        """
        Load a trained agent.
        
        Args:
            agent_path: Path to the saved agent file
        """
        # Load using standardized utility
        loaded_data = load_model_standardized(agent_path, framework="pytorch")
        
        # Extract model and metadata
        self.q_network = loaded_data["model"]
        metadata = loaded_data["metadata"]
        
        # Update agent state
        self.grid_size = metadata["grid_size"]
        self.input_size = metadata["input_size"]
        self.hidden_size = metadata["hidden_size"]
        self.learning_rate = metadata["learning_rate"]
        self.gamma = metadata["gamma"]
        self.epsilon = metadata["epsilon"]
        self.memory_size = metadata["memory_size"]
        self.target_update = metadata["target_update"]
        self.device = torch.device(metadata["device"])
        self.q_network.to(self.device)
        
        # Update target network
        self.target_network = DQNNetwork(self.input_size, self.hidden_size, 4).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.is_trained = True
        
        print(f"DQN agent loaded from: {agent_path}")
        print(f"Grid size: {self.grid_size}, Hidden size: {self.hidden_size}")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the agent.
        
        Returns:
            Dictionary containing agent information
        """
        return {
            "agent_type": "DQN",
            "grid_size": self.grid_size,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "memory_size": self.memory_size,
            "target_update": self.target_update,
            "is_trained": self.is_trained,
            "device": str(self.device),
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "total_parameters": sum(p.numel() for p in self.q_network.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
        } 