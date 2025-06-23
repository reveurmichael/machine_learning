"""
Reinforcement Learning Game Manager
==================================

This module extends BaseGameManager to support reinforcement learning training.
It manages training sessions and coordinates between the RL agent and environment.

Design Patterns Used:
1. Template Method Pattern: Extends base game manager with RL-specific behavior
2. Strategy Pattern: Different training strategies can be plugged in
3. Observer Pattern: Training progress is observed and logged
4. Factory Pattern: Agent creation is abstracted

Motivation for Design Patterns:
- Template Method: Ensures consistent training workflow
- Strategy Pattern: Enables different training approaches
- Observer Pattern: Training progress is observed and logged
- Factory Pattern: Agent creation is abstracted

Trade-offs:
- Template Method: Less flexibility but ensures consistency
- Strategy Pattern: Slight overhead for simple cases
- Observer Pattern: Potential memory leaks if not managed properly
- Factory Pattern: Additional complexity for simple cases

Author: Snake Game Extensions
Version: 0.01
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import argparse
import time
import json
from datetime import datetime

# Add project root to path for imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from extensions.common.path_utils import setup_extension_paths
setup_extension_paths()

from core.game_manager import BaseGameManager
from game_logic import RLGameLogic, RLGameData
from agent_dqn import DQNAgent


class RLGameManager(BaseGameManager):
    """
    Reinforcement learning game manager.
    
    Extends BaseGameManager to coordinate RL training sessions.
    Manages the interaction between RL agents and the game environment.
    
    Design Pattern: Template Method Pattern
    - Extends base game manager with RL-specific behavior
    - Maintains consistent session management
    - Adds RL training coordination
    
    Design Pattern: Strategy Pattern
    - Configurable training strategies
    - Different training approaches can be plugged in
    
    Design Pattern: Observer Pattern
    - Training progress is observed and logged
    - Enables monitoring without tight coupling
    
    Motivation:
    - Provides consistent training session management
    - Coordinates between agent and environment
    - Enables easy experimentation with different training approaches
    
    Trade-offs:
    - Template Method: Less flexibility but ensures consistency
    - Strategy Pattern: Slight overhead for simple cases
    - Observer Pattern: Potential memory leaks if not managed properly
    """
    
    GAME_LOGIC_CLS = RLGameLogic
    GAME_DATA_CLS = RLGameData
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize RL game manager.
        
        Args:
            args: Command line arguments
        """
        super().__init__(args)
        
        # RL-specific attributes
        self.agent: Optional[DQNAgent] = None
        self.episodes = getattr(args, 'episodes', 1000)
        self.eval_frequency = getattr(args, 'eval_frequency', 100)
        self.save_frequency = getattr(args, 'save_frequency', 500)
        self.output_dir = getattr(args, 'output_dir', './models')
        
        # Training state
        self.current_episode = 0
        self.best_reward = float('-inf')
        self.training_start_time = None
        
        # Observers for monitoring
        self.observers: List[Callable] = []
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging directory and stats manager."""
        # Create log directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/reinforcement-dqn_{timestamp}"
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Logging to: {self.log_dir}")
        print(f"Models will be saved to: {self.output_dir}")
    
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
    
    def initialize_agent(self, agent_config: Dict[str, Any] = None) -> None:
        """
        Initialize the RL agent.
        
        Args:
            agent_config: Agent configuration dictionary
        """
        if agent_config is None:
            agent_config = {}
        
        # Default agent configuration
        default_config = {
            'state_size': 19,
            'action_size': 4,
            'hidden_sizes': [128, 64, 32],
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'buffer_size': 100000,
            'batch_size': 32,
            'target_update_freq': 1000
        }
        
        # Update with provided config
        default_config.update(agent_config)
        
        # Create agent
        self.agent = DQNAgent(**default_config)
        
        # Add agent as observer
        self.agent.add_observer(self._on_agent_update)
        
        print(f"Initialized DQN agent with config: {default_config}")
    
    def _on_agent_update(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle agent update events.
        
        Args:
            event_type: Type of agent event
            data: Event data
        """
        if event_type == 'episode_complete':
            self._handle_episode_complete(data)
    
    def _handle_episode_complete(self, episode_data: Dict[str, Any]) -> None:
        """
        Handle episode completion.
        
        Args:
            episode_data: Episode completion data
        """
        self.current_episode = episode_data['episode']
        
        # Update best reward
        if episode_data['reward'] > self.best_reward:
            self.best_reward = episode_data['reward']
        
        # Periodic evaluation
        if self.current_episode % self.eval_frequency == 0:
            self._evaluate_agent()
        
        # Periodic saving
        if self.current_episode % self.save_frequency == 0:
            self._save_agent()
        
        # Log progress
        if self.current_episode % 10 == 0:
            self._log_progress(episode_data)
        
        # Notify observers
        self._notify_observers('episode_complete', episode_data)
    
    def _evaluate_agent(self) -> None:
        """Evaluate agent performance."""
        if self.agent is None:
            return
        
        # Run evaluation episodes
        eval_episodes = 10
        eval_rewards = []
        
        for _ in range(eval_episodes):
            state = self.game.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.agent.get_action(state, training=False)
                state, reward, done, _ = self.game.step(action)
                total_reward += reward
            
            eval_rewards.append(total_reward)
        
        avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
        
        # Log evaluation results
        eval_data = {
            'episode': self.current_episode,
            'avg_eval_reward': avg_eval_reward,
            'eval_rewards': eval_rewards
        }
        
        self._notify_observers('evaluation_complete', eval_data)
        
        print(f"Episode {self.current_episode}: Avg eval reward = {avg_eval_reward:.2f}")
    
    def _save_agent(self) -> None:
        """Save agent model."""
        if self.agent is None:
            return
        
        # Create model filename
        model_filename = f"dqn_episode_{self.current_episode}.pth"
        model_path = os.path.join(self.output_dir, model_filename)
        
        # Save agent
        self.agent.save_model(model_path)
        
        # Save training stats
        stats_filename = f"training_stats_episode_{self.current_episode}.json"
        stats_path = os.path.join(self.log_dir, stats_filename)
        
        stats = {
            'episode': self.current_episode,
            'agent_stats': self.agent.get_stats(),
            'game_stats': self.game.get_stats(),
            'best_reward': self.best_reward,
            'training_time': time.time() - self.training_start_time
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved model and stats at episode {self.current_episode}")
    
    def _log_progress(self, episode_data: Dict[str, Any]) -> None:
        """
        Log training progress.
        
        Args:
            episode_data: Episode data to log
        """
        agent_stats = self.agent.get_stats() if self.agent else {}
        
        progress_data = {
            'episode': episode_data['episode'],
            'reward': episode_data['reward'],
            'steps': episode_data['steps'],
            'epsilon': agent_stats.get('epsilon', 0),
            'avg_reward': agent_stats.get('avg_reward', 0),
            'avg_loss': agent_stats.get('avg_loss', 0),
            'best_reward': self.best_reward
        }
        
        print(f"Episode {progress_data['episode']:4d} | "
              f"Reward: {progress_data['reward']:6.1f} | "
              f"Steps: {progress_data['steps']:3d} | "
              f"Epsilon: {progress_data['epsilon']:.3f} | "
              f"Avg Reward: {progress_data['avg_reward']:6.1f} | "
              f"Avg Loss: {progress_data['avg_loss']:.4f}")
    
    def run(self) -> None:
        """Run the training session."""
        if self.agent is None:
            raise ValueError("Agent not initialized. Call initialize_agent() first.")
        
        print(f"Starting RL training for {self.episodes} episodes...")
        self.training_start_time = time.time()
        
        # Notify observers of training start
        self._notify_observers('training_start', {
            'episodes': self.episodes,
            'start_time': self.training_start_time
        })
        
        try:
            for episode in range(1, self.episodes + 1):
                # Train one episode
                episode_data = self.agent.train_episode(self.game)
                
                # Check for early stopping
                if episode_data['reward'] > 1000:  # Very good performance
                    print("Early stopping: Excellent performance achieved!")
                    break
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        
        finally:
            # Final save
            self._save_agent()
            
            # Training summary
            training_time = time.time() - self.training_start_time
            print("\nTraining completed!")
            print(f"Total episodes: {self.current_episode}")
            print(f"Training time: {training_time:.1f} seconds")
            print(f"Best reward: {self.best_reward:.1f}")
            
            # Notify observers of training end
            self._notify_observers('training_end', {
                'episodes_completed': self.current_episode,
                'training_time': training_time,
                'best_reward': self.best_reward
            })
    
    def load_agent(self, model_path: str) -> None:
        """
        Load trained agent from file.
        
        Args:
            model_path: Path to saved model
        """
        if self.agent is None:
            self.initialize_agent()
        
        self.agent.load_model(model_path)
        print(f"Loaded agent from: {model_path}")
    
    def evaluate_agent(self, num_episodes: int = 100) -> Dict[str, Any]:
        """
        Evaluate agent performance.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results
        """
        if self.agent is None:
            raise ValueError("Agent not initialized.")
        
        print(f"Evaluating agent for {num_episodes} episodes...")
        
        rewards = []
        scores = []
        lengths = []
        
        for episode in range(num_episodes):
            state = self.game.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                action = self.agent.get_action(state, training=False)
                state, reward, done, info = self.game.step(action)
                total_reward += reward
                steps += 1
            
            rewards.append(total_reward)
            scores.append(info['score'])
            lengths.append(steps)
        
        results = {
            'num_episodes': num_episodes,
            'avg_reward': sum(rewards) / len(rewards),
            'avg_score': sum(scores) / len(scores),
            'avg_length': sum(lengths) / len(lengths),
            'max_reward': max(rewards),
            'min_reward': min(rewards),
            'rewards': rewards,
            'scores': scores,
            'lengths': lengths
        }
        
        print("Evaluation Results:")
        print(f"  Avg Reward: {results['avg_reward']:.2f}")
        print(f"  Avg Score: {results['avg_score']:.2f}")
        print(f"  Avg Length: {results['avg_length']:.1f}")
        print(f"  Max Reward: {results['max_reward']:.2f}")
        print(f"  Min Reward: {results['min_reward']:.2f}")
        
        return results 