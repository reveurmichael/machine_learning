"""
Reinforcement Learning Game Logic
================================

This module extends BaseGameLogic to support reinforcement learning training.
It provides the environment interface that RL agents interact with.

Design Patterns Used:
1. Template Method Pattern: Extends base game logic with RL-specific behavior
2. Strategy Pattern: Different reward functions can be plugged in
3. Observer Pattern: Training progress is observed and logged
4. State Pattern: Different training phases (exploration vs evaluation)

Motivation for Design Patterns:
- Template Method: Ensures consistent game logic while allowing RL extensions
- Strategy Pattern: Enables experimentation with different reward functions
- Observer Pattern: Enables monitoring without tight coupling
- State Pattern: Manages complex training state transitions

Trade-offs:
- Template Method: Less flexibility but ensures consistency
- Strategy Pattern: Slight overhead for simple cases
- Observer Pattern: Potential memory leaks if not managed properly
- State Pattern: More classes but cleaner state management

Author: Snake Game Extensions
Version: 0.01
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Callable
import numpy as np

# Add project root to path for imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from extensions.common.path_utils import setup_extension_paths
setup_extension_paths()

from core.game_logic import BaseGameLogic
from core.game_data import BaseGameData
from config.game_constants import VALID_MOVES


class RLGameData(BaseGameData):
    """
    Game data container for reinforcement learning.
    
    Extends BaseGameData with RL-specific tracking and statistics.
    Maintains training episode information and performance metrics.
    
    Design Pattern: Extension of Base Class
    - Inherits all generic game state functionality
    - Adds RL-specific data tracking
    - Maintains clean separation of concerns
    
    Attributes:
        episode_reward: Current episode reward
        episode_steps: Current episode step count
        total_reward: Total reward across all episodes
        episode_count: Number of completed episodes
        training_stats: Dictionary of training statistics
    """
    
    def __init__(self):
        """Initialize RL game data."""
        super().__init__()
        
        # RL-specific tracking
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.total_reward = 0.0
        self.episode_count = 0
        self.training_stats = {}
        
        # Episode history
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []
    
    def reset_episode(self) -> None:
        """Reset episode-specific data."""
        self.episode_reward = 0.0
        self.episode_steps = 0
    
    def record_step(self, reward: float, score: int) -> None:
        """
        Record step data for current episode.
        
        Args:
            reward: Reward for this step
            score: Current game score
        """
        self.episode_reward += reward
        self.episode_steps += 1
        self.score = score
    
    def end_episode(self) -> None:
        """Record episode completion."""
        self.episode_count += 1
        self.total_reward += self.episode_reward
        
        # Store episode statistics
        self.episode_rewards.append(self.episode_reward)
        self.episode_lengths.append(self.episode_steps)
        self.episode_scores.append(self.score)
        
        # Update training stats
        self.training_stats = {
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
            'avg_score': np.mean(self.episode_scores[-100:]) if self.episode_scores else 0,
            'total_episodes': self.episode_count,
            'total_reward': self.total_reward
        }


class RewardFunction:
    """
    Abstract base class for reward functions.
    
    Defines the interface for different reward function implementations.
    This enables easy experimentation with different reward schemes.
    
    Design Pattern: Strategy Pattern
    - Different reward functions can be plugged in
    - Enables easy comparison of reward schemes
    - Maintains clean separation of reward logic
    
    Motivation:
    - Reward function design is crucial for RL success
    - Different reward schemes may work better for different scenarios
    - Enables systematic experimentation and comparison
    
    Trade-offs:
    - Slight overhead from interface abstraction
    - More classes to maintain
    - Enables better experimentation and comparison
    """
    
    @abstractmethod
    def calculate_reward(self, game_state: Dict[str, Any], 
                        action: int, next_state: Dict[str, Any], 
                        done: bool, info: Dict[str, Any]) -> float:
        """
        Calculate reward for given state transition.
        
        Args:
            game_state: Current game state
            action: Action taken
            next_state: Next game state
            done: Whether episode ended
            info: Additional information
            
        Returns:
            Reward value
        """
        pass


class SnakeRewardFunction(RewardFunction):
    """
    Standard reward function for Snake game.
    
    Implements a reward scheme that encourages:
    - Eating apples (positive reward)
    - Surviving (small positive reward)
    - Avoiding death (negative reward)
    - Efficient movement (small negative reward for steps)
    
    Design Pattern: Concrete Strategy
    - Implements the reward function interface
    - Provides sensible default reward scheme
    - Configurable reward values
    
    Motivation:
    - Balances exploration and exploitation
    - Encourages efficient apple collection
    - Penalizes dangerous behavior
    
    Trade-offs:
    - Fixed reward scheme may not be optimal for all scenarios
    - Requires tuning of reward values
    - May need adjustment for different grid sizes
    """
    
    def __init__(self, apple_reward: float = 10.0, step_reward: float = -0.1,
                 death_reward: float = -10.0, survival_reward: float = 0.1):
        """
        Initialize reward function with given values.
        
        Args:
            apple_reward: Reward for eating apple
            step_reward: Reward for each step (usually negative)
            death_reward: Reward for dying (usually negative)
            survival_reward: Reward for surviving (usually small positive)
        """
        self.apple_reward = apple_reward
        self.step_reward = step_reward
        self.death_reward = death_reward
        self.survival_reward = survival_reward
    
    def calculate_reward(self, game_state: Dict[str, Any], 
                        action: int, next_state: Dict[str, Any], 
                        done: bool, info: Dict[str, Any]) -> float:
        """
        Calculate reward based on state transition.
        
        Args:
            game_state: Current game state
            action: Action taken
            next_state: Next game state
            done: Whether episode ended
            info: Additional information
            
        Returns:
            Calculated reward value
        """
        reward = 0.0
        
        # Check if apple was eaten
        if info.get('apple_eaten', False):
            reward += self.apple_reward
        
        # Check if game ended
        if done:
            if info.get('game_end_reason') == 'wall_collision':
                reward += self.death_reward
            elif info.get('game_end_reason') == 'body_collision':
                reward += self.death_reward
            elif info.get('game_end_reason') == 'max_steps_reached':
                reward += self.step_reward  # Small penalty for not finishing
        else:
            # Survival reward
            reward += self.survival_reward
        
        # Step penalty
        reward += self.step_reward
        
        return reward


class RLGameLogic(BaseGameLogic):
    """
    Reinforcement learning game logic.
    
    Extends BaseGameLogic to provide an environment interface for RL agents.
    Implements the standard RL environment interface with reset, step, and
    observation methods.
    
    Design Pattern: Template Method Pattern
    - Extends base game logic with RL-specific behavior
    - Maintains consistent game mechanics
    - Adds RL environment interface
    
    Design Pattern: Strategy Pattern
    - Configurable reward function
    - Different reward schemes can be plugged in
    
    Design Pattern: Observer Pattern
    - Training progress is observed and logged
    - Enables monitoring without tight coupling
    
    Motivation:
    - Provides standard RL environment interface
    - Maintains game consistency across different agents
    - Enables easy experimentation with different reward schemes
    
    Trade-offs:
    - Template Method: Less flexibility but ensures consistency
    - Strategy Pattern: Slight overhead for simple cases
    - Observer Pattern: Potential memory leaks if not managed properly
    """
    
    def __init__(self, grid_size: int = 10, use_gui: bool = False,
                 reward_function: RewardFunction = None):
        """
        Initialize RL game logic.
        
        Args:
            grid_size: Size of game grid
            use_gui: Whether to use GUI (usually False for RL training)
            reward_function: Reward function to use
        """
        super().__init__(grid_size, use_gui)
        
        # Use RL-specific game data
        self.game_state = RLGameData()
        
        # Set up reward function
        if reward_function is None:
            reward_function = SnakeRewardFunction()
        self.reward_function = reward_function
        
        # RL environment state
        self.current_episode = 0
        self.observers: List[Callable] = []
        
        # State encoding
        self.state_encoder = SnakeStateEncoder(grid_size)
    
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
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Returns:
            Initial state observation
        """
        # Reset game state
        super().reset()
        
        # Reset RL-specific data
        self.game_state.reset_episode()
        self.current_episode += 1
        
        # Get initial state
        state = self.get_observation()
        
        # Notify observers
        self._notify_observers('episode_start', {
            'episode': self.current_episode,
            'state': state
        })
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take action in environment.
        
        Args:
            action: Action to take (0-3 for UP, DOWN, LEFT, RIGHT)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Convert action to direction
        direction = VALID_MOVES[action]
        
        # Get current state
        current_state = self.get_observation()
        
        # Take action in game
        game_ended, end_reason = self.move_snake(direction)
        
        # Get next state
        next_state = self.get_observation()
        
        # Calculate reward
        info = {
            'apple_eaten': self.game_state.apple_eaten,
            'game_end_reason': end_reason,
            'score': self.game_state.score,
            'steps': self.game_state.steps
        }
        
        reward = self.reward_function.calculate_reward(
            current_state, action, next_state, game_ended, info
        )
        
        # Record step data
        self.game_state.record_step(reward, self.game_state.score)
        
        # Check if episode should end
        done = game_ended or self.game_state.steps >= 1000  # Max steps
        
        if done:
            self.game_state.end_episode()
            
            # Notify observers
            self._notify_observers('episode_end', {
                'episode': self.current_episode,
                'reward': self.game_state.episode_reward,
                'steps': self.game_state.episode_steps,
                'score': self.game_state.score,
                'end_reason': end_reason
            })
        
        return next_state, reward, done, info
    
    def get_observation(self) -> np.ndarray:
        """
        Get current state observation.
        
        Returns:
            State vector for RL agent
        """
        return self.state_encoder.encode_state(self)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary of training statistics
        """
        stats = super().get_stats()
        stats.update(self.game_state.training_stats)
        return stats


class SnakeStateEncoder:
    """
    Encodes Snake game state for RL agents.
    
    Converts the game state into a feature vector that RL agents can use
    for learning. Implements the same feature encoding as the supervised
    learning extensions for consistency.
    
    Design Pattern: Encoder Pattern
    - Separates state encoding logic from game logic
    - Enables easy experimentation with different encodings
    - Maintains consistency across different learning approaches
    
    Motivation:
    - Provides consistent state representation
    - Enables easy comparison between RL and supervised learning
    - Separates encoding logic for maintainability
    
    Trade-offs:
    - Additional complexity for state encoding
    - May need adjustment for different grid sizes
    - Enables better experimentation and comparison
    """
    
    def __init__(self, grid_size: int = 10):
        """
        Initialize state encoder.
        
        Args:
            grid_size: Size of game grid
        """
        self.grid_size = grid_size
        self.feature_size = 19  # Fixed feature size for 10x10 grid
    
    def encode_state(self, game_logic: RLGameLogic) -> np.ndarray:
        """
        Encode game state into feature vector.
        
        Args:
            game_logic: Game logic instance
            
        Returns:
            Feature vector for RL agent
        """
        # Get game state
        head_pos = game_logic.head_position
        apple_pos = game_logic.apple_position
        snake_length = len(game_logic.snake_positions)
        
        # Basic state features
        features = [
            head_pos[0],  # head_x
            head_pos[1],  # head_y
            apple_pos[0],  # apple_x
            apple_pos[1],  # apple_y
            snake_length,  # snake_length
        ]
        
        # Apple direction flags
        apple_dir_up = 1 if apple_pos[1] > head_pos[1] else 0
        apple_dir_down = 1 if apple_pos[1] < head_pos[1] else 0
        apple_dir_left = 1 if apple_pos[0] < head_pos[0] else 0
        apple_dir_right = 1 if apple_pos[0] > head_pos[0] else 0
        
        features.extend([apple_dir_up, apple_dir_down, apple_dir_left, apple_dir_right])
        
        # Danger flags
        danger_straight, danger_left, danger_right = self._calculate_danger_flags(game_logic)
        features.extend([danger_straight, danger_left, danger_right])
        
        # Free space counts
        free_space_up, free_space_down, free_space_left, free_space_right = \
            self._calculate_free_space(game_logic)
        features.extend([free_space_up, free_space_down, free_space_left, free_space_right])
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_danger_flags(self, game_logic: RLGameLogic) -> Tuple[int, int, int]:
        """
        Calculate danger flags for current direction.
        
        Args:
            game_logic: Game logic instance
            
        Returns:
            Tuple of (danger_straight, danger_left, danger_right)
        """
        head_pos = game_logic.head_position
        snake_body = set(map(tuple, game_logic.snake_positions[:-1]))  # Exclude head
        
        # Get current direction
        if len(game_logic.snake_positions) > 1:
            prev_pos = game_logic.snake_positions[-2]
            current_dir = (head_pos[0] - prev_pos[0], head_pos[1] - prev_pos[1])
        else:
            current_dir = (1, 0)  # Default to right
        
        # Calculate positions in each direction
        straight_pos = (head_pos[0] + current_dir[0], head_pos[1] + current_dir[1])
        left_dir = (-current_dir[1], current_dir[0])  # Rotate 90 degrees left
        right_dir = (current_dir[1], -current_dir[0])  # Rotate 90 degrees right
        left_pos = (head_pos[0] + left_dir[0], head_pos[1] + left_dir[1])
        right_pos = (head_pos[0] + right_dir[0], head_pos[1] + right_dir[1])
        
        # Check for collisions
        def is_dangerous(pos):
            return (pos[0] < 0 or pos[0] >= self.grid_size or
                    pos[1] < 0 or pos[1] >= self.grid_size or
                    pos in snake_body)
        
        return (int(is_dangerous(straight_pos)),
                int(is_dangerous(left_pos)),
                int(is_dangerous(right_pos)))
    
    def _calculate_free_space(self, game_logic: RLGameLogic) -> Tuple[int, int, int, int]:
        """
        Calculate free space in each direction.
        
        Args:
            game_logic: Game logic instance
            
        Returns:
            Tuple of (free_space_up, free_space_down, free_space_left, free_space_right)
        """
        head_pos = game_logic.head_position
        snake_body = set(map(tuple, game_logic.snake_positions[:-1]))
        
        def count_free_space(direction):
            count = 0
            pos = head_pos
            for _ in range(self.grid_size):
                pos = (pos[0] + direction[0], pos[1] + direction[1])
                if (pos[0] < 0 or pos[0] >= self.grid_size or
                    pos[1] < 0 or pos[1] >= self.grid_size or
                    pos in snake_body):
                    break
                count += 1
            return count
        
        return (count_free_space((0, 1)),   # up
                count_free_space((0, -1)),  # down
                count_free_space((-1, 0)),  # left
                count_free_space((1, 0)))   # right 