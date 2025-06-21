"""
Consecutive Limits Management System
===================================

This module provides an elegant, centralized system for managing consecutive move limits
in the Snake game. It handles tracking and enforcement of limits for various sentinel moves
like EMPTY, SOMETHING_IS_WRONG, NO_PATH_FOUND, and INVALID_REVERSAL.

Design Philosophy:
- Single Responsibility: Each limit type has dedicated tracking
- Observer Pattern: Game events trigger limit updates
- Strategy Pattern: Different limit enforcement strategies
- Template Method: Common limit checking workflow
- Factory Pattern: Limit manager creation

Educational Value:
This implementation demonstrates multiple design patterns working together to solve
a real-world problem of game state management and error handling.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Protocol, TYPE_CHECKING
from colorama import Fore

if TYPE_CHECKING:
    import argparse

# Import configuration constants
from config.game_constants import (
    MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED,
    MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED,
    MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED,
    MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED,
    SLEEP_AFTER_EMPTY_STEP,
    MAX_STEPS_ALLOWED
)


class LimitType(Enum):
    """
    Enumeration of consecutive limit types.
    
    Each type represents a different kind of problematic game state that
    needs to be tracked and limited to prevent infinite loops or degraded
    game experience.
    """
    EMPTY_MOVES = "empty_moves"                    # LLM produced no valid moves
    SOMETHING_IS_WRONG = "something_is_wrong"      # Parsing/LLM errors
    INVALID_REVERSALS = "invalid_reversals"        # Blocked reversal attempts
    NO_PATH_FOUND = "no_path_found"               # Algorithm couldn't find path
    MAX_STEPS = "max_steps"                       # Game exceeded maximum steps


class LimitViolationAction(Enum):
    """Actions to take when a consecutive limit is violated."""
    END_GAME = "end_game"                         # Terminate the current game
    APPLY_PENALTY = "apply_penalty"               # Apply time penalty but continue
    LOG_WARNING = "log_warning"                   # Log warning but continue
    RESET_COUNTER = "reset_counter"               # Reset counter and continue


@dataclass
class LimitConfiguration:
    """
    Configuration for a specific consecutive limit.
    
    This class encapsulates all the settings needed to track and enforce
    a particular type of consecutive limit, following the Single Responsibility
    Principle.
    
    Attributes:
        limit_type: Type of limit being configured
        max_consecutive: Maximum allowed consecutive occurrences
        violation_action: Action to take when limit is exceeded
        sleep_after_occurrence: Time to sleep after each occurrence (minutes)
        end_reason_code: Code to record when game ends due to this limit
        warning_threshold: Threshold for issuing warnings (fraction of max)
    """
    limit_type: LimitType
    max_consecutive: int
    violation_action: LimitViolationAction = LimitViolationAction.END_GAME
    sleep_after_occurrence: float = 0.0  # minutes
    end_reason_code: str = ""
    warning_threshold: float = 0.75  # Issue warning at 75% of limit
    
    def __post_init__(self):
        """Validate configuration and set defaults."""
        if self.max_consecutive < 0:
            raise ValueError(f"max_consecutive must be non-negative, got {self.max_consecutive}")
        
        if not (0.0 <= self.warning_threshold <= 1.0):
            raise ValueError(f"warning_threshold must be between 0.0 and 1.0, got {self.warning_threshold}")
        
        # Set default end reason code if not provided
        if not self.end_reason_code:
            self.end_reason_code = f"MAX_CONSECUTIVE_{self.limit_type.value.upper()}_REACHED"


@dataclass
class LimitStatus:
    """
    Current status of a consecutive limit.
    
    This immutable data class tracks the current state of a limit,
    following the Value Object pattern for clean state management.
    """
    consecutive_count: int = 0
    total_count: int = 0
    last_occurrence_time: Optional[float] = None
    is_violated: bool = False
    warning_issued: bool = False
    
    def increment(self) -> 'LimitStatus':
        """Create new status with incremented counters."""
        return LimitStatus(
            consecutive_count=self.consecutive_count + 1,
            total_count=self.total_count + 1,
            last_occurrence_time=time.time(),
            is_violated=self.is_violated,
            warning_issued=self.warning_issued
        )
    
    def reset_consecutive(self) -> 'LimitStatus':
        """Create new status with reset consecutive counter."""
        return LimitStatus(
            consecutive_count=0,
            total_count=self.total_count,
            last_occurrence_time=self.last_occurrence_time,
            is_violated=False,
            warning_issued=False
        )


class GameStateProvider(Protocol):
    """
    Protocol defining the interface for game state access.
    
    This protocol allows the limits manager to interact with different
    game implementations without tight coupling, following the Dependency
    Inversion Principle.
    """
    
    def record_game_end(self, reason: str) -> None:
        """Record the end of a game with the specified reason."""
        ...
    
    def is_game_active(self) -> bool:
        """Check if the game is currently active."""
        ...


class LimitEnforcementStrategy(ABC):
    """
    Abstract base class for limit enforcement strategies.
    
    This implements the Strategy Pattern, allowing different approaches
    to handling limit violations while maintaining a consistent interface.
    """
    
    @abstractmethod
    def enforce_limit(self, config: LimitConfiguration, status: LimitStatus, 
                     game_state: GameStateProvider) -> bool:
        """
        Enforce the limit based on current status.
        
        Args:
            config: Limit configuration
            status: Current limit status
            game_state: Game state provider for actions
            
        Returns:
            True if game should continue, False if game should end
        """
        pass


class StandardLimitEnforcement(LimitEnforcementStrategy):
    """
    Standard limit enforcement strategy.
    
    This strategy implements the traditional approach:
    - Issue warnings at threshold
    - End game when limit is exceeded
    - Apply sleep penalties as configured
    """
    
    def enforce_limit(self, config: LimitConfiguration, status: LimitStatus, 
                     game_state: GameStateProvider) -> bool:
        """Enforce limit using standard strategy."""
        
        # Check if we've hit the warning threshold
        if (not status.warning_issued and 
            status.consecutive_count >= config.max_consecutive * config.warning_threshold):
            self._issue_warning(config, status)
        
        # Check if limit is violated
        if status.consecutive_count >= config.max_consecutive:
            return self._handle_violation(config, status, game_state)
        
        # Apply sleep penalty if configured
        if config.sleep_after_occurrence > 0:
            self._apply_sleep_penalty(config, status)
        
        return True  # Continue game
    
    def _issue_warning(self, config: LimitConfiguration, status: LimitStatus) -> None:
        """Issue a warning when approaching the limit."""
        limit_name = config.limit_type.value.replace('_', ' ').title()
        progress = f"{status.consecutive_count}/{config.max_consecutive}"
        
        print(Fore.YELLOW + f"⚠️  Approaching {limit_name} limit: {progress}")
        print(Fore.YELLOW + f"   Consider reviewing game strategy or increasing limits")
    
    def _handle_violation(self, config: LimitConfiguration, status: LimitStatus, 
                         game_state: GameStateProvider) -> bool:
        """Handle limit violation based on configured action."""
        limit_name = config.limit_type.value.replace('_', ' ').title()
        
        if config.violation_action == LimitViolationAction.END_GAME:
            print(Fore.RED + f"❌ Maximum consecutive {limit_name.lower()} reached "
                             f"({config.max_consecutive}). Game over.")
            
            if game_state.is_game_active():
                game_state.record_game_end(config.end_reason_code)
            return False  # End game
        
        elif config.violation_action == LimitViolationAction.LOG_WARNING:
            print(Fore.YELLOW + f"⚠️  Consecutive {limit_name.lower()} limit exceeded "
                               f"({status.consecutive_count}), but continuing game")
            return True  # Continue game
        
        elif config.violation_action == LimitViolationAction.APPLY_PENALTY:
            penalty_time = config.sleep_after_occurrence * 2  # Double penalty
            print(Fore.CYAN + f"⏸️  Applying penalty sleep ({penalty_time:.1f} minutes) "
                             f"for excessive {limit_name.lower()}")
            time.sleep(penalty_time * 60)
            return True  # Continue game
        
        return True  # Default: continue game
    
    def _apply_sleep_penalty(self, config: LimitConfiguration, status: LimitStatus) -> None:
        """Apply configured sleep penalty."""
        if config.sleep_after_occurrence <= 0:
            return
        
        limit_name = config.limit_type.value.replace('_', ' ').title()
        sleep_time = config.sleep_after_occurrence
        plural = "s" if sleep_time != 1 else ""
        
        print(Fore.CYAN + f"⏸️  Sleeping {sleep_time} minute{plural} after {limit_name.lower()} occurrence...")
        time.sleep(sleep_time * 60)


class ConsecutiveLimitsManager:
    """
    Elegant manager for all consecutive move limits.
    
    This class implements the Facade Pattern, providing a clean interface
    for managing multiple consecutive limits while hiding the complexity
    of individual limit tracking and enforcement.
    
    Design Patterns Used:
    - Facade: Simplifies complex limit management
    - Strategy: Pluggable enforcement strategies
    - Observer: Responds to game events
    - Template Method: Common limit checking workflow
    
    Key Features:
    - Centralized limit configuration
    - Automatic counter management
    - Configurable enforcement strategies
    - Comprehensive logging and reporting
    - Thread-safe operations
    """
    
    def __init__(self, args: 'argparse.Namespace', 
                 enforcement_strategy: Optional[LimitEnforcementStrategy] = None):
        """
        Initialize the consecutive limits manager.
        
        Args:
            args: Command line arguments containing limit configurations
            enforcement_strategy: Strategy for enforcing limits (defaults to standard)
        """
        self.args = args
        self.enforcement_strategy = enforcement_strategy or StandardLimitEnforcement()
        
        # Initialize limit configurations from command line arguments
        self.configurations: Dict[LimitType, LimitConfiguration] = self._create_configurations()
        
        # Initialize limit status tracking
        self.statuses: Dict[LimitType, LimitStatus] = {
            limit_type: LimitStatus() for limit_type in LimitType
        }
        
        # Track last move for intelligent counter resets
        self.last_move: Optional[str] = None
        
        print(Fore.GREEN + "✅ Consecutive Limits Manager initialized with elegant tracking")
        self._log_configuration()
    
    def _create_configurations(self) -> Dict[LimitType, LimitConfiguration]:
        """
        Create limit configurations from command line arguments.
        
        This method follows the Factory Pattern to create appropriate
        configurations based on the provided arguments.
        """
        configs = {}
        
        # Empty moves configuration (LLM-specific)
        configs[LimitType.EMPTY_MOVES] = LimitConfiguration(
            limit_type=LimitType.EMPTY_MOVES,
            max_consecutive=getattr(self.args, 'max_consecutive_empty_moves_allowed', 
                                  MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED),
            sleep_after_occurrence=getattr(self.args, 'sleep_after_empty_step', 
                                         SLEEP_AFTER_EMPTY_STEP),
            end_reason_code="MAX_CONSECUTIVE_EMPTY_MOVES_REACHED"
        )
        
        # Something is wrong configuration (LLM-specific)
        configs[LimitType.SOMETHING_IS_WRONG] = LimitConfiguration(
            limit_type=LimitType.SOMETHING_IS_WRONG,
            max_consecutive=getattr(self.args, 'max_consecutive_something_is_wrong_allowed',
                                  MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED),
            end_reason_code="MAX_CONSECUTIVE_SOMETHING_IS_WRONG_REACHED"
        )
        
        # Invalid reversals configuration (Universal)
        configs[LimitType.INVALID_REVERSALS] = LimitConfiguration(
            limit_type=LimitType.INVALID_REVERSALS,
            max_consecutive=getattr(self.args, 'max_consecutive_invalid_reversals_allowed',
                                  MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED),
            end_reason_code="MAX_CONSECUTIVE_INVALID_REVERSALS_REACHED"
        )
        
        # No path found configuration (Universal)
        configs[LimitType.NO_PATH_FOUND] = LimitConfiguration(
            limit_type=LimitType.NO_PATH_FOUND,
            max_consecutive=getattr(self.args, 'max_consecutive_no_path_found_allowed',
                                  MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED),
            end_reason_code="MAX_CONSECUTIVE_NO_PATH_FOUND_REACHED"
        )
        
        # Max steps configuration (Absolute limit, not consecutive)
        configs[LimitType.MAX_STEPS] = LimitConfiguration(
            limit_type=LimitType.MAX_STEPS,
            max_consecutive=getattr(self.args, 'max_steps', MAX_STEPS_ALLOWED),
            end_reason_code="MAX_STEPS_REACHED",
            warning_threshold=0.9  # Warning at 90% for absolute limits
        )
        
        return configs
    
    def _log_configuration(self) -> None:
        """Log the current configuration for debugging and monitoring."""
        print(Fore.BLUE + "📊 Consecutive Limits Configuration:")
        for limit_type, config in self.configurations.items():
            limit_name = limit_type.value.replace('_', ' ').title()
            if limit_type == LimitType.MAX_STEPS:
                print(Fore.BLUE + f"   • {limit_name}: Max {config.max_consecutive} steps")
            else:
                sleep_info = f", Sleep: {config.sleep_after_occurrence}min" if config.sleep_after_occurrence > 0 else ""
                print(Fore.BLUE + f"   • {limit_name}: Max {config.max_consecutive}{sleep_info}")
    
    def record_move(self, move: str, game_state: GameStateProvider) -> bool:
        """
        Record a move and update consecutive counters.
        
        This method implements the Template Method Pattern, providing
        a consistent workflow for processing any type of move.
        
        Args:
            move: The move that was made (including sentinel moves)
            game_state: Game state provider for enforcement actions
            
        Returns:
            True if game should continue, False if game should end
        """
        # Store the move for intelligent counter management
        previous_move = self.last_move
        self.last_move = move
        
        # Process the move based on its type
        if move == "EMPTY":
            return self._handle_limit_occurrence(LimitType.EMPTY_MOVES, game_state)
        elif move == "SOMETHING_IS_WRONG":
            return self._handle_limit_occurrence(LimitType.SOMETHING_IS_WRONG, game_state)
        elif move == "INVALID_REVERSAL":
            return self._handle_limit_occurrence(LimitType.INVALID_REVERSALS, game_state)
        elif move == "NO_PATH_FOUND":
            return self._handle_limit_occurrence(LimitType.NO_PATH_FOUND, game_state)
        else:
            # Valid move - reset appropriate counters
            self._reset_counters_for_valid_move(move, previous_move)
            return True
    
    def check_step_limit(self, current_steps: int, game_state: GameStateProvider) -> bool:
        """
        Check if the current step count exceeds the maximum allowed steps.
        
        This method handles absolute limits (like max steps) rather than
        consecutive limits. It follows the same elegant pattern as consecutive
        limit checking but with different logic.
        
        Args:
            current_steps: Current number of steps in the game
            game_state: Game state provider for enforcement actions
            
        Returns:
            True if game should continue, False if game should end
        """
        config = self.configurations[LimitType.MAX_STEPS]
        
        # Update the status to reflect current step count
        # For absolute limits, we use consecutive_count to track the current value
        old_status = self.statuses[LimitType.MAX_STEPS]
        new_status = LimitStatus(
            consecutive_count=current_steps,
            total_count=current_steps,
            last_occurrence_time=time.time(),
            is_violated=current_steps >= config.max_consecutive,
            warning_issued=old_status.warning_issued
        )
        self.statuses[LimitType.MAX_STEPS] = new_status
        
        # Check for warnings and violations
        if (not old_status.warning_issued and 
            current_steps >= config.max_consecutive * config.warning_threshold):
            self._issue_step_warning(config, new_status)
            # Update status to mark warning issued
            new_status = LimitStatus(
                consecutive_count=current_steps,
                total_count=current_steps,
                last_occurrence_time=new_status.last_occurrence_time,
                is_violated=new_status.is_violated,
                warning_issued=True
            )
            self.statuses[LimitType.MAX_STEPS] = new_status
        
        # Check if limit is exceeded
        if current_steps >= config.max_consecutive:
            print(Fore.RED + f"❌ Maximum steps reached ({config.max_consecutive}). Game over.")
            game_state.record_game_end(config.end_reason_code)
            return False
        
        return True
    
    def _issue_step_warning(self, config: LimitConfiguration, status: LimitStatus) -> None:
        """Issue a warning for step limit approaching."""
        warning_threshold = int(config.max_consecutive * config.warning_threshold)
        remaining_steps = config.max_consecutive - status.consecutive_count
        print(Fore.YELLOW + 
              f"⚠️  Approaching step limit: {status.consecutive_count}/{config.max_consecutive} "
              f"({remaining_steps} steps remaining)")
        print(Fore.YELLOW + f"   Warning threshold: {warning_threshold} steps")
    
    def _handle_limit_occurrence(self, limit_type: LimitType, 
                                game_state: GameStateProvider) -> bool:
        """
        Handle the occurrence of a specific limit type.
        
        This is the core of the Template Method Pattern implementation.
        """
        # Update status
        old_status = self.statuses[limit_type]
        new_status = old_status.increment()
        self.statuses[limit_type] = new_status
        
        # Get configuration
        config = self.configurations[limit_type]
        
        # Log the occurrence
        self._log_occurrence(limit_type, new_status, config)
        
        # Enforce the limit
        return self.enforcement_strategy.enforce_limit(config, new_status, game_state)
    
    def _reset_counters_for_valid_move(self, current_move: str, previous_move: Optional[str]) -> None:
        """
        Intelligently reset consecutive counters based on move patterns.
        
        This method implements sophisticated logic for determining which
        counters should be reset when a valid move is made.
        """
        # Reset counters based on move type and context
        if current_move in ["UP", "DOWN", "LEFT", "RIGHT"]:
            # Valid directional move - reset most counters
            self._reset_counter(LimitType.EMPTY_MOVES)
            self._reset_counter(LimitType.SOMETHING_IS_WRONG)
            self._reset_counter(LimitType.NO_PATH_FOUND)
            # Note: INVALID_REVERSALS counter is reset elsewhere in game logic
        
        # Additional intelligent resets based on move patterns
        if previous_move == "EMPTY" and current_move != "EMPTY":
            self._reset_counter(LimitType.EMPTY_MOVES)
        
        if previous_move == "SOMETHING_IS_WRONG" and current_move != "SOMETHING_IS_WRONG":
            self._reset_counter(LimitType.SOMETHING_IS_WRONG)
    
    def _reset_counter(self, limit_type: LimitType) -> None:
        """Reset the consecutive counter for a specific limit type."""
        old_status = self.statuses[limit_type]
        new_status = old_status.reset_consecutive()
        self.statuses[limit_type] = new_status
    
    def _log_occurrence(self, limit_type: LimitType, status: LimitStatus, 
                       config: LimitConfiguration) -> None:
        """Log the occurrence of a limit event with appropriate formatting."""
        limit_name = limit_type.value.replace('_', ' ').title()
        progress = f"{status.consecutive_count}/{config.max_consecutive}"
        
        if status.consecutive_count == 1:
            print(Fore.YELLOW + f"⚠️  {limit_name} occurred. Count: {progress}")
        else:
            print(Fore.YELLOW + f"⚠️  Consecutive {limit_name.lower()}: {progress}")
    
    def get_status_summary(self) -> Dict[str, Dict[str, int]]:
        """
        Get a comprehensive summary of all limit statuses.
        
        Returns:
            Dictionary containing status information for all limits
        """
        summary = {}
        for limit_type, status in self.statuses.items():
            config = self.configurations[limit_type]
            if limit_type == LimitType.MAX_STEPS:
                summary[limit_type.value] = {
                    "current_steps": status.consecutive_count,
                    "max_steps": config.max_consecutive,
                    "is_violated": status.consecutive_count >= config.max_consecutive,
                    "warning_threshold": int(config.max_consecutive * config.warning_threshold),
                    "remaining_steps": max(0, config.max_consecutive - status.consecutive_count)
                }
            else:
                summary[limit_type.value] = {
                    "consecutive_count": status.consecutive_count,
                    "total_count": status.total_count,
                    "max_allowed": config.max_consecutive,
                    "is_violated": status.consecutive_count >= config.max_consecutive,
                    "warning_threshold": int(config.max_consecutive * config.warning_threshold)
                }
        return summary
    
    def reset_all_counters(self) -> None:
        """Reset all consecutive counters (typically called at game start)."""
        for limit_type in LimitType:
            self._reset_counter(limit_type)
        print(Fore.GREEN + "🔄 All consecutive counters reset")
    
    def __str__(self) -> str:
        """String representation for debugging."""
        status_lines = []
        for limit_type, status in self.statuses.items():
            config = self.configurations[limit_type]
            limit_name = limit_type.value.replace('_', ' ').title()
            progress = f"{status.consecutive_count}/{config.max_consecutive}"
            status_lines.append(f"{limit_name}: {progress}")
        
        return f"ConsecutiveLimitsManager({', '.join(status_lines)})"


def create_limits_manager(args: 'argparse.Namespace') -> ConsecutiveLimitsManager:
    """
    Factory function for creating a consecutive limits manager.
    
    This function implements the Factory Pattern, providing a clean
    way to create properly configured limits managers.
    
    Args:
        args: Command line arguments containing configuration
        
    Returns:
        Configured ConsecutiveLimitsManager instance
    """
    return ConsecutiveLimitsManager(args) 