"""
Game data tracking and statistics management for the Snake game.
Provides centralized collection and reporting of game statistics.
"""

import json
import time
from datetime import datetime
import numpy as np
from utils.json_utils import NumPyJSONEncoder
import os
import re

class GameData:
    """Tracks and manages statistics for Snake game sessions."""
    
    def __init__(self):
        """Initialize the game data tracking."""
        self.reset()
    
    def reset(self):
        """Reset all tracking data to initial state."""
        # Game metadata
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.game_number = 0
        
        # Game stats
        self.score = 0
        self.steps = 0
        self.empty_steps = 0
        self.error_steps = 0
        self.invalid_reversals = 0  # Counter for invalid reversals
        self.consecutive_empty_moves = 0
        self.max_consecutive_empty_moves_reached = 0
        
        # Game state
        self.game_over = False
        # Removed win attribute as it's redundant in Snake (there is no win, only game over)
        
        # Other tracking data
        self.start_time = time.time()
        self.end_time = None
        self.rounds_data = {}
        # Initialize round_count to 1 to make round numbering more intuitive (1, 2, 3, ...)
        self.round_count = 1
        self.current_round_data = self._create_empty_round_data()
        self.apple_positions = []
        self.game_end_reason = None
        self.last_move = None
        self.last_action_time = None
        
        # Time tracking
        self.llm_communication_time = 0  # Total time spent communicating with LLMs
        self.game_movement_time = 0      # Time spent in actual game movement
        self.waiting_time = 0            # Time spent waiting (pauses, etc.)
        self.last_action_time = self.start_time  # For tracking time between actions
        
        # Basic game stats
        self.snake_length = 1
        self.max_empty_moves_allowed = 3
        
        # Game history
        self.moves = []
        
        # Step statistics
        self.valid_steps = 0
        
        # Response times
        self.primary_response_times = []
        self.secondary_response_times = []
        
        # Token statistics - Simplified to just track totals
        self.primary_token_stats = []
        self.secondary_token_stats = []
        
        # LLM error statistics
        self.primary_llm_errors = 0
        self.secondary_llm_errors = 0
        self.primary_llm_requests = 0
        self.secondary_llm_requests = 0
        
        # JSON parsing statistics
        self.total_extraction_attempts = 0
        self.successful_extractions = 0
        self.failed_extractions = 0
        self.json_decode_errors = 0
        self.format_validation_errors = 0
        self.code_block_extraction_errors = 0
        self.text_extraction_errors = 0
        self.pattern_extraction_success = 0
        
        # Parser usage
        self.parser_usage_count = 0
    
    def _create_empty_round_data(self):
        return {
            "apple_position": None,
            "moves": [],
            "primary_response_times": [],
            "secondary_response_times": [],
            "primary_token_stats": [],
            "secondary_token_stats": []
        }
    
    def start_new_round(self, apple_position):
        """Start a new round of moves.
        
        Args:
            apple_position: The position of the apple as [x, y]
        """
        # Always save the current round data to rounds_data, even if it's empty
        # This ensures we don't lose any data during transitions
        if self.current_round_data:
            # Use the current round_count (will be incremented later in communication_utils.py)
            round_key = f"round_{self.round_count}"
            
            # Make a deep copy of the current round data to store in rounds_data
            # This prevents changes to current_round_data from affecting the saved data
            self.rounds_data[round_key] = self.current_round_data.copy()
            
            # Log that we saved round data
            print(f"💾 Saved data for round {self.round_count} with {len(self.current_round_data.get('moves', []))} moves")
        
        # IMPORTANT: We DO NOT increment round_count here anymore
        # Round count is ONLY incremented in llm/communication_utils.py after getting a valid move from the LLM
        # This ensures rounds in JSON files match the prompt/response file counts
        
        # Reset current round data
        self.current_round_data = {
            "apple_position": apple_position,
            "moves": [],
            "primary_response_times": [],
            "secondary_response_times": [],
            "primary_token_stats": [],
            "secondary_token_stats": []
        }
    
    def record_move(self, move, apple_eaten=False):
        """Record a move and update relevant statistics.
        
        Args:
            move: The direction moved ("UP", "DOWN", "LEFT", "RIGHT")
            apple_eaten: Whether an apple was eaten on this move
        """
        self.last_move = move
        self.moves.append(move)
        self.current_round_data["moves"].append(move)
        self.steps += 1
        self.valid_steps += 1
        self.consecutive_empty_moves = 0  # Reset on valid move
        
        # Also update the rounds_data for the current round
        round_data = self._get_or_create_round_data(self.round_count)
        
        # Ensure the move is recorded in rounds_data
        if "moves" not in round_data:
            round_data["moves"] = []
        round_data["moves"].append(move)
        
        if apple_eaten:
            self.score += 1
            self.snake_length += 1
            # Note: We DO NOT increment round_count here
            # Round count is ONLY incremented in llm/communication_utils.py after getting a valid move from the LLM
            # This ensures round numbers align with LLM queries, not with apple eating events
        
    def record_apple_position(self, position):
        """Record an apple position.
        
        Args:
            position: The position of the apple as [x, y]
        """
        x, y = position
        self.apple_positions.append({"x": x, "y": y})
        self.current_round_data["apple_position"] = [x, y]
        
        # Also store directly in the rounds_data for the current round
        round_data = self._get_or_create_round_data(self.round_count)
        
        # Update apple position in the round data
        round_data["apple_position"] = [x, y]
    
    def record_empty_move(self):
        """Record an empty move (no valid direction)."""
        self.empty_steps += 1
        self.steps += 1
        self.consecutive_empty_moves += 1
        self.max_consecutive_empty_moves_reached = max(
            self.max_consecutive_empty_moves_reached, 
            self.consecutive_empty_moves
        )
    
    def record_invalid_reversal(self, attempted_move, current_direction):
        """Record an invalid reversal move.
        
        Args:
            attempted_move: The direction that was attempted ("UP", "DOWN", etc.)
            current_direction: The current direction of the snake
        """
        self.invalid_reversals += 1
        
        # Add to the current round data if we're tracking a round
        if self.current_round_data:
            if "invalid_reversals" not in self.current_round_data:
                self.current_round_data["invalid_reversals"] = []
            
            self.current_round_data["invalid_reversals"].append({
                "attempted_move": attempted_move,
                "current_direction": current_direction,
                "step": self.steps
            })
    
    def record_error_move(self):
        """Record an error move (error in LLM response)."""
        self.error_steps += 1
        self.steps += 1
    
    def record_llm_communication_start(self):
        """Mark the start of communication with an LLM."""
        self.last_action_time = time.time()
        
    def record_llm_communication_end(self):
        """Record time spent communicating with an LLM."""
        current_time = time.time()
        self.llm_communication_time += (current_time - self.last_action_time)
        self.last_action_time = current_time
        
    def record_game_movement_start(self):
        """Mark the start of game movement."""
        self.last_action_time = time.time()
        
    def record_game_movement_end(self):
        """Record time spent on game movement."""
        current_time = time.time()
        self.game_movement_time += (current_time - self.last_action_time)
        self.last_action_time = current_time
        
    def record_waiting_start(self):
        """Mark the start of waiting time."""
        self.last_action_time = time.time()
        
    def record_waiting_end(self):
        """Record time spent waiting."""
        current_time = time.time()
        self.waiting_time += (current_time - self.last_action_time)
        self.last_action_time = current_time
    
    def record_game_end(self, reason):
        """Record the end of a game.
        
        Args:
            reason: The reason the game ended ("WALL", "SELF", "MAX_STEPS_REACHED", 
                   "MAX_EMPTY_MOVES_REACHED", "MAX_CONSECUTIVE_ERRORS_REACHED")
        """
        # Standardize reason naming to ensure consistency
        if reason == "MAX_STEPS":
            reason = "MAX_STEPS_REACHED"
        elif reason == "EMPTY_MOVES":
            reason = "MAX_EMPTY_MOVES_REACHED"
        elif reason == "ERROR_THRESHOLD":
            reason = "MAX_CONSECUTIVE_ERRORS_REACHED"
        
        self.game_end_reason = reason
        self.game_over = True
        self.game_number += 1
        self.end_time = time.time()
        
        # Save current round data without incrementing round count
        # This ensures we capture the final state without creating an extra round at game end
        if self.current_round_data:
            # Get round data for the current round
            round_data = self._get_or_create_round_data(self.round_count)
            
            # Save the round data by copying the current round data
            for key, value in self.current_round_data.items():
                if value is not None:
                    round_data[key] = value
            
            # Log that we saved the final round data
            print(f"💾 Saved final data for round {self.round_count} with {len(self.current_round_data.get('moves', []))} moves")
    
    def record_primary_response_time(self, duration):
        """Record a primary LLM response time.
        
        Args:
            duration: Response time in seconds
        """
        self.primary_response_times.append(duration)
        self.current_round_data["primary_response_times"].append(duration)
        self.primary_llm_requests += 1
        
        # Also store directly in the rounds_data for the current round
        round_data = self._get_or_create_round_data(self.round_count)
                
        # Update response times in the round data
        round_data["primary_response_times"].append(duration)
    
    def record_secondary_response_time(self, duration):
        """Record a secondary LLM response time.
        
        Args:
            duration: Response time in seconds
        """
        self.secondary_response_times.append(duration)
        self.current_round_data["secondary_response_times"].append(duration)
        self.secondary_llm_requests += 1
        self.parser_usage_count += 1
        
        # Also store directly in the rounds_data for the current round
        round_data = self._get_or_create_round_data(self.round_count)
                
        # Update response times in the round data
        round_data["secondary_response_times"].append(duration)
    
    def record_primary_token_stats(self, prompt_tokens, completion_tokens):
        """Record token usage statistics for the primary LLM.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
        """
        token_stats = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
        
        # Add to global token stats
        self.primary_token_stats.append(token_stats)
        
        # Add to current round data
        self.current_round_data["primary_token_stats"].append(token_stats)
        
        # Also store directly in the rounds_data for the current round
        round_data = self._get_or_create_round_data(self.round_count)
                
        # Update token stats in the round data
        round_data["primary_token_stats"].append(token_stats)
    
    def record_secondary_token_stats(self, prompt_tokens, completion_tokens):
        """Record token usage statistics for the secondary LLM.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
        """
        token_stats = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
        
        # Add to global token stats
        self.secondary_token_stats.append(token_stats)
        
        # Add to current round data
        self.current_round_data["secondary_token_stats"].append(token_stats)
        
        # Also store directly in the rounds_data for the current round
        round_data = self._get_or_create_round_data(self.round_count)
                
        # Update token stats in the round data
        round_data["secondary_token_stats"].append(token_stats)
    
    def record_json_extraction_attempt(self, success, error_type=None):
        """Record an attempt to extract JSON from LLM response.
        
        Args:
            success: Whether the extraction was successful
            error_type: Type of error if unsuccessful (decode, validation, etc.)
        """
        self.total_extraction_attempts += 1
        
        if success:
            self.successful_extractions += 1
        else:
            self.failed_extractions += 1
            
            # Record specific error type if provided
            if error_type == "decode":
                self.json_decode_errors += 1
            elif error_type == "validation":
                self.format_validation_errors += 1
            elif error_type == "code_block":
                self.code_block_extraction_errors += 1
            elif error_type == "text":
                self.text_extraction_errors += 1
    
    def record_continuation(self, previous_session_data=None):
        """Record that this game is a continuation of a previous session.
        
        Args:
            previous_session_data: Optional dictionary containing data from the previous session
        """
        # Initialize continuation tracking attributes if they don't exist
        if not hasattr(self, 'is_continuation'):
            self.is_continuation = True
            self.continuation_count = 1
            self.continuation_timestamps = []
            self.continuation_metadata = []
        else:
            self.continuation_count += 1
        
        # Record continuation timestamp
        continuation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.continuation_timestamps.append(continuation_timestamp)
        
        # Store metadata about this continuation
        continuation_meta = {
            "timestamp": continuation_timestamp,
            "continuation_number": self.continuation_count
        }
        
        # Add previous session data if provided
        if previous_session_data:
            # Extract game statistics from summary.json
            if 'game_count' in previous_session_data:
                continuation_meta['previous_session'] = {
                    'total_games': previous_session_data.get('game_count', 0),
                    'total_score': previous_session_data.get('total_score', 0),
                    'total_steps': previous_session_data.get('total_steps', 0),
                    'scores': previous_session_data.get('game_scores', [])
                }
        
        # Add this continuation's metadata
        self.continuation_metadata.append(continuation_meta)
    
    def synchronize_with_summary_json(self, summary_data):
        """Synchronize game state with data from summary.json.
        
        Args:
            summary_data: Dictionary containing data from summary.json
        """
        # Import settings from summary.json
        if 'max_empty_moves_allowed' in summary_data:
            self.max_empty_moves_allowed = summary_data['max_empty_moves_allowed']
        
        # Import step statistics if available
        if 'step_stats' in summary_data:
            step_stats = summary_data['step_stats']
            # Don't overwrite empty_steps and error_steps as they're already tracked from game files
            # But import valid_steps and invalid_reversals
            self.valid_steps = step_stats.get('valid_steps', 0)
            self.invalid_reversals = step_stats.get('invalid_reversals', 0)
        
        # Import JSON parsing stats if available
        if 'json_parsing_stats' in summary_data:
            json_stats = summary_data['json_parsing_stats']
            self.total_extraction_attempts = json_stats.get('total_extraction_attempts', 0)
            self.successful_extractions = json_stats.get('successful_extractions', 0)
            self.failed_extractions = json_stats.get('failed_extractions', 0)
            self.json_decode_errors = json_stats.get('json_decode_errors', 0)
            self.format_validation_errors = json_stats.get('format_validation_errors', 0)
            self.code_block_extraction_errors = json_stats.get('code_block_extraction_errors', 0)
            self.text_extraction_errors = json_stats.get('text_extraction_errors', 0)
            self.pattern_extraction_success = json_stats.get('pattern_extraction_success', 0)
        
        # Initialize continuation attributes if needed
        if not hasattr(self, 'is_continuation'):
            self.is_continuation = False
            self.continuation_count = 0
            self.continuation_timestamps = []
            self.continuation_metadata = []
    
    def record_round_data(self, round_data):
        """Record data for a game round.
        
        Args:
            round_data: Dictionary containing round data:
                - round_number: Round number
                - apple_position: Position of the apple as [x, y]
                - moves: List of moves made in this round
                - primary_response_time: Time taken to get primary LLM response (optional)
                - secondary_response_time: Time taken to get secondary LLM response (optional)
                - primary_tokens: Dictionary with primary token stats (optional)
                - secondary_tokens: Dictionary with secondary token stats (optional)
        """
        round_number = round_data.get('round_number')
        apple_position = round_data.get('apple_position')
        moves = round_data.get('moves', [])
        
        # Format the round key
        round_key = f"round_{round_number}"
        
        # Initialize round data if not exists
        if round_key not in self.rounds_data:
            self.rounds_data[round_key] = self._create_empty_round_data()
        
        # Update round data
        self.rounds_data[round_key]["apple_position"] = apple_position
        
        # Update moves - store the full array of moves from the LLM
        if moves:
            # Store the entire array of moves from the LLM
            self.rounds_data[round_key]["moves"] = moves
        
        # Update response times if provided
        if 'primary_response_time' in round_data and round_data['primary_response_time'] is not None:
            self.rounds_data[round_key]["primary_response_times"].append(round_data['primary_response_time'])
        
        if 'secondary_response_time' in round_data and round_data['secondary_response_time'] is not None:
            self.rounds_data[round_key]["secondary_response_times"].append(round_data['secondary_response_time'])
        
        # Update token stats if provided
        if 'primary_tokens' in round_data and round_data['primary_tokens'] is not None:
            self.rounds_data[round_key]["primary_token_stats"].append(round_data['primary_tokens'])
        
        if 'secondary_tokens' in round_data and round_data['secondary_tokens'] is not None:
            self.rounds_data[round_key]["secondary_token_stats"].append(round_data['secondary_tokens'])
    
    def get_step_stats(self):
        """Calculate step statistics.
        
        Returns:
            Dictionary of step statistics
        """
        return {
            "total_steps": self.steps,
            "valid_steps": self.valid_steps,
            "empty_steps": self.empty_steps,
            "error_steps": self.error_steps,
            "invalid_reversals": self.invalid_reversals,
            "consecutive_empty_moves": self.consecutive_empty_moves,
            "max_consecutive_empty_moves": self.max_consecutive_empty_moves_reached
        }
    
    def get_prompt_response_stats(self):
        """Calculate prompt response statistics.
        
        Returns:
            Dictionary of prompt response statistics
        """
        # Primary LLM stats
        primary_times = self.primary_response_times
        avg_primary = np.mean(primary_times) if primary_times else 0
        min_primary = np.min(primary_times) if primary_times else 0
        max_primary = np.max(primary_times) if primary_times else 0
        
        # Secondary LLM stats
        secondary_times = self.secondary_response_times
        avg_secondary = np.mean(secondary_times) if secondary_times else 0
        min_secondary = np.min(secondary_times) if secondary_times else 0
        max_secondary = np.max(secondary_times) if secondary_times else 0
        
        return {
            "avg_primary_response_time": avg_primary,
            "min_primary_response_time": min_primary,
            "max_primary_response_time": max_primary,
            "avg_secondary_response_time": avg_secondary,
            "min_secondary_response_time": min_secondary,
            "max_secondary_response_time": max_secondary
        }
    
    def get_token_stats(self):
        """Calculate token statistics.
        
        Returns:
            Dictionary of token statistics
        """
        # Calculate primary LLM token stats
        primary_prompt = [stat["prompt_tokens"] for stat in self.primary_token_stats]
        primary_completion = [stat["completion_tokens"] for stat in self.primary_token_stats]
        primary_total = [stat["total_tokens"] for stat in self.primary_token_stats]
        
        # Calculate secondary LLM token stats
        secondary_prompt = [stat["prompt_tokens"] for stat in self.secondary_token_stats]
        secondary_completion = [stat["completion_tokens"] for stat in self.secondary_token_stats]
        secondary_total = [stat["total_tokens"] for stat in self.secondary_token_stats]
        
        return {
            "primary": {
                "total_tokens": sum(primary_total) if primary_total else 0,
                "total_prompt_tokens": sum(primary_prompt) if primary_prompt else 0,
                "total_completion_tokens": sum(primary_completion) if primary_completion else 0,
                "avg_total_tokens": np.mean(primary_total) if primary_total else 0,
                "avg_prompt_tokens": np.mean(primary_prompt) if primary_prompt else 0,
                "avg_completion_tokens": np.mean(primary_completion) if primary_completion else 0,
                "request_count": len(primary_total)
            },
            "secondary": {
                "total_tokens": sum(secondary_total) if secondary_total else 0,
                "total_prompt_tokens": sum(secondary_prompt) if secondary_prompt else 0,
                "total_completion_tokens": sum(secondary_completion) if secondary_completion else 0,
                "avg_total_tokens": np.mean(secondary_total) if secondary_total else 0,
                "avg_prompt_tokens": np.mean(secondary_prompt) if secondary_prompt else 0,
                "avg_completion_tokens": np.mean(secondary_completion) if secondary_completion else 0,
                "request_count": len(secondary_total)
            }
        }
    
    def get_error_stats(self):
        """Calculate error statistics.
        
        Returns:
            Dictionary of error statistics
        """
        primary_error_rate = self.primary_llm_errors / max(1, self.primary_llm_requests) * 100
        secondary_error_rate = self.secondary_llm_errors / max(1, self.secondary_llm_requests) * 100
        
        return {
            "total_errors_from_primary_llm": self.primary_llm_errors,
            "total_errors_from_secondary_llm": self.secondary_llm_errors,
            "error_rate_from_primary_llm": primary_error_rate,
            "error_rate_from_secondary_llm": secondary_error_rate
        }
    
    def get_json_parsing_stats(self):
        """Calculate JSON parsing statistics.
        
        Returns:
            Dictionary of JSON parsing statistics
        """
        success_rate = self.successful_extractions / max(1, self.total_extraction_attempts) * 100
        failure_rate = self.failed_extractions / max(1, self.total_extraction_attempts) * 100
        
        return {
            "total_extraction_attempts": self.total_extraction_attempts,
            "successful_extractions": self.successful_extractions,
            "success_rate": success_rate,
            "failed_extractions": self.failed_extractions,
            "failure_rate": failure_rate,
            "json_decode_errors": self.json_decode_errors,
            "format_validation_errors": self.format_validation_errors,
            "code_block_extraction_errors": self.code_block_extraction_errors,
            "text_extraction_errors": self.text_extraction_errors,
            "pattern_extraction_success": self.pattern_extraction_success
        }
    
    def get_time_stats(self):
        """Calculate time-related statistics.
        
        Returns:
            Dictionary of time statistics
        """
        # Calculate total game duration
        if self.end_time is None:
            self.end_time = time.time()  # If game isn't over yet, use current time
        
        total_duration = self.end_time - self.start_time
        
        # Calculate percentages based on active time
        divisor = max(0.001, total_duration)  # Avoid division by zero
        llm_percent = (self.llm_communication_time / divisor * 100)
        movement_percent = (self.game_movement_time / divisor * 100)
        waiting_percent = (self.waiting_time / divisor * 100)
        
        # Calculate other time (time not accounted for in the other categories)
        other_time = total_duration - (self.llm_communication_time + self.game_movement_time + self.waiting_time)
        other_percent = (other_time / divisor * 100)
        
        return {
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(self.end_time).strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": total_duration,
            "llm_communication_time": self.llm_communication_time,
            "game_movement_time": self.game_movement_time,
            "waiting_time": self.waiting_time,
            "other_time": other_time,
            "llm_communication_percent": llm_percent,
            "game_movement_percent": movement_percent,
            "waiting_percent": waiting_percent,
            "other_percent": other_percent
        }
    
    def generate_game_summary(self, primary_provider, primary_model, parser_provider, parser_model, max_consecutive_errors_allowed=5):
        """Generate a summary of the game.
        
        Args:
            primary_provider: The provider of the primary LLM
            primary_model: The model of the primary LLM
            parser_provider: The provider of the parser LLM
            parser_model: The model of the parser LLM
            max_consecutive_errors_allowed: Maximum consecutive errors allowed before game over
            
        Returns:
            Dictionary with game summary
        """
        # Create the base summary
        summary = {
            # Core game data
            "score": self.score,
            "steps": self.steps,
            "snake_length": self.snake_length,
            "game_over": self.game_over,
            "game_end_reason": getattr(self, 'game_end_reason', "UNKNOWN"),
            "round_count": self.round_count,  # Add round_count at the top level
            
            # Time statistics
            "time_stats": self.get_time_stats(),
            
            # Provider and model info
            "llm_info": {
                "primary_provider": primary_provider,
                "primary_model": primary_model,
                "parser_provider": parser_provider if parser_provider and hasattr(parser_provider, 'lower') and parser_provider.lower() != "none" else None,
                "parser_model": parser_model if parser_provider and hasattr(parser_provider, 'lower') and parser_provider.lower() != "none" else None
            },
            
            # Important statistics
            "prompt_response_stats": self.get_prompt_response_stats(),
            "token_stats": self.get_token_stats(),
            "step_stats": self.get_step_stats(),
            "error_stats": self.get_error_stats(),
            "json_parsing_stats": self.get_json_parsing_stats(),
            
            # Metadata
            "metadata": {
                "game_number": self.game_number,
                "timestamp": self.timestamp,
                "last_move": self.last_move,
                "round_count": self.round_count,  # Keep it here for backward compatibility
                "max_empty_moves_allowed": self.max_empty_moves_allowed,
                "max_consecutive_errors_allowed": max_consecutive_errors_allowed,
                "parser_usage_count": self.parser_usage_count
            },
            
            # Raw token stats data (for detailed analysis)
            "primary_token_stats": self.primary_token_stats,
            "secondary_token_stats": self.secondary_token_stats,
            
            # Detailed game history (at bottom)
            # Create an ordered version of rounds_data
            "detailed_history": {
                "apple_positions": self.apple_positions,
                "moves": self.moves,
                "rounds_data": self._get_ordered_rounds_data()
            }
        }
        
        # Add continuation data if this is a continuation
        if hasattr(self, 'is_continuation') and self.is_continuation:
            summary["continuation_info"] = {
                "is_continuation": True,
                "continuation_count": self.continuation_count,
                "continuation_timestamps": self.continuation_timestamps,
                "continuation_metadata": self.continuation_metadata
            }
        
        return summary
    
    def _get_ordered_rounds_data(self):
        """Get an ordered version of rounds_data with keys sorted numerically.
        
        Returns:
            Ordered dictionary with rounds_data sorted by round number
        """
        # Extract round numbers from keys
        round_keys = list(self.rounds_data.keys())
        
        # Log information about round data for debugging
        print(f"Round data keys before ordering: {round_keys}")
        print(f"Current round_count: {self.round_count}")
        print(f"Total rounds to include: {len(round_keys)}")
        
        # Sort the keys numerically (extract the number from 'round_X')
        sorted_keys = sorted(round_keys, key=lambda k: int(k.split('_')[1]))
        
        # Create new ordered dictionary
        ordered_rounds_data = {}
        for key in sorted_keys:
            ordered_rounds_data[key] = self.rounds_data[key]
            
        return ordered_rounds_data
    
    def load_game_data(self, game_controller=None):
        """Load game data directly from the game controller instance or ensure all rounds are present.
        
        This method ensures that all rounds up to round_count are properly represented in the
        rounds_data dictionary, preserving all LLM interactions.
        
        Args:
            game_controller: GameController or GameLogic instance (optional)
        """
        # Calculate expected round count based on game activity
        calculated_round_count = self._calculate_expected_round_count() + 1  # +1 for current round
        
        # Update the round_count to match the actual game state
        self.round_count = calculated_round_count
        
        # If we already have the expected number of rounds, no need to do anything
        if self.rounds_data and len(self.rounds_data) >= self.round_count:
            return
            
        # No game controller provided or reconstructing from existing data
        if not game_controller:
            # Create placeholder entries for all rounds up to round_count
            for round_num in range(1, self.round_count + 1):  # Include the current round
                self._get_or_create_round_data(round_num)
            return
            
        # Clear existing rounds_data to ensure clean state only if we're reconstructing everything
        if len(self.rounds_data) == 0:
            self.rounds_data = {}
        
        # The rest of the implementation remains for backward compatibility
        # Create round data for each apple position we have
        for i, apple_pos in enumerate(self.apple_positions):
            round_num = i + 1  # Convert to 1-based index for round numbering
            round_key = f'round_{round_num}'
            
            # Skip if we already have data for this round
            if round_key in self.rounds_data:
                continue
            
            # Get the moves for this round
            moves = []
            if i < len(self.moves):
                moves = [self.moves[i]]
            
            # Create round data with the apple position and move
            round_data = {
                "apple_position": [apple_pos["x"], apple_pos["y"]],
                "moves": moves,
                "primary_response_times": [],
                "secondary_response_times": [],
                "primary_token_stats": [],
                "secondary_token_stats": []
            }
            
            # Add token stats and response times if available
            if i < len(self.primary_response_times):
                round_data["primary_response_times"] = [self.primary_response_times[i]]
            
            if i < len(self.primary_token_stats):
                round_data["primary_token_stats"] = [self.primary_token_stats[i]]
            
            if i < len(self.secondary_response_times):
                round_data["secondary_response_times"] = [self.secondary_response_times[i]]
            
            if i < len(self.secondary_token_stats):
                round_data["secondary_token_stats"] = [self.secondary_token_stats[i]]
            
            # Save the round data
            self.rounds_data[round_key] = round_data
        
        # Ensure we have entries for all rounds up to round_count
        for round_num in range(1, self.round_count + 1):  # Include the current round
            self._get_or_create_round_data(round_num)
    
    def save_game_summary(self, filepath, primary_provider, primary_model, parser_provider, parser_model, max_consecutive_errors_allowed=5):
        """Save the game summary to a JSON file.
        
        Ensures that all rounds data is properly included in the JSON, matching the number of
        rounds in the prompt/response files.
        
        Args:
            filepath: Path to save the JSON file
            primary_provider: The provider of the primary LLM
            primary_model: The model of the primary LLM
            parser_provider: The provider of the parser LLM
            parser_model: The model of the parser LLM
            max_consecutive_errors_allowed: Maximum consecutive errors allowed before game over
            
        Returns:
            Path to the saved file
        """
        # Get the game number from the filepath
        match = re.search(r'game_(\d+)\.json', os.path.basename(filepath))
        if match:
            game_number = int(match.group(1))
        
        # Calculate the expected round_count
        calculated_round_count = self._calculate_expected_round_count()
        
        # Update round_count if it's inconsistent with game activity
        if calculated_round_count > self.round_count:
            print(f"⚠️ Adjusting round_count from {self.round_count} to {calculated_round_count} based on game activity")
            self.round_count = calculated_round_count
        
        # Force regeneration of rounds_data to ensure all rounds are included
        # This is essential for matching the JSON rounds with prompt/response files
        self.load_game_data()
        
        # Double-check that we have entries for all rounds up to and including the current round
        for round_num in range(1, self.round_count + 1):
            self._get_or_create_round_data(round_num)
        
        # Generate and save the summary
        summary = self.generate_game_summary(primary_provider, primary_model, parser_provider, parser_model, max_consecutive_errors_allowed)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, cls=NumPyJSONEncoder)
            
        return filepath
    
    def record_planned_moves(self, moves):
        """Record the full array of moves planned by the LLM for the current round.
        
        Args:
            moves: List of moves returned by the LLM (["UP", "DOWN", "LEFT", "RIGHT", ...])
        """
        if moves and isinstance(moves, list):
            # Store the entire array of planned moves for the current round
            self.current_round_data["moves"] = moves.copy()
            
            # Also store directly in the rounds_data for the current round
            round_data = self._get_or_create_round_data(self.round_count)
            
            # Update moves in the round data
            round_data["moves"] = moves.copy()
    
    def sync_round_data(self):
        """Synchronize the current round data with rounds_data.
        
        This ensures that the current round's data is properly saved to rounds_data
        before continuing with game operations.
        """
        if not self.current_round_data:
            return
            
        # Get or create round data for current round
        round_data = self._get_or_create_round_data(self.round_count)
        
        # Copy all data from current_round_data to round_data
        for key, value in self.current_round_data.items():
            # Skip None values
            if value is None:
                continue
                
            # Handle array-like objects (lists, numpy arrays)
            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) > 0:  # Only copy non-empty arrays
                    round_data[key] = value
            # Handle dictionaries
            elif isinstance(value, dict):
                if value:  # Only copy non-empty dicts
                    round_data[key] = value
            # Handle other types (strings, numbers, etc.)
            else:
                round_data[key] = value
                
        # Log the sync operation
        print(f"🔄 Synchronized data for round {self.round_count}")
    
    def _get_or_create_round_data(self, round_num):
        """Get existing round data or create new round data for the specified round number.
        
        Args:
            round_num: The round number
            
        Returns:
            Dictionary containing the round data
        """
        round_key = f"round_{round_num}"
        
        # Create the round entry if it doesn't exist
        if round_key not in self.rounds_data:
            self.rounds_data[round_key] = self._create_empty_round_data()
            
            # If we have an apple position for this round, add it
            if (self.apple_positions is not None and 
                len(self.apple_positions) > 0 and 
                round_num <= len(self.apple_positions)):
                
                # Handle numpy array specifically
                if isinstance(self.apple_positions, np.ndarray):
                    if round_num <= len(self.apple_positions) and len(self.apple_positions[0]) >= 2:
                        x, y = self.apple_positions[0][0], self.apple_positions[0][1]
                        self.rounds_data[round_key]["apple_position"] = [x, y]
                # Handle list of dictionaries
                elif isinstance(self.apple_positions[round_num-1], dict):
                    apple_pos = self.apple_positions[round_num-1]
                    self.rounds_data[round_key]["apple_position"] = [apple_pos["x"], apple_pos["y"]]
                # Handle list of lists or tuples
                elif isinstance(self.apple_positions[round_num-1], (list, tuple)):
                    x, y = self.apple_positions[round_num-1]
                    self.rounds_data[round_key]["apple_position"] = [x, y]
                
        return self.rounds_data[round_key]
    
    def _calculate_expected_round_count(self):
        """Calculate the expected round count based on game activity.
        
        Returns:
            The expected round count
        """
        expected_rounds_from_moves = len(self.moves) if self.moves else 0
        expected_rounds_from_apples = len(self.apple_positions) if self.apple_positions is not None and len(self.apple_positions) > 0 else 0
        expected_rounds_from_responses = len(self.primary_response_times) if self.primary_response_times else 0
        
        # The round_count should be at least the maximum of these values
        return max(expected_rounds_from_moves, 
                  expected_rounds_from_apples,
                  expected_rounds_from_responses,
                  self.round_count) 