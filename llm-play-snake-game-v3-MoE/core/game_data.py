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
from colorama import Fore

class GameData:
    """Tracks and manages statistics for Snake game sessions."""
    
    def __init__(self):
        """Initialize the game data tracking."""
        # Initialize last_action_time as None to prevent timing issues
        self.last_action_time = None
        self.reset()
    
    def reset(self):
        """Reset all tracking data to initial state."""
        # Import config at reset time to avoid circular imports
        from config import MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED, MAX_CONSECUTIVE_ERRORS_ALLOWED
        
        # Game state
        self.game_number = 0
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.apple_positions = []
        self.score = 0
        self.steps = 0
        self.empty_steps = 0
        self.error_steps = 0
        self.last_move = None
        self.consecutive_empty_moves = 0
        self.max_consecutive_empty_moves_reached = 0
        self.game_over = False
        self.game_end_reason = None
        self.round_count = 1  # Start at 1 for first round
        self.round_data = []  # Track LLM outputs for each round
        
        # Rounds data structure
        self.rounds_data = {}
        self.current_round_data = {}
        
        # Game history data
        self.snake_positions = []
        self.apple_position = None
        
        # Time tracking - initialize with defensive defaults
        self.start_time = time.time()
        self.end_time = None
        self.llm_communication_time = 0   # Total time spent communicating with LLMs
        self.game_movement_time = 0      # Time spent in actual game movement
        self.waiting_time = 0            # Time spent waiting (pauses, etc.)
        self.other_time = 0              # Defensive initialization for other time
        
        # Basic game stats
        self.max_consecutive_empty_moves_allowed = MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED
        
        # Game history
        self.moves = []
        
        # Step statistics
        self.valid_steps = 0
        self.invalid_reversals = 0
        
        # Response times
        self.primary_response_times = []
        self.secondary_response_times = []
        
        # Token statistics - Simplified to just track totals
        self.primary_token_stats = []
        self.secondary_token_stats = []
        
        # Running totals and averages for token usage
        self.primary_total_tokens = 0
        self.primary_total_prompt_tokens = 0
        self.primary_total_completion_tokens = 0
        self.primary_avg_total_tokens = 0
        self.primary_avg_prompt_tokens = 0
        self.primary_avg_completion_tokens = 0
        
        self.secondary_total_tokens = 0
        self.secondary_total_prompt_tokens = 0
        self.secondary_total_completion_tokens = 0
        self.secondary_avg_total_tokens = 0
        self.secondary_avg_prompt_tokens = 0
        self.secondary_avg_completion_tokens = 0
        
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
            "secondary_token_stats": [],
            "invalid_reversals": []  # Ensure this starts as an empty list for each round
        }
    
    def record_move(self, move, apple_eaten=False):
        """Record a move and update relevant statistics.
        
        Args:
            move: The direction moved ("UP", "DOWN", "LEFT", "RIGHT")
            apple_eaten: Whether an apple was eaten on this move
        """
        # Standardize move to uppercase for consistency
        if isinstance(move, str):
            move = move.upper()
            
        # Always update critical game state values regardless of duplicate moves
        self.steps += 1
        self.valid_steps += 1
        self.consecutive_empty_moves = 0  # Reset on valid move
        
        # Always update score if an apple was eaten
        if apple_eaten:
            self.score += 1
            # Snake length is now calculated through the property getter
            
        
        # Update global game state
        self.last_move = move
        self.moves.append(move)
        
        # Update current round data
        if "moves" not in self.current_round_data:
            self.current_round_data["moves"] = []
            
        # Only add the move to current_round_data, not directly to rounds_data
        # This prevents duplication as sync_round_data will handle copying to rounds_data
        self.current_round_data["moves"].append(move)
        
        # Note: We removed the direct append to round_data["moves"] to prevent duplication
        
        # Note: Apple eaten handling moved to top of function
        # Note: Round count is ONLY incremented in one place:
        # 1. llm/communication_utils.py after getting a valid move from the LLM
        # It is NOT incremented when an apple is eaten during planned moves
    
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
        self.steps += 1  # Increment steps for invalid reversals
        
        # ------------------------------------------------------------------
        # Keep the invariant: len(self.moves) == self.steps
        # Treat the blocked reversal as a distinctive pseudo-move so that
        # replay and statistics stay consistent.  We use the sentinel string
        # "INVALID_REVERSAL" – consumers can ignore or colour-code it.
        # ------------------------------------------------------------------

        self.moves.append("INVALID_REVERSAL")

        # Also make sure the current round's executed-moves list stays aligned
        if "moves" not in self.current_round_data:
            self.current_round_data["moves"] = []
        self.current_round_data["moves"].append("INVALID_REVERSAL")
        
        # Add to the current round data if we're tracking a round
        if self.current_round_data:
            if "invalid_reversals" not in self.current_round_data:
                self.current_round_data["invalid_reversals"] = []
            
            # Create a new invalid reversal record
            invalid_reversal = {
                "attempted_move": attempted_move,
                "current_direction": current_direction,
                "step": self.steps
            }
            
            # Check if this exact invalid reversal is already recorded for this step
            # to prevent duplicates
            duplicate_exists = False
            for existing_reversal in self.current_round_data["invalid_reversals"]:
                if (existing_reversal["attempted_move"] == attempted_move and
                    existing_reversal["current_direction"] == current_direction and
                    existing_reversal["step"] == self.steps):
                    duplicate_exists = True
                    break
                
            # Only add if it's not a duplicate
            if not duplicate_exists:
                # Add to current round data
                self.current_round_data["invalid_reversals"].append(invalid_reversal)
                
                # Get or create the round data for the current round
                round_data = self._get_or_create_round_data(self.round_count)
                
                # Ensure invalid_reversals exists in round data
                if "invalid_reversals" not in round_data:
                    round_data["invalid_reversals"] = []
                    
                # Add only this specific invalid reversal to the round data
                # instead of copying the entire list to prevent duplicates
                round_data["invalid_reversals"].append(invalid_reversal)
    
    def record_error_move(self):
        """Record an error move (error in LLM response)."""
        self.error_steps += 1
        self.steps += 1
    
    def record_llm_communication_start(self):
        """Mark the start of communication with an LLM."""
        self.last_action_time = time.perf_counter()
        
    def record_llm_communication_end(self):
        """Record time spent communicating with an LLM."""
        if self.last_action_time is not None:
            current_time = time.perf_counter()
            self.llm_communication_time += (current_time - self.last_action_time)
            self.last_action_time = None
        
    def record_game_movement_start(self):
        """Mark the start of game movement."""
        self.last_action_time = time.perf_counter()
        
    def record_game_movement_end(self):
        """Record time spent on game movement."""
        if self.last_action_time is not None:
            current_time = time.perf_counter()
            self.game_movement_time += (current_time - self.last_action_time)
            self.last_action_time = None
        
    def record_waiting_start(self):
        """Mark the start of waiting time."""
        self.last_action_time = time.perf_counter()
        
    def record_waiting_end(self):
        """Record time spent waiting."""
        if self.last_action_time is not None:
            current_time = time.perf_counter()
            self.waiting_time += (current_time - self.last_action_time)
            self.last_action_time = None
    
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
        
        # IMPORTANT: We do NOT increment round_count here
        # Round count is ONLY incremented in llm/communication_utils.py after getting a valid move from the LLM
        # This ensures rounds in JSON files match the prompt/response file counts
        
        # Save current round data without incrementing round count
        # This ensures we capture the final state without creating an extra round at game end
        if self.current_round_data:
            # Process the current round data
            print(f"Processing {self.round_count} rounds with data")
            
            # Only save if there's meaningful data to save
            if (self.current_round_data.get("moves") or 
                self.current_round_data.get("apple_position") is not None or
                self.current_round_data.get("primary_response_times") or
                self.current_round_data.get("secondary_response_times")):
                
                # Get round key for the current round
                round_key = f"round_{self.round_count}"
                
                # Initialize round data if it doesn't exist
                if round_key not in self.rounds_data:
                    self.rounds_data[round_key] = {
                        "apple_position": None,
                        "moves": [],
                        "primary_response_times": [],
                        "secondary_response_times": [],
                        "primary_token_stats": [],
                        "secondary_token_stats": [],
                        "invalid_reversals": []
                    }
                
                # Save the round data by copying the current round data
                for key, value in self.current_round_data.items():
                    if value is not None:
                        # Make a deep copy to prevent future modifications from affecting saved data
                        if isinstance(value, list):
                            self.rounds_data[round_key][key] = value.copy()
                        elif isinstance(value, dict):
                            self.rounds_data[round_key][key] = value.copy()
                        else:
                            self.rounds_data[round_key][key] = value
                
                # Log that we saved the final round data
                print(f"💾 Saved data for round {self.round_count} with {len(self.current_round_data.get('moves', []))} moves")
    
    def record_primary_response_time(self, duration):
        """Record the time taken for a primary LLM response.
        
        Args:
            duration: The duration in seconds
        """
        # Add to the global list
        self.primary_response_times.append(duration)
        
        # Also store in current_round_data for the round
        self.current_round_data.setdefault("primary_response_times", []).append(duration)
        
        # Get primary response stats
        primary_times = self.primary_response_times
        
        # Update prompt_response_stats for reporting
        if primary_times:
            self.avg_primary_response_time = sum(primary_times) / len(primary_times)
            self.min_primary_response_time = min(primary_times)
            self.max_primary_response_time = max(primary_times)
    
    def record_secondary_response_time(self, duration):
        """Record the time taken for a secondary LLM response.
        
        Args:
            duration: The duration in seconds
        """
        # Add to the global list
        self.secondary_response_times.append(duration)
        
        # Also store in current_round_data for the round
        self.current_round_data.setdefault("secondary_response_times", []).append(duration)
        
        # Get secondary response stats
        secondary_times = self.secondary_response_times
        
        # Update prompt_response_stats for reporting
        if secondary_times:
            self.avg_secondary_response_time = sum(secondary_times) / len(secondary_times)
            self.min_secondary_response_time = min(secondary_times)
            self.max_secondary_response_time = max(secondary_times)
    
    def record_primary_token_stats(self, prompt_tokens, completion_tokens):
        """Record token usage for the primary LLM.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
        """
        # Calculate total tokens
        total_tokens = prompt_tokens + completion_tokens
        
        # Create token stats dict
        token_stats = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        
        # Store in current_round_data: keep only the latest entry for this round
        primary_list = self.current_round_data.setdefault("primary_token_stats", [])
        if primary_list:
            primary_list[0] = token_stats  # replace existing
        else:
            primary_list.append(token_stats)
        
        # Update running totals
        self.primary_total_tokens += total_tokens
        self.primary_total_prompt_tokens += prompt_tokens
        self.primary_total_completion_tokens += completion_tokens
        
        # Update averages
        request_count = len(self.primary_token_stats)
        if request_count > 0:
            self.primary_avg_total_tokens = self.primary_total_tokens / request_count
            self.primary_avg_prompt_tokens = self.primary_total_prompt_tokens / request_count
            self.primary_avg_completion_tokens = self.primary_total_completion_tokens / request_count
    
    def record_secondary_token_stats(self, prompt_tokens, completion_tokens):
        """Record token usage for the secondary LLM.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
        """
        # Calculate total tokens
        total_tokens = prompt_tokens + completion_tokens
        
        # Create token stats dict
        token_stats = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        
        # Store in current_round_data: keep only the latest entry for this round
        secondary_list = self.current_round_data.setdefault("secondary_token_stats", [])
        if secondary_list:
            secondary_list[0] = token_stats  # replace existing
        else:
            secondary_list.append(token_stats)
        
        # Update running totals
        self.secondary_total_tokens += total_tokens
        self.secondary_total_prompt_tokens += prompt_tokens
        self.secondary_total_completion_tokens += completion_tokens
        
        # Update averages
        request_count = len(self.secondary_token_stats)
        if request_count > 0:
            self.secondary_avg_total_tokens = self.secondary_total_tokens / request_count
            self.secondary_avg_prompt_tokens = self.secondary_total_prompt_tokens / request_count
            self.secondary_avg_completion_tokens = self.secondary_total_completion_tokens / request_count
    
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
        if 'max_consecutive_empty_moves_allowed' in summary_data:
            self.max_consecutive_empty_moves_allowed = summary_data['max_consecutive_empty_moves_allowed']
        
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
            self.rounds_data[round_key] = {
                "apple_position": None,
                "moves": [],
                "primary_response_times": [],
                "secondary_response_times": [],
                "primary_token_stats": [],
                "secondary_token_stats": [],
                "invalid_reversals": []
            }
        
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
        """Get the statistics for game steps.
        
        Returns:
            Dictionary with step statistics
        """
        return {
            "valid_steps": self.valid_steps,
            "empty_steps": self.empty_steps, 
            "error_steps": self.error_steps,
            "invalid_reversals": self.invalid_reversals,  # Include the count of invalid reversals
            "max_consecutive_empty_moves_reached": self.max_consecutive_empty_moves_reached
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
        self.other_time = total_duration - (self.llm_communication_time + self.game_movement_time + self.waiting_time)
        other_percent = (self.other_time / divisor * 100)
        
        return {
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(self.end_time).strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": total_duration,
            "llm_communication_time": self.llm_communication_time,
            "game_movement_time": self.game_movement_time,
            "waiting_time": self.waiting_time,
            "other_time": self.other_time,
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
        # Snake length is now calculated through the property getter
        
        # Create planned_moves_stats for debugging - recalculate safely
        ordered_rounds = self._get_ordered_rounds_data()
        planned_moves_stats = {
            rk: len(rd.get("planned_moves", []))
            for rk, rd in ordered_rounds.items()
        }
        
        # Create the base summary
        summary = {
            # Core game data
            "score": self.score,
            "steps": self.steps,
            "snake_length": self.snake_length,  # Using the property
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
            "planned_moves_stats": planned_moves_stats,  # Add planned_moves_stats
            
            # Metadata
            "metadata": {
                "game_number": self.game_number,
                "timestamp": self.timestamp,
                "last_move": self.last_move,
                "round_count": self.round_count,  # Keep it here for backward compatibility
                "max_consecutive_empty_moves_allowed": self.max_consecutive_empty_moves_allowed,
                "max_consecutive_errors_allowed": max_consecutive_errors_allowed,
                "parser_usage_count": self.parser_usage_count
            },
            
            # Raw token stats data (for detailed analysis)
            "primary_token_stats": self.primary_token_stats,
            "secondary_token_stats": self.secondary_token_stats,
            
            # Detailed game history (at bottom)
            # Use self.moves directly for the authoritative move list
            "detailed_history": {
                "apple_positions": self.apple_positions,
                "moves": self.moves.copy(),  # Flat list for replay, use copy to avoid mutation
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
        
        # Sanity check - validate that the global moves list length matches steps
        assert len(summary["detailed_history"]["moves"]) == summary["steps"], \
            f"Moves length ({len(summary['detailed_history']['moves'])}) doesn't match steps ({summary['steps']})"
        
        # Additional validation - each round's moves should be a subset of the global moves
        for rk, rd in summary["detailed_history"]["rounds_data"].items():
            assert len(rd.get("moves", [])) <= summary["steps"], \
                f"Round {rk} moves ({len(rd.get('moves', []))}) exceed total steps ({summary['steps']})"
        
        return summary
    
    def _collect_all_moves_from_rounds(self):
        """Collect all moves from all rounds to create a complete move history.
        
        This ensures that the moves list in detailed_history contains all moves
        from all rounds, not just individual moves recorded during execution.
        
        Returns:
            List of all moves across all rounds
        """
        # This method is now obsolete and will be removed
        # We now use self.moves directly in generate_game_summary
        # which is the single source of truth for executed moves
        # This list matches 'steps' exactly as both are incremented together in record_move()
        return self.moves.copy()
    
    def _get_ordered_rounds_data(self):
        """Get an ordered version of rounds_data with keys sorted numerically.
        
        Returns:
            Ordered dictionary with rounds_data sorted by round number
        """
        # Extract round numbers from keys
        round_keys = list(self.rounds_data.keys())
        
        # Log information about round data for debugging
        print(f"Processing {len(round_keys)} rounds with data")
        
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
        
        # Calculate the actual round_count based ONLY on LLM communications
        actual_round_count = self._calculate_actual_round_count()
        
        # Update round_count if it doesn't match LLM communication count
        if actual_round_count != self.round_count:
            print(f"🔄 Setting round_count to {actual_round_count} based strictly on LLM communication count")
            self.round_count = actual_round_count
        
        # CRITICAL: Only include rounds that correspond to actual LLM communications
        # This ensures that the number of rounds in the JSON exactly matches the number
        # of prompt/response file pairs
        valid_keys = set()
        for i in range(1, self.round_count + 1):
            valid_keys.add(f"round_{i}")
        
        # Create a clean copy of rounds_data with ONLY the valid rounds
        clean_rounds_data = {}
        for key in valid_keys:
            if key in self.rounds_data:
                clean_rounds_data[key] = self.rounds_data[key]
        
        # Replace the original rounds_data with our clean version
        self.rounds_data = clean_rounds_data
        
        # Get ordered rounds data for the JSON
        ordered_rounds_data = self._get_ordered_rounds_data()
        
        # Generate and save the summary
        summary = self.generate_game_summary(primary_provider, primary_model, parser_provider, parser_model, max_consecutive_errors_allowed)
        
        # Validate the summary before saving
        from utils.json_utils import validate_game_summary
        is_valid, error_message = validate_game_summary(summary)
        if not is_valid:
            print(f"⚠️ Warning: Game summary validation failed: {error_message}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, cls=NumPyJSONEncoder)
            
        # Only print this message if debug mode is enabled, to avoid duplicate messages with record_game_end
        if os.getenv("SNAKE_DEBUG"):
            print(f"💾 Saved data for round {self.round_count} with {len(self.current_round_data.get('moves', []))} moves")
            
        return filepath
    
    def record_planned_moves(self, moves):
        """Record the full array of moves planned by the LLM for the current round.
        
        Args:
            moves: List of moves returned by the LLM (["UP", "DOWN", "LEFT", "RIGHT", ...])
        """
        if moves and isinstance(moves, list):
            # Standardize all moves to uppercase for consistency
            standardized_moves = [move.upper() if isinstance(move, str) else move for move in moves]
            
            # Store the planned moves for the current round
            # but DON'T update the current_round_data["moves"] to avoid duplication
            # The moves will be added individually as they're executed by record_move
            
            # Add to current_round_data (used for the current round)
            self.current_round_data["planned_moves"] = standardized_moves.copy()
            
            # Store directly in the rounds_data for reference only
            round_data = self._get_or_create_round_data(self.round_count)
            
            # Store the planned moves in a separate field to avoid duplication
            round_data["planned_moves"] = standardized_moves.copy()
            
            # Important: We don't add these to self.moves yet - they'll be added
            # individually as they're executed by record_move
    
    def sync_round_data(self):
        """Synchronize the current round data with rounds_data.
        
        This ensures that the current round's data is properly saved to rounds_data
        before continuing with game operations.
        """
        if not self.current_round_data:
            return
            
        # Get or create round data for current round
        round_key = f"round_{self.round_count}"
        if round_key not in self.rounds_data:
            self.rounds_data[round_key] = {
                "apple_position": None,
                "moves": [],
                "planned_moves": [],
                "primary_response_times": [],
                "secondary_response_times": [],
                "primary_token_stats": [],
                "secondary_token_stats": [],
                "invalid_reversals": []  # Initialize with empty list
            }
            
        round_data = self.rounds_data[round_key]
        
        # Track if any changes were made
        changes_made = False
        
        # Copy all data from current_round_data to round_data
        for key, value in self.current_round_data.items():
            # Skip None values
            if value is None:
                continue
                
            # Handle array-like objects (lists, numpy arrays)
            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) > 0:  # Only copy non-empty arrays
                    # Create a deep copy to prevent future modifications from affecting this data
                    if isinstance(value, (list, tuple)):
                        if key == "invalid_reversals":
                            # Special handling for invalid_reversals to prevent duplicates
                            # Store as a completely new list instead of merging
                            existing_reversals = round_data.get(key, [])
                            
                            # Create a list of unique invalid reversals by comparing fields
                            unique_reversals = []
                            seen_keys = set()
                            
                            # Process existing reversals first
                            for reversal in existing_reversals:
                                # Create a key based on the reversal's properties
                                rev_key = f"{reversal.get('attempted_move')}:{reversal.get('current_direction')}:{reversal.get('step')}"
                                if rev_key not in seen_keys:
                                    seen_keys.add(rev_key)
                                    unique_reversals.append(reversal)
                            
                            # Then add new reversals from current_round_data if not already present
                            for reversal in value:
                                rev_key = f"{reversal.get('attempted_move')}:{reversal.get('current_direction')}:{reversal.get('step')}"
                                if rev_key not in seen_keys:
                                    seen_keys.add(rev_key)
                                    unique_reversals.append(reversal)
                                    changes_made = True
                            
                            # Set the deduplicated list back to round_data
                            round_data[key] = unique_reversals
                        elif key == "moves":
                            # Special handling for moves to prevent duplicates
                            # Use append-only delta to preserve proper order
                            existing = round_data.get(key, [])
                            if len(value) > len(existing):
                                # Copy ONLY the new slice
                                round_data[key] = existing + value[len(existing):]
                                changes_made = True
                        else:
                            # For other lists, check if there are actual changes
                            if key not in round_data or round_data[key] != value:
                                # For other lists, just copy the entire list
                                round_data[key] = value.copy()
                                changes_made = True
                    else:  # numpy array
                        # For numpy arrays, always assume changes (difficult to compare efficiently)
                        round_data[key] = value.copy()
                        changes_made = True
            # Handle dictionaries
            elif isinstance(value, dict):
                if value:  # Only copy non-empty dicts
                    if key not in round_data or round_data[key] != value:
                        round_data[key] = value.copy()
                        changes_made = True
            # Handle other types (strings, numbers, etc.)
            else:
                if key not in round_data or round_data[key] != value:
                    round_data[key] = value
                    changes_made = True
                
        # Log the sync operation only if changes were made
        if changes_made:
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
            self.rounds_data[round_key] = {
                "apple_position": None,
                "moves": [],
                "planned_moves": [],
                "primary_response_times": [],
                "secondary_response_times": [],
                "primary_token_stats": [],
                "secondary_token_stats": [],
                "invalid_reversals": []  # Initialize an empty list for each round's invalid reversals
            }
            
            # If we have an apple position for this round, add it
            if (self.apple_positions is not None and 
                len(self.apple_positions) > 0):
                
                # Use the latest apple position available
                apple_index = min(round_num - 1, len(self.apple_positions) - 1)
                
                # Handle numpy array specifically
                if isinstance(self.apple_positions, np.ndarray):
                    if apple_index < len(self.apple_positions) and len(self.apple_positions[0]) >= 2:
                        x, y = self.apple_positions[0][0], self.apple_positions[0][1]
                        self.rounds_data[round_key]["apple_position"] = [x, y]
                # Handle list of dictionaries
                elif isinstance(self.apple_positions[apple_index], dict):
                    apple_pos = self.apple_positions[apple_index]
                    self.rounds_data[round_key]["apple_position"] = [apple_pos["x"], apple_pos["y"]]
                # Handle list of lists or tuples
                elif isinstance(self.apple_positions[apple_index], (list, tuple)):
                    x, y = self.apple_positions[apple_index]
                    self.rounds_data[round_key]["apple_position"] = [x, y]
                
        # Get the current apple position for this round
        apple_pos = self.rounds_data[round_key].get("apple_position")
        
        # Ensure we always have a valid apple position
        apple_pos_is_empty = (apple_pos is None or 
                             (isinstance(apple_pos, (list, tuple, np.ndarray)) and len(apple_pos) == 0))
        
        if apple_pos_is_empty:
            # First try to use an apple from apple_positions
            if self.apple_positions and len(self.apple_positions) > 0:
                # Use the most recent apple position as a fallback
                latest_apple = self.apple_positions[-1]
                if isinstance(latest_apple, dict):
                    self.rounds_data[round_key]["apple_position"] = [latest_apple["x"], latest_apple["y"]]
                elif isinstance(latest_apple, (list, tuple)):
                    self.rounds_data[round_key]["apple_position"] = list(latest_apple)
                elif isinstance(latest_apple, np.ndarray):
                    self.rounds_data[round_key]["apple_position"] = latest_apple.tolist()
            else:
                # If all else fails, use a default position
                self.rounds_data[round_key]["apple_position"] = [5, 5]  # Default position
                
        return self.rounds_data[round_key]
    
    def _calculate_expected_round_count(self):
        """Calculate the expected round count based on LLM interactions only.
        
        Returns:
            The expected round count based on LLM interactions
        """
        # Base expected rounds only on LLM interactions and apple positions, not on moves
        # This prevents huge round counts when there are many duplicate moves
        expected_rounds_from_apples = len(self.apple_positions) if self.apple_positions is not None and len(self.apple_positions) > 0 else 0
        expected_rounds_from_responses = len(self.primary_response_times) if self.primary_response_times else 0
        
        # The round_count should be at least the maximum of these values
        return max(expected_rounds_from_responses,
                  expected_rounds_from_apples,
                  self.round_count) 
    
    def _calculate_actual_round_count(self):
        """Calculate the actual round count based on LLM communications only.
        
        This method ensures round count is strictly based on the number of 
        LLM interactions (send/receive), corresponding to prompt/response files.
        
        Returns:
            The count of LLM interactions
        """
        # ONLY use LLM response times as the source of truth
        # This directly corresponds to the number of prompt/response files
        return len(self.primary_response_times) if self.primary_response_times else 0 

    @property
    def snake_length(self):
        """Calculate the snake length based on score.
        
        Returns:
            The current length of the snake (score + initial length of 1)
        """
        return self.score + 1 

    def to_json(self, primary_provider=None, primary_model=None, parser_provider=None, parser_model=None, max_consecutive_errors_allowed=5):
        """Wrapper method for generate_game_summary to ensure compatibility with process_game_over.
        
        Args:
            primary_provider: The provider of the primary LLM
            primary_model: The model of the primary LLM
            parser_provider: The provider of the parser LLM
            parser_model: The model of the parser LLM
            max_consecutive_errors_allowed: Maximum consecutive errors allowed before game over
            
        Returns:
            Dictionary with game summary
        """
        return self.generate_game_summary(primary_provider, primary_model, parser_provider, parser_model, max_consecutive_errors_allowed) 

    def _flush_current_round(self):
        """Persist the buffered data of the active round and reset the buffer.

        Must be invoked BEFORE round_count is incremented, otherwise the data
        would end up under the wrong round key.
        """
        # First, make sure any pending deltas are pushed to rounds_data
        if self.current_round_data:
            self.sync_round_data()

        # Start a pristine buffer for the next round
        self.current_round_data = {
            "apple_position": None,
            "moves": [],
            "primary_response_times": [],
            "secondary_response_times": [],
            "primary_token_stats": [],
            "secondary_token_stats": [],
            "invalid_reversals": []
        } 