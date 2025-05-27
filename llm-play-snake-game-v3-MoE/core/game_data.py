"""
Game data tracking and statistics management for the Snake game.
Provides centralized collection and reporting of game statistics.
"""

import json
import time
from datetime import datetime
import numpy as np
from utils.json_utils import NumPyJSONEncoder

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
        self.is_continuation = False
        self.continuation_count = 0
        self.continuation_timestamps = []
        self.continuation_metadata = []
        
        # Game stats
        self.score = 0
        self.steps = 0
        self.empty_steps = 0
        self.error_steps = 0
        self.invalid_reversals = 0  # New counter for invalid reversals
        self.consecutive_empty_moves = 0
        self.max_consecutive_empty_moves_reached = 0
        
        # Other tracking data
        self.start_time = time.time()
        self.end_time = None
        self.rounds_data = {}
        self.current_round = 0
        self.current_round_data = self._create_empty_round_data()
        self.apple_positions = []
        self.round_count = 0
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
        self.max_empty_moves = 3
        
        # Game history
        self.moves = []
        
        # Step statistics
        self.valid_steps = 0
        
        # Response times
        self.primary_response_times = []
        self.secondary_response_times = []
        
        # Token statistics
        self.primary_token_stats = {
            "prompt_tokens": [],
            "completion_tokens": [],
            "total_tokens": []
        }
        self.secondary_token_stats = {
            "prompt_tokens": [],
            "completion_tokens": [],
            "total_tokens": []
        }
        
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
        self.fallback_extraction_success = 0
        
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
        # Save previous round data if we have moves
        if self.current_round_data["moves"]:
            self.rounds_data[f"round_{self.current_round}"] = self.current_round_data.copy()
            self.current_round += 1
        
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
        
        if apple_eaten:
            self.score += 1
            self.snake_length += 1
            
            # Save the current round data
            self.rounds_data[f"round_{self.current_round}"] = self.current_round_data.copy()
            self.current_round += 1
            
            # Reset for next round
            self.current_round_data = {
                "apple_position": None,  # Will be set when new apple is generated
                "moves": [],
                "primary_response_times": [],
                "secondary_response_times": [],
                "primary_token_stats": [],
                "secondary_token_stats": []
            }
    
    def record_apple_position(self, position):
        """Record an apple position.
        
        Args:
            position: The position of the apple as [x, y]
        """
        x, y = position
        self.apple_positions.append({"x": x, "y": y})
        self.current_round_data["apple_position"] = [x, y]
    
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
        self.consecutive_empty_moves = 0  # Reset on error (as per game rules)
    
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
            reason: The reason the game ended ("WALL", "SELF", "MAX_STEPS", "EMPTY_MOVES")
        """
        self.game_end_reason = reason
        self.game_number += 1
        self.end_time = time.time()
        
        # Save any remaining round data
        if self.current_round_data["moves"]:
            self.rounds_data[f"round_{self.current_round}"] = self.current_round_data.copy()
    
    def record_primary_response_time(self, duration):
        """Record a primary LLM response time.
        
        Args:
            duration: Response time in seconds
        """
        self.primary_response_times.append(duration)
        self.current_round_data["primary_response_times"].append(duration)
        self.primary_llm_requests += 1
    
    def record_secondary_response_time(self, duration):
        """Record a secondary LLM response time.
        
        Args:
            duration: Response time in seconds
        """
        self.secondary_response_times.append(duration)
        self.current_round_data["secondary_response_times"].append(duration)
        self.secondary_llm_requests += 1
        self.parser_usage_count += 1
    
    def record_primary_token_stats(self, prompt_tokens, completion_tokens):
        """Record token usage statistics for the primary LLM.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
        """
        self.primary_token_stats["prompt_tokens"].append(prompt_tokens)
        self.primary_token_stats["completion_tokens"].append(completion_tokens)
        self.primary_token_stats["total_tokens"].append(prompt_tokens + completion_tokens)
        
        # Add to current round data
        self.current_round_data["primary_token_stats"].append({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        })
    
    def record_secondary_token_stats(self, prompt_tokens, completion_tokens):
        """Record token usage statistics for the secondary LLM.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
        """
        self.secondary_token_stats["prompt_tokens"].append(prompt_tokens)
        self.secondary_token_stats["completion_tokens"].append(completion_tokens)
        self.secondary_token_stats["total_tokens"].append(prompt_tokens + completion_tokens)
        
        # Add to current round data
        self.current_round_data["secondary_token_stats"].append({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        })
    
    def record_primary_llm_error(self):
        """Record an error from the primary LLM."""
        self.primary_llm_errors += 1
    
    def record_secondary_llm_error(self):
        """Record an error from the secondary LLM."""
        self.secondary_llm_errors += 1
    
    def record_json_extraction_attempt(self, success, error_type=None):
        """Record a JSON extraction attempt.
        
        Args:
            success: Whether the extraction was successful
            error_type: Type of error if extraction failed ("decode", "validation", "code_block", "text")
        """
        self.total_extraction_attempts += 1
        
        if success:
            self.successful_extractions += 1
        else:
            self.failed_extractions += 1
            
            if error_type == "decode":
                self.json_decode_errors += 1
            elif error_type == "validation":
                self.format_validation_errors += 1
            elif error_type == "code_block":
                self.code_block_extraction_errors += 1
            elif error_type == "text":
                self.text_extraction_errors += 1
    
    def record_fallback_extraction_success(self):
        """Record a successful fallback extraction."""
        self.fallback_extraction_success += 1
    
    def record_round_data(self, round_number, apple_position, moves, 
                         primary_response_time=None, secondary_response_time=None,
                         primary_tokens=None, secondary_tokens=None):
        """Record data for a specific round.
        
        Args:
            round_number: The round number
            apple_position: Position of the apple as [x, y]
            moves: List of moves made in this round
            primary_response_time: Response time of the primary LLM
            secondary_response_time: Response time of the secondary LLM
            primary_tokens: Token stats for primary LLM as dict with prompt_tokens, completion_tokens, total_tokens
            secondary_tokens: Token stats for secondary LLM as dict with prompt_tokens, completion_tokens, total_tokens
        """
        round_data = {
            "apple_position": apple_position,
            "moves": moves
        }
        
        if primary_response_time is not None:
            round_data["primary_response_times"] = [primary_response_time]
            
        if secondary_response_time is not None:
            round_data["secondary_response_times"] = [secondary_response_time]
            
        if primary_tokens is not None:
            round_data["primary_token_stats"] = [primary_tokens]
            
        if secondary_tokens is not None:
            round_data["secondary_token_stats"] = [secondary_tokens]
            
        self.rounds_data[f"round_{round_number}"] = round_data
        self.round_count = max(self.round_count, round_number)
    
    def get_efficiency_metrics(self):
        """Calculate efficiency metrics.
        
        Returns:
            Dictionary of efficiency metrics
        """
        apples_per_step = self.score / max(1, self.steps)
        valid_move_ratio = self.valid_steps / max(1, self.steps) * 100
        
        return {
            "apples_per_step": apples_per_step,
            "steps_per_game": self.steps,
            "valid_move_ratio": valid_move_ratio
        }
    
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
        # Primary LLM token stats
        primary_prompt = self.primary_token_stats["prompt_tokens"]
        primary_completion = self.primary_token_stats["completion_tokens"]
        primary_total = self.primary_token_stats["total_tokens"]
        
        # Secondary LLM token stats
        secondary_prompt = self.secondary_token_stats["prompt_tokens"]
        secondary_completion = self.secondary_token_stats["completion_tokens"]
        secondary_total = self.secondary_token_stats["total_tokens"]
        
        return {
            "primary": {
                "total_tokens": sum(primary_total) if primary_total else 0,
                "total_prompt_tokens": sum(primary_prompt) if primary_prompt else 0,
                "total_completion_tokens": sum(primary_completion) if primary_completion else 0,
                "avg_total_tokens": np.mean(primary_total) if primary_total else 0,
                "avg_prompt_tokens": np.mean(primary_prompt) if primary_prompt else 0,
                "avg_completion_tokens": np.mean(primary_completion) if primary_completion else 0
            },
            "secondary": {
                "total_tokens": sum(secondary_total) if secondary_total else 0,
                "total_prompt_tokens": sum(secondary_prompt) if secondary_prompt else 0,
                "total_completion_tokens": sum(secondary_completion) if secondary_completion else 0,
                "avg_total_tokens": np.mean(secondary_total) if secondary_total else 0,
                "avg_prompt_tokens": np.mean(secondary_prompt) if secondary_prompt else 0,
                "avg_completion_tokens": np.mean(secondary_completion) if secondary_completion else 0
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
            "fallback_extraction_success": self.fallback_extraction_success
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
        
        # Calculate pause time from continuations if applicable
        pause_time = 0
        if hasattr(self, 'continuation_metadata') and self.continuation_metadata:
            for meta in self.continuation_metadata:
                if 'time_since_last_action' in meta:
                    pause_time += meta['time_since_last_action']
        
        # Calculate active duration (excluding pauses from continuations)
        active_duration = total_duration - pause_time
        
        # Calculate percentages based on active time
        divisor = max(0.001, active_duration)  # Avoid division by zero
        llm_percent = (self.llm_communication_time / divisor * 100)
        movement_percent = (self.game_movement_time / divisor * 100)
        waiting_percent = (self.waiting_time / divisor * 100)
        
        # Calculate other time (time not accounted for in the other categories)
        other_time = active_duration - (self.llm_communication_time + self.game_movement_time + self.waiting_time)
        other_percent = (other_time / divisor * 100)
        
        return {
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(self.end_time).strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": total_duration,
            "active_duration_seconds": active_duration,
            "pause_duration_seconds": pause_time,
            "is_continuation": self.is_continuation,
            "continuation_count": self.continuation_count,
            "llm_communication_time": self.llm_communication_time,
            "game_movement_time": self.game_movement_time,
            "waiting_time": self.waiting_time,
            "other_time": other_time,
            "llm_communication_percent": llm_percent,
            "game_movement_percent": movement_percent,
            "waiting_percent": waiting_percent,
            "other_percent": other_percent
        }
    
    def generate_game_summary(self, primary_provider, primary_model, parser_provider, parser_model):
        """Generate a summary of the game.
        
        Args:
            primary_provider: The provider of the primary LLM
            primary_model: The model of the primary LLM
            parser_provider: The provider of the parser LLM
            parser_model: The model of the parser LLM
            
        Returns:
            Dictionary with game summary
        """
        # Create the base summary
        summary = {
            # Core game data
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_segments) if hasattr(self, 'snake_segments') else 1,
            "win": self.win,
            "game_over": self.game_over,
            "game_over_reason": self.game_over_reason,
            
            # Time statistics
            "time_stats": self.get_time_stats(),
            
            # Provider and model info
            "llm_info": {
                "primary_provider": primary_provider,
                "primary_model": primary_model,
                "parser_provider": parser_provider if parser_provider.lower() != "none" else None,
                "parser_model": parser_model if parser_provider.lower() != "none" else None
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
                "round_count": self.round_count,
                "max_empty_moves": self.max_empty_moves,
                "parser_usage_count": self.parser_usage_count
            },
            
            # Detailed game history (at bottom)
            "detailed_history": {
                "apple_positions": self.apple_positions,
                "moves": self.moves,
                "rounds_data": self.rounds_data
            }
        }
        
        # Update with continuation info (this will place it at the bottom of the metadata section)
        if self.is_continuation:
            summary = self.update_continuation_info_in_summary(summary)
            
        return summary
    
    def get_aggregated_stats_for_summary_json(self, game_count, game_scores, game_durations=None):
        """Generate aggregated statistics for summary.json file.
        
        Args:
            game_count: Total number of games played
            game_scores: List of scores for all games
            game_durations: List of game durations in seconds
            
        Returns:
            Dictionary with aggregated statistics
        """
        # Calculate aggregate statistics
        total_score = sum(game_scores) if game_scores else 0
        mean_score = total_score / max(1, game_count)
        max_score = max(game_scores) if game_scores else 0
        min_score = min(game_scores) if game_scores else 0
        
        # Time statistics
        session_start_time = datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S")
        session_end_time = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
        session_duration = time.time() - self.start_time
        
        # Game duration statistics
        time_stats = {}
        if game_durations and len(game_durations) > 0:
            time_stats = {
                "total_session_duration_seconds": session_duration,
                "session_start_time": session_start_time,
                "session_end_time": session_end_time,
                "avg_game_duration_seconds": np.mean(game_durations),
                "min_game_duration_seconds": np.min(game_durations),
                "max_game_duration_seconds": np.max(game_durations),
                "total_llm_communication_time": self.llm_communication_time,
                "total_game_movement_time": self.game_movement_time,
                "total_waiting_time": self.waiting_time,
                "llm_communication_percent": (self.llm_communication_time / session_duration * 100) if session_duration > 0 else 0,
                "game_movement_percent": (self.game_movement_time / session_duration * 100) if session_duration > 0 else 0,
                "waiting_percent": (self.waiting_time / session_duration * 100) if session_duration > 0 else 0
            }
        else:
            time_stats = {
                "total_session_duration_seconds": session_duration,
                "session_start_time": session_start_time,
                "session_end_time": session_end_time
            }
        
        # Create the base summary data
        summary = {
            "game_statistics": {
                "total_games": game_count,
                "total_score": total_score,
                "total_steps": self.steps,
                "mean_score": mean_score,
                "max_score": max_score,
                "min_score": min_score,
                "steps_per_game": self.steps / max(1, game_count),
                "steps_per_apple": self.steps / max(1, total_score) if total_score > 0 else self.steps,
                "apples_per_step": total_score / max(1, self.steps),
                "valid_move_ratio": self.valid_steps / max(1, self.steps) * 100,
                "invalid_reversals": self.invalid_reversals,
                "invalid_reversal_ratio": self.invalid_reversals / max(1, self.steps) * 100
            },
            "time_statistics": time_stats,
            "response_time_stats": {
                "primary_llm": {
                    "avg_response_time": np.mean(self.primary_response_times) if self.primary_response_times else 0,
                    "min_response_time": np.min(self.primary_response_times) if self.primary_response_times else 0,
                    "max_response_time": np.max(self.primary_response_times) if self.primary_response_times else 0,
                    "total_response_time": sum(self.primary_response_times) if self.primary_response_times else 0,
                    "response_count": len(self.primary_response_times)
                },
                "secondary_llm": {
                    "avg_response_time": np.mean(self.secondary_response_times) if self.secondary_response_times else 0,
                    "min_response_time": np.min(self.secondary_response_times) if self.secondary_response_times else 0,
                    "max_response_time": np.max(self.secondary_response_times) if self.secondary_response_times else 0,
                    "total_response_time": sum(self.secondary_response_times) if self.secondary_response_times else 0,
                    "response_count": len(self.secondary_response_times)
                }
            },
            "token_usage_stats": {
                "primary_llm": {
                    "total_tokens": self.get_token_stats()["primary"]["total_tokens"],
                    "total_prompt_tokens": self.get_token_stats()["primary"]["total_prompt_tokens"],
                    "total_completion_tokens": self.get_token_stats()["primary"]["total_completion_tokens"],
                    "avg_tokens_per_request": self.get_token_stats()["primary"]["avg_total_tokens"],
                    "avg_prompt_tokens": self.get_token_stats()["primary"]["avg_prompt_tokens"],
                    "avg_completion_tokens": self.get_token_stats()["primary"]["avg_completion_tokens"]
                },
                "secondary_llm": {
                    "total_tokens": self.get_token_stats()["secondary"]["total_tokens"],
                    "total_prompt_tokens": self.get_token_stats()["secondary"]["total_prompt_tokens"],
                    "total_completion_tokens": self.get_token_stats()["secondary"]["total_completion_tokens"],
                    "avg_tokens_per_request": self.get_token_stats()["secondary"]["avg_total_tokens"],
                    "avg_prompt_tokens": self.get_token_stats()["secondary"]["avg_prompt_tokens"],
                    "avg_completion_tokens": self.get_token_stats()["secondary"]["avg_completion_tokens"]
                }
            },
            "step_stats": self.get_step_stats(),
            "json_parsing_stats": {
                "success_rate": self.successful_extractions / max(1, self.total_extraction_attempts) * 100,
                "total_extraction_attempts": self.total_extraction_attempts,
                "successful_extractions": self.successful_extractions,
                "failed_extractions": self.failed_extractions,
                "failure_rate": self.failed_extractions / max(1, self.total_extraction_attempts) * 100
            },
            "game_scores": game_scores
        }
        
        # Add continuation info at the bottom
        if self.is_continuation:
            # Create a dedicated continuation section
            continuation_info = {
                "is_continued": True,
                "continuation_count": self.continuation_count
            }
            
            # Add timestamps if available
            if hasattr(self, 'continuation_timestamps') and self.continuation_timestamps:
                continuation_info['continuation_timestamps'] = self.continuation_timestamps
                
            # Add metadata if available
            if hasattr(self, 'continuation_metadata') and self.continuation_metadata:
                continuation_info['continuation_metadata'] = self.continuation_metadata
                
            # Add to summary at the end
            summary['continuation_info'] = continuation_info
            
        return summary
    
    def save_game_summary(self, filepath, primary_provider, primary_model, parser_provider, parser_model):
        """Save the game summary to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
            primary_provider: The provider of the primary LLM
            primary_model: The model of the primary LLM
            parser_provider: The provider of the parser LLM
            parser_model: The model of the parser LLM
            
        Returns:
            Path to the saved file
        """
        summary = self.generate_game_summary(primary_provider, primary_model, parser_provider, parser_model)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumPyJSONEncoder)
            
        return filepath

    def record_continuation(self, previous_session_data=None):
        """Record that this game is a continuation of a previous session.
        
        Args:
            previous_session_data: Optional dictionary with data from previous session
        """
        self.is_continuation = True
        self.continuation_count += 1
        continuation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Initialize continuation timestamps if needed
        if not hasattr(self, 'continuation_timestamps'):
            self.continuation_timestamps = []
            
        # Initialize continuation metadata if needed
        if not hasattr(self, 'continuation_metadata'):
            self.continuation_metadata = []
            
        # Add current timestamp
        self.continuation_timestamps.append(continuation_timestamp)
        
        # Reset the start time for accurate time tracking during this continuation session
        # This ensures we don't include the time between sessions in our calculations
        current_time = time.time()
        
        # Store the time delta between the last action and the continuation
        # This helps in accurately tracking time spent in the game vs. paused time
        if hasattr(self, 'last_action_time') and self.last_action_time:
            time_delta = current_time - self.last_action_time
            
            # Record this continuation event with metadata for analytics
            continuation_meta = {
                "timestamp": continuation_timestamp,
                "time_since_last_action": time_delta,
                "previous_duration": current_time - self.start_time if hasattr(self, 'start_time') else 0,
                "game_state": {
                    "score": self.score,
                    "steps": self.steps,
                    "snake_length": len(self.snake_segments) if hasattr(self, 'snake_segments') else 1
                }
            }
            
            # Add additional data from previous session if provided
            if previous_session_data:
                # Add previous game statistics if available
                if 'game_statistics' in previous_session_data:
                    game_stats = previous_session_data['game_statistics']
                    continuation_meta['previous_session'] = {
                        'total_games': game_stats.get('total_games', 0),
                        'total_score': game_stats.get('total_score', 0),
                        'total_steps': game_stats.get('total_steps', 0)
                    }
                
                # Add previous continuation info if available
                if 'continuation_info' in previous_session_data:
                    cont_info = previous_session_data['continuation_info']
                    
                    # Import timestamps from previous session if they're not already in our list
                    if 'continuation_timestamps' in cont_info:
                        for timestamp in cont_info['continuation_timestamps']:
                            if timestamp not in self.continuation_timestamps:
                                self.continuation_timestamps.append(timestamp)
            
            # Add the metadata
            self.continuation_metadata.append(continuation_meta)
            
        # Update the start time to the current time
        # This essentially "pauses" the timer when continuing
        self.last_action_time = current_time
        
        # Update continuation count to match actual number of continuations
        self.continuation_count = len(self.continuation_timestamps)
        
    def synchronize_with_summary_json(self, summary_data):
        """Synchronize game state with data from summary.json.
        
        This helps ensure continuation data is accurate.
        
        Args:
            summary_data: Dictionary loaded from summary.json
        """
        # Check if continuation info exists in summary
        if 'continuation_info' in summary_data:
            cont_info = summary_data['continuation_info']
            
            # Initialize continuation attributes if needed
            if not hasattr(self, 'continuation_timestamps'):
                self.continuation_timestamps = []
                
            if not hasattr(self, 'continuation_metadata'):
                self.continuation_metadata = []
                
            # Import timestamps from summary
            if 'continuation_timestamps' in cont_info:
                for timestamp in cont_info['continuation_timestamps']:
                    if timestamp not in self.continuation_timestamps:
                        self.continuation_timestamps.append(timestamp)
            
            # Import session metadata from summary
            if 'session_metadata' in cont_info:
                for metadata in cont_info['session_metadata']:
                    # Check if this metadata is already in our list
                    timestamp = metadata.get('timestamp')
                    if timestamp:
                        # Check if we already have this timestamp in our metadata
                        existing = [m for m in self.continuation_metadata if m.get('timestamp') == timestamp]
                        if not existing:
                            self.continuation_metadata.append(metadata)
            
            # Update continuation count
            self.continuation_count = max(
                len(self.continuation_timestamps),
                cont_info.get('continuation_count', 0)
            )
            
            # Mark as continuation if it has happened before
            if self.continuation_count > 0:
                self.is_continuation = True 

    def update_continuation_info_in_summary(self, summary_dict):
        """Update the continuation information in a game summary dictionary.
        
        This ensures continuation information is properly organized and placed at the bottom
        of the JSON file.
        
        Args:
            summary_dict: Dictionary representing a game summary to update
            
        Returns:
            Updated summary dictionary with organized continuation info
        """
        # Initialize continuation section if needed
        if 'metadata' not in summary_dict:
            summary_dict['metadata'] = {}
            
        # Add continuation info to metadata section
        continuation_data = {
            'is_continuation': self.is_continuation,
            'continuation_count': self.continuation_count
        }
        
        # Add timestamps if available
        if hasattr(self, 'continuation_timestamps') and self.continuation_timestamps:
            continuation_data['continuation_timestamps'] = self.continuation_timestamps
            
        # Add metadata if available
        if hasattr(self, 'continuation_metadata') and self.continuation_metadata:
            continuation_data['continuation_metadata'] = self.continuation_metadata
            
        # Update the metadata
        summary_dict['metadata']['continuation_info'] = continuation_data
        
        return summary_dict 