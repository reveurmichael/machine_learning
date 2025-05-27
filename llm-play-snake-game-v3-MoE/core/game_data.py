"""
Game data tracking and statistics management for the Snake game.
Provides centralized collection and reporting of game statistics.
"""

import json
import time
from datetime import datetime
import numpy as np

class NumPyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class GameData:
    """Tracks and manages statistics for Snake game sessions."""
    
    def __init__(self):
        """Initialize the game data tracking."""
        # Game metadata
        self.game_number = 0
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Basic game stats
        self.score = 0
        self.steps = 0
        self.game_end_reason = None  # "WALL", "SELF", "MAX_STEPS", "EMPTY_MOVES"
        self.snake_length = 1
        self.round_count = 0
        self.max_empty_moves = 3
        self.last_move = None
        
        # Game history
        self.apple_positions = []
        self.moves = []
        self.rounds_data = {}
        
        # Step statistics
        self.empty_steps = 0
        self.error_steps = 0
        self.valid_steps = 0
        self.consecutive_empty_moves = 0
        self.max_consecutive_empty_moves_reached = 0
        
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
        
        # Current round data tracking
        self.current_round_data = {
            "apple_position": None,
            "moves": [],
            "primary_response_times": [],
            "secondary_response_times": [],
            "primary_token_stats": [],
            "secondary_token_stats": []
        }
        self.current_round = 1
    
    def reset(self):
        """Reset the game data for a new game."""
        self.score = 0
        self.steps = 0
        self.game_end_reason = None
        self.snake_length = 1
        self.round_count = 0
        self.last_move = None
        
        # Reset game history for this game
        self.apple_positions = []
        self.moves = []
        self.rounds_data = {}
        
        # Reset step statistics for this game
        self.empty_steps = 0
        self.error_steps = 0
        self.valid_steps = 0
        self.consecutive_empty_moves = 0
        self.max_consecutive_empty_moves_reached = 0
        
        # Reset current round data
        self.current_round_data = {
            "apple_position": None,
            "moves": [],
            "primary_response_times": [],
            "secondary_response_times": [],
            "primary_token_stats": [],
            "secondary_token_stats": []
        }
        self.current_round = 1
        
        # Keep response times and token stats as they are cumulative
    
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
    
    def record_error_move(self):
        """Record an error move (error in LLM response)."""
        self.error_steps += 1
        self.steps += 1
        self.consecutive_empty_moves = 0  # Reset on error (as per game rules)
    
    def record_game_end(self, reason):
        """Record the end of a game.
        
        Args:
            reason: The reason the game ended ("WALL", "SELF", "MAX_STEPS", "EMPTY_MOVES")
        """
        self.game_end_reason = reason
        self.game_number += 1
        
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
        empty_step_percentage = self.empty_steps / max(1, self.steps) * 100
        error_step_percentage = self.error_steps / max(1, self.steps) * 100
        valid_step_percentage = self.valid_steps / max(1, self.steps) * 100
        
        return {
            "empty_steps": self.empty_steps,
            "empty_step_percentage": empty_step_percentage,
            "error_steps": self.error_steps,
            "error_step_percentage": error_step_percentage,
            "valid_steps": self.valid_steps,
            "valid_step_percentage": valid_step_percentage,
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
    
    def generate_game_summary(self, primary_provider, primary_model, parser_provider, parser_model):
        """Generate a complete game summary.
        
        Args:
            primary_provider: The provider of the primary LLM
            primary_model: The model of the primary LLM
            parser_provider: The provider of the parser LLM
            parser_model: The model of the parser LLM
            
        Returns:
            Dictionary with complete game summary
        """
        return {
            # Core performance metrics (most important/abstract at top)
            "score": self.score,
            "steps": self.steps,
            "snake_length": self.snake_length,
            "game_end_reason": self.game_end_reason,
            "efficiency_metrics": self.get_efficiency_metrics(),
            
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
    
    def get_aggregated_stats_for_summary_json(self, game_count, game_scores):
        """Generate aggregated statistics for summary.json file.
        
        Args:
            game_count: Total number of games played
            game_scores: List of scores for all games
            
        Returns:
            Dictionary with aggregated statistics
        """
        # Calculate aggregate statistics
        total_score = sum(game_scores) if game_scores else 0
        mean_score = total_score / max(1, game_count)
        max_score = max(game_scores) if game_scores else 0
        min_score = min(game_scores) if game_scores else 0
        
        # Primary LLM response times
        primary_times = self.primary_response_times
        avg_primary = np.mean(primary_times) if primary_times else 0
        min_primary = np.min(primary_times) if primary_times else 0
        max_primary = np.max(primary_times) if primary_times else 0
        total_primary_time = sum(primary_times) if primary_times else 0
        
        # Secondary LLM response times
        secondary_times = self.secondary_response_times
        avg_secondary = np.mean(secondary_times) if secondary_times else 0
        min_secondary = np.min(secondary_times) if secondary_times else 0
        max_secondary = np.max(secondary_times) if secondary_times else 0
        total_secondary_time = sum(secondary_times) if secondary_times else 0
        
        # Token statistics
        primary_token_stats = self.get_token_stats()["primary"]
        secondary_token_stats = self.get_token_stats()["secondary"]
        
        return {
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
                "valid_move_ratio": self.valid_steps / max(1, self.steps) * 100
            },
            "response_time_stats": {
                "primary_llm": {
                    "avg_response_time": avg_primary,
                    "min_response_time": min_primary,
                    "max_response_time": max_primary,
                    "total_response_time": total_primary_time,
                    "response_count": len(primary_times)
                },
                "secondary_llm": {
                    "avg_response_time": avg_secondary,
                    "min_response_time": min_secondary,
                    "max_response_time": max_secondary,
                    "total_response_time": total_secondary_time,
                    "response_count": len(secondary_times)
                }
            },
            "token_usage_stats": {
                "primary_llm": {
                    "total_tokens": primary_token_stats["total_tokens"],
                    "total_prompt_tokens": primary_token_stats["total_prompt_tokens"],
                    "total_completion_tokens": primary_token_stats["total_completion_tokens"],
                    "avg_tokens_per_request": primary_token_stats["avg_total_tokens"],
                    "avg_prompt_tokens": primary_token_stats["avg_prompt_tokens"],
                    "avg_completion_tokens": primary_token_stats["avg_completion_tokens"]
                },
                "secondary_llm": {
                    "total_tokens": secondary_token_stats["total_tokens"],
                    "total_prompt_tokens": secondary_token_stats["total_prompt_tokens"],
                    "total_completion_tokens": secondary_token_stats["total_completion_tokens"],
                    "avg_tokens_per_request": secondary_token_stats["avg_total_tokens"],
                    "avg_prompt_tokens": secondary_token_stats["avg_prompt_tokens"],
                    "avg_completion_tokens": secondary_token_stats["avg_completion_tokens"]
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