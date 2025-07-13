"""
Universal Game Summary Generator for all Snake Game AI tasks (0-5).

Implements the Template Method and Strategy patterns to provide a single source of truth
for game and session summary generation. All extensions (Task-0, heuristics, RL, etc.)
should use this generator, with extension-specific fields handled via hooks or subclassing.

Design Patterns:
- Template Method: Defines the summary generation workflow
- Strategy: Allows extension-specific summary fields via hooks

Usage:
- Instantiate or subclass for extension-specific needs
- Call generate_game_summary() and generate_session_summary() as needed

Extension Points:
- _add_task_specific_game_fields
- _add_task_specific_session_fields
"""
from typing import Any, Dict, List, Optional
from datetime import datetime

class BaseGameSummaryGenerator:
    """
    Universal summary generator for all tasks/extensions.
    """
    def generate_game_summary(self, game_data: Dict[str, Any], game_duration: float) -> Dict[str, Any]:
        """
        Generate a summary for a single game.
        Args:
            game_data: The canonical game state dictionary
            game_duration: Duration of the game in seconds
        Returns:
            Dictionary containing the game summary
        """
        summary = {
            "game_number": game_data.get("game_number"),
            "timestamp": game_data.get("timestamp", datetime.now().isoformat()),
            "score": game_data.get("score"),
            "steps": game_data.get("steps"),
            "game_over": game_data.get("game_over"),
            "game_end_reason": game_data.get("game_end_reason"),
            "duration_seconds": round(game_duration, 2),
            "snake_positions": game_data.get("snake_positions"),
            "apple_positions": game_data.get("apple_positions"),
            "moves": game_data.get("moves"),
            "rounds": game_data.get("rounds"),
        }
        # Extension hook
        self._add_task_specific_game_fields(summary, game_data)
        return summary

    def generate_session_summary(self, session_data: Dict[str, Any], session_duration: float) -> Dict[str, Any]:
        """
        Generate a summary for a session (multiple games).
        Args:
            session_data: Aggregated session statistics
            session_duration: Duration of the session in seconds
        Returns:
            Dictionary containing the session summary
        """
        summary = {
            "session_timestamp": session_data.get("session_timestamp", datetime.now().isoformat()),
            "total_games": session_data.get("total_games"),
            "total_score": session_data.get("total_score"),
            "average_score": session_data.get("average_score"),
            "total_steps": session_data.get("total_steps"),
            "total_rounds": session_data.get("total_rounds"),
            "session_duration_seconds": round(session_duration, 2),
            "score_per_step": session_data.get("score_per_step"),
            "score_per_round": session_data.get("score_per_round"),
            "game_scores": session_data.get("game_scores"),
            "game_steps": session_data.get("game_steps"),
            "round_counts": session_data.get("round_counts"),
            "configuration": session_data.get("configuration"),
        }
        # Extension hook
        self._add_task_specific_session_fields(summary, session_data)
        return summary

    def _add_task_specific_game_fields(self, summary: Dict[str, Any], game_data: Dict[str, Any]) -> None:
        """
        Hook for extension-specific game summary fields.
        Override in subclasses as needed.
        """
        pass

    def _add_task_specific_session_fields(self, summary: Dict[str, Any], session_data: Dict[str, Any]) -> None:
        """
        Hook for extension-specific session summary fields.
        Override in subclasses as needed.
        """
        pass 

# TODO: it's NOT used by Task0 yet. Should be used by Task0. And should be used by heuristics-v0.04.