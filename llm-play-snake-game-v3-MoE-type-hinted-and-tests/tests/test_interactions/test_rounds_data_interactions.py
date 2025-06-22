"""
Tests for GameRounds ↔ GameData interactions.

Focuses on testing how GameRounds and GameData maintain consistency
during round transitions, state persistence, and data synchronization.
"""

import pytest
import time
import threading
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch
import numpy as np
from numpy.typing import NDArray

from core.game_rounds import GameRounds
from core.game_data import GameData
from core.game_controller import GameController


class TestRoundsDataInteractions:
    """Test interactions between GameRounds and GameData."""

    def test_round_transition_data_consistency(self) -> None:
        """Test data consistency during round transitions."""
        controller: GameController = GameController(grid_size=12, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Mock or create GameRounds if it exists
        try:
            from core.game_rounds import GameRounds
            rounds: GameRounds = GameRounds(controller)
        except ImportError:
            # If GameRounds doesn't exist, simulate its behavior
            rounds = Mock()
            rounds.current_round = 1
            rounds.round_history = []
        
        round_transitions: List[Dict[str, Any]] = []
        
        # Simulate multiple rounds
        for round_num in range(1, 6):
            round_start_time: float = time.time()
            
            # Record round start state
            start_state: Dict[str, Any] = {
                "round": round_num,
                "score": game_data.score,
                "steps": game_data.steps,
                "snake_length": game_data.snake_length,
                "moves_count": len(game_data.moves),
                "llm_communications": len(game_data.llm_communication)
            }
            
            if hasattr(rounds, 'start_round'):
                rounds.start_round(round_num)
            else:
                rounds.current_round = round_num
            
            # Play round
            for step in range(20):
                move: str = ["UP", "RIGHT", "DOWN", "LEFT"][step % 4]
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move(move)
                
                # Verify round-data consistency during gameplay
                if hasattr(rounds, 'current_round'):
                    assert rounds.current_round == round_num, f"Round number inconsistent at step {step}"
                
                # Add some data
                if step % 5 == 0:
                    game_data.add_llm_communication(
                        f"Round {round_num} step {step} prompt",
                        f"Round {round_num} step {step} response"
                    )
                
                if collision:
                    break
            
            # Record round end state
            end_state: Dict[str, Any] = {
                "round": round_num,
                "final_score": game_data.score,
                "final_steps": game_data.steps,
                "final_snake_length": game_data.snake_length,
                "final_moves_count": len(game_data.moves),
                "final_llm_communications": len(game_data.llm_communication),
                "round_duration": time.time() - round_start_time
            }
            
            if hasattr(rounds, 'end_round'):
                round_summary = rounds.end_round()
                if round_summary:
                    end_state["round_summary"] = round_summary
            
            round_transitions.append({
                "start_state": start_state,
                "end_state": end_state,
                "score_gained": end_state["final_score"] - start_state["score"],
                "steps_taken": end_state["final_steps"] - start_state["steps"]
            })
            
            # Reset for next round
            controller.reset()
        
        # Verify round transitions
        assert len(round_transitions) == 5, "Should have 5 round transitions"
        
        for i, transition in enumerate(round_transitions):
            start = transition["start_state"]
            end = transition["end_state"]
            
            # Verify round progression
            assert start["round"] == i + 1, f"Round {i+1} start state incorrect"
            assert end["round"] == i + 1, f"Round {i+1} end state incorrect"
            
            # Verify data consistency
            assert end["final_score"] >= start["score"], f"Score decreased in round {i+1}"
            assert end["final_steps"] >= start["steps"], f"Steps decreased in round {i+1}"
            assert end["final_moves_count"] >= start["moves_count"], f"Moves decreased in round {i+1}"

    def test_round_data_persistence_coordination(self) -> None:
        """Test coordination between round management and data persistence."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Mock rounds with persistence capabilities
        rounds = Mock()
        rounds.current_round = 0
        rounds.round_data = {}
        
        def mock_save_round_data(round_num: int, data: Dict[str, Any]) -> bool:
            rounds.round_data[round_num] = data.copy()
            return True
        
        def mock_load_round_data(round_num: int) -> Optional[Dict[str, Any]]:
            return rounds.round_data.get(round_num)
        
        rounds.save_round_data = mock_save_round_data
        rounds.load_round_data = mock_load_round_data
        
        # Test round data persistence
        persistence_results: List[Dict[str, Any]] = []
        
        for round_num in range(1, 4):
            rounds.current_round = round_num
            
            # Generate round data
            for step in range(15):
                move: str = ["UP", "DOWN", "LEFT", "RIGHT"][step % 4]
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move(move)
                
                if step % 3 == 0:
                    game_data.add_llm_communication(f"Round {round_num} prompt", f"Response {step}")
                
                if collision:
                    break
            
            # Prepare round data for persistence
            round_data: Dict[str, Any] = {
                "round_number": round_num,
                "final_score": game_data.score,
                "total_steps": game_data.steps,
                "snake_length": game_data.snake_length,
                "moves": game_data.moves.copy(),
                "llm_communications": game_data.llm_communication.copy(),
                "timestamp": time.time()
            }
            
            # Save round data
            save_success: bool = rounds.save_round_data(round_num, round_data)
            assert save_success, f"Failed to save round {round_num} data"
            
            # Verify data can be loaded
            loaded_data: Optional[Dict[str, Any]] = rounds.load_round_data(round_num)
            assert loaded_data is not None, f"Failed to load round {round_num} data"
            
            # Verify data integrity
            assert loaded_data["round_number"] == round_num
            assert loaded_data["final_score"] == game_data.score
            assert loaded_data["total_steps"] == game_data.steps
            assert len(loaded_data["moves"]) == len(game_data.moves)
            
            persistence_results.append({
                "round": round_num,
                "saved": save_success,
                "loaded": loaded_data is not None,
                "data_size": len(str(loaded_data)) if loaded_data else 0
            })
            
            # Reset for next round
            controller.reset()
        
        # Verify all rounds persisted correctly
        assert all(result["saved"] and result["loaded"] for result in persistence_results)
        
        # Verify round data isolation
        for round_num in range(1, 4):
            round_data = rounds.load_round_data(round_num)
            assert round_data["round_number"] == round_num, f"Round {round_num} data corrupted"

    def test_concurrent_round_data_access(self) -> None:
        """Test concurrent access to round and data systems."""
        controller: GameController = GameController(grid_size=8, use_gui=False)
        
        # Shared round management
        shared_rounds = Mock()
        shared_rounds.current_round = 0
        shared_rounds.round_operations = []
        shared_rounds.access_lock = threading.Lock()
        
        access_results: List[Dict[str, Any]] = []
        access_errors: List[Exception] = []
        
        def concurrent_round_player(player_id: int) -> None:
            """Simulate concurrent round playing."""
            try:
                with shared_rounds.access_lock:
                    shared_rounds.current_round += 1
                    current_round = shared_rounds.current_round
                    shared_rounds.round_operations.append(f"Player {player_id} started round {current_round}")
                
                # Play round
                local_controller = GameController(grid_size=8, use_gui=False)
                round_moves: List[str] = []
                
                for step in range(10):
                    move: str = ["UP", "RIGHT", "DOWN", "LEFT"][step % 4]
                    collision: bool
                    apple_eaten: bool
                    collision, apple_eaten = local_controller.make_move(move)
                    round_moves.append(move)
                    
                    if collision:
                        break
                
                with shared_rounds.access_lock:
                    shared_rounds.round_operations.append(f"Player {player_id} finished round {current_round}")
                
                access_results.append({
                    "player_id": player_id,
                    "round": current_round,
                    "moves_made": len(round_moves),
                    "final_score": local_controller.score,
                    "success": True
                })
                
            except Exception as e:
                access_errors.append(e)
        
        # Start concurrent round players
        threads: List[threading.Thread] = []
        
        for player_id in range(5):
            thread = threading.Thread(target=concurrent_round_player, args=(player_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Verify results
        assert len(access_errors) == 0, f"Concurrent round access errors: {access_errors}"
        assert len(access_results) == 5, "Should have results from all players"
        
        # Verify round progression
        rounds_played = [result["round"] for result in access_results]
        assert len(set(rounds_played)) == 5, "Each player should have unique round"
        assert min(rounds_played) == 1, "Rounds should start from 1"
        assert max(rounds_played) == 5, "Should have 5 rounds total"
        
        # Verify operation ordering
        assert len(shared_rounds.round_operations) == 10, "Should have start/end for each round"

    def test_round_statistics_data_aggregation(self) -> None:
        """Test aggregation of statistics across rounds and data."""
        controller: GameController = GameController(grid_size=12, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Mock round statistics aggregator
        round_stats = Mock()
        round_stats.round_summaries = {}
        round_stats.aggregate_stats = {
            "total_rounds": 0,
            "total_score": 0,
            "total_steps": 0,
            "average_score_per_round": 0.0,
            "average_steps_per_round": 0.0
        }
        
        def update_round_stats(round_num: int, round_data: Dict[str, Any]) -> None:
            round_stats.round_summaries[round_num] = round_data
            round_stats.aggregate_stats["total_rounds"] += 1
            round_stats.aggregate_stats["total_score"] += round_data["score"]
            round_stats.aggregate_stats["total_steps"] += round_data["steps"]
            
            total_rounds = round_stats.aggregate_stats["total_rounds"]
            round_stats.aggregate_stats["average_score_per_round"] = (
                round_stats.aggregate_stats["total_score"] / total_rounds
            )
            round_stats.aggregate_stats["average_steps_per_round"] = (
                round_stats.aggregate_stats["total_steps"] / total_rounds
            )
        
        # Play multiple rounds and aggregate
        for round_num in range(1, 6):
            round_start_score = game_data.score
            round_start_steps = game_data.steps
            
            # Play round
            for step in range(25):
                move: str = ["UP", "RIGHT", "DOWN", "LEFT"][step % 4]
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move(move)
                
                # Add round-specific data
                if step % 8 == 0:
                    game_data.add_llm_communication(
                        f"Round {round_num} strategic prompt",
                        f"Round {round_num} strategic response"
                    )
                
                if collision:
                    break
            
            # Calculate round statistics
            round_data: Dict[str, Any] = {
                "round": round_num,
                "score": game_data.score - round_start_score,
                "steps": game_data.steps - round_start_steps,
                "snake_growth": game_data.snake_length - 1,
                "llm_interactions": len([comm for comm in game_data.llm_communication 
                                      if f"Round {round_num}" in comm["prompt"]]),
                "duration": 1.0  # Mock duration
            }
            
            # Update aggregated statistics
            update_round_stats(round_num, round_data)
            
            # Reset for next round
            controller.reset()
        
        # Verify aggregated statistics
        assert round_stats.aggregate_stats["total_rounds"] == 5
        assert round_stats.aggregate_stats["total_score"] > 0
        assert round_stats.aggregate_stats["total_steps"] > 0
        assert round_stats.aggregate_stats["average_score_per_round"] > 0
        assert round_stats.aggregate_stats["average_steps_per_round"] > 0
        
        # Verify individual round data
        for round_num in range(1, 6):
            assert round_num in round_stats.round_summaries
            round_data = round_stats.round_summaries[round_num]
            assert round_data["round"] == round_num
            assert round_data["score"] >= 0
            assert round_data["steps"] >= 0

    def test_round_buffer_data_synchronization(self) -> None:
        """Test round buffer synchronization with game data."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Mock round buffer system
        class MockRoundBuffer:
            def __init__(self, max_size: int = 3):
                self.max_size = max_size
                self.buffer: List[Dict[str, Any]] = []
                self.current_index = 0
            
            def add_round(self, round_data: Dict[str, Any]) -> None:
                if len(self.buffer) >= self.max_size:
                    # Circular buffer - overwrite oldest
                    self.buffer[self.current_index] = round_data
                    self.current_index = (self.current_index + 1) % self.max_size
                else:
                    self.buffer.append(round_data)
            
            def get_recent_rounds(self, count: int) -> List[Dict[str, Any]]:
                return self.buffer[-count:] if count <= len(self.buffer) else self.buffer
            
            def is_full(self) -> bool:
                return len(self.buffer) >= self.max_size
        
        round_buffer = MockRoundBuffer(max_size=3)
        
        # Test buffer synchronization over multiple rounds
        for round_num in range(1, 7):  # More rounds than buffer size
            # Play round
            for step in range(15):
                move: str = ["UP", "DOWN", "LEFT", "RIGHT"][step % 4]
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move(move)
                
                if collision:
                    break
            
            # Create round data snapshot
            round_snapshot: Dict[str, Any] = {
                "round": round_num,
                "score": game_data.score,
                "steps": game_data.steps,
                "snake_length": game_data.snake_length,
                "moves_snapshot": game_data.moves.copy(),
                "llm_count": len(game_data.llm_communication),
                "timestamp": time.time()
            }
            
            # Add to buffer
            round_buffer.add_round(round_snapshot)
            
            # Verify buffer state
            if round_num <= 3:
                assert len(round_buffer.buffer) == round_num, f"Buffer size incorrect at round {round_num}"
                assert not round_buffer.is_full() or round_num == 3
            else:
                assert len(round_buffer.buffer) == 3, "Buffer should be at max size"
                assert round_buffer.is_full()
            
            # Verify recent rounds retrieval
            recent_rounds = round_buffer.get_recent_rounds(2)
            assert len(recent_rounds) <= 2, "Should return at most 2 recent rounds"
            
            if len(recent_rounds) > 0:
                latest_round = recent_rounds[-1]
                assert latest_round["round"] == round_num, "Latest round should match current"
            
            # Reset for next round
            controller.reset()
        
        # Final verification
        final_buffer = round_buffer.get_recent_rounds(3)
        assert len(final_buffer) == 3, "Should have 3 rounds in final buffer"
        
        # Should contain rounds 4, 5, 6 (latest)
        buffer_rounds = [r["round"] for r in final_buffer]
        expected_rounds = [4, 5, 6]
        assert buffer_rounds == expected_rounds, f"Buffer should contain {expected_rounds}, got {buffer_rounds}"

    def test_round_transition_error_handling(self) -> None:
        """Test error handling during round transitions."""
        controller: GameController = GameController(grid_size=8, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Mock round manager with error injection
        round_manager = Mock()
        round_manager.current_round = 0
        round_manager.error_conditions = {}
        
        def mock_start_round(round_num: int) -> bool:
            if round_num in round_manager.error_conditions:
                raise round_manager.error_conditions[round_num]
            round_manager.current_round = round_num
            return True
        
        def mock_end_round() -> Optional[Dict[str, Any]]:
            if round_manager.current_round in round_manager.error_conditions:
                raise round_manager.error_conditions[round_manager.current_round]
            return {"round": round_manager.current_round, "status": "completed"}
        
        round_manager.start_round = mock_start_round
        round_manager.end_round = mock_end_round
        
        # Test various error scenarios
        error_scenarios: List[Tuple[int, Exception, str]] = [
            (2, ValueError("Invalid round configuration"), "start_error"),
            (4, RuntimeError("Round state corruption"), "end_error"),
            (5, IOError("Round data save failed"), "persistence_error"),
        ]
        
        # Set up error conditions
        for round_num, error, error_type in error_scenarios:
            round_manager.error_conditions[round_num] = error
        
        successful_rounds: List[int] = []
        error_recoveries: List[Dict[str, Any]] = []
        
        for round_num in range(1, 7):
            try:
                # Attempt to start round
                round_manager.start_round(round_num)
                
                # Play round
                for step in range(10):
                    move: str = ["UP", "RIGHT", "DOWN", "LEFT"][step % 4]
                    collision: bool
                    apple_eaten: bool
                    collision, apple_eaten = controller.make_move(move)
                    
                    if collision:
                        break
                
                # Attempt to end round
                round_summary = round_manager.end_round()
                
                successful_rounds.append(round_num)
                
            except Exception as e:
                # Handle round error
                error_recoveries.append({
                    "round": round_num,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "data_state": {
                        "score": game_data.score,
                        "steps": game_data.steps,
                        "snake_length": game_data.snake_length
                    }
                })
                
                # Attempt recovery
                try:
                    # Reset round state
                    round_manager.current_round = round_num - 1 if round_num > 1 else 0
                    
                    # Verify game data is still consistent
                    assert game_data.score >= 0, "Score became invalid during error"
                    assert game_data.steps >= 0, "Steps became invalid during error"
                    assert game_data.snake_length >= 1, "Snake length became invalid during error"
                    
                except Exception as recovery_error:
                    error_recoveries[-1]["recovery_failed"] = str(recovery_error)
            
            # Reset for next round attempt
            controller.reset()
        
        # Verify error handling
        assert len(successful_rounds) > 0, "Should have some successful rounds"
        assert len(error_recoveries) == len(error_scenarios), "Should have handled all error scenarios"
        
        # Verify successful rounds
        expected_successful = [1, 3, 6]  # Rounds without errors
        assert successful_rounds == expected_successful, f"Expected {expected_successful}, got {successful_rounds}"
        
        # Verify error recovery
        for recovery in error_recoveries:
            assert "error" in recovery, "Error should be recorded"
            assert "data_state" in recovery, "Data state should be preserved"
            assert recovery["data_state"]["score"] >= 0, "Data should remain valid after error" 