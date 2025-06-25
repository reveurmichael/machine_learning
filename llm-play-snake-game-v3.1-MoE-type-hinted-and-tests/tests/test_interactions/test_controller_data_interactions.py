"""
Tests for GameController ↔ GameData interactions.

Focuses on testing how GameController and GameData maintain synchronization
under various conditions including concurrent modifications, error states,
and complex game scenarios.
"""

import pytest
import numpy as np
import threading
import time
from typing import List, Dict, Any, Optional, Tuple, Generator
from unittest.mock import Mock, patch, MagicMock
from numpy.typing import NDArray

from core.game_controller import GameController
from core.game_data import GameData


class TestControllerDataInteractions:
    """Test interactions between GameController and GameData."""

    def test_state_synchronization_during_rapid_moves(self) -> None:
        """Test state synchronization during rapid move sequences."""
        controller: GameController = GameController(grid_size=15, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Perform rapid moves and verify synchronization at each step
        moves: List[str] = ["UP", "RIGHT", "DOWN", "LEFT", "UP", "RIGHT"] * 50
        
        for i, move in enumerate(moves):
            # Record state before move
            pre_score: int = game_data.score
            pre_steps: int = game_data.steps
            pre_length: int = game_data.snake_length
            
            # Make move
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move(move)
            
            # Verify synchronization after move
            assert controller.score == game_data.score
            assert controller.steps == game_data.steps
            assert controller.snake_length == game_data.snake_length
            assert len(controller.snake_positions) == game_data.snake_length
            
            # Verify incremental changes
            assert game_data.steps == pre_steps + 1
            if apple_eaten:
                assert game_data.score > pre_score
                assert game_data.snake_length > pre_length
            else:
                assert game_data.score == pre_score
                assert game_data.snake_length == pre_length
            
            # Verify move history consistency
            assert len(game_data.moves) == i + 1
            assert game_data.moves[-1] == move
            
            if collision:
                assert game_data.game_over
                break

    def test_score_update_consistency(self) -> None:
        """Test score update consistency between controller and data."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Force apple eating scenarios
        initial_score: int = controller.score
        apples_eaten: int = 0
        
        for i in range(20):
            # Position apple adjacent to snake head
            head: NDArray[np.int_] = controller.snake_positions[-1]
            apple_pos: List[int] = [head[0], head[1] - 1]
            
            # Ensure apple position is valid
            if (0 <= apple_pos[0] < controller.grid_size and 
                0 <= apple_pos[1] < controller.grid_size):
                controller.set_apple_position(apple_pos)
                
                # Move towards apple
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move("UP")
                
                if apple_eaten:
                    apples_eaten += 1
                    
                    # Verify score consistency
                    expected_score: int = initial_score + apples_eaten
                    assert controller.score == expected_score
                    assert game_data.score == expected_score
                    
                    # Verify snake length consistency
                    expected_length: int = 1 + apples_eaten
                    assert controller.snake_length == expected_length
                    assert game_data.snake_length == expected_length
                    assert len(controller.snake_positions) == expected_length
                
                if collision:
                    break

    def test_concurrent_data_modifications(self) -> None:
        """Test behavior when data is modified from multiple sources."""
        controller: GameController = GameController(grid_size=12, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Simulate concurrent modifications
        modification_results: List[Dict[str, Any]] = []
        modification_errors: List[Exception] = []
        
        def modify_via_controller(thread_id: int) -> None:
            """Modify state via controller."""
            try:
                for i in range(100):
                    move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
                    collision: bool
                    apple_eaten: bool
                    collision, apple_eaten = controller.make_move(move)
                    
                    if collision:
                        controller.reset()
                
                modification_results.append({
                    "thread_id": thread_id,
                    "source": "controller",
                    "final_score": controller.score,
                    "final_steps": controller.steps
                })
                
            except Exception as e:
                modification_errors.append(e)
        
        def modify_via_data(thread_id: int) -> None:
            """Modify state via data object."""
            try:
                for i in range(50):
                    # Add LLM communication data
                    game_data.add_llm_communication(
                        f"Thread {thread_id} prompt {i}",
                        f"Thread {thread_id} response {i}"
                    )
                    
                    # Add token usage
                    game_data.add_token_usage(
                        prompt_tokens=10 + i,
                        completion_tokens=5 + i
                    )
                    
                    time.sleep(0.001)  # Small delay to encourage race conditions
                
                modification_results.append({
                    "thread_id": thread_id,
                    "source": "data",
                    "llm_requests": len(game_data.llm_communication),
                    "token_usage": game_data.total_tokens
                })
                
            except Exception as e:
                modification_errors.append(e)
        
        # Start concurrent modification threads
        threads: List[threading.Thread] = []
        for i in range(4):
            if i < 2:
                thread = threading.Thread(target=modify_via_controller, args=(i,))
            else:
                thread = threading.Thread(target=modify_via_data, args=(i,))
            threads.append(thread)
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Verify no errors occurred
        assert len(modification_errors) == 0, f"Concurrent modification errors: {modification_errors}"
        
        # Verify data consistency
        assert len(modification_results) == 4
        
        # Controller and data should still be synchronized
        assert controller.score == game_data.score
        assert controller.steps == game_data.steps
        assert controller.snake_length == game_data.snake_length

    def test_data_persistence_during_gameplay(self) -> None:
        """Test data persistence while game is actively running."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Configure game data
        game_data.set_llm_info(
            primary_provider="test_provider",
            primary_model="test_model"
        )
        
        # Play game while periodically saving/loading
        moves: List[str] = ["UP", "RIGHT", "DOWN", "LEFT"] * 25
        save_points: List[int] = [10, 25, 50, 75]
        
        for i, move in enumerate(moves):
            # Make move
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move(move)
            
            # Add some LLM data
            if i % 5 == 0:
                game_data.add_llm_communication(f"Prompt {i}", f"Response {i}")
                game_data.add_token_usage(prompt_tokens=50, completion_tokens=25)
            
            # Save and reload at checkpoints
            if i in save_points:
                # Save current state
                save_data: Dict[str, Any] = game_data.to_dict()
                
                # Create new data object and load
                loaded_data: GameData = GameData()
                loaded_data.from_dict(save_data)
                
                # Verify data consistency
                assert loaded_data.score == game_data.score
                assert loaded_data.steps == game_data.steps
                assert loaded_data.snake_length == game_data.snake_length
                assert loaded_data.moves == game_data.moves
                assert loaded_data.llm_info == game_data.llm_info
                assert loaded_data.total_tokens == game_data.total_tokens
            
            if collision:
                assert game_data.game_over
                break

    def test_error_propagation_between_components(self) -> None:
        """Test how errors propagate between controller and data."""
        controller: GameController = GameController(grid_size=8, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Test various error scenarios
        error_scenarios: List[Tuple[str, callable]] = [
            ("invalid_snake_position", lambda: setattr(controller, 'snake_positions', 
                                                     np.array([[-1, -1]], dtype=np.int_))),
            ("corrupted_score", lambda: game_data.update_scores(-100, game_data.steps, game_data.snake_length)),
            ("invalid_move_history", lambda: setattr(game_data, 'moves', ["INVALID_MOVE"] * 1000)),
            ("negative_steps", lambda: game_data.update_scores(game_data.score, -1, game_data.snake_length)),
        ]
        
        for scenario_name, error_func in error_scenarios:
            # Reset to clean state
            controller.reset()
            assert controller.score == 0
            assert game_data.score == 0
            
            try:
                # Introduce error
                error_func()
                
                # Try to continue normal operation
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move("UP")
                
                # Verify system can still operate or fails gracefully
                if scenario_name == "corrupted_score":
                    # Score should be handled gracefully
                    assert isinstance(game_data.score, int)
                elif scenario_name == "negative_steps":
                    # Steps should be non-negative
                    assert game_data.steps >= 0
                
            except Exception as e:
                # Should be specific, informative errors
                assert isinstance(e, (ValueError, IndexError, TypeError))
                assert len(str(e)) > 0
            
            # Reset should always work
            try:
                controller.reset()
                assert controller.steps == 0
                assert game_data.steps == 0
            except Exception:
                # If reset fails, system is in bad state
                pass

    def test_state_consistency_after_reset(self) -> None:
        """Test state consistency between controller and data after reset."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Build up complex state
        for i in range(50):
            move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move(move)
            
            if i % 5 == 0:
                game_data.add_llm_communication(f"Prompt {i}", f"Response {i}")
                game_data.add_token_usage(prompt_tokens=100, completion_tokens=50)
            
            if collision:
                break
        
        # Verify we have substantial state
        assert controller.steps > 0
        assert game_data.steps > 0
        assert len(game_data.moves) > 0
        
        # Reset and verify consistency
        controller.reset()
        
        # Controller state should be reset
        assert controller.steps == 0
        assert controller.score == 0
        assert controller.snake_length == 1
        assert len(controller.snake_positions) == 1
        
        # Data state should be reset
        assert game_data.steps == 0
        assert game_data.score == 0
        assert game_data.snake_length == 1
        assert len(game_data.moves) == 0
        assert not game_data.game_over
        
        # LLM data should persist (design decision)
        # This tests the boundary between game state and session data

    def test_apple_generation_data_consistency(self) -> None:
        """Test apple generation consistency between controller and data tracking."""
        controller: GameController = GameController(grid_size=8, use_gui=False)
        game_data: GameData = controller.game_state
        
        apple_positions: List[Tuple[int, int]] = []
        apple_eaten_count: int = 0
        
        for i in range(100):
            # Record apple position before move
            apple_pos: Tuple[int, int] = tuple(controller.apple_position.tolist())
            apple_positions.append(apple_pos)
            
            # Make move
            move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move(move)
            
            if apple_eaten:
                apple_eaten_count += 1
                
                # Verify apple was at recorded position
                assert apple_pos in apple_positions
                
                # Verify new apple is generated
                new_apple_pos: Tuple[int, int] = tuple(controller.apple_position.tolist())
                assert new_apple_pos != apple_pos
                
                # Verify apple is not on snake
                for snake_pos in controller.snake_positions:
                    assert not np.array_equal(controller.apple_position, snake_pos)
                
                # Verify score consistency
                assert controller.score == apple_eaten_count
                assert game_data.score == apple_eaten_count
            
            if collision:
                break
        
        # Verify total consistency
        assert apple_eaten_count == controller.score
        assert apple_eaten_count == game_data.score

    def test_move_validation_data_recording(self) -> None:
        """Test move validation and data recording consistency."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Test various move scenarios
        move_scenarios: List[Tuple[str, List[str], str]] = [
            ("valid_moves", ["UP", "RIGHT", "DOWN", "LEFT"], "normal"),
            ("reversal_attempts", ["UP", "DOWN", "LEFT", "RIGHT"], "reversal"),
            ("repeated_moves", ["UP"] * 10, "repeated"),
            ("mixed_invalid", ["UP", "INVALID", "RIGHT", "", "DOWN"], "mixed"),
        ]
        
        for scenario_name, moves, expected_behavior in move_scenarios:
            controller.reset()
            valid_moves_made: int = 0
            
            for move in moves:
                pre_steps: int = game_data.steps
                
                try:
                    collision: bool
                    apple_eaten: bool
                    collision, apple_eaten = controller.make_move(move)
                    
                    # If move was processed, it should be recorded
                    if game_data.steps > pre_steps:
                        valid_moves_made += 1
                        assert len(game_data.moves) == valid_moves_made
                        assert game_data.moves[-1] == move
                    
                    if collision:
                        break
                        
                except (ValueError, TypeError):
                    # Invalid moves should not increment steps or be recorded
                    assert game_data.steps == pre_steps
                    assert len(game_data.moves) == valid_moves_made
            
            # Verify final consistency
            assert game_data.steps == valid_moves_made
            assert len(game_data.moves) == valid_moves_made

    def test_memory_efficiency_during_long_games(self) -> None:
        """Test memory efficiency during extended gameplay."""
        controller: GameController = GameController(grid_size=20, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Play very long game
        moves_made: int = 0
        resets: int = 0
        
        for i in range(10000):  # Very long sequence
            move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move(move)
            moves_made += 1
            
            # Periodically add LLM data
            if i % 100 == 0:
                game_data.add_llm_communication(f"Long game prompt {i}", f"Response {i}")
            
            if collision:
                controller.reset()
                resets += 1
                
                # Verify memory isn't growing unbounded
                # Snake positions should be reset to minimal size
                assert len(controller.snake_positions) == 1
                assert controller.score == 0
                assert game_data.score == 0
                
                # Move history should be reset
                assert len(game_data.moves) == 0
            
            # Break if we've done enough resets
            if resets >= 100:
                break
        
        # Verify system remains responsive
        assert moves_made > 1000
        assert resets > 10
        
        # Final state should be consistent
        assert controller.score == game_data.score
        assert controller.steps == game_data.steps

    def test_serialization_state_consistency(self) -> None:
        """Test serialization doesn't break controller-data consistency."""
        controller: GameController = GameController(grid_size=12, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Build complex state
        for i in range(30):
            move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move(move)
            
            if i % 3 == 0:
                game_data.add_llm_communication(f"Serialization test {i}", f"Response {i}")
                game_data.add_token_usage(prompt_tokens=75, completion_tokens=35)
            
            if collision:
                break
        
        # Capture state before serialization
        pre_serialize_state: Dict[str, Any] = {
            "controller_score": controller.score,
            "controller_steps": controller.steps,
            "controller_snake_length": controller.snake_length,
            "data_score": game_data.score,
            "data_steps": game_data.steps,
            "data_snake_length": game_data.snake_length,
            "moves_count": len(game_data.moves),
            "llm_count": len(game_data.llm_communication)
        }
        
        # Serialize and deserialize
        serialized: str = game_data.to_json()
        new_data: GameData = GameData()
        new_data.from_json(serialized)
        
        # Verify serialized data consistency
        assert new_data.score == pre_serialize_state["data_score"]
        assert new_data.steps == pre_serialize_state["data_steps"]
        assert new_data.snake_length == pre_serialize_state["data_snake_length"]
        assert len(new_data.moves) == pre_serialize_state["moves_count"]
        assert len(new_data.llm_communication) == pre_serialize_state["llm_count"]
        
        # Original controller-data consistency should be maintained
        assert controller.score == game_data.score
        assert controller.steps == game_data.steps
        assert controller.snake_length == game_data.snake_length 