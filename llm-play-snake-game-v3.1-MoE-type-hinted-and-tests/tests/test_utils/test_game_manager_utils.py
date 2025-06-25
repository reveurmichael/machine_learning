"""
Tests for utils.game_manager_utils module.

Focuses on testing game manager utility functions for game lifecycle management,
session coordination, batch processing, and state management.
"""

import pytest
import time
import tempfile
import json
import os
from typing import List, Dict, Any, Optional, Tuple, Generator
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from numpy.typing import NDArray

from utils.game_manager_utils import GameManagerUtils
from core.game_controller import GameController
from core.game_data import GameData


class TestGameManagerUtils:
    """Test game manager utility functions."""

    def test_game_session_initialization(self) -> None:
        """Test game session initialization and configuration."""
        
        # Mock game manager utils
        manager_utils: GameManagerUtils = GameManagerUtils()
        
        # Test session configuration scenarios
        session_configs: List[Dict[str, Any]] = [
            {
                "session_id": "test_session_1",
                "grid_size": 10,
                "max_games": 5,
                "llm_provider": "deepseek",
                "difficulty": "medium"
            },
            {
                "session_id": "test_session_2", 
                "grid_size": 12,
                "max_games": 3,
                "llm_provider": "mistral",
                "difficulty": "hard"
            },
            {
                "session_id": "test_session_3",
                "grid_size": 8,
                "max_games": 10,
                "llm_provider": "hunyuan", 
                "difficulty": "easy"
            }
        ]
        
        initialization_results: List[Dict[str, Any]] = []
        
        for config in session_configs:
            start_time = time.time()
            
            # Initialize session
            session_result = manager_utils.initialize_game_session(config)
            
            end_time = time.time()
            
            # Verify session initialization
            assert session_result is not None, f"Failed to initialize session {config['session_id']}"
            assert session_result.get("success", False), f"Session initialization failed for {config['session_id']}"
            
            # Verify configuration applied
            session_data = session_result.get("session_data", {})
            assert session_data.get("session_id") == config["session_id"], "Session ID mismatch"
            assert session_data.get("grid_size") == config["grid_size"], "Grid size mismatch"
            assert session_data.get("max_games") == config["max_games"], "Max games mismatch"
            
            initialization_results.append({
                "session_id": config["session_id"],
                "config": config,
                "result": session_result,
                "initialization_time": end_time - start_time,
                "success": True
            })
        
        # Verify all sessions initialized successfully
        assert len(initialization_results) == 3, "Should initialize all test sessions"
        
        # Verify initialization performance
        avg_init_time = sum(r["initialization_time"] for r in initialization_results) / len(initialization_results)
        assert avg_init_time < 0.1, f"Session initialization too slow: {avg_init_time}s"

    def test_batch_game_execution(self) -> None:
        """Test batch execution of multiple games."""
        
        manager_utils: GameManagerUtils = GameManagerUtils()
        
        # Create batch configuration
        batch_config: Dict[str, Any] = {
            "session_id": "batch_test_session",
            "games_to_run": 5,
            "grid_size": 10,
            "max_steps_per_game": 100,
            "llm_provider": "deepseek",
            "save_intermediate": True
        }
        
        # Mock game execution
        executed_games: List[Dict[str, Any]] = []
        
        def mock_execute_single_game(game_config: Dict[str, Any]) -> Dict[str, Any]:
            """Mock single game execution."""
            game_id = game_config.get("game_id", "unknown")
            
            # Simulate game execution
            controller = GameController(
                grid_size=game_config.get("grid_size", 10),
                use_gui=False
            )
            
            # Play some moves
            moves_made: List[str] = []
            for step in range(20):
                move = ["UP", "RIGHT", "DOWN", "LEFT"][step % 4]
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move(move)
                moves_made.append(move)
                
                if collision:
                    break
            
            game_result = {
                "game_id": game_id,
                "final_score": controller.score,
                "total_steps": controller.steps,
                "moves_made": moves_made,
                "snake_length": controller.snake_length,
                "execution_time": time.time() - game_config.get("start_time", time.time()),
                "success": True
            }
            
            executed_games.append(game_result)
            return game_result
        
        # Execute batch
        batch_start_time = time.time()
        
        batch_results: List[Dict[str, Any]] = []
        
        for game_num in range(batch_config["games_to_run"]):
            game_config = {
                "game_id": f"game_{game_num + 1}",
                "grid_size": batch_config["grid_size"],
                "max_steps": batch_config["max_steps_per_game"],
                "start_time": time.time()
            }
            
            game_result = mock_execute_single_game(game_config)
            batch_results.append(game_result)
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        
        # Verify batch execution
        assert len(batch_results) == 5, "Should execute all 5 games in batch"
        assert all(result["success"] for result in batch_results), "All games should execute successfully"
        
        # Verify game data consistency
        for i, result in enumerate(batch_results):
            expected_game_id = f"game_{i + 1}"
            assert result["game_id"] == expected_game_id, f"Game ID mismatch: expected {expected_game_id}, got {result['game_id']}"
            assert result["final_score"] >= 0, f"Invalid score for {result['game_id']}"
            assert result["total_steps"] >= 0, f"Invalid steps for {result['game_id']}"
            assert len(result["moves_made"]) > 0, f"No moves recorded for {result['game_id']}"
        
        # Verify batch performance
        avg_game_duration = batch_duration / len(batch_results)
        assert avg_game_duration < 1.0, f"Average game execution too slow: {avg_game_duration}s"

    def test_game_state_management(self) -> None:
        """Test game state management across multiple games."""
        
        manager_utils: GameManagerUtils = GameManagerUtils()
        
        # Mock state manager
        state_manager: Mock = Mock()
        state_manager.game_states = {}
        state_manager.state_history = []
        
        def mock_save_game_state(game_id: str, state: Dict[str, Any]) -> bool:
            """Mock game state saving."""
            state_copy = state.copy()
            state_copy["save_timestamp"] = time.time()
            state_manager.game_states[game_id] = state_copy
            state_manager.state_history.append({
                "action": "save",
                "game_id": game_id,
                "timestamp": time.time()
            })
            return True
        
        def mock_load_game_state(game_id: str) -> Optional[Dict[str, Any]]:
            """Mock game state loading."""
            if game_id in state_manager.game_states:
                state_manager.state_history.append({
                    "action": "load",
                    "game_id": game_id,
                    "timestamp": time.time()
                })
                return state_manager.game_states[game_id].copy()
            return None
        
        def mock_delete_game_state(game_id: str) -> bool:
            """Mock game state deletion."""
            if game_id in state_manager.game_states:
                del state_manager.game_states[game_id]
                state_manager.state_history.append({
                    "action": "delete",
                    "game_id": game_id,
                    "timestamp": time.time()
                })
                return True
            return False
        
        state_manager.save_game_state = mock_save_game_state
        state_manager.load_game_state = mock_load_game_state
        state_manager.delete_game_state = mock_delete_game_state
        
        # Test state management operations
        test_games: List[Dict[str, Any]] = []
        
        for game_num in range(3):
            game_id = f"state_test_game_{game_num}"
            
            # Create game and generate state
            controller = GameController(grid_size=8, use_gui=False)
            
            # Play some moves to create interesting state
            for step in range(15):
                move = ["UP", "DOWN", "LEFT", "RIGHT"][step % 4]
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move(move)
                
                if collision:
                    break
            
            # Create state snapshot
            game_state = {
                "game_id": game_id,
                "score": controller.score,
                "steps": controller.steps,
                "snake_length": controller.snake_length,
                "snake_positions": controller.snake_positions.tolist(),
                "apple_position": controller.apple_position.tolist(),
                "moves_history": controller.moves.copy()
            }
            
            # Save state
            save_success = state_manager.save_game_state(game_id, game_state)
            assert save_success, f"Failed to save state for {game_id}"
            
            test_games.append({
                "game_id": game_id,
                "original_state": game_state,
                "controller": controller
            })
        
        # Test state loading and verification
        for test_game in test_games:
            game_id = test_game["game_id"]
            original_state = test_game["original_state"]
            
            # Load state
            loaded_state = state_manager.load_game_state(game_id)
            assert loaded_state is not None, f"Failed to load state for {game_id}"
            
            # Verify state integrity
            assert loaded_state["game_id"] == original_state["game_id"], "Game ID mismatch"
            assert loaded_state["score"] == original_state["score"], "Score mismatch"
            assert loaded_state["steps"] == original_state["steps"], "Steps mismatch"
            assert loaded_state["snake_length"] == original_state["snake_length"], "Snake length mismatch"
            assert loaded_state["snake_positions"] == original_state["snake_positions"], "Snake positions mismatch"
            assert loaded_state["apple_position"] == original_state["apple_position"], "Apple position mismatch"
            
            # Verify save timestamp was added
            assert "save_timestamp" in loaded_state, "Save timestamp missing"
            assert loaded_state["save_timestamp"] > 0, "Invalid save timestamp"
        
        # Test state deletion
        first_game_id = test_games[0]["game_id"]
        delete_success = state_manager.delete_game_state(first_game_id)
        assert delete_success, f"Failed to delete state for {first_game_id}"
        
        # Verify deletion
        deleted_state = state_manager.load_game_state(first_game_id)
        assert deleted_state is None, f"State still exists after deletion for {first_game_id}"
        
        # Verify state history
        assert len(state_manager.state_history) == 7, "Should have 7 state operations (3 saves + 3 loads + 1 delete)"
        
        save_operations = [op for op in state_manager.state_history if op["action"] == "save"]
        load_operations = [op for op in state_manager.state_history if op["action"] == "load"]
        delete_operations = [op for op in state_manager.state_history if op["action"] == "delete"]
        
        assert len(save_operations) == 3, "Should have 3 save operations"
        assert len(load_operations) == 3, "Should have 3 load operations"
        assert len(delete_operations) == 1, "Should have 1 delete operation"

    def test_concurrent_game_management(self) -> None:
        """Test concurrent game management and coordination."""
        
        manager_utils: GameManagerUtils = GameManagerUtils()
        
        import threading
        
        # Shared resources for concurrent testing
        concurrent_results: List[Dict[str, Any]] = []
        result_lock = threading.Lock()
        
        # Mock concurrent game manager
        concurrent_manager: Mock = Mock()
        concurrent_manager.active_games = {}
        concurrent_manager.game_locks = {}
        concurrent_manager.resource_usage = {"memory": 0, "cpu": 0}
        
        def mock_start_concurrent_game(game_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
            """Mock starting a game in concurrent environment."""
            
            # Create game-specific lock
            if game_id not in concurrent_manager.game_locks:
                concurrent_manager.game_locks[game_id] = threading.Lock()
            
            with concurrent_manager.game_locks[game_id]:
                if game_id in concurrent_manager.active_games:
                    return {"success": False, "error": f"Game {game_id} already active"}
                
                # Simulate resource allocation
                memory_needed = config.get("memory_requirement", 100)
                cpu_needed = config.get("cpu_requirement", 10)
                
                concurrent_manager.resource_usage["memory"] += memory_needed
                concurrent_manager.resource_usage["cpu"] += cpu_needed
                
                # Create game
                controller = GameController(
                    grid_size=config.get("grid_size", 8),
                    use_gui=False
                )
                
                game_info = {
                    "controller": controller,
                    "config": config,
                    "start_time": time.time(),
                    "memory_used": memory_needed,
                    "cpu_used": cpu_needed
                }
                
                concurrent_manager.active_games[game_id] = game_info
                
                return {
                    "success": True,
                    "game_id": game_id,
                    "start_time": game_info["start_time"]
                }
        
        def mock_play_concurrent_game(game_id: str, moves: List[str]) -> Dict[str, Any]:
            """Mock playing moves in concurrent game."""
            
            if game_id not in concurrent_manager.game_locks:
                return {"success": False, "error": f"Game {game_id} not found"}
            
            with concurrent_manager.game_locks[game_id]:
                if game_id not in concurrent_manager.active_games:
                    return {"success": False, "error": f"Game {game_id} not active"}
                
                game_info = concurrent_manager.active_games[game_id]
                controller = game_info["controller"]
                
                move_results: List[Dict[str, Any]] = []
                
                for move in moves:
                    if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
                        collision: bool
                        apple_eaten: bool
                        collision, apple_eaten = controller.make_move(move.upper())
                        
                        move_results.append({
                            "move": move.upper(),
                            "collision": collision,
                            "apple_eaten": apple_eaten,
                            "score": controller.score
                        })
                        
                        if collision:
                            break
                
                return {
                    "success": True,
                    "game_id": game_id,
                    "moves_played": move_results,
                    "final_score": controller.score
                }
        
        def mock_stop_concurrent_game(game_id: str) -> Dict[str, Any]:
            """Mock stopping concurrent game and cleanup."""
            
            if game_id not in concurrent_manager.game_locks:
                return {"success": False, "error": f"Game {game_id} not found"}
            
            with concurrent_manager.game_locks[game_id]:
                if game_id not in concurrent_manager.active_games:
                    return {"success": False, "error": f"Game {game_id} not active"}
                
                game_info = concurrent_manager.active_games[game_id]
                controller = game_info["controller"]
                
                # Release resources
                concurrent_manager.resource_usage["memory"] -= game_info["memory_used"]
                concurrent_manager.resource_usage["cpu"] -= game_info["cpu_used"]
                
                # Get final stats
                final_stats = {
                    "final_score": controller.score,
                    "total_steps": controller.steps,
                    "snake_length": controller.snake_length,
                    "duration": time.time() - game_info["start_time"]
                }
                
                # Remove from active games
                del concurrent_manager.active_games[game_id]
                
                return {
                    "success": True,
                    "game_id": game_id,
                    "final_stats": final_stats
                }
        
        concurrent_manager.start_game = mock_start_concurrent_game
        concurrent_manager.play_game = mock_play_concurrent_game
        concurrent_manager.stop_game = mock_stop_concurrent_game
        
        def concurrent_game_worker(worker_id: int) -> None:
            """Worker function for concurrent game testing."""
            try:
                game_id = f"concurrent_game_{worker_id}"
                
                # Start game
                config = {
                    "grid_size": 8,
                    "memory_requirement": 50,
                    "cpu_requirement": 5
                }
                
                start_result = concurrent_manager.start_game(game_id, config)
                if not start_result["success"]:
                    with result_lock:
                        concurrent_results.append({
                            "worker_id": worker_id,
                            "game_id": game_id,
                            "error": "Failed to start game",
                            "success": False
                        })
                    return
                
                # Play game
                moves = ["UP", "RIGHT", "DOWN", "LEFT", "UP"] * 3
                play_result = concurrent_manager.play_game(game_id, moves)
                
                if not play_result["success"]:
                    with result_lock:
                        concurrent_results.append({
                            "worker_id": worker_id,
                            "game_id": game_id,
                            "error": "Failed to play game",
                            "success": False
                        })
                    return
                
                # Stop game
                stop_result = concurrent_manager.stop_game(game_id)
                
                with result_lock:
                    concurrent_results.append({
                        "worker_id": worker_id,
                        "game_id": game_id,
                        "start_result": start_result,
                        "play_result": play_result,
                        "stop_result": stop_result,
                        "success": True
                    })
                    
            except Exception as e:
                with result_lock:
                    concurrent_results.append({
                        "worker_id": worker_id,
                        "game_id": f"concurrent_game_{worker_id}",
                        "error": str(e),
                        "success": False
                    })
        
        # Start concurrent workers
        threads: List[threading.Thread] = []
        
        for worker_id in range(5):
            thread = threading.Thread(target=concurrent_game_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Verify concurrent game management
        assert len(concurrent_results) == 5, "Should handle all concurrent workers"
        
        successful_results = [r for r in concurrent_results if r.get("success", False)]
        failed_results = [r for r in concurrent_results if not r.get("success", False)]
        
        assert len(successful_results) == 5, f"All concurrent games should succeed, but {len(failed_results)} failed"
        
        # Verify resource cleanup
        assert concurrent_manager.resource_usage["memory"] == 0, "Memory should be fully released"
        assert concurrent_manager.resource_usage["cpu"] == 0, "CPU should be fully released"
        
        # Verify no active games remain
        assert len(concurrent_manager.active_games) == 0, "No games should remain active"

    def test_session_statistics_aggregation(self) -> None:
        """Test aggregation of statistics across game sessions."""
        
        manager_utils: GameManagerUtils = GameManagerUtils()
        
        # Mock session statistics aggregator
        stats_aggregator: Mock = Mock()
        stats_aggregator.session_stats = {}
        stats_aggregator.global_stats = {
            "total_sessions": 0,
            "total_games": 0,
            "total_score": 0,
            "average_score": 0.0,
            "best_session": None
        }
        
        def mock_aggregate_session_stats(session_id: str, games_data: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Mock session statistics aggregation."""
            
            if not games_data:
                return {"success": False, "error": "No games data provided"}
            
            # Calculate session statistics
            total_score = sum(game.get("final_score", 0) for game in games_data)
            total_steps = sum(game.get("total_steps", 0) for game in games_data)
            total_games = len(games_data)
            
            avg_score = total_score / total_games if total_games > 0 else 0
            avg_steps = total_steps / total_games if total_games > 0 else 0
            
            best_game = max(games_data, key=lambda g: g.get("final_score", 0))
            worst_game = min(games_data, key=lambda g: g.get("final_score", 0))
            
            session_summary = {
                "session_id": session_id,
                "total_games": total_games,
                "total_score": total_score,
                "total_steps": total_steps,
                "average_score": avg_score,
                "average_steps": avg_steps,
                "best_game": best_game,
                "worst_game": worst_game,
                "session_duration": sum(game.get("duration", 0) for game in games_data),
                "aggregation_time": time.time()
            }
            
            # Store session stats
            stats_aggregator.session_stats[session_id] = session_summary
            
            # Update global stats
            stats_aggregator.global_stats["total_sessions"] += 1
            stats_aggregator.global_stats["total_games"] += total_games
            stats_aggregator.global_stats["total_score"] += total_score
            
            if stats_aggregator.global_stats["total_games"] > 0:
                stats_aggregator.global_stats["average_score"] = (
                    stats_aggregator.global_stats["total_score"] / 
                    stats_aggregator.global_stats["total_games"]
                )
            
            # Update best session
            if (stats_aggregator.global_stats["best_session"] is None or 
                avg_score > stats_aggregator.global_stats["best_session"]["average_score"]):
                stats_aggregator.global_stats["best_session"] = session_summary
            
            return {
                "success": True,
                "session_summary": session_summary,
                "global_stats": stats_aggregator.global_stats.copy()
            }
        
        stats_aggregator.aggregate_session_stats = mock_aggregate_session_stats
        
        # Test statistics aggregation with multiple sessions
        test_sessions: List[Dict[str, Any]] = [
            {
                "session_id": "stats_session_1",
                "games": [
                    {"game_id": "game_1", "final_score": 150, "total_steps": 75, "duration": 30.0},
                    {"game_id": "game_2", "final_score": 200, "total_steps": 90, "duration": 35.0},
                    {"game_id": "game_3", "final_score": 100, "total_steps": 60, "duration": 25.0}
                ]
            },
            {
                "session_id": "stats_session_2",
                "games": [
                    {"game_id": "game_1", "final_score": 300, "total_steps": 120, "duration": 45.0},
                    {"game_id": "game_2", "final_score": 250, "total_steps": 100, "duration": 40.0}
                ]
            },
            {
                "session_id": "stats_session_3",
                "games": [
                    {"game_id": "game_1", "final_score": 80, "total_steps": 50, "duration": 20.0},
                    {"game_id": "game_2", "final_score": 120, "total_steps": 70, "duration": 28.0},
                    {"game_id": "game_3", "final_score": 180, "total_steps": 85, "duration": 32.0},
                    {"game_id": "game_4", "final_score": 90, "total_steps": 55, "duration": 22.0}
                ]
            }
        ]
        
        aggregation_results: List[Dict[str, Any]] = []
        
        for session in test_sessions:
            session_id = session["session_id"]
            games_data = session["games"]
            
            # Aggregate session statistics
            agg_result = stats_aggregator.aggregate_session_stats(session_id, games_data)
            
            assert agg_result["success"], f"Failed to aggregate stats for {session_id}"
            
            session_summary = agg_result["session_summary"]
            
            # Verify session summary
            assert session_summary["session_id"] == session_id, "Session ID mismatch"
            assert session_summary["total_games"] == len(games_data), "Total games mismatch"
            
            expected_total_score = sum(game["final_score"] for game in games_data)
            assert session_summary["total_score"] == expected_total_score, "Total score mismatch"
            
            expected_avg_score = expected_total_score / len(games_data)
            assert abs(session_summary["average_score"] - expected_avg_score) < 0.01, "Average score mismatch"
            
            aggregation_results.append({
                "session_id": session_id,
                "summary": session_summary,
                "global_stats": agg_result["global_stats"]
            })
        
        # Verify global statistics
        final_global_stats = stats_aggregator.global_stats
        
        assert final_global_stats["total_sessions"] == 3, "Should have 3 sessions"
        assert final_global_stats["total_games"] == 9, "Should have 9 total games"  # 3+2+4
        
        expected_total_score = 150+200+100+300+250+80+120+180+90  # 1470
        assert final_global_stats["total_score"] == expected_total_score, "Global total score mismatch"
        
        expected_global_avg = expected_total_score / 9
        assert abs(final_global_stats["average_score"] - expected_global_avg) < 0.01, "Global average mismatch"
        
        # Verify best session identification
        best_session = final_global_stats["best_session"]
        assert best_session is not None, "Best session should be identified"
        assert best_session["session_id"] == "stats_session_2", "Session 2 should be best (avg 275)"
        
        # Verify individual session stats stored correctly
        assert len(stats_aggregator.session_stats) == 3, "Should store all session stats"
        
        for session_id in ["stats_session_1", "stats_session_2", "stats_session_3"]:
            assert session_id in stats_aggregator.session_stats, f"Session {session_id} stats missing" 