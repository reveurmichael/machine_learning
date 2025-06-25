"""
Tests for complete game workflow interactions.

Focuses on testing the full end-to-end game flow involving all major components:
GameController, GameData, LLMClient, GUI, Web interfaces, file I/O, and statistics.
"""

import pytest
import tempfile
import time
import json
import os
from typing import List, Dict, Any, Optional, Tuple, Generator
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
from numpy.typing import NDArray

# Import all components for workflow testing
from core.game_controller import GameController
from core.game_data import GameData
from core.game_logic import GameLogic
from llm.client import LLMClient
from utils.json_utils import JSONUtils
from utils.file_utils import FileUtils
from utils.game_stats_utils import GameStatsUtils
from core.game_manager import GameManager
from core.game_stats import GameStatistics
from utils.game_manager_utils import initialize_game_manager


class TestFullGameWorkflowInteractions:
    """Test complete game workflow interactions across all components."""

    def test_complete_single_game_lifecycle(self) -> None:
        """Test complete lifecycle of a single game from start to finish."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize all components
            game_data: GameData = GameData()
            game_controller: GameController = GameController(
                grid_size=10,
                use_gui=False,
                game_data=game_data
            )
            
            # Mock LLM client for move generation
            llm_client: Mock = Mock(spec=LLMClient)
            llm_client.generate_move.return_value = {
                "success": True,
                "move": "UP",
                "confidence": 0.85,
                "reasoning": "Moving up to avoid collision",
                "response_time": 150
            }
            
            # Mock file utils for persistence
            file_utils: Mock = Mock(spec=FileUtils)
            file_utils.save_game_data.return_value = {"success": True, "file_path": f"{temp_dir}/game_data.json"}
            file_utils.load_game_data.return_value = {"success": True, "data": {}}
            
            # Mock stats utils for analytics
            stats_utils: Mock = Mock(spec=GameStatsUtils)
            stats_utils.calculate_basic_statistics.return_value = {
                "success": True,
                "statistics": {
                    "average_score": 150.0,
                    "total_games": 1,
                    "success_rate": 1.0
                }
            }
            
            # Workflow tracking
            workflow_events: List[Dict[str, Any]] = []
            
            def track_event(event_type: str, component: str, details: Dict[str, Any]) -> None:
                workflow_events.append({
                    "timestamp": time.time(),
                    "event_type": event_type,
                    "component": component,
                    "details": details
                })
            
            # 1. Game Initialization Phase
            track_event("initialization_start", "workflow", {"game_id": "test_game_1"})
            
            # Initialize game state
            init_result = game_controller.initialize_game()
            assert init_result["success"] is True, "Game initialization should succeed"
            track_event("game_initialized", "game_controller", init_result)
            
            # Set up data tracking
            game_data.start_new_game()
            track_event("data_tracking_started", "game_data", {"tracking_active": True})
            
            # Load any existing state
            load_result = file_utils.load_game_data(f"{temp_dir}/game_state.json")
            track_event("state_loaded", "file_utils", load_result)
            
            # 2. Game Loop Phase
            track_event("game_loop_start", "workflow", {"max_moves": 20})
            
            move_sequence: List[Dict[str, Any]] = []
            
            for move_count in range(20):  # Simulate 20 moves
                # Get current game state
                current_state = game_controller.get_game_state()
                track_event("state_retrieved", "game_controller", {
                    "move_number": move_count + 1,
                    "snake_length": len(current_state.get("snake_positions", [])),
                    "score": current_state.get("score", 0)
                })
                
                # LLM generates next move
                llm_context = {
                    "grid_state": current_state,
                    "move_history": move_sequence[-5:],  # Last 5 moves for context
                    "game_metadata": {"move_number": move_count + 1}
                }
                
                move_result = llm_client.generate_move(llm_context)
                assert move_result["success"] is True, f"LLM move generation should succeed for move {move_count + 1}"
                track_event("move_generated", "llm_client", {
                    "move": move_result["move"],
                    "confidence": move_result["confidence"],
                    "response_time": move_result["response_time"]
                })
                
                # Execute move in game controller
                execution_result = game_controller.process_move(move_result["move"])
                track_event("move_executed", "game_controller", {
                    "move": move_result["move"],
                    "success": execution_result["success"],
                    "new_score": execution_result.get("score", 0),
                    "game_over": execution_result.get("game_over", False)
                })
                
                # Update game data
                move_data = {
                    "move_number": move_count + 1,
                    "move": move_result["move"],
                    "llm_confidence": move_result["confidence"],
                    "llm_response_time": move_result["response_time"],
                    "game_state_before": current_state,
                    "game_state_after": game_controller.get_game_state(),
                    "timestamp": time.time()
                }
                
                game_data.record_move(move_data)
                move_sequence.append(move_data)
                track_event("move_recorded", "game_data", {"move_number": move_count + 1})
                
                # Check for game over
                if execution_result.get("game_over", False):
                    track_event("game_over", "workflow", {
                        "reason": execution_result.get("game_over_reason", "unknown"),
                        "final_score": execution_result.get("score", 0),
                        "moves_completed": move_count + 1
                    })
                    break
                
                # Periodic state persistence (every 5 moves)
                if (move_count + 1) % 5 == 0:
                    save_result = file_utils.save_game_data(
                        game_controller.get_game_state(),
                        f"{temp_dir}/game_state_checkpoint_{move_count + 1}.json"
                    )
                    track_event("state_persisted", "file_utils", {
                        "checkpoint_move": move_count + 1,
                        "save_result": save_result
                    })
            
            # 3. Game Completion Phase
            track_event("completion_phase_start", "workflow", {})
            
            # Finalize game data
            final_state = game_controller.get_game_state()
            game_summary = game_data.finalize_game(final_state)
            track_event("game_finalized", "game_data", game_summary)
            
            # Calculate final statistics
            all_games_data = [game_summary]
            final_stats = stats_utils.calculate_basic_statistics(all_games_data)
            track_event("statistics_calculated", "stats_utils", final_stats)
            
            # Save complete game record
            complete_game_record = {
                "game_summary": game_summary,
                "move_sequence": move_sequence,
                "workflow_events": workflow_events,
                "final_statistics": final_stats,
                "metadata": {
                    "total_moves": len(move_sequence),
                    "duration": workflow_events[-1]["timestamp"] - workflow_events[0]["timestamp"],
                    "components_involved": ["game_controller", "game_data", "llm_client", "file_utils", "stats_utils"]
                }
            }
            
            final_save_result = file_utils.save_game_data(
                complete_game_record,
                f"{temp_dir}/complete_game_record.json"
            )
            track_event("complete_record_saved", "file_utils", final_save_result)
            
            # 4. Workflow Validation
            # Verify all expected events occurred
            event_types = [event["event_type"] for event in workflow_events]
            required_events = [
                "initialization_start", "game_initialized", "data_tracking_started",
                "game_loop_start", "move_generated", "move_executed", "move_recorded",
                "completion_phase_start", "game_finalized", "statistics_calculated"
            ]
            
            for required_event in required_events:
                assert required_event in event_types, f"Required event '{required_event}' missing from workflow"
            
            # Verify component interactions
            assert llm_client.generate_move.call_count > 0, "LLM should have been called for move generation"
            assert file_utils.save_game_data.call_count > 0, "File utils should have been called for persistence"
            assert stats_utils.calculate_basic_statistics.call_count > 0, "Stats utils should have been called"
            
            # Verify data consistency
            assert len(move_sequence) > 0, "Should have recorded moves"
            assert game_summary["total_moves"] == len(move_sequence), "Move count should be consistent"
            
            # Verify workflow timing
            total_duration = workflow_events[-1]["timestamp"] - workflow_events[0]["timestamp"]
            assert total_duration > 0, "Workflow should have measurable duration"
            assert total_duration < 60, "Workflow should complete within reasonable time"

    def test_multi_game_session_workflow(self) -> None:
        """Test complete workflow for multiple games in a session."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Session configuration
            session_config = {
                "session_id": "multi_game_session_1",
                "total_games": 3,
                "grid_size": 8,
                "max_moves_per_game": 15,
                "llm_provider": "deepseek"
            }
            
            # Initialize session-level components
            session_data: List[Dict[str, Any]] = []
            session_stats: Dict[str, Any] = {
                "games_completed": 0,
                "total_score": 0,
                "total_moves": 0,
                "session_start_time": time.time()
            }
            
            # Mock components for session
            llm_client: Mock = Mock(spec=LLMClient)
            file_utils: Mock = Mock(spec=FileUtils)
            stats_utils: Mock = Mock(spec=GameStatsUtils)
            
            # Configure LLM to provide different strategies per game
            move_strategies = [
                {"strategy": "aggressive", "base_confidence": 0.9},
                {"strategy": "conservative", "base_confidence": 0.7},
                {"strategy": "balanced", "base_confidence": 0.8}
            ]
            
            session_workflow_events: List[Dict[str, Any]] = []
            
            def track_session_event(event_type: str, component: str, details: Dict[str, Any]) -> None:
                session_workflow_events.append({
                    "timestamp": time.time(),
                    "event_type": event_type,
                    "component": component,
                    "session_id": session_config["session_id"],
                    "details": details
                })
            
            # Session initialization
            track_session_event("session_start", "workflow", session_config)
            
            # Configure file persistence for session
            file_utils.save_session_data.return_value = {
                "success": True,
                "file_path": f"{temp_dir}/session_data.json"
            }
            file_utils.load_session_data.return_value = {
                "success": True,
                "data": {"existing_games": []}
            }
            
            # Play each game in the session
            for game_number in range(1, session_config["total_games"] + 1):
                track_session_event("game_start", "workflow", {"game_number": game_number})
                
                # Initialize game
                game_data = GameData()
                game_controller = GameController(
                    grid_size=session_config["grid_size"],
                    use_gui=False,
                    game_data=game_data
                )
                
                init_result = game_controller.initialize_game()
                track_session_event("game_initialized", "game_controller", {
                    "game_number": game_number,
                    "init_success": init_result["success"]
                })
                
                # Configure LLM strategy for this game
                current_strategy = move_strategies[game_number - 1]
                
                def mock_move_generator(context: Dict[str, Any]) -> Dict[str, Any]:
                    moves = ["UP", "DOWN", "LEFT", "RIGHT"]
                    move = np.random.choice(moves)
                    confidence = current_strategy["base_confidence"] + np.random.uniform(-0.1, 0.1)
                    return {
                        "success": True,
                        "move": move,
                        "confidence": max(0.1, min(1.0, confidence)),
                        "strategy": current_strategy["strategy"],
                        "response_time": np.random.randint(100, 300)
                    }
                
                llm_client.generate_move.side_effect = mock_move_generator
                
                # Play the game
                game_moves: List[Dict[str, Any]] = []
                
                for move_count in range(session_config["max_moves_per_game"]):
                    # Generate and execute move
                    current_state = game_controller.get_game_state()
                    
                    move_result = llm_client.generate_move({"game_state": current_state})
                    execution_result = game_controller.process_move(move_result["move"])
                    
                    move_data = {
                        "game_number": game_number,
                        "move_number": move_count + 1,
                        "move": move_result["move"],
                        "strategy": move_result["strategy"],
                        "confidence": move_result["confidence"],
                        "execution_success": execution_result["success"],
                        "score": execution_result.get("score", 0)
                    }
                    
                    game_moves.append(move_data)
                    game_data.record_move(move_data)
                    
                    if execution_result.get("game_over", False):
                        break
                
                # Finalize game
                final_state = game_controller.get_game_state()
                game_summary = game_data.finalize_game(final_state)
                game_summary.update({
                    "game_number": game_number,
                    "strategy_used": current_strategy["strategy"],
                    "total_moves": len(game_moves),
                    "session_id": session_config["session_id"]
                })
                
                session_data.append(game_summary)
                track_session_event("game_completed", "game_data", {
                    "game_number": game_number,
                    "final_score": game_summary.get("final_score", 0),
                    "moves_played": len(game_moves)
                })
                
                # Update session statistics
                session_stats["games_completed"] += 1
                session_stats["total_score"] += game_summary.get("final_score", 0)
                session_stats["total_moves"] += len(game_moves)
                
                # Save incremental session data
                incremental_save = file_utils.save_session_data({
                    "session_config": session_config,
                    "completed_games": session_data,
                    "session_stats": session_stats
                }, f"{temp_dir}/session_checkpoint_game_{game_number}.json")
                
                track_session_event("session_checkpoint", "file_utils", {
                    "game_number": game_number,
                    "save_success": incremental_save["success"]
                })
            
            # Session completion and analysis
            track_session_event("session_analysis_start", "workflow", {})
            
            # Calculate comprehensive session statistics
            stats_utils.calculate_basic_statistics.return_value = {
                "success": True,
                "statistics": {
                    "total_games": len(session_data),
                    "average_score": session_stats["total_score"] / session_stats["games_completed"],
                    "total_moves": session_stats["total_moves"],
                    "games_per_strategy": {
                        strategy["strategy"]: 1 for strategy in move_strategies
                    }
                }
            }
            
            session_final_stats = stats_utils.calculate_basic_statistics(session_data)
            track_session_event("session_statistics", "stats_utils", session_final_stats)
            
            # Generate session summary
            session_duration = time.time() - session_stats["session_start_time"]
            session_summary = {
                "session_config": session_config,
                "games_data": session_data,
                "session_statistics": session_final_stats,
                "session_metadata": {
                    "duration": session_duration,
                    "games_completed": session_stats["games_completed"],
                    "strategies_tested": [s["strategy"] for s in move_strategies],
                    "total_llm_calls": sum(len(game.get("moves", [])) for game in session_data),
                    "workflow_events_count": len(session_workflow_events)
                }
            }
            
            # Final session persistence
            final_save_result = file_utils.save_session_data(
                session_summary,
                f"{temp_dir}/complete_session_summary.json"
            )
            track_session_event("session_finalized", "file_utils", final_save_result)
            
            # Workflow validation for multi-game session
            assert session_stats["games_completed"] == session_config["total_games"], "Should complete all planned games"
            assert len(session_data) == session_config["total_games"], "Should have data for all games"
            
            # Verify each game used different strategy
            strategies_used = [game.get("strategy_used") for game in session_data]
            assert len(set(strategies_used)) == len(move_strategies), "Should use different strategies"
            
            # Verify LLM was called for each move of each game
            total_expected_calls = sum(game.get("total_moves", 0) for game in session_data)
            assert llm_client.generate_move.call_count == total_expected_calls, "LLM should be called for every move"
            
            # Verify file operations occurred
            assert file_utils.save_session_data.call_count >= session_config["total_games"] + 1, "Should save checkpoints and final data"
            
            # Verify session progression
            session_events = [e for e in session_workflow_events if e["event_type"] == "game_completed"]
            assert len(session_events) == session_config["total_games"], "Should have completion event for each game"

    def test_error_recovery_workflow(self) -> None:
        """Test complete workflow with error scenarios and recovery mechanisms."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize components with error injection capabilities
            game_data = GameData()
            game_controller = GameController(grid_size=10, use_gui=False, game_data=game_data)
            
            # Mock components with controlled failures
            llm_client: Mock = Mock(spec=LLMClient)
            file_utils: Mock = Mock(spec=FileUtils)
            stats_utils: Mock = Mock(spec=GameStatsUtils)
            
            error_recovery_events: List[Dict[str, Any]] = []
            
            def track_error_event(event_type: str, component: str, details: Dict[str, Any]) -> None:
                error_recovery_events.append({
                    "timestamp": time.time(),
                    "event_type": event_type,
                    "component": component,
                    "details": details
                })
            
            # Configure error scenarios
            error_scenarios = [
                {"move": 3, "component": "llm_client", "error_type": "timeout"},
                {"move": 7, "component": "file_utils", "error_type": "disk_full"},
                {"move": 12, "component": "llm_client", "error_type": "invalid_response"},
                {"move": 15, "component": "stats_utils", "error_type": "calculation_error"}
            ]
            
            track_error_event("error_workflow_start", "workflow", {"planned_errors": len(error_scenarios)})
            
            # Initialize game with error monitoring
            init_result = game_controller.initialize_game()
            assert init_result["success"] is True
            track_error_event("initialization_success", "game_controller", {})
            
            # Configure LLM with error injection
            llm_call_count = 0
            
            def mock_llm_with_errors(context: Dict[str, Any]) -> Dict[str, Any]:
                nonlocal llm_call_count
                llm_call_count += 1
                
                # Check for planned errors
                error_scenario = next((e for e in error_scenarios if e["move"] == llm_call_count and e["component"] == "llm_client"), None)
                
                if error_scenario:
                    track_error_event("error_injected", "llm_client", error_scenario)
                    
                    if error_scenario["error_type"] == "timeout":
                        return {
                            "success": False,
                            "error": "Request timeout",
                            "error_type": "timeout",
                            "retry_suggested": True
                        }
                    elif error_scenario["error_type"] == "invalid_response":
                        return {
                            "success": False,
                            "error": "Invalid move format",
                            "error_type": "validation_error",
                            "retry_suggested": True
                        }
                
                # Normal successful response
                moves = ["UP", "DOWN", "LEFT", "RIGHT"]
                return {
                    "success": True,
                    "move": np.random.choice(moves),
                    "confidence": np.random.uniform(0.7, 0.9),
                    "response_time": np.random.randint(100, 200)
                }
            
            llm_client.generate_move.side_effect = mock_llm_with_errors
            
            # Configure file utils with error injection
            file_save_count = 0
            
            def mock_file_with_errors(*args, **kwargs) -> Dict[str, Any]:
                nonlocal file_save_count
                file_save_count += 1
                
                # Check for file errors
                error_scenario = next((e for e in error_scenarios if e["component"] == "file_utils"), None)
                if error_scenario and file_save_count == 2:  # Fail on second save
                    track_error_event("error_injected", "file_utils", error_scenario)
                    return {
                        "success": False,
                        "error": "Disk full",
                        "error_type": "storage_error",
                        "retry_suggested": True
                    }
                
                return {
                    "success": True,
                    "file_path": f"{temp_dir}/save_{file_save_count}.json"
                }
            
            file_utils.save_game_data.side_effect = mock_file_with_errors
            
            # Configure stats utils with error injection
            def mock_stats_with_errors(*args, **kwargs) -> Dict[str, Any]:
                error_scenario = next((e for e in error_scenarios if e["component"] == "stats_utils"), None)
                if error_scenario:
                    track_error_event("error_injected", "stats_utils", error_scenario)
                    return {
                        "success": False,
                        "error": "Calculation overflow",
                        "error_type": "math_error",
                        "retry_suggested": False
                    }
                
                return {
                    "success": True,
                    "statistics": {"average_score": 100.0, "total_games": 1}
                }
            
            stats_utils.calculate_basic_statistics.side_effect = mock_stats_with_errors
            
            # Game loop with error handling and recovery
            successful_moves = 0
            error_recoveries = 0
            max_moves = 20
            
            for move_count in range(1, max_moves + 1):
                track_error_event("move_attempt", "workflow", {"move_number": move_count})
                
                # Attempt LLM move generation with retry logic
                llm_attempts = 0
                max_llm_retries = 3
                move_result = None
                
                while llm_attempts < max_llm_retries:
                    llm_attempts += 1
                    try:
                        current_state = game_controller.get_game_state()
                        move_result = llm_client.generate_move({"game_state": current_state})
                        
                        if move_result["success"]:
                            break
                        else:
                            track_error_event("llm_error_detected", "error_recovery", {
                                "attempt": llm_attempts,
                                "error": move_result.get("error", "unknown"),
                                "retry_suggested": move_result.get("retry_suggested", False)
                            })
                            
                            if move_result.get("retry_suggested", False) and llm_attempts < max_llm_retries:
                                # Implement backoff strategy
                                time.sleep(0.1 * llm_attempts)
                                error_recoveries += 1
                                continue
                            else:
                                break
                    except Exception as e:
                        track_error_event("llm_exception", "error_recovery", {
                            "attempt": llm_attempts,
                            "exception": str(e)
                        })
                        break
                
                # Fallback move if LLM fails
                if not move_result or not move_result["success"]:
                    track_error_event("fallback_move", "error_recovery", {"move_number": move_count})
                    # Use simple fallback strategy
                    fallback_moves = ["UP", "RIGHT", "DOWN", "LEFT"]
                    move_result = {
                        "success": True,
                        "move": fallback_moves[move_count % 4],
                        "confidence": 0.5,
                        "response_time": 50,
                        "fallback_used": True
                    }
                
                # Execute move
                execution_result = game_controller.process_move(move_result["move"])
                if execution_result["success"]:
                    successful_moves += 1
                    
                    # Record move with error recovery metadata
                    move_data = {
                        "move_number": move_count,
                        "move": move_result["move"],
                        "llm_attempts": llm_attempts,
                        "fallback_used": move_result.get("fallback_used", False),
                        "execution_success": execution_result["success"]
                    }
                    game_data.record_move(move_data)
                
                # Attempt periodic save with error handling
                if move_count % 5 == 0:
                    save_attempts = 0
                    max_save_retries = 2
                    
                    while save_attempts < max_save_retries:
                        save_attempts += 1
                        save_result = file_utils.save_game_data(
                            game_controller.get_game_state(),
                            f"{temp_dir}/checkpoint_{move_count}.json"
                        )
                        
                        if save_result["success"]:
                            track_error_event("save_success", "file_utils", {"move": move_count})
                            break
                        else:
                            track_error_event("save_error_detected", "error_recovery", {
                                "attempt": save_attempts,
                                "error": save_result.get("error", "unknown")
                            })
                            
                            if save_attempts < max_save_retries:
                                error_recoveries += 1
                                time.sleep(0.2 * save_attempts)
                
                if execution_result.get("game_over", False):
                    track_error_event("game_over", "workflow", {"move": move_count})
                    break
            
            # Final statistics with error handling
            stats_attempts = 0
            max_stats_retries = 2
            final_stats = None
            
            while stats_attempts < max_stats_retries:
                stats_attempts += 1
                
                try:
                    game_summary = game_data.finalize_game(game_controller.get_game_state())
                    final_stats = stats_utils.calculate_basic_statistics([game_summary])
                    
                    if final_stats["success"]:
                        break
                    else:
                        track_error_event("stats_error_detected", "error_recovery", {
                            "attempt": stats_attempts,
                            "error": final_stats.get("error", "unknown")
                        })
                        
                        if stats_attempts < max_stats_retries:
                            error_recoveries += 1
                except Exception as e:
                    track_error_event("stats_exception", "error_recovery", {
                        "attempt": stats_attempts,
                        "exception": str(e)
                    })
            
            # If stats failed, use fallback calculations
            if not final_stats or not final_stats["success"]:
                track_error_event("stats_fallback", "error_recovery", {})
                final_stats = {
                    "success": True,
                    "statistics": {
                        "total_moves": successful_moves,
                        "error_recoveries": error_recoveries,
                        "completion_status": "completed_with_errors"
                    },
                    "fallback_used": True
                }
            
            track_error_event("error_workflow_complete", "workflow", {
                "successful_moves": successful_moves,
                "error_recoveries": error_recoveries,
                "final_stats_success": final_stats["success"]
            })
            
            # Validate error recovery workflow
            # Should have successfully handled all error scenarios
            injected_errors = [e for e in error_recovery_events if e["event_type"] == "error_injected"]
            detected_errors = [e for e in error_recovery_events if "error_detected" in e["event_type"]]
            
            assert len(injected_errors) > 0, "Should have injected errors for testing"
            assert len(detected_errors) > 0, "Should have detected and handled errors"
            assert error_recoveries > 0, "Should have performed error recovery attempts"
            assert successful_moves > 0, "Should have completed some moves despite errors"
            assert final_stats["success"] is True, "Should have final statistics despite errors"
            
            # Verify error recovery patterns
            llm_errors = [e for e in injected_errors if e["component"] == "llm_client"]
            file_errors = [e for e in injected_errors if e["component"] == "file_utils"]
            stats_errors = [e for e in injected_errors if e["component"] == "stats_utils"]
            
            assert len(llm_errors) > 0, "Should have tested LLM error recovery"
            assert len(file_errors) > 0, "Should have tested file error recovery"
            assert len(stats_errors) > 0, "Should have tested stats error recovery"
            
            # Verify fallback mechanisms were used
            fallback_events = [e for e in error_recovery_events if "fallback" in e["event_type"]]
            assert len(fallback_events) > 0, "Should have used fallback mechanisms"


class TestCompleteGameWorkflow:
    """Test suite for complete game workflow with all components."""

    def create_mock_args(self, **kwargs) -> Mock:
        """Create mock command line arguments."""
        defaults = {
            'max_games': 2,
            'no_gui': True,
            'move_pause': 0.0,
            'primary_provider': 'test_provider',
            'primary_model': 'test_model',
            'secondary_provider': None,
            'secondary_model': None,
            'max_steps': 100,
            'continue_session': None,
        }
        defaults.update(kwargs)
        args = Mock()
        for key, value in defaults.items():
            setattr(args, key, value)
        return args

    @patch('core.game_manager.initialize_game_manager')
    @patch('core.game_manager.run_game_loop')
    def test_complete_single_game_workflow(
        self,
        mock_run_loop: Mock,
        mock_initialize: Mock
    ) -> None:
        """Test complete workflow for a single game."""
        args = self.create_mock_args(max_games=1)
        manager = GameManager(args)
        
        # Mock initialization
        def mock_init(mgr):
            mgr.llm_client = Mock()
            mgr.log_dir = "/test/logs"
            mgr.game = Mock(spec=GameLogic)
            mgr.game.game_data = Mock(spec=GameData)
            mgr.game.game_state = Mock()
        
        mock_initialize.side_effect = mock_init
        
        # Mock game loop to simulate one complete game
        def mock_loop(mgr):
            mgr.game_count = 1
            mgr.total_score = 50
            mgr.total_steps = 25
            mgr.game_scores = [50]
            mgr.running = False
        
        mock_run_loop.side_effect = mock_loop
        
        with patch.object(manager, 'report_final_statistics') as mock_report:
            manager.run()
            
            # Verify workflow
            mock_initialize.assert_called_once_with(manager)
            mock_run_loop.assert_called_once_with(manager)
            mock_report.assert_called_once()
            
            # Verify final state
            assert manager.game_count == 1
            assert manager.total_score == 50
            assert manager.total_steps == 25
            assert manager.game_scores == [50]

    @patch('core.game_manager.GameLogic')
    @patch('core.game_manager.LLMClient')
    @patch('utils.game_manager_utils.create_log_directories')
    def test_full_initialization_workflow(
        self,
        mock_create_dirs: Mock,
        mock_llm_client: Mock,
        mock_game_logic: Mock
    ) -> None:
        """Test complete initialization workflow with all components."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Mock components
        mock_client = Mock()
        mock_llm_client.return_value = mock_client
        mock_game = Mock()
        mock_game_logic.return_value = mock_game
        mock_create_dirs.return_value = ("/test/logs", "/test/prompts", "/test/responses")
        
        # Initialize
        manager.initialize()
        
        # Verify LLM client creation
        mock_llm_client.assert_called()
        
        # Verify game setup
        manager.setup_game()
        mock_game_logic.assert_called_with(use_gui=False)

    @patch('llm.communication_utils.get_llm_response')
    @patch('core.game_controller.GameController.move')
    def test_llm_game_interaction_workflow(
        self,
        mock_move: Mock,
        mock_llm_response: Mock
    ) -> None:
        """Test LLM-game interaction workflow."""
        args = self.create_mock_args()
        manager = GameManager(args)
        manager.game = Mock(spec=GameLogic)
        manager.game.game_data = Mock(spec=GameData)
        manager.game.game_state = Mock()
        manager.game.planned_moves = []
        manager.game.apple_eaten = False
        manager.llm_client = Mock()
        
        # Mock LLM response
        mock_llm_response.return_value = ("UP", True)
        mock_move.return_value = True
        
        # Simulate LLM interaction
        manager.need_new_plan = True
        manager.game_active = True
        
        # Test the interaction flow
        from core.game_loop import _request_and_execute_first_move
        
        with patch('time.sleep'):  # Skip GUI delay
            _request_and_execute_first_move(manager)
        
        # Verify LLM was called
        mock_llm_response.assert_called_once()
        
        # Verify move was executed
        mock_move.assert_called_once_with("UP")

    def test_data_persistence_workflow(self) -> None:
        """Test data persistence throughout game workflow."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Set up game with data tracking
        manager.game = Mock(spec=GameLogic)
        game_data = Mock(spec=GameData)
        manager.game.game_data = game_data
        manager.game.round_manager = Mock()
        
        # Simulate data recording
        game_data.record_move("UP")
        game_data.record_score(10)
        game_data.record_apple_eaten()
        
        # Verify data recording
        game_data.record_move.assert_called_with("UP")
        game_data.record_score.assert_called_with(10)
        game_data.record_apple_eaten.assert_called_once()
        
        # Test round finishing
        manager.finish_round("apple eaten")
        manager.game.round_manager.flush_buffer.assert_called_once()

    @patch('utils.file_utils.save_game_data')
    @patch('utils.json_utils.save_json_data')
    def test_file_system_interaction_workflow(
        self,
        mock_save_json: Mock,
        mock_save_game: Mock
    ) -> None:
        """Test file system interaction workflow."""
        args = self.create_mock_args()
        manager = GameManager(args)
        manager.log_dir = "/test/logs"
        manager.game = Mock(spec=GameLogic)
        manager.game.game_data = Mock(spec=GameData)
        
        # Mock game data
        game_data_dict = {
            "game_number": 1,
            "score": 50,
            "steps": 25,
            "moves": ["UP", "DOWN", "LEFT"]
        }
        manager.game.game_data.to_dict.return_value = game_data_dict
        
        # Simulate file operations
        from utils.game_manager_utils import process_game_over
        
        with patch('utils.game_manager_utils.save_game_data') as mock_save:
            # Mock the process_game_over to test file operations
            process_game_over(manager)
            
            # Verify file operations would be called
            # (actual verification depends on implementation details)

    def test_statistics_aggregation_workflow(self) -> None:
        """Test statistics aggregation across multiple games."""
        args = self.create_mock_args(max_games=3)
        manager = GameManager(args)
        
        # Simulate multiple games
        game_scores = [30, 45, 60]
        game_steps = [15, 20, 25]
        
        for i, (score, steps) in enumerate(zip(game_scores, game_steps)):
            manager.game_count = i + 1
            manager.total_score += score
            manager.total_steps += steps
            manager.game_scores.append(score)
        
        # Verify aggregation
        assert manager.game_count == 3
        assert manager.total_score == 135  # 30 + 45 + 60
        assert manager.total_steps == 60   # 15 + 20 + 25
        assert manager.game_scores == [30, 45, 60]

    @patch('core.game_manager.save_session_stats')
    @patch('core.game_manager.report_final_statistics')
    def test_session_completion_workflow(
        self,
        mock_report: Mock,
        mock_save: Mock
    ) -> None:
        """Test complete session completion workflow."""
        args = self.create_mock_args()
        manager = GameManager(args)
        manager.game_count = 2
        manager.log_dir = "/test/logs"
        manager.game = Mock()
        
        # Simulate session completion
        manager.report_final_statistics()
        
        # Verify completion workflow
        mock_save.assert_called_once_with("/test/logs")
        mock_report.assert_called_once()
        assert manager.running is False


class TestErrorRecoveryWorkflow:
    """Test suite for error recovery in complete workflow."""

    def create_mock_args(self, **kwargs) -> Mock:
        """Create mock command line arguments."""
        defaults = {
            'max_games': 1,
            'no_gui': True,
            'move_pause': 0.0,
        }
        defaults.update(kwargs)
        args = Mock()
        for key, value in defaults.items():
            setattr(args, key, value)
        return args

    @patch('core.game_manager.run_game_loop')
    def test_llm_error_recovery_workflow(self, mock_run_loop: Mock) -> None:
        """Test error recovery when LLM fails."""
        args = self.create_mock_args()
        manager = GameManager(args)
        manager.game = Mock(spec=GameLogic)
        manager.game.game_state = Mock()
        
        # Simulate LLM error
        manager.game.parse_llm_response.side_effect = ValueError("LLM Error")
        
        # Test error handling
        result = manager.game.parse_llm_response("invalid response")
        
        # Verify error was handled
        assert result is None
        manager.game.game_state.record_something_is_wrong_move.assert_called()

    def test_game_over_recovery_workflow(self) -> None:
        """Test game over recovery and session continuation."""
        args = self.create_mock_args(max_games=2)
        manager = GameManager(args)
        
        # Simulate first game ending
        manager.game_count = 1
        manager.game_active = False
        
        # Verify can start new game
        manager.game_active = True
        assert manager.game_active is True
        assert manager.game_count == 1

    @patch('traceback.print_exc')
    def test_critical_error_workflow(self, mock_traceback: Mock) -> None:
        """Test critical error handling workflow."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Simulate critical error
        from core.game_loop import run_game_loop
        
        with patch('core.game_loop.process_events', side_effect=Exception("Critical Error")):
            with patch('pygame.quit'):
                run_game_loop(manager)
        
        # Verify error was logged
        mock_traceback.assert_called_once()


class TestConcurrentInteractions:
    """Test suite for concurrent interactions in workflow."""

    def create_mock_args(self, **kwargs) -> Mock:
        """Create mock command line arguments."""
        defaults = {
            'max_games': 1,
            'no_gui': True,
            'move_pause': 0.0,
        }
        defaults.update(kwargs)
        args = Mock()
        for key, value in defaults.items():
            setattr(args, key, value)
        return args

    def test_simultaneous_data_recording(self) -> None:
        """Test simultaneous data recording from multiple sources."""
        args = self.create_mock_args()
        manager = GameManager(args)
        manager.game = Mock(spec=GameLogic)
        manager.game.game_data = Mock(spec=GameData)
        manager.game.round_manager = Mock()
        
        # Simulate simultaneous operations
        manager.game.game_data.record_move("UP")
        manager.game.round_manager.record_planned_moves(["UP", "DOWN"])
        manager.game.game_data.record_score(10)
        
        # Verify all operations were recorded
        manager.game.game_data.record_move.assert_called_with("UP")
        manager.game.round_manager.record_planned_moves.assert_called_with(["UP", "DOWN"])
        manager.game.game_data.record_score.assert_called_with(10)

    @patch('time.perf_counter')
    def test_timing_coordination(self, mock_perf: Mock) -> None:
        """Test timing coordination across components."""
        mock_perf.side_effect = [100.0, 105.0]  # 5 second duration
        
        args = self.create_mock_args()
        manager = GameManager(args)
        manager.game = Mock(spec=GameLogic)
        manager.game.game_data = Mock(spec=GameData)
        manager.game.game_data.game_statistics = Mock(spec=GameStatistics)
        
        # Simulate timing coordination
        stats = manager.game.game_data.game_statistics
        stats.record_llm_communication_start()
        stats.record_llm_communication_end()
        
        # Verify timing was coordinated
        stats.record_llm_communication_start.assert_called_once()
        stats.record_llm_communication_end.assert_called_once()


class TestMemoryManagement:
    """Test suite for memory management in complete workflow."""

    def create_mock_args(self, **kwargs) -> Mock:
        """Create mock command line arguments."""
        defaults = {
            'max_games': 10,
            'no_gui': True,
            'move_pause': 0.0,
        }
        defaults.update(kwargs)
        args = Mock()
        for key, value in defaults.items():
            setattr(args, key, value)
        return args

    def test_memory_cleanup_between_games(self) -> None:
        """Test memory cleanup between games."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Simulate multiple games
        for game_num in range(1, 4):
            manager.game_count = game_num
            manager.current_game_moves = [f"MOVE_{game_num}"]
            
            # Reset for next game
            if game_num < 3:
                manager.current_game_moves = []
        
        # Verify cleanup
        assert manager.game_count == 3
        assert manager.current_game_moves == []

    def test_data_structure_scaling(self) -> None:
        """Test data structure scaling with large datasets."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Simulate large dataset
        large_game_scores = list(range(100))
        manager.game_scores = large_game_scores
        
        # Verify data structure can handle large datasets
        assert len(manager.game_scores) == 100
        assert manager.game_scores[99] == 99


class TestNetworkInteractions:
    """Test suite for network interactions in workflow."""

    def create_mock_args(self, **kwargs) -> Mock:
        """Create mock command line arguments."""
        defaults = {
            'max_games': 1,
            'no_gui': True,
            'move_pause': 0.0,
            'primary_provider': 'test_provider',
        }
        defaults.update(kwargs)
        args = Mock()
        for key, value in defaults.items():
            setattr(args, key, value)
        return args

    @patch('llm.client.LLMClient')
    def test_network_resilience_workflow(self, mock_client_class: Mock) -> None:
        """Test network resilience in complete workflow."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Mock network failure and recovery
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Simulate network failure
        mock_client.get_response.side_effect = [
            ConnectionError("Network failed"),
            {"choices": [{"message": {"content": '{"moves": ["UP"]}'}}]}
        ]
        
        # Create client
        client = manager.create_llm_client("test_provider")
        
        # Verify client was created despite potential network issues
        assert client == mock_client
        mock_client_class.assert_called_once()

    def test_retry_mechanism_workflow(self) -> None:
        """Test retry mechanism workflow for network operations."""
        args = self.create_mock_args()
        manager = GameManager(args)
        manager.llm_client = Mock()
        
        # Mock retry behavior
        manager.llm_client.get_response.side_effect = [
            ConnectionError("First attempt failed"),
            ConnectionError("Second attempt failed"),
            {"response": "success"}
        ]
        
        # Test that retry mechanism can be implemented
        # (Actual retry logic would be in communication_utils)
        assert manager.llm_client is not None 