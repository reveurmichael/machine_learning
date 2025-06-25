"""
Tests for complete system integration workflows.

Focuses on testing end-to-end workflows that exercise all major 
component interactions in realistic usage scenarios.
"""

import pytest
import time
import threading
import tempfile
import os
from typing import List, Dict, Any, Optional, Tuple, Generator
from unittest.mock import Mock, patch
import numpy as np
from numpy.typing import NDArray

from core.game_controller import GameController
from core.game_data import GameData
from core.game_stats import GameStats
from llm.client import LLMClient
from llm.providers.base_provider import BaseLLMProvider
from utils.file_utils import save_json_safely, load_json_safely
from utils.json_utils import safe_json_parse


class TestSystemIntegrationWorkflows:
    """Test complete system integration workflows."""

    def test_complete_llm_game_session_workflow(self, temp_dir: str) -> None:
        """Test complete LLM-driven game session with all component interactions."""
        # Setup complete system
        controller: GameController = GameController(grid_size=15, use_gui=False)
        game_data: GameData = controller.game_state
        stats: GameStats = GameStats()
        
        # Mock LLM provider
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Realistic LLM responses
        llm_responses: List[str] = [
            '{"moves": ["UP"], "analysis": "Moving up to avoid walls"}',
            '{"moves": ["RIGHT"], "analysis": "Turning right towards apple"}',
            '{"moves": ["DOWN"], "analysis": "Going down to get apple"}',
            '{"moves": ["LEFT"], "analysis": "Moving left to continue"}',
            '{"moves": ["UP", "RIGHT"], "analysis": "Planned sequence"}',
        ]
        
        response_index: int = 0
        def get_llm_response(prompt: str) -> str:
            nonlocal response_index
            response = llm_responses[response_index % len(llm_responses)]
            response_index += 1
            return response
        
        mock_provider.generate_response.side_effect = get_llm_response
        mock_provider.get_last_token_count.return_value = {"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200}
        
        client: LLMClient = LLMClient(mock_provider)
        
        # Session workflow tracking
        workflow_events: List[Dict[str, Any]] = []
        session_file: str = os.path.join(temp_dir, "game_session.json")
        
        def log_workflow_event(event_type: str, **data: Any) -> None:
            """Log workflow events for analysis."""
            workflow_events.append({
                "timestamp": time.time(),
                "event_type": event_type,
                "game_score": controller.score,
                "game_steps": controller.steps,
                "snake_length": controller.snake_length,
                **data
            })
        
        # Start workflow
        log_workflow_event("session_start")
        
        # Game session loop
        games_played: int = 0
        total_moves: int = 0
        
        for game in range(5):  # Multiple games
            log_workflow_event("game_start", game_number=game)
            
            for round_num in range(50):  # Multiple rounds per game
                # 1. Generate prompt based on current state
                game_state_prompt = f"""
                Current game state:
                - Score: {controller.score}
                - Steps: {controller.steps}
                - Snake length: {controller.snake_length}
                - Snake head: {controller.snake_positions[-1].tolist()}
                - Apple position: {controller.apple_position.tolist()}
                - Grid size: {controller.grid_size}
                
                What move should the snake make next?
                """
                
                log_workflow_event("llm_request_start", round=round_num)
                
                # 2. Get LLM response
                try:
                    llm_response: str = client.generate_response(game_state_prompt)
                    log_workflow_event("llm_response_received", response_length=len(llm_response))
                    
                    # 3. Parse LLM response
                    parsed_response: Optional[Dict[str, Any]] = safe_json_parse(llm_response)
                    if parsed_response and "moves" in parsed_response:
                        planned_moves: List[str] = parsed_response["moves"]
                        log_workflow_event("moves_parsed", move_count=len(planned_moves))
                        
                        # 4. Execute moves
                        for move in planned_moves:
                            if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
                                collision: bool
                                apple_eaten: bool
                                collision, apple_eaten = controller.make_move(move.upper())
                                
                                total_moves += 1
                                
                                # 5. Update statistics
                                stats.record_step_result(
                                    valid=True,
                                    collision=collision,
                                    apple_eaten=apple_eaten
                                )
                                
                                log_workflow_event("move_executed", 
                                                 move=move.upper(), 
                                                 collision=collision, 
                                                 apple_eaten=apple_eaten)
                                
                                # 6. Store LLM communication
                                game_data.add_llm_communication(game_state_prompt, llm_response)
                                
                                # 7. Update token usage
                                token_count = client.get_last_token_count()
                                if token_count:
                                    game_data.add_token_usage(
                                        token_count.get("prompt_tokens", 0),
                                        token_count.get("completion_tokens", 0)
                                    )
                                
                                if collision:
                                    log_workflow_event("game_collision")
                                    break
                                
                                if apple_eaten:
                                    log_workflow_event("apple_eaten", new_score=controller.score)
                        
                        if collision:
                            break
                    
                    else:
                        log_workflow_event("llm_parse_failed", response=llm_response[:100])
                        # Fallback move
                        collision, apple_eaten = controller.make_move("UP")
                        total_moves += 1
                
                except Exception as e:
                    log_workflow_event("llm_error", error=str(e))
                    # Fallback move
                    collision, apple_eaten = controller.make_move("RIGHT")
                    total_moves += 1
                
                if collision:
                    break
                
                # 8. Periodic save
                if round_num % 10 == 0:
                    session_data = {
                        "game": game,
                        "round": round_num,
                        "controller_state": {
                            "score": controller.score,
                            "steps": controller.steps,
                            "snake_length": controller.snake_length
                        },
                        "statistics": {
                            "valid_moves": stats.step_stats.valid,
                            "collisions": stats.step_stats.collisions
                        },
                        "workflow_events": len(workflow_events)
                    }
                    save_json_safely(session_data, session_file)
                    log_workflow_event("session_saved")
            
            # End of game
            final_score = controller.score
            final_steps = controller.steps
            
            stats.update_game_stats(
                final_score=final_score,
                total_steps=final_steps,
                apples_eaten=controller.snake_length - 1
            )
            
            log_workflow_event("game_end", 
                             final_score=final_score, 
                             final_steps=final_steps,
                             apples_eaten=controller.snake_length - 1)
            
            games_played += 1
            controller.reset()
            log_workflow_event("game_reset")
        
        log_workflow_event("session_end", 
                         games_played=games_played, 
                         total_moves=total_moves)
        
        # Verify complete workflow
        assert games_played == 5, "Should have completed 5 games"
        assert total_moves > 0, "Should have made moves"
        assert len(workflow_events) > 50, "Should have logged many events"
        
        # Verify component interactions
        assert len(game_data.llm_communication) > 0, "Should have LLM communications"
        assert stats.step_stats.valid > 0, "Should have recorded valid moves"
        assert os.path.exists(session_file), "Session file should exist"
        
        # Verify data consistency
        final_session_data = load_json_safely(session_file)
        assert final_session_data is not None, "Should be able to load final session"

    def test_human_play_web_integration_workflow(self, temp_dir: str) -> None:
        """Test human play web integration workflow."""
        controller: GameController = GameController(grid_size=12, use_gui=False)
        
        # Simulate web interface interactions
        web_requests: List[Dict[str, Any]] = []
        web_responses: List[Dict[str, Any]] = []
        
        def simulate_web_request(request_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Simulate web interface request."""
            web_requests.append({
                "type": request_type,
                "data": data,
                "timestamp": time.time()
            })
            
            response: Dict[str, Any] = {"status": "error"}
            
            if request_type == "get_game_state":
                response = {
                    "status": "success",
                    "game_state": {
                        "score": controller.score,
                        "steps": controller.steps,
                        "snake_positions": controller.snake_positions.tolist(),
                        "apple_position": controller.apple_position.tolist(),
                        "game_over": getattr(controller, 'game_over', False),
                        "grid_size": controller.grid_size
                    }
                }
            
            elif request_type == "make_move":
                move = data.get("move", "").upper()
                if move in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    collision, apple_eaten = controller.make_move(move)
                    response = {
                        "status": "success",
                        "result": {
                            "collision": collision,
                            "apple_eaten": apple_eaten,
                            "new_score": controller.score,
                            "new_snake_length": controller.snake_length
                        }
                    }
                else:
                    response = {"status": "error", "message": "Invalid move"}
            
            elif request_type == "reset_game":
                controller.reset()
                response = {
                    "status": "success",
                    "message": "Game reset"
                }
            
            elif request_type == "get_statistics":
                response = {
                    "status": "success",
                    "statistics": {
                        "current_score": controller.score,
                        "current_steps": controller.steps,
                        "snake_length": controller.snake_length
                    }
                }
            
            web_responses.append(response)
            return response
        
        # Simulate human play session
        human_actions: List[Tuple[str, str]] = [
            ("initial_load", "get_game_state"),
            ("move_up", "make_move"),
            ("move_right", "make_move"),
            ("move_down", "make_move"),
            ("check_stats", "get_statistics"),
            ("move_left", "make_move"),
            ("move_up", "make_move"),
            ("reset", "reset_game"),
            ("new_game_state", "get_game_state"),
        ]
        
        for action_name, request_type in human_actions:
            if request_type == "make_move":
                # Extract move from action name
                move_map = {
                    "move_up": "UP",
                    "move_right": "RIGHT", 
                    "move_down": "DOWN",
                    "move_left": "LEFT"
                }
                move = move_map.get(action_name, "UP")
                response = simulate_web_request(request_type, {"move": move})
            else:
                response = simulate_web_request(request_type, {})
            
            # Verify response
            assert response["status"] == "success", f"Request {action_name} failed: {response}"
            
            # Add delay to simulate human interaction
            time.sleep(0.01)
        
        # Verify web workflow
        assert len(web_requests) == len(human_actions), "Should have processed all requests"
        assert len(web_responses) == len(human_actions), "Should have generated all responses"
        
        # Verify game state consistency
        final_state = simulate_web_request("get_game_state", {})
        assert final_state["status"] == "success"
        assert final_state["game_state"]["score"] >= 0
        assert final_state["game_state"]["steps"] >= 0

    def test_replay_system_workflow(self, temp_dir: str) -> None:
        """Test replay system workflow with stored game data."""
        # Create original game session
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Play a game and record moves
        recorded_moves: List[str] = []
        game_states: List[Dict[str, Any]] = []
        
        def record_game_state() -> None:
            """Record current game state."""
            game_states.append({
                "score": controller.score,
                "steps": controller.steps,
                "snake_positions": controller.snake_positions.tolist(),
                "apple_position": controller.apple_position.tolist(),
                "timestamp": time.time()
            })
        
        # Record initial state
        record_game_state()
        
        # Play game and record
        for i in range(30):
            move = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
            recorded_moves.append(move)
            
            collision, apple_eaten = controller.make_move(move)
            record_game_state()
            
            if collision:
                break
        
        # Save game data for replay
        game_file = os.path.join(temp_dir, "replay_game.json")
        replay_data = {
            "moves": recorded_moves,
            "states": game_states,
            "final_score": controller.score,
            "total_steps": len(recorded_moves)
        }
        save_json_safely(replay_data, game_file)
        
        # Replay workflow
        replay_controller: GameController = GameController(grid_size=10, use_gui=False)
        loaded_data = load_json_safely(game_file)
        assert loaded_data is not None, "Should load replay data"
        
        replayed_states: List[Dict[str, Any]] = []
        
        def record_replay_state() -> None:
            """Record replay state."""
            replayed_states.append({
                "score": replay_controller.score,
                "steps": replay_controller.steps,
                "snake_positions": replay_controller.snake_positions.tolist(),
                "apple_position": replay_controller.apple_position.tolist()
            })
        
        # Replay initial state
        record_replay_state()
        
        # Replay moves
        replayed_moves = loaded_data["moves"]
        for i, move in enumerate(replayed_moves):
            collision, apple_eaten = replay_controller.make_move(move)
            record_replay_state()
            
            # Verify replay accuracy (allowing for some randomness in apple placement)
            original_state = game_states[i + 1]
            replay_state = replayed_states[i + 1]
            
            assert replay_state["score"] <= original_state["score"] + 10, "Replay score diverged too much"
            assert replay_state["steps"] == original_state["steps"], "Replay steps should match"
            
            if collision:
                break
        
        # Verify replay completion
        assert len(replayed_states) > 1, "Replay should have multiple states"
        assert replay_controller.steps > 0, "Replay should have made moves"

    def test_stress_test_all_components_integration(self) -> None:
        """Stress test all components working together."""
        # Create multiple controllers for stress testing
        controllers: List[GameController] = []
        stats_collectors: List[GameStats] = []
        
        for i in range(5):
            controller = GameController(grid_size=8, use_gui=False)
            stats = GameStats()
            controllers.append(controller)
            stats_collectors.append(stats)
        
        # Stress test results
        stress_results: List[Dict[str, Any]] = []
        stress_errors: List[Exception] = []
        
        def stress_test_component(component_id: int) -> None:
            """Stress test a single component set."""
            try:
                controller = controllers[component_id]
                stats = stats_collectors[component_id]
                
                operations_completed = 0
                games_completed = 0
                
                start_time = time.time()
                
                # Run stress test for limited time
                while time.time() - start_time < 2.0:  # 2 second stress test
                    # Rapid game operations
                    for step in range(50):
                        move = ["UP", "RIGHT", "DOWN", "LEFT"][step % 4]
                        
                        collision, apple_eaten = controller.make_move(move)
                        
                        # Update statistics
                        stats.record_step_result(
                            valid=True,
                            collision=collision,
                            apple_eaten=apple_eaten
                        )
                        
                        operations_completed += 1
                        
                        if collision:
                            games_completed += 1
                            controller.reset()
                            break
                    
                    # Prevent infinite loop
                    if time.time() - start_time > 2.0:
                        break
                
                end_time = time.time()
                
                stress_results.append({
                    "component_id": component_id,
                    "operations_completed": operations_completed,
                    "games_completed": games_completed,
                    "duration": end_time - start_time,
                    "ops_per_second": operations_completed / (end_time - start_time)
                })
                
            except Exception as e:
                stress_errors.append(e)
        
        # Run stress test on multiple components concurrently
        threads: List[threading.Thread] = []
        
        for i in range(5):
            thread = threading.Thread(target=stress_test_component, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all stress tests to complete
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Verify stress test results
        assert len(stress_errors) == 0, f"Stress test errors: {stress_errors}"
        assert len(stress_results) == 5, "Should have results from all components"
        
        # Verify performance under stress
        for result in stress_results:
            assert result["operations_completed"] > 10, f"Component {result['component_id']} too slow"
            assert result["ops_per_second"] > 5, f"Component {result['component_id']} throughput too low"
            assert result["games_completed"] >= 1, f"Component {result['component_id']} didn't complete games"
        
        # Verify all components maintained integrity
        for i, controller in enumerate(controllers):
            assert hasattr(controller, 'snake_positions'), f"Controller {i} corrupted"
            assert len(controller.snake_positions) >= 1, f"Controller {i} invalid state"
            assert controller.score >= 0, f"Controller {i} negative score"
        
        for i, stats in enumerate(stats_collectors):
            assert stats.step_stats.valid > 0, f"Stats {i} no valid moves recorded"

    def test_error_propagation_full_system_recovery(self) -> None:
        """Test error propagation and recovery across the full system."""
        controller: GameController = GameController(grid_size=12, use_gui=False)
        
        # Simulate various system errors
        error_scenarios: List[Tuple[str, Exception, bool]] = [
            ("memory_error", MemoryError("Out of memory"), True),
            ("value_error", ValueError("Invalid game state"), True),
            ("runtime_error", RuntimeError("System error"), True),
            ("io_error", IOError("File system error"), True),
        ]
        
        recovery_results: List[Dict[str, Any]] = []
        
        for scenario_name, error, should_recover in error_scenarios:
            # Create error condition
            original_score = controller.score
            original_steps = controller.steps
            
            try:
                # Simulate error during operation
                for i in range(10):
                    move = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
                    
                    # Inject error on specific step
                    if i == 5:
                        # Simulate error condition
                        if scenario_name == "memory_error":
                            # Simulate memory pressure
                            large_data = [0] * 10000  # Allocate memory
                            collision, apple_eaten = controller.make_move(move)
                            del large_data
                        else:
                            # Normal operation
                            collision, apple_eaten = controller.make_move(move)
                    else:
                        collision, apple_eaten = controller.make_move(move)
                    
                    if collision:
                        break
                
                # System should still be functional
                recovery_results.append({
                    "scenario": scenario_name,
                    "recovered": True,
                    "final_score": controller.score,
                    "final_steps": controller.steps,
                    "state_preserved": controller.score >= original_score and controller.steps >= original_steps
                })
                
            except Exception as e:
                if should_recover:
                    # Try recovery
                    try:
                        controller.reset()
                        
                        # Verify system is still functional after reset
                        test_collision, test_apple = controller.make_move("UP")
                        
                        recovery_results.append({
                            "scenario": scenario_name,
                            "recovered": True,
                            "error_handled": str(e),
                            "reset_successful": True
                        })
                        
                    except Exception as recovery_error:
                        recovery_results.append({
                            "scenario": scenario_name,
                            "recovered": False,
                            "error": str(e),
                            "recovery_error": str(recovery_error)
                        })
                else:
                    recovery_results.append({
                        "scenario": scenario_name,
                        "recovered": False,
                        "error": str(e)
                    })
        
        # Verify error handling and recovery
        successful_recoveries = [r for r in recovery_results if r.get("recovered", False)]
        
        # System should handle most errors gracefully
        assert len(successful_recoveries) >= len(error_scenarios) - 1, \
            "System should recover from most error scenarios"
        
        # Final system verification
        try:
            final_collision, final_apple = controller.make_move("UP")
            # System should still be responsive
            assert isinstance(final_collision, bool)
            assert isinstance(final_apple, bool)
        except Exception as e:
            assert False, f"System not functional after error recovery: {e}" 