"""
Tests for LLMClient ↔ GameController interactions.

Focuses on testing how LLMClient and GameController coordinate in
game state to prompt conversion, move application, and decision loops.
"""

import pytest
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch
import numpy as np
from numpy.typing import NDArray

from core.game_controller import GameController
from llm.client import LLMClient
from llm.providers.base_provider import BaseLLMProvider


class TestClientControllerInteractions:
    """Test interactions between LLMClient and GameController."""

    def test_game_state_prompt_generation_cycle(self) -> None:
        """Test complete cycle of game state to prompt to move application."""
        controller: GameController = GameController(grid_size=12, use_gui=False)
        
        # Mock LLM provider
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Realistic LLM responses based on game state
        def generate_contextual_response(prompt: str) -> str:
            # Extract game state from prompt
            if "snake head" in prompt.lower():
                # Respond based on context
                if "score: 0" in prompt:
                    return '{"moves": ["UP"], "reasoning": "Starting game, move up"}'
                elif "apple" in prompt.lower():
                    return '{"moves": ["RIGHT", "DOWN"], "reasoning": "Moving toward apple"}'
                else:
                    return '{"moves": ["LEFT"], "reasoning": "Exploring"}'
            return '{"moves": ["UP"], "reasoning": "Default move"}'
        
        mock_provider.generate_response.side_effect = generate_contextual_response
        mock_provider.get_last_token_count.return_value = {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250}
        
        client: LLMClient = LLMClient(mock_provider)
        
        # Test complete interaction cycles
        interaction_cycles: List[Dict[str, Any]] = []
        
        for cycle in range(10):
            cycle_start: float = time.time()
            
            # 1. Capture current game state
            game_state: Dict[str, Any] = {
                "score": controller.score,
                "steps": controller.steps,
                "snake_length": controller.snake_length,
                "snake_head": controller.snake_positions[-1].tolist(),
                "apple_position": controller.apple_position.tolist(),
                "grid_size": controller.grid_size
            }
            
            # 2. Generate context-aware prompt
            prompt: str = f"""
            Current Game State:
            - Score: {game_state['score']}
            - Steps: {game_state['steps']}
            - Snake Length: {game_state['snake_length']}
            - Snake Head Position: {game_state['snake_head']}
            - Apple Position: {game_state['apple_position']}
            - Grid Size: {game_state['grid_size']}
            
            Please analyze the current state and provide the next move(s).
            Consider the distance to the apple and avoid collisions.
            """
            
            # 3. Get LLM response
            try:
                llm_response: str = client.generate_response(prompt)
                
                # 4. Parse response
                try:
                    parsed_response: Dict[str, Any] = json.loads(llm_response)
                    moves: List[str] = parsed_response.get("moves", [])
                    reasoning: str = parsed_response.get("reasoning", "No reasoning provided")
                except json.JSONDecodeError:
                    moves = ["UP"]  # Fallback
                    reasoning = "JSON parse failed"
                
                # 5. Apply moves to controller
                moves_applied: List[str] = []
                total_collision: bool = False
                total_apples_eaten: int = 0
                
                for move in moves:
                    if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
                        collision: bool
                        apple_eaten: bool
                        collision, apple_eaten = controller.make_move(move.upper())
                        
                        moves_applied.append(move.upper())
                        
                        if apple_eaten:
                            total_apples_eaten += 1
                        
                        if collision:
                            total_collision = True
                            break
                
                # 6. Record interaction cycle
                cycle_end: float = time.time()
                
                interaction_cycles.append({
                    "cycle": cycle,
                    "initial_state": game_state,
                    "prompt_length": len(prompt),
                    "llm_response": llm_response,
                    "parsed_moves": moves,
                    "applied_moves": moves_applied,
                    "reasoning": reasoning,
                    "collision": total_collision,
                    "apples_eaten": total_apples_eaten,
                    "final_score": controller.score,
                    "cycle_duration": cycle_end - cycle_start
                })
                
                # 7. Handle game over
                if total_collision:
                    interaction_cycles[-1]["game_over"] = True
                    controller.reset()
                
            except Exception as e:
                interaction_cycles.append({
                    "cycle": cycle,
                    "initial_state": game_state,
                    "error": str(e),
                    "cycle_duration": time.time() - cycle_start
                })
        
        # Verify interaction cycles
        successful_cycles = [c for c in interaction_cycles if "error" not in c]
        assert len(successful_cycles) > 0, "Should have successful interaction cycles"
        
        # Verify state progression
        for i, cycle in enumerate(successful_cycles):
            if i > 0:
                prev_cycle = successful_cycles[i-1]
                
                # Score should not decrease (unless game reset)
                if not cycle.get("game_over", False) and not prev_cycle.get("game_over", False):
                    assert cycle["final_score"] >= prev_cycle["final_score"], \
                        f"Score decreased unexpectedly in cycle {i}"
            
            # Verify moves were applied
            assert len(cycle["applied_moves"]) > 0, f"No moves applied in cycle {i}"
            assert all(move in ["UP", "DOWN", "LEFT", "RIGHT"] for move in cycle["applied_moves"]), \
                f"Invalid moves in cycle {i}"
        
        # Verify performance
        avg_cycle_duration = sum(c["cycle_duration"] for c in successful_cycles) / len(successful_cycles)
        assert avg_cycle_duration < 0.1, f"Interaction cycles too slow: {avg_cycle_duration}s"

    def test_decision_loop_error_handling(self) -> None:
        """Test error handling in the LLM-Controller decision loop."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Mock provider with various error conditions
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        
        # Error injection scenarios
        error_scenarios: List[Tuple[int, Exception, str]] = [
            (2, ConnectionError("Network timeout"), "network_error"),
            (4, ValueError("Invalid API response"), "api_error"),
            (6, TimeoutError("LLM request timeout"), "timeout_error"),
            (8, RuntimeError("Provider unavailable"), "provider_error"),
        ]
        
        error_responses: Dict[int, Exception] = {step: error for step, error, _ in error_scenarios}
        normal_response = '{"moves": ["RIGHT"], "reasoning": "Safe move"}'
        
        def error_injecting_response(prompt: str) -> str:
            # Get current call count (approximate)
            call_count = mock_provider.generate_response.call_count
            if call_count in error_responses:
                raise error_responses[call_count]
            return normal_response
        
        mock_provider.generate_response.side_effect = error_injecting_response
        mock_provider.get_last_token_count.return_value = {"total_tokens": 100}
        
        client: LLMClient = LLMClient(mock_provider)
        
        # Test decision loop with error handling
        decision_results: List[Dict[str, Any]] = []
        
        for step in range(10):
            step_start: float = time.time()
            
            # Generate prompt based on current state
            state_prompt = f"""
            Game Step {step}:
            Score: {controller.score}
            Snake Head: {controller.snake_positions[-1].tolist()}
            Apple: {controller.apple_position.tolist()}
            
            What should the snake do next?
            """
            
            # Decision loop with error handling
            move_decided: Optional[str] = None
            error_encountered: Optional[str] = None
            fallback_used: bool = False
            
            try:
                # Primary decision path
                response: str = client.generate_response(state_prompt)
                parsed: Dict[str, Any] = json.loads(response)
                moves: List[str] = parsed.get("moves", [])
                
                if moves and moves[0].upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    move_decided = moves[0].upper()
                else:
                    raise ValueError("No valid moves in response")
                
            except json.JSONDecodeError:
                # JSON parsing fallback
                move_decided = "UP"  # Safe fallback
                fallback_used = True
                error_encountered = "json_parse_error"
                
            except (ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
                # LLM error fallback
                # Use simple heuristic: move toward apple
                snake_head = controller.snake_positions[-1]
                apple_pos = controller.apple_position
                
                if apple_pos[0] > snake_head[0]:
                    move_decided = "RIGHT"
                elif apple_pos[0] < snake_head[0]:
                    move_decided = "LEFT"
                elif apple_pos[1] > snake_head[1]:
                    move_decided = "DOWN"
                else:
                    move_decided = "UP"
                
                fallback_used = True
                error_encountered = type(e).__name__
            
            # Apply decided move
            if move_decided:
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move(move_decided)
                
                decision_results.append({
                    "step": step,
                    "move": move_decided,
                    "collision": collision,
                    "apple_eaten": apple_eaten,
                    "fallback_used": fallback_used,
                    "error": error_encountered,
                    "final_score": controller.score,
                    "duration": time.time() - step_start
                })
                
                if collision:
                    controller.reset()
            else:
                decision_results.append({
                    "step": step,
                    "error": "no_move_decided",
                    "duration": time.time() - step_start
                })
        
        # Verify error handling effectiveness
        successful_decisions = [r for r in decision_results if "move" in r]
        assert len(successful_decisions) == 10, "Should make decisions even with errors"
        
        # Verify fallbacks were used appropriately
        fallback_decisions = [r for r in successful_decisions if r.get("fallback_used", False)]
        error_steps = [step for step, _, _ in error_scenarios]
        
        # Should have fallbacks for error scenarios
        for result in fallback_decisions:
            if result["step"] in error_steps:
                assert result["error"] is not None, f"Error should be recorded for step {result['step']}"
        
        # Verify game continued functioning
        total_score = max(r["final_score"] for r in successful_decisions)
        assert total_score > 0, "Should have achieved some score despite errors"

    def test_move_validation_coordination(self) -> None:
        """Test coordination between LLM move suggestions and controller validation."""
        controller: GameController = GameController(grid_size=8, use_gui=False)
        
        # Mock provider with various move suggestions
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.get_last_token_count.return_value = {"total_tokens": 75}
        
        # Test various move suggestion scenarios
        move_scenarios: List[Tuple[str, List[str], str]] = [
            ("valid_single", ["UP"], "normal"),
            ("valid_sequence", ["RIGHT", "DOWN"], "sequence"),
            ("invalid_move", ["INVALID"], "invalid"),
            ("mixed_moves", ["UP", "INVALID", "RIGHT"], "mixed"),
            ("empty_moves", [], "empty"),
            ("case_issues", ["up", "RIGHT"], "case"),
            ("extra_data", ["LEFT", "UP", "DOWN", "RIGHT", "UP"], "long_sequence"),
        ]
        
        scenario_index: int = 0
        
        def scenario_response(prompt: str) -> str:
            nonlocal scenario_index
            scenario_name, moves, _ = move_scenarios[scenario_index % len(move_scenarios)]
            scenario_index += 1
            
            return json.dumps({
                "moves": moves,
                "scenario": scenario_name,
                "reasoning": f"Testing {scenario_name} scenario"
            })
        
        mock_provider.generate_response.side_effect = scenario_response
        client: LLMClient = LLMClient(mock_provider)
        
        # Test move validation coordination
        validation_results: List[Dict[str, Any]] = []
        
        for i, (scenario_name, expected_moves, scenario_type) in enumerate(move_scenarios):
            # Get LLM suggestion
            prompt = f"Scenario {i}: What moves should the snake make?"
            response: str = client.generate_response(prompt)
            parsed: Dict[str, Any] = json.loads(response)
            suggested_moves: List[str] = parsed.get("moves", [])
            
            # Validate and apply moves
            valid_moves: List[str] = []
            invalid_moves: List[str] = []
            applied_moves: List[str] = []
            
            for move in suggested_moves:
                # Validate move format
                if isinstance(move, str) and move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    valid_moves.append(move.upper())
                else:
                    invalid_moves.append(str(move))
            
            # Apply valid moves to controller
            for move in valid_moves:
                try:
                    collision: bool
                    apple_eaten: bool
                    collision, apple_eaten = controller.make_move(move)
                    applied_moves.append(move)
                    
                    if collision:
                        break
                except Exception as e:
                    # Move was valid but couldn't be applied (shouldn't happen normally)
                    break
            
            validation_results.append({
                "scenario": scenario_name,
                "suggested_moves": suggested_moves,
                "valid_moves": valid_moves,
                "invalid_moves": invalid_moves,
                "applied_moves": applied_moves,
                "moves_applied_count": len(applied_moves),
                "controller_score": controller.score,
                "scenario_type": scenario_type
            })
        
        # Verify validation coordination
        for result in validation_results:
            scenario_type = result["scenario_type"]
            
            if scenario_type == "normal":
                assert len(result["valid_moves"]) == 1, "Should have one valid move"
                assert len(result["invalid_moves"]) == 0, "Should have no invalid moves"
                assert len(result["applied_moves"]) >= 1, "Should apply the valid move"
            
            elif scenario_type == "sequence":
                assert len(result["valid_moves"]) >= 2, "Should have multiple valid moves"
                assert len(result["invalid_moves"]) == 0, "Should have no invalid moves"
            
            elif scenario_type == "invalid":
                assert len(result["valid_moves"]) == 0, "Should have no valid moves"
                assert len(result["invalid_moves"]) >= 1, "Should detect invalid moves"
                assert len(result["applied_moves"]) == 0, "Should not apply invalid moves"
            
            elif scenario_type == "mixed":
                assert len(result["valid_moves"]) >= 1, "Should extract valid moves"
                assert len(result["invalid_moves"]) >= 1, "Should detect invalid moves"
                assert len(result["applied_moves"]) >= 1, "Should apply valid moves only"
            
            elif scenario_type == "empty":
                assert len(result["valid_moves"]) == 0, "Should have no moves"
                assert len(result["applied_moves"]) == 0, "Should not apply any moves"
            
            elif scenario_type == "case":
                # Should normalize case
                assert all(move.isupper() for move in result["valid_moves"]), "Should normalize to uppercase"

    def test_performance_optimization_coordination(self) -> None:
        """Test performance optimization coordination between LLM and controller."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Mock high-performance provider
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.get_last_token_count.return_value = {"total_tokens": 50}
        
        # Fast, simple responses for performance testing
        def fast_response(prompt: str) -> str:
            # Minimize processing time
            return '{"moves": ["UP"]}'
        
        mock_provider.generate_response.side_effect = fast_response
        client: LLMClient = LLMClient(mock_provider)
        
        # Performance optimization test
        performance_results: List[Dict[str, Any]] = []
        
        # Test high-frequency LLM-Controller interaction
        start_time: float = time.time()
        
        for iteration in range(100):  # High frequency
            iteration_start: float = time.time()
            
            # Minimal prompt for speed
            prompt = f"Move {iteration}: UP/DOWN/LEFT/RIGHT?"
            
            # LLM call
            llm_start: float = time.time()
            response: str = client.generate_response(prompt)
            llm_end: float = time.time()
            
            # Controller application
            controller_start: float = time.time()
            try:
                parsed: Dict[str, Any] = json.loads(response)
                move: str = parsed.get("moves", ["UP"])[0]
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move(move)
            except:
                collision, apple_eaten = controller.make_move("UP")
            
            controller_end: float = time.time()
            
            iteration_end: float = time.time()
            
            performance_results.append({
                "iteration": iteration,
                "llm_time": llm_end - llm_start,
                "controller_time": controller_end - controller_start,
                "total_time": iteration_end - iteration_start,
                "collision": collision
            })
            
            if collision:
                controller.reset()
        
        end_time: float = time.time()
        total_duration: float = end_time - start_time
        
        # Verify performance optimization
        avg_llm_time: float = sum(r["llm_time"] for r in performance_results) / len(performance_results)
        avg_controller_time: float = sum(r["controller_time"] for r in performance_results) / len(performance_results)
        avg_total_time: float = sum(r["total_time"] for r in performance_results) / len(performance_results)
        
        # Performance assertions
        assert avg_llm_time < 0.01, f"LLM calls too slow: {avg_llm_time}s"
        assert avg_controller_time < 0.001, f"Controller operations too slow: {avg_controller_time}s"
        assert avg_total_time < 0.02, f"Total interaction too slow: {avg_total_time}s"
        assert total_duration < 5.0, f"Overall test too slow: {total_duration}s"
        
        # Verify functionality maintained
        successful_iterations = [r for r in performance_results if not r.get("error")]
        assert len(successful_iterations) == 100, "Should complete all iterations successfully"
        
        # Verify controller remained functional
        final_score = controller.score
        assert final_score >= 0, "Controller should remain in valid state"

    def test_state_consistency_long_interaction(self) -> None:
        """Test state consistency over long LLM-Controller interaction sessions."""
        controller: GameController = GameController(grid_size=12, use_gui=False)
        
        # Mock provider with state-aware responses
        mock_provider: Mock = Mock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.get_last_token_count.return_value = {"total_tokens": 100}
        
        # State-tracking response generation
        interaction_count: int = 0
        
        def state_aware_response(prompt: str) -> str:
            nonlocal interaction_count
            interaction_count += 1
            
            # Vary strategy based on interaction count
            if interaction_count % 10 == 0:
                return '{"moves": ["DOWN", "LEFT"], "strategy": "exploration"}'
            elif interaction_count % 7 == 0:
                return '{"moves": ["UP", "UP", "RIGHT"], "strategy": "aggressive"}'
            else:
                return '{"moves": ["RIGHT"], "strategy": "conservative"}'
        
        mock_provider.generate_response.side_effect = state_aware_response
        client: LLMClient = LLMClient(mock_provider)
        
        # Long interaction session
        session_snapshots: List[Dict[str, Any]] = []
        
        for round_num in range(20):  # Long session
            round_moves: List[str] = []
            round_start_score: int = controller.score
            round_start_steps: int = controller.steps
            
            # Multiple interactions per round
            for interaction in range(10):
                # Create detailed state prompt
                state_prompt = f"""
                Round {round_num}, Interaction {interaction}
                Current State:
                - Score: {controller.score}
                - Steps: {controller.steps}
                - Snake Length: {controller.snake_length}
                - Head Position: {controller.snake_positions[-1].tolist()}
                - Apple Position: {controller.apple_position.tolist()}
                
                Previous moves this round: {round_moves}
                Please provide next move considering game progression.
                """
                
                # Get LLM decision
                response: str = client.generate_response(state_prompt)
                
                try:
                    parsed: Dict[str, Any] = json.loads(response)
                    moves: List[str] = parsed.get("moves", ["UP"])
                    strategy: str = parsed.get("strategy", "unknown")
                    
                    # Apply moves
                    for move in moves:
                        if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
                            collision: bool
                            apple_eaten: bool
                            collision, apple_eaten = controller.make_move(move.upper())
                            round_moves.append(move.upper())
                            
                            if collision:
                                break
                    
                    if collision:
                        break
                        
                except json.JSONDecodeError:
                    # Fallback
                    collision, apple_eaten = controller.make_move("UP")
                    round_moves.append("UP")
            
            # Record round snapshot
            session_snapshots.append({
                "round": round_num,
                "moves_made": round_moves,
                "score_gained": controller.score - round_start_score,
                "steps_taken": controller.steps - round_start_steps,
                "final_snake_length": controller.snake_length,
                "interaction_count": interaction_count,
                "controller_state": {
                    "score": controller.score,
                    "steps": controller.steps,
                    "snake_positions": controller.snake_positions.tolist(),
                    "apple_position": controller.apple_position.tolist()
                }
            })
            
            # Reset for next round
            controller.reset()
        
        # Verify long-term consistency
        assert len(session_snapshots) == 20, "Should have all round snapshots"
        
        # Verify state progression
        for i, snapshot in enumerate(session_snapshots):
            # Each round should have activity
            assert len(snapshot["moves_made"]) > 0, f"Round {i} had no moves"
            assert snapshot["steps_taken"] > 0, f"Round {i} had no steps"
            
            # Controller state should be valid
            state = snapshot["controller_state"]
            assert state["score"] >= 0, f"Invalid score in round {i}"
            assert state["steps"] >= 0, f"Invalid steps in round {i}"
            assert len(state["snake_positions"]) >= 1, f"Invalid snake in round {i}"
            
            # Apple should be within bounds
            apple_pos = state["apple_position"]
            assert 0 <= apple_pos[0] < 12, f"Apple x out of bounds in round {i}"
            assert 0 <= apple_pos[1] < 12, f"Apple y out of bounds in round {i}"
        
        # Verify interaction count progression
        total_interactions = session_snapshots[-1]["interaction_count"]
        assert total_interactions > 100, "Should have many interactions over long session"
        
        # Verify no state corruption over time
        final_snapshot = session_snapshots[-1]
        assert final_snapshot["controller_state"]["score"] >= 0, "Final state should be valid" 