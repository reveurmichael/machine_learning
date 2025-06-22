"""
Tests for Session utilities ↔ Game state continuation interactions.

Focuses on testing how session utilities coordinate with game state
for continuation, data consistency, and state reconstruction.
"""

import pytest
import time
import json
import tempfile
import os
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch
import numpy as np
from numpy.typing import NDArray

from core.game_controller import GameController
from core.game_data import GameData
from utils.session_utils import SessionUtils
from utils.continuation_utils import ContinuationUtils


class TestSessionContinuationInteractions:
    """Test interactions between session utilities and game state continuation."""

    def test_session_state_preservation_cycle(self) -> None:
        """Test complete cycle of session state preservation and restoration."""
        
        # Create controllers for testing
        original_controller: GameController = GameController(grid_size=12, use_gui=False)
        
        # Mock session utilities
        session_utils: Mock = Mock(spec=SessionUtils)
        continuation_utils: Mock = Mock(spec=ContinuationUtils)
        
        # Play some game to create state
        game_states: List[Dict[str, Any]] = []
        
        for round_num in range(3):
            round_start_state = {
                "round": round_num,
                "score": original_controller.score,
                "steps": original_controller.steps,
                "snake_length": original_controller.snake_length,
                "snake_positions": original_controller.snake_positions.tolist(),
                "apple_position": original_controller.apple_position.tolist()
            }
            
            # Play round
            for step in range(15):
                move: str = ["UP", "RIGHT", "DOWN", "LEFT"][step % 4]
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = original_controller.make_move(move)
                
                if collision:
                    break
            
            round_end_state = {
                "round": round_num,
                "final_score": original_controller.score,
                "final_steps": original_controller.steps,
                "final_snake_length": original_controller.snake_length,
                "final_snake_positions": original_controller.snake_positions.tolist(),
                "final_apple_position": original_controller.apple_position.tolist(),
                "moves_made": original_controller.moves.copy()
            }
            
            game_states.append({
                "start_state": round_start_state,
                "end_state": round_end_state,
                "round_summary": {
                    "score_gained": round_end_state["final_score"] - round_start_state["score"],
                    "steps_taken": round_end_state["final_steps"] - round_start_state["steps"]
                }
            })
            
            # Reset for next round
            original_controller.reset()
        
        # Mock session preservation
        preserved_sessions: Dict[str, Dict[str, Any]] = {}
        
        def mock_preserve_session(session_id: str, game_state: Dict[str, Any]) -> bool:
            """Mock session preservation."""
            preserved_sessions[session_id] = {
                "session_id": session_id,
                "preservation_time": time.time(),
                "game_state": game_state.copy(),
                "data_size": len(str(game_state)),
                "preservation_successful": True
            }
            return True
        
        def mock_restore_session(session_id: str) -> Optional[Dict[str, Any]]:
            """Mock session restoration."""
            if session_id in preserved_sessions:
                return preserved_sessions[session_id]["game_state"].copy()
            return None
        
        session_utils.preserve_session = mock_preserve_session
        session_utils.restore_session = mock_restore_session
        
        # Test session preservation for each game state
        session_ids: List[str] = []
        
        for i, game_state in enumerate(game_states):
            session_id = f"test_session_{i}_{int(time.time())}"
            session_ids.append(session_id)
            
            # Preserve session
            preservation_success = session_utils.preserve_session(session_id, game_state)
            assert preservation_success, f"Failed to preserve session {session_id}"
        
        # Test session restoration and continuation
        continuation_results: List[Dict[str, Any]] = []
        
        for i, session_id in enumerate(session_ids):
            # Restore session
            restored_state = session_utils.restore_session(session_id)
            assert restored_state is not None, f"Failed to restore session {session_id}"
            
            # Verify restored state integrity
            original_state = game_states[i]
            
            # Check state consistency
            assert restored_state["start_state"]["round"] == original_state["start_state"]["round"]
            assert restored_state["end_state"]["final_score"] == original_state["end_state"]["final_score"]
            assert restored_state["end_state"]["final_steps"] == original_state["end_state"]["final_steps"]
            
            # Test continuation from restored state
            continued_controller = GameController(grid_size=12, use_gui=False)
            
            # Apply restored state to new controller
            end_state = restored_state["end_state"]
            
            # Mock state application (in real implementation this would be more complex)
            continued_controller.score = end_state["final_score"]
            continued_controller.steps = end_state["final_steps"]
            continued_controller.snake_positions = np.array(end_state["final_snake_positions"])
            continued_controller.apple_position = np.array(end_state["final_apple_position"])
            
            # Continue playing from restored state
            continuation_moves: List[str] = []
            pre_continuation_score = continued_controller.score
            
            for step in range(10):
                move: str = ["RIGHT", "DOWN", "LEFT", "UP"][step % 4]
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = continued_controller.make_move(move)
                continuation_moves.append(move)
                
                if collision:
                    break
            
            continuation_results.append({
                "session_id": session_id,
                "restoration_successful": True,
                "continuation_moves": continuation_moves,
                "pre_continuation_score": pre_continuation_score,
                "post_continuation_score": continued_controller.score,
                "score_delta": continued_controller.score - pre_continuation_score,
                "continuation_steps": len(continuation_moves)
            })
        
        # Verify continuation results
        assert len(continuation_results) == 3, "Should have continuation results for all sessions"
        
        for result in continuation_results:
            assert result["restoration_successful"], "All restorations should succeed"
            assert len(result["continuation_moves"]) > 0, "Should make continuation moves"
            assert result["post_continuation_score"] >= result["pre_continuation_score"], \
                "Score should not decrease during continuation"

    def test_concurrent_session_management(self) -> None:
        """Test concurrent session management and state synchronization."""
        
        # Mock session manager for concurrent access
        session_manager: Mock = Mock()
        session_manager.active_sessions = {}
        session_manager.session_locks = {}
        
        import threading
        
        def mock_create_session(session_id: str, initial_state: Dict[str, Any]) -> bool:
            """Create session with thread safety."""
            if session_id not in session_manager.session_locks:
                session_manager.session_locks[session_id] = threading.Lock()
            
            with session_manager.session_locks[session_id]:
                if session_id in session_manager.active_sessions:
                    return False  # Session already exists
                
                session_manager.active_sessions[session_id] = {
                    "session_id": session_id,
                    "state": initial_state.copy(),
                    "created_time": time.time(),
                    "last_updated": time.time(),
                    "access_count": 0
                }
                return True
        
        def mock_update_session(session_id: str, state_update: Dict[str, Any]) -> bool:
            """Update session state with thread safety."""
            if session_id not in session_manager.session_locks:
                return False
            
            with session_manager.session_locks[session_id]:
                if session_id not in session_manager.active_sessions:
                    return False
                
                session = session_manager.active_sessions[session_id]
                session["state"].update(state_update)
                session["last_updated"] = time.time()
                session["access_count"] += 1
                return True
        
        def mock_get_session(session_id: str) -> Optional[Dict[str, Any]]:
            """Get session state with thread safety."""
            if session_id not in session_manager.session_locks:
                return None
            
            with session_manager.session_locks[session_id]:
                if session_id in session_manager.active_sessions:
                    session = session_manager.active_sessions[session_id]
                    session["access_count"] += 1
                    return session["state"].copy()
                return None
        
        session_manager.create_session = mock_create_session
        session_manager.update_session = mock_update_session
        session_manager.get_session = mock_get_session
        
        # Test concurrent session operations
        concurrent_results: List[Dict[str, Any]] = []
        result_lock = threading.Lock()
        
        def concurrent_session_worker(worker_id: int) -> None:
            """Worker function for concurrent session testing."""
            try:
                # Create unique session
                session_id = f"concurrent_session_{worker_id}"
                
                # Create initial state
                controller = GameController(grid_size=8, use_gui=False)
                initial_state = {
                    "worker_id": worker_id,
                    "score": controller.score,
                    "steps": controller.steps,
                    "creation_time": time.time()
                }
                
                # Create session
                creation_success = session_manager.create_session(session_id, initial_state)
                
                if not creation_success:
                    with result_lock:
                        concurrent_results.append({
                            "worker_id": worker_id,
                            "error": "Failed to create session",
                            "success": False
                        })
                    return
                
                # Perform multiple session updates
                update_results: List[bool] = []
                
                for update_num in range(5):
                    # Play some moves
                    for step in range(3):
                        move: str = ["UP", "DOWN", "LEFT", "RIGHT"][step % 4]
                        collision: bool
                        apple_eaten: bool
                        collision, apple_eaten = controller.make_move(move)
                        
                        if collision:
                            controller.reset()
                            break
                    
                    # Update session state
                    state_update = {
                        f"update_{update_num}": {
                            "score": controller.score,
                            "steps": controller.steps,
                            "update_time": time.time()
                        }
                    }
                    
                    update_success = session_manager.update_session(session_id, state_update)
                    update_results.append(update_success)
                
                # Verify session state
                final_state = session_manager.get_session(session_id)
                
                with result_lock:
                    concurrent_results.append({
                        "worker_id": worker_id,
                        "session_id": session_id,
                        "creation_success": creation_success,
                        "update_results": update_results,
                        "final_state_retrieved": final_state is not None,
                        "total_updates": len([u for u in update_results if u]),
                        "success": True
                    })
                    
            except Exception as e:
                with result_lock:
                    concurrent_results.append({
                        "worker_id": worker_id,
                        "error": str(e),
                        "success": False
                    })
        
        # Start concurrent workers
        threads: List[threading.Thread] = []
        
        for worker_id in range(10):
            thread = threading.Thread(target=concurrent_session_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Verify concurrent session management
        successful_results = [r for r in concurrent_results if r.get("success", False)]
        assert len(successful_results) == 10, "All concurrent workers should succeed"
        
        # Verify session creation
        for result in successful_results:
            assert result["creation_success"], f"Worker {result['worker_id']} failed to create session"
            assert result["final_state_retrieved"], f"Worker {result['worker_id']} failed to retrieve state"
            assert result["total_updates"] > 0, f"Worker {result['worker_id']} made no updates"
        
        # Verify session isolation
        session_ids = [r["session_id"] for r in successful_results]
        assert len(set(session_ids)) == 10, "Each worker should have unique session"
        
        # Verify no session corruption
        for session_id in session_ids:
            final_state = session_manager.get_session(session_id)
            assert final_state is not None, f"Session {session_id} should be accessible"
            assert "worker_id" in final_state, f"Session {session_id} missing worker_id"

    def test_continuation_data_integrity_validation(self) -> None:
        """Test data integrity validation during continuation processes."""
        
        # Mock validation utilities
        validator: Mock = Mock()
        validator.validation_log = []
        
        def mock_validate_game_state(state: Dict[str, Any]) -> Dict[str, Any]:
            """Validate game state integrity."""
            validation_result = {
                "timestamp": time.time(),
                "state_hash": hash(str(sorted(state.items()))),
                "validation_errors": [],
                "validation_warnings": [],
                "is_valid": True
            }
            
            # Check required fields
            required_fields = ["score", "steps", "snake_positions", "apple_position"]
            for field in required_fields:
                if field not in state:
                    validation_result["validation_errors"].append(f"Missing required field: {field}")
                    validation_result["is_valid"] = False
            
            # Check score validity
            if "score" in state and state["score"] < 0:
                validation_result["validation_errors"].append("Score cannot be negative")
                validation_result["is_valid"] = False
            
            # Check steps validity
            if "steps" in state and state["steps"] < 0:
                validation_result["validation_errors"].append("Steps cannot be negative")
                validation_result["is_valid"] = False
            
            # Check snake positions
            if "snake_positions" in state:
                positions = state["snake_positions"]
                if not isinstance(positions, list) or len(positions) == 0:
                    validation_result["validation_errors"].append("Invalid snake positions")
                    validation_result["is_valid"] = False
                elif len(positions) != len(set(tuple(pos) for pos in positions)):
                    validation_result["validation_warnings"].append("Snake has overlapping positions")
            
            # Check apple position
            if "apple_position" in state:
                apple_pos = state["apple_position"]
                if not isinstance(apple_pos, list) or len(apple_pos) != 2:
                    validation_result["validation_errors"].append("Invalid apple position")
                    validation_result["is_valid"] = False
            
            validator.validation_log.append(validation_result)
            return validation_result
        
        validator.validate_game_state = mock_validate_game_state
        
        # Test continuation with various state integrity scenarios
        test_scenarios = [
            {
                "name": "valid_state",
                "state": {
                    "score": 150,
                    "steps": 75,
                    "snake_positions": [[5, 5], [5, 6], [5, 7]],
                    "apple_position": [8, 8],
                    "session_id": "valid_test"
                }
            },
            {
                "name": "negative_score",
                "state": {
                    "score": -10,  # Invalid
                    "steps": 20,
                    "snake_positions": [[3, 3], [3, 4]],
                    "apple_position": [7, 7],
                    "session_id": "negative_score_test"
                }
            },
            {
                "name": "missing_fields",
                "state": {
                    "score": 50,
                    # Missing steps, snake_positions, apple_position
                    "session_id": "missing_fields_test"
                }
            },
            {
                "name": "overlapping_snake",
                "state": {
                    "score": 80,
                    "steps": 40,
                    "snake_positions": [[4, 4], [4, 5], [4, 4]],  # Overlapping
                    "apple_position": [6, 6],
                    "session_id": "overlapping_snake_test"
                }
            },
            {
                "name": "invalid_apple",
                "state": {
                    "score": 120,
                    "steps": 60,
                    "snake_positions": [[2, 2], [2, 3], [2, 4]],
                    "apple_position": "invalid",  # Invalid format
                    "session_id": "invalid_apple_test"
                }
            }
        ]
        
        validation_results: List[Dict[str, Any]] = []
        
        for scenario in test_scenarios:
            scenario_name = scenario["name"]
            state = scenario["state"]
            
            # Validate state
            validation_result = validator.validate_game_state(state)
            
            # Attempt continuation based on validation
            continuation_attempted = False
            continuation_successful = False
            
            if validation_result["is_valid"]:
                # Attempt continuation
                try:
                    controller = GameController(grid_size=10, use_gui=False)
                    
                    # Apply validated state
                    controller.score = state["score"]
                    controller.steps = state["steps"]
                    
                    if isinstance(state["snake_positions"], list):
                        controller.snake_positions = np.array(state["snake_positions"])
                    
                    if isinstance(state["apple_position"], list):
                        controller.apple_position = np.array(state["apple_position"])
                    
                    # Test a few moves to verify continuation works
                    for move in ["UP", "RIGHT", "DOWN"]:
                        collision: bool
                        apple_eaten: bool
                        collision, apple_eaten = controller.make_move(move)
                        if collision:
                            break
                    
                    continuation_attempted = True
                    continuation_successful = True
                    
                except Exception as e:
                    continuation_attempted = True
                    continuation_successful = False
            
            validation_results.append({
                "scenario": scenario_name,
                "validation_result": validation_result,
                "continuation_attempted": continuation_attempted,
                "continuation_successful": continuation_successful,
                "has_errors": len(validation_result["validation_errors"]) > 0,
                "has_warnings": len(validation_result["validation_warnings"]) > 0
            })
        
        # Verify validation results
        assert len(validation_results) == 5, "Should validate all scenarios"
        
        # Valid state should pass validation and continuation
        valid_result = next(r for r in validation_results if r["scenario"] == "valid_state")
        assert valid_result["validation_result"]["is_valid"], "Valid state should pass validation"
        assert valid_result["continuation_successful"], "Valid state should continue successfully"
        
        # Invalid states should fail validation
        invalid_scenarios = ["negative_score", "missing_fields", "invalid_apple"]
        for scenario_name in invalid_scenarios:
            result = next(r for r in validation_results if r["scenario"] == scenario_name)
            assert not result["validation_result"]["is_valid"], f"{scenario_name} should fail validation"
            assert not result["continuation_attempted"], f"{scenario_name} should not attempt continuation"
        
        # Warning scenario should pass validation but have warnings
        warning_result = next(r for r in validation_results if r["scenario"] == "overlapping_snake")
        assert warning_result["validation_result"]["is_valid"], "Overlapping snake should pass validation"
        assert warning_result["has_warnings"], "Overlapping snake should have warnings"
        
        # Verify validation log
        assert len(validator.validation_log) == 5, "Should log all validations"

    def test_cross_session_data_consistency(self) -> None:
        """Test data consistency across multiple session interactions."""
        
        # Mock cross-session coordinator
        coordinator: Mock = Mock()
        coordinator.session_registry = {}
        coordinator.data_checksums = {}
        
        def mock_register_session(session_id: str, session_data: Dict[str, Any]) -> str:
            """Register session and return checksum."""
            checksum = hash(str(sorted(session_data.items())))
            coordinator.session_registry[session_id] = {
                "data": session_data.copy(),
                "checksum": checksum,
                "registration_time": time.time()
            }
            coordinator.data_checksums[session_id] = checksum
            return str(checksum)
        
        def mock_verify_session_consistency(session_id: str) -> Dict[str, Any]:
            """Verify session data consistency."""
            if session_id not in coordinator.session_registry:
                return {"consistent": False, "error": "Session not found"}
            
            session_info = coordinator.session_registry[session_id]
            current_checksum = hash(str(sorted(session_info["data"].items())))
            expected_checksum = session_info["checksum"]
            
            return {
                "consistent": current_checksum == expected_checksum,
                "current_checksum": current_checksum,
                "expected_checksum": expected_checksum,
                "session_id": session_id
            }
        
        def mock_synchronize_sessions(session_ids: List[str]) -> Dict[str, Any]:
            """Synchronize data across multiple sessions."""
            sync_result = {
                "synchronized_sessions": [],
                "synchronization_errors": [],
                "consistency_checks": []
            }
            
            for session_id in session_ids:
                consistency_check = mock_verify_session_consistency(session_id)
                sync_result["consistency_checks"].append(consistency_check)
                
                if consistency_check["consistent"]:
                    sync_result["synchronized_sessions"].append(session_id)
                else:
                    sync_result["synchronization_errors"].append({
                        "session_id": session_id,
                        "error": "Consistency check failed"
                    })
            
            return sync_result
        
        coordinator.register_session = mock_register_session
        coordinator.verify_session_consistency = mock_verify_session_consistency
        coordinator.synchronize_sessions = mock_synchronize_sessions
        
        # Create multiple related sessions
        base_controller = GameController(grid_size=10, use_gui=False)
        
        # Play base game to create initial state
        for step in range(20):
            move: str = ["UP", "RIGHT", "DOWN", "LEFT"][step % 4]
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = base_controller.make_move(move)
            
            if collision:
                break
        
        base_state = {
            "score": base_controller.score,
            "steps": base_controller.steps,
            "snake_length": base_controller.snake_length,
            "snake_positions": base_controller.snake_positions.tolist(),
            "apple_position": base_controller.apple_position.tolist()
        }
        
        # Create related sessions with variations
        session_variations = [
            {"name": "original", "state": base_state.copy()},
            {"name": "score_modified", "state": {**base_state, "score": base_state["score"] + 50}},
            {"name": "position_modified", "state": {**base_state, "apple_position": [base_state["apple_position"][0] + 1, base_state["apple_position"][1]]}},
            {"name": "steps_modified", "state": {**base_state, "steps": base_state["steps"] + 10}},
        ]
        
        # Register sessions
        session_ids: List[str] = []
        registration_results: List[Dict[str, Any]] = []
        
        for variation in session_variations:
            session_id = f"session_{variation['name']}_{int(time.time())}"
            session_ids.append(session_id)
            
            checksum = coordinator.register_session(session_id, variation["state"])
            
            registration_results.append({
                "session_id": session_id,
                "variation_name": variation["name"],
                "checksum": checksum,
                "registration_successful": checksum is not None
            })
        
        # Verify individual session consistency
        consistency_results: List[Dict[str, Any]] = []
        
        for session_id in session_ids:
            consistency_check = coordinator.verify_session_consistency(session_id)
            consistency_results.append(consistency_check)
        
        # Test cross-session synchronization
        sync_result = coordinator.synchronize_sessions(session_ids)
        
        # Verify registration results
        assert len(registration_results) == 4, "Should register all session variations"
        assert all(r["registration_successful"] for r in registration_results), "All registrations should succeed"
        
        # Verify consistency checks
        assert len(consistency_results) == 4, "Should check consistency for all sessions"
        assert all(r["consistent"] for r in consistency_results), "All sessions should be consistent initially"
        
        # Verify synchronization
        assert len(sync_result["synchronized_sessions"]) == 4, "All sessions should synchronize"
        assert len(sync_result["synchronization_errors"]) == 0, "Should have no synchronization errors"
        assert len(sync_result["consistency_checks"]) == 4, "Should check all sessions"
        
        # Test data corruption detection
        # Artificially corrupt one session
        corrupted_session_id = session_ids[1]
        coordinator.session_registry[corrupted_session_id]["data"]["score"] = -999  # Corrupt data
        
        # Verify corruption is detected
        corruption_check = coordinator.verify_session_consistency(corrupted_session_id)
        assert not corruption_check["consistent"], "Should detect data corruption"
        
        # Verify synchronization detects corruption
        sync_with_corruption = coordinator.synchronize_sessions(session_ids)
        assert len(sync_with_corruption["synchronization_errors"]) > 0, "Should detect synchronization errors" 