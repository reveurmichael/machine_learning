"""
Tests for Web interface ↔ Core components interactions.

Focuses on testing how web interface components interact with core
game components for HTTP request handling and state synchronization.
"""

import pytest
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch
import numpy as np
from numpy.typing import NDArray

from core.game_controller import GameController
from core.game_data import GameData


class TestWebCoreInteractions:
    """Test interactions between web interface and core components."""

    def test_http_request_game_state_sync(self) -> None:
        """Test HTTP request handling with game state synchronization."""
        
        # Mock web request handler
        web_handler: Mock = Mock()
        web_handler.active_sessions = {}
        web_handler.request_log = []
        
        # Create core game components
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        def mock_handle_web_request(request_type: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
            """Mock web request handler."""
            request_start = time.time()
            
            response = {
                "request_type": request_type,
                "timestamp": request_start,
                "success": False,
                "data": {},
                "error": None
            }
            
            try:
                if request_type == "get_game_state":
                    response["data"] = {
                        "score": controller.score,
                        "steps": controller.steps,
                        "snake_length": controller.snake_length,
                        "snake_positions": controller.snake_positions.tolist(),
                        "apple_position": controller.apple_position.tolist(),
                        "grid_size": controller.grid_size
                    }
                    response["success"] = True
                
                elif request_type == "make_move":
                    move = request_data.get("move", "").upper()
                    if move in ["UP", "DOWN", "LEFT", "RIGHT"]:
                        collision: bool
                        apple_eaten: bool
                        collision, apple_eaten = controller.make_move(move)
                        
                        response["data"] = {
                            "move_applied": move,
                            "collision": collision,
                            "apple_eaten": apple_eaten,
                            "new_score": controller.score,
                            "new_steps": controller.steps
                        }
                        response["success"] = True
                    else:
                        response["error"] = f"Invalid move: {move}"
                
                elif request_type == "reset_game":
                    controller.reset()
                    response["data"] = {
                        "game_reset": True,
                        "initial_state": {
                            "score": controller.score,
                            "steps": controller.steps,
                            "snake_length": controller.snake_length
                        }
                    }
                    response["success"] = True
                
                elif request_type == "get_statistics":
                    response["data"] = {
                        "total_score": controller.score,
                        "total_steps": controller.steps,
                        "snake_efficiency": controller.score / max(1, controller.steps)
                    }
                    response["success"] = True
                
                else:
                    response["error"] = f"Unknown request type: {request_type}"
                
            except Exception as e:
                response["error"] = str(e)
            
            response["processing_time"] = time.time() - request_start
            web_handler.request_log.append(response.copy())
            
            return response
        
        web_handler.handle_request = mock_handle_web_request
        
        # Test various web-core interaction scenarios
        test_requests = [
            {"type": "get_game_state", "data": {}},
            {"type": "make_move", "data": {"move": "UP"}},
            {"type": "get_game_state", "data": {}},
            {"type": "make_move", "data": {"move": "RIGHT"}},
            {"type": "make_move", "data": {"move": "DOWN"}},
            {"type": "get_statistics", "data": {}},
            {"type": "make_move", "data": {"move": "INVALID"}},  # Should fail
            {"type": "reset_game", "data": {}},
            {"type": "get_game_state", "data": {}},
            {"type": "unknown_request", "data": {}}  # Should fail
        ]
        
        request_results: List[Dict[str, Any]] = []
        
        for i, request in enumerate(test_requests):
            result = web_handler.handle_request(request["type"], request["data"])
            result["request_index"] = i
            request_results.append(result)
        
        # Verify web-core interactions
        successful_requests = [r for r in request_results if r["success"]]
        failed_requests = [r for r in request_results if not r["success"]]
        
        assert len(successful_requests) == 8, "Should have 8 successful requests"
        assert len(failed_requests) == 2, "Should have 2 failed requests"
        
        # Verify game state requests
        state_requests = [r for r in request_results if r["request_type"] == "get_game_state"]
        assert len(state_requests) == 3, "Should have 3 state requests"
        
        for state_request in state_requests:
            assert "snake_positions" in state_request["data"], "State should include snake positions"
            assert "apple_position" in state_request["data"], "State should include apple position"
            assert state_request["data"]["score"] >= 0, "Score should be non-negative"
        
        # Verify move requests
        move_requests = [r for r in request_results if r["request_type"] == "make_move" and r["success"]]
        assert len(move_requests) == 3, "Should have 3 successful move requests"
        
        for move_request in move_requests:
            assert "move_applied" in move_request["data"], "Move response should include applied move"
            assert "collision" in move_request["data"], "Move response should include collision status"
        
        # Verify performance
        avg_processing_time = sum(r["processing_time"] for r in request_results) / len(request_results)
        assert avg_processing_time < 0.01, f"Request processing too slow: {avg_processing_time}s"

    def test_concurrent_web_requests(self) -> None:
        """Test handling of concurrent web requests to core components."""
        
        # Mock concurrent web server
        web_server: Mock = Mock()
        web_server.concurrent_sessions = {}
        web_server.request_queue = []
        
        import threading
        
        # Create multiple game controllers for different sessions
        controllers = {
            f"session_{i}": GameController(grid_size=8, use_gui=False) 
            for i in range(5)
        }
        
        def mock_concurrent_request_handler(
            session_id: str, 
            request_type: str, 
            request_data: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Handle concurrent requests per session."""
            
            if session_id not in controllers:
                return {
                    "success": False,
                    "error": f"Invalid session: {session_id}",
                    "session_id": session_id
                }
            
            controller = controllers[session_id]
            response = {
                "session_id": session_id,
                "request_type": request_type,
                "timestamp": time.time(),
                "success": False
            }
            
            try:
                if request_type == "play_sequence":
                    moves = request_data.get("moves", [])
                    results = []
                    
                    for move in moves:
                        if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
                            collision: bool
                            apple_eaten: bool
                            collision, apple_eaten = controller.make_move(move.upper())
                            
                            results.append({
                                "move": move.upper(),
                                "collision": collision,
                                "apple_eaten": apple_eaten,
                                "score": controller.score
                            })
                            
                            if collision:
                                break
                    
                    response["data"] = {
                        "moves_played": results,
                        "final_score": controller.score,
                        "final_steps": controller.steps
                    }
                    response["success"] = True
                
                elif request_type == "get_session_info":
                    response["data"] = {
                        "session_score": controller.score,
                        "session_steps": controller.steps,
                        "snake_length": controller.snake_length
                    }
                    response["success"] = True
                
            except Exception as e:
                response["error"] = str(e)
            
            return response
        
        web_server.handle_concurrent_request = mock_concurrent_request_handler
        
        # Test concurrent request scenarios
        concurrent_results: List[Dict[str, Any]] = []
        result_lock = threading.Lock()
        
        def concurrent_request_worker(worker_id: int) -> None:
            """Worker for concurrent request testing."""
            try:
                session_id = f"session_{worker_id % 5}"  # Distribute across 5 sessions
                
                # Different request patterns per worker
                if worker_id % 3 == 0:
                    # Play sequence requests
                    request_type = "play_sequence"
                    request_data = {"moves": ["UP", "RIGHT", "DOWN", "LEFT", "UP"]}
                else:
                    # Info requests
                    request_type = "get_session_info"
                    request_data = {}
                
                response = web_server.handle_concurrent_request(session_id, request_type, request_data)
                response["worker_id"] = worker_id
                
                with result_lock:
                    concurrent_results.append(response)
                    
            except Exception as e:
                with result_lock:
                    concurrent_results.append({
                        "worker_id": worker_id,
                        "session_id": f"session_{worker_id % 5}",
                        "success": False,
                        "error": str(e)
                    })
        
        # Start concurrent workers
        threads: List[threading.Thread] = []
        
        for worker_id in range(15):  # 15 workers across 5 sessions
            thread = threading.Thread(target=concurrent_request_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Verify concurrent request handling
        assert len(concurrent_results) == 15, "Should handle all concurrent requests"
        
        successful_results = [r for r in concurrent_results if r.get("success", False)]
        assert len(successful_results) == 15, "All concurrent requests should succeed"
        
        # Verify session isolation
        session_results = {}
        for result in successful_results:
            session_id = result["session_id"]
            if session_id not in session_results:
                session_results[session_id] = []
            session_results[session_id].append(result)
        
        assert len(session_results) == 5, "Should use all 5 sessions"
        
        # Verify each session maintained consistency
        for session_id, results in session_results.items():
            controller = controllers[session_id]
            
            # All results for this session should be consistent
            for result in results:
                if result["request_type"] == "get_session_info":
                    # Info should match controller state at some point
                    assert result["data"]["session_score"] >= 0, f"Valid score for {session_id}"
                elif result["request_type"] == "play_sequence":
                    # Sequence should be valid
                    assert len(result["data"]["moves_played"]) > 0, f"Moves played for {session_id}"

    def test_web_error_handling_propagation(self) -> None:
        """Test error handling and propagation from core to web interface."""
        
        # Mock web error handler
        error_handler: Mock = Mock()
        error_handler.error_log = []
        
        # Create controller that can be corrupted for testing
        controller: GameController = GameController(grid_size=6, use_gui=False)
        
        def mock_error_prone_request_handler(
            request_type: str, 
            request_data: Dict[str, Any],
            inject_error: Optional[str] = None
        ) -> Dict[str, Any]:
            """Request handler with error injection capabilities."""
            
            response = {
                "request_type": request_type,
                "timestamp": time.time(),
                "success": False,
                "error": None,
                "error_details": {}
            }
            
            try:
                # Inject specific errors for testing
                if inject_error == "controller_corruption":
                    # Simulate controller corruption
                    controller.snake_positions = np.array([])  # Invalid state
                    raise ValueError("Game controller in invalid state")
                
                elif inject_error == "memory_error":
                    raise MemoryError("Insufficient memory for game operation")
                
                elif inject_error == "network_timeout":
                    raise TimeoutError("Network timeout during game state sync")
                
                elif inject_error == "invalid_operation":
                    raise RuntimeError("Invalid game operation requested")
                
                # Normal request processing
                if request_type == "get_state_with_validation":
                    # Validate controller state before returning
                    if len(controller.snake_positions) == 0:
                        raise ValueError("Invalid snake state detected")
                    
                    response["data"] = {
                        "score": controller.score,
                        "snake_length": len(controller.snake_positions),
                        "state_valid": True
                    }
                    response["success"] = True
                
                elif request_type == "safe_move":
                    move = request_data.get("move", "").upper()
                    
                    # Validate move
                    if move not in ["UP", "DOWN", "LEFT", "RIGHT"]:
                        raise ValueError(f"Invalid move direction: {move}")
                    
                    # Validate controller state
                    if len(controller.snake_positions) == 0:
                        raise RuntimeError("Cannot make move: invalid game state")
                    
                    collision: bool
                    apple_eaten: bool
                    collision, apple_eaten = controller.make_move(move)
                    
                    response["data"] = {
                        "move": move,
                        "success": True,
                        "collision": collision
                    }
                    response["success"] = True
                
                else:
                    raise ValueError(f"Unknown request type: {request_type}")
                
            except (ValueError, RuntimeError, MemoryError, TimeoutError) as e:
                error_type = type(e).__name__
                error_message = str(e)
                
                response["error"] = error_message
                response["error_details"] = {
                    "error_type": error_type,
                    "error_source": "core_component",
                    "recovery_possible": error_type in ["ValueError", "RuntimeError"]
                }
                
                # Log error for analysis
                error_handler.error_log.append({
                    "request_type": request_type,
                    "error_type": error_type,
                    "error_message": error_message,
                    "timestamp": time.time(),
                    "request_data": request_data.copy()
                })
            
            return response
        
        error_handler.handle_request = mock_error_prone_request_handler
        
        # Test error handling scenarios
        error_scenarios = [
            {
                "name": "normal_operation",
                "request_type": "get_state_with_validation",
                "request_data": {},
                "inject_error": None,
                "should_succeed": True
            },
            {
                "name": "controller_corruption",
                "request_type": "get_state_with_validation",
                "request_data": {},
                "inject_error": "controller_corruption",
                "should_succeed": False
            },
            {
                "name": "memory_error",
                "request_type": "safe_move",
                "request_data": {"move": "UP"},
                "inject_error": "memory_error",
                "should_succeed": False
            },
            {
                "name": "invalid_move",
                "request_type": "safe_move",
                "request_data": {"move": "INVALID"},
                "inject_error": None,
                "should_succeed": False
            },
            {
                "name": "network_timeout",
                "request_type": "get_state_with_validation",
                "request_data": {},
                "inject_error": "network_timeout",
                "should_succeed": False
            }
        ]
        
        error_test_results: List[Dict[str, Any]] = []
        
        for scenario in error_scenarios:
            # Reset controller for each test
            controller = GameController(grid_size=6, use_gui=False)
            
            response = error_handler.handle_request(
                scenario["request_type"],
                scenario["request_data"],
                scenario["inject_error"]
            )
            
            error_test_results.append({
                "scenario": scenario["name"],
                "expected_success": scenario["should_succeed"],
                "actual_success": response["success"],
                "error_message": response.get("error"),
                "error_details": response.get("error_details", {}),
                "response": response
            })
        
        # Verify error handling
        assert len(error_test_results) == 5, "Should test all error scenarios"
        
        # Verify expected successes and failures
        for result in error_test_results:
            expected = result["expected_success"]
            actual = result["actual_success"]
            scenario = result["scenario"]
            
            assert actual == expected, f"Scenario {scenario}: expected {expected}, got {actual}"
        
        # Verify error details for failed requests
        failed_results = [r for r in error_test_results if not r["actual_success"]]
        assert len(failed_results) == 4, "Should have 4 failed scenarios"
        
        for failed_result in failed_results:
            error_details = failed_result["error_details"]
            assert "error_type" in error_details, f"Missing error type for {failed_result['scenario']}"
            assert "error_source" in error_details, f"Missing error source for {failed_result['scenario']}"
        
        # Verify error logging
        assert len(error_handler.error_log) == 4, "Should log all errors"
        
        # Verify specific error types
        error_types = [log["error_type"] for log in error_handler.error_log]
        assert "ValueError" in error_types, "Should capture ValueError"
        assert "MemoryError" in error_types, "Should capture MemoryError"
        assert "TimeoutError" in error_types, "Should capture TimeoutError"

    def test_web_session_lifecycle_management(self) -> None:
        """Test web session lifecycle management with core components."""
        
        # Mock session lifecycle manager
        session_manager: Mock = Mock()
        session_manager.active_sessions = {}
        session_manager.session_history = []
        
        def mock_create_web_session(session_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
            """Create web session with core component binding."""
            
            if session_id in session_manager.active_sessions:
                return {
                    "success": False,
                    "error": f"Session {session_id} already exists"
                }
            
            # Create core components for session
            grid_size = config.get("grid_size", 10)
            controller = GameController(grid_size=grid_size, use_gui=False)
            
            session_info = {
                "session_id": session_id,
                "controller": controller,
                "created_time": time.time(),
                "last_activity": time.time(),
                "request_count": 0,
                "config": config.copy()
            }
            
            session_manager.active_sessions[session_id] = session_info
            
            return {
                "success": True,
                "session_id": session_id,
                "initial_state": {
                    "score": controller.score,
                    "grid_size": controller.grid_size,
                    "snake_length": controller.snake_length
                }
            }
        
        def mock_session_request(session_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
            """Process request within session context."""
            
            if session_id not in session_manager.active_sessions:
                return {
                    "success": False,
                    "error": f"Session {session_id} not found"
                }
            
            session_info = session_manager.active_sessions[session_id]
            controller = session_info["controller"]
            
            # Update session activity
            session_info["last_activity"] = time.time()
            session_info["request_count"] += 1
            
            # Process request
            request_type = request.get("type", "unknown")
            
            if request_type == "play_moves":
                moves = request.get("data", {}).get("moves", [])
                move_results = []
                
                for move in moves:
                    if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
                        collision: bool
                        apple_eaten: bool
                        collision, apple_eaten = controller.make_move(move.upper())
                        
                        move_results.append({
                            "move": move.upper(),
                            "collision": collision,
                            "apple_eaten": apple_eaten
                        })
                        
                        if collision:
                            break
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "move_results": move_results,
                    "current_score": controller.score
                }
            
            elif request_type == "get_session_stats":
                return {
                    "success": True,
                    "session_id": session_id,
                    "stats": {
                        "score": controller.score,
                        "steps": controller.steps,
                        "snake_length": controller.snake_length,
                        "request_count": session_info["request_count"],
                        "session_duration": time.time() - session_info["created_time"]
                    }
                }
            
            return {
                "success": False,
                "error": f"Unknown request type: {request_type}"
            }
        
        def mock_cleanup_session(session_id: str) -> Dict[str, Any]:
            """Clean up session and core components."""
            
            if session_id not in session_manager.active_sessions:
                return {
                    "success": False,
                    "error": f"Session {session_id} not found"
                }
            
            session_info = session_manager.active_sessions[session_id]
            
            # Archive session info
            final_stats = {
                "session_id": session_id,
                "duration": time.time() - session_info["created_time"],
                "final_score": session_info["controller"].score,
                "total_requests": session_info["request_count"],
                "cleanup_time": time.time()
            }
            
            session_manager.session_history.append(final_stats)
            
            # Remove active session
            del session_manager.active_sessions[session_id]
            
            return {
                "success": True,
                "session_id": session_id,
                "final_stats": final_stats
            }
        
        session_manager.create_session = mock_create_web_session
        session_manager.session_request = mock_session_request
        session_manager.cleanup_session = mock_cleanup_session
        
        # Test session lifecycle
        session_configs = [
            {"session_id": "web_session_1", "config": {"grid_size": 8}},
            {"session_id": "web_session_2", "config": {"grid_size": 12}},
            {"session_id": "web_session_3", "config": {"grid_size": 10}},
        ]
        
        lifecycle_results: List[Dict[str, Any]] = []
        
        # Create sessions
        for session_config in session_configs:
            session_id = session_config["session_id"]
            config = session_config["config"]
            
            create_result = session_manager.create_session(session_id, config)
            assert create_result["success"], f"Failed to create session {session_id}"
            
            # Use session
            for i in range(3):
                request = {
                    "type": "play_moves",
                    "data": {"moves": ["UP", "RIGHT", "DOWN", "LEFT"]}
                }
                
                request_result = session_manager.session_request(session_id, request)
                assert request_result["success"], f"Request failed for session {session_id}"
            
            # Get session stats
            stats_request = {"type": "get_session_stats", "data": {}}
            stats_result = session_manager.session_request(session_id, stats_request)
            assert stats_result["success"], f"Stats request failed for session {session_id}"
            
            # Clean up session
            cleanup_result = session_manager.cleanup_session(session_id)
            assert cleanup_result["success"], f"Cleanup failed for session {session_id}"
            
            lifecycle_results.append({
                "session_id": session_id,
                "creation": create_result,
                "final_stats": stats_result["stats"],
                "cleanup": cleanup_result
            })
        
        # Verify lifecycle management
        assert len(lifecycle_results) == 3, "Should manage all session lifecycles"
        assert len(session_manager.active_sessions) == 0, "All sessions should be cleaned up"
        assert len(session_manager.session_history) == 3, "All sessions should be archived"
        
        # Verify session isolation
        for result in lifecycle_results:
            stats = result["final_stats"]
            assert stats["score"] >= 0, "Session should have valid final score"
            assert stats["request_count"] == 4, "Session should have correct request count"  # 3 move requests + 1 stats request 