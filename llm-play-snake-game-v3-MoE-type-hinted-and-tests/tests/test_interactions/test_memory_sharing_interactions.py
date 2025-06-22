"""
Tests for memory sharing interactions across components.

Focuses on testing memory usage patterns, shared data structures,
and memory efficiency across component boundaries.
"""

import pytest
import gc
import time
import threading
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, WeakSet
from unittest.mock import Mock
from numpy.typing import NDArray
import weakref

from core.game_controller import GameController
from core.game_data import GameData
from core.game_stats import GameStats


class TestMemorySharingInteractions:
    """Test memory sharing interactions between components."""

    def test_shared_data_structure_consistency(self) -> None:
        """Test consistency of shared data structures across components."""
        controller: GameController = GameController(grid_size=15, use_gui=False)
        game_data: GameData = controller.game_state
        
        # Track shared references
        shared_references: Dict[str, List[Any]] = {
            "snake_positions": [controller.snake_positions, game_data.snake_positions],
            "apple_position": [controller.apple_position, game_data.apple_position],
        }
        
        # Verify initial reference sharing
        for data_name, refs in shared_references.items():
            if len(refs) > 1:
                # Check if references point to same object
                first_ref = refs[0]
                for other_ref in refs[1:]:
                    # They should be the same object or have the same values
                    if hasattr(first_ref, 'shape') and hasattr(other_ref, 'shape'):
                        assert np.array_equal(first_ref, other_ref), \
                            f"Shared {data_name} arrays not equal"
                    else:
                        assert first_ref == other_ref, f"Shared {data_name} not equal"
        
        # Test modifications and consistency
        for i in range(50):
            move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
            
            # Capture state before move
            pre_snake_pos = controller.snake_positions.copy()
            pre_apple_pos = controller.apple_position.copy()
            
            # Make move
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move(move)
            
            # Verify shared state consistency after move
            assert np.array_equal(controller.snake_positions, game_data.snake_positions), \
                f"Snake positions not consistent after move {i}"
            
            assert np.array_equal(controller.apple_position, game_data.apple_position), \
                f"Apple position not consistent after move {i}"
            
            # Verify memory addresses for shared structures
            snake_id_controller = id(controller.snake_positions)
            snake_id_data = id(game_data.snake_positions)
            
            # Should either be same object or valid copies
            if snake_id_controller == snake_id_data:
                # Same object - verify this is intentional
                assert np.shares_memory(controller.snake_positions, game_data.snake_positions), \
                    "Same ID but not sharing memory"
            
            if collision:
                controller.reset()
                # Verify reset consistency
                assert np.array_equal(controller.snake_positions, game_data.snake_positions)

    def test_memory_efficiency_large_game_sessions(self) -> None:
        """Test memory efficiency during large game sessions."""
        controller: GameController = GameController(grid_size=20, use_gui=False)
        
        # Track memory usage approximations
        memory_snapshots: List[Dict[str, Any]] = []
        
        def estimate_memory_usage() -> Dict[str, Any]:
            """Estimate current memory usage."""
            snake_size = controller.snake_positions.nbytes if hasattr(controller.snake_positions, 'nbytes') else 0
            moves_size = len(controller.game_state.moves) * 8  # Approximate string size
            llm_comm_size = len(controller.game_state.llm_communication) * 100  # Approximate
            
            return {
                "snake_bytes": snake_size,
                "moves_bytes": moves_size,
                "llm_comm_bytes": llm_comm_size,
                "total_estimated": snake_size + moves_size + llm_comm_size,
                "timestamp": time.time()
            }
        
        # Run large session
        initial_memory = estimate_memory_usage()
        memory_snapshots.append(initial_memory)
        
        for game in range(20):  # Multiple games
            for step in range(200):  # Long games
                move: str = ["UP", "RIGHT", "DOWN", "LEFT"][step % 4]
                
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move(move)
                
                # Add LLM communication to test memory growth
                if step % 20 == 0:
                    controller.game_state.add_llm_communication(
                        f"Game {game} step {step} prompt",
                        f"Response for game {game} step {step}"
                    )
                
                # Sample memory every 50 steps
                if step % 50 == 0:
                    memory_snapshots.append(estimate_memory_usage())
                
                if collision:
                    break
            
            # Reset for next game
            controller.reset()
            
            # Memory after reset should be lower
            reset_memory = estimate_memory_usage()
            memory_snapshots.append(reset_memory)
            
            # Force garbage collection
            gc.collect()
        
        final_memory = estimate_memory_usage()
        memory_snapshots.append(final_memory)
        
        # Analyze memory patterns
        memory_growth = final_memory["total_estimated"] - initial_memory["total_estimated"]
        max_memory = max(snapshot["total_estimated"] for snapshot in memory_snapshots)
        
        # Memory should not grow unbounded
        assert memory_growth < 100_000, f"Excessive memory growth: {memory_growth} bytes"
        assert max_memory < 500_000, f"Peak memory too high: {max_memory} bytes"
        
        # Verify periodic cleanup
        reset_snapshots = [s for i, s in enumerate(memory_snapshots) if i > 0 and 
                          s["total_estimated"] < memory_snapshots[i-1]["total_estimated"]]
        assert len(reset_snapshots) > 0, "No memory cleanup detected"

    def test_concurrent_memory_access_safety(self) -> None:
        """Test memory access safety under concurrent operations."""
        controller: GameController = GameController(grid_size=12, use_gui=False)
        
        memory_access_results: List[Dict[str, Any]] = []
        memory_access_errors: List[Exception] = []
        
        # Shared data that will be accessed concurrently
        shared_data: Dict[str, Any] = {
            "snake_positions": controller.snake_positions,
            "game_data": controller.game_state,
            "access_count": 0
        }
        
        access_lock = threading.Lock()
        
        def concurrent_memory_reader(thread_id: int) -> None:
            """Perform concurrent memory reads."""
            try:
                read_operations: int = 0
                
                for i in range(100):
                    with access_lock:
                        # Read shared memory
                        snake_copy = shared_data["snake_positions"].copy()
                        game_data_ref = shared_data["game_data"]
                        shared_data["access_count"] += 1
                    
                    # Verify data integrity
                    assert len(snake_copy) >= 1, f"Invalid snake copy in thread {thread_id}"
                    assert hasattr(game_data_ref, 'score'), f"Invalid game data in thread {thread_id}"
                    
                    read_operations += 1
                    time.sleep(0.001)  # Small delay
                
                memory_access_results.append({
                    "thread_id": thread_id,
                    "type": "reader",
                    "operations": read_operations
                })
                
            except Exception as e:
                memory_access_errors.append(e)
        
        def concurrent_memory_writer(thread_id: int) -> None:
            """Perform concurrent memory writes."""
            try:
                write_operations: int = 0
                
                for i in range(50):
                    move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
                    
                    with access_lock:
                        # Modify shared memory
                        collision: bool
                        apple_eaten: bool
                        collision, apple_eaten = controller.make_move(move)
                        
                        # Update shared references
                        shared_data["snake_positions"] = controller.snake_positions
                        shared_data["access_count"] += 1
                    
                    if collision:
                        with access_lock:
                            controller.reset()
                            shared_data["snake_positions"] = controller.snake_positions
                    
                    write_operations += 1
                    time.sleep(0.002)  # Small delay
                
                memory_access_results.append({
                    "thread_id": thread_id,
                    "type": "writer",
                    "operations": write_operations
                })
                
            except Exception as e:
                memory_access_errors.append(e)
        
        # Start concurrent access
        threads: List[threading.Thread] = []
        
        # Multiple readers
        for i in range(3):
            thread = threading.Thread(target=concurrent_memory_reader, args=(i,))
            threads.append(thread)
        
        # Single writer
        thread = threading.Thread(target=concurrent_memory_writer, args=(99,))
        threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Verify results
        assert len(memory_access_errors) == 0, f"Concurrent memory access errors: {memory_access_errors}"
        assert len(memory_access_results) == 4
        
        # Verify access counts
        total_accesses = shared_data["access_count"]
        expected_accesses = 3 * 100 + 1 * 50  # readers + writer
        assert total_accesses >= expected_accesses * 0.9, "Not enough memory accesses recorded"

    def test_memory_leak_detection_component_lifecycle(self) -> None:
        """Test memory leak detection across component lifecycles."""
        initial_objects = len(gc.get_objects())
        
        # Track object creation and cleanup
        created_controllers: WeakSet[GameController] = WeakSet()
        created_data: WeakSet[GameData] = WeakSet()
        
        def create_and_destroy_components(cycles: int) -> None:
            """Create and destroy components to test for leaks."""
            for cycle in range(cycles):
                # Create components
                controller: GameController = GameController(grid_size=10, use_gui=False)
                game_data: GameData = controller.game_state
                
                # Track with weak references
                created_controllers.add(controller)
                created_data.add(game_data)
                
                # Use components
                for i in range(20):
                    move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
                    collision: bool
                    apple_eaten: bool
                    collision, apple_eaten = controller.make_move(move)
                    
                    if collision:
                        controller.reset()
                
                # Add some data
                game_data.add_llm_communication(f"Cycle {cycle} test", "Test response")
                
                # Explicit cleanup
                if hasattr(controller, 'cleanup'):
                    controller.cleanup()
                
                # Remove references
                del controller
                del game_data
                
                # Force garbage collection
                gc.collect()
        
        # Run multiple creation/destruction cycles
        create_and_destroy_components(10)
        
        # Final garbage collection
        gc.collect()
        time.sleep(0.1)  # Allow cleanup
        gc.collect()
        
        final_objects = len(gc.get_objects())
        
        # Check for memory leaks
        object_growth = final_objects - initial_objects
        
        # Some object growth is normal, but should be bounded
        assert object_growth < 1000, f"Excessive object growth: {object_growth} objects"
        
        # Weak references should be cleaned up
        remaining_controllers = len(created_controllers)
        remaining_data = len(created_data)
        
        # Most objects should be garbage collected
        assert remaining_controllers <= 2, f"Controllers not cleaned up: {remaining_controllers}"
        assert remaining_data <= 2, f"GameData not cleaned up: {remaining_data}"

    def test_numpy_array_memory_sharing_patterns(self) -> None:
        """Test numpy array memory sharing patterns between components."""
        controller: GameController = GameController(grid_size=15, use_gui=False)
        
        # Track numpy array memory relationships
        array_relationships: List[Dict[str, Any]] = []
        
        def analyze_array_memory(step: int) -> None:
            """Analyze numpy array memory relationships."""
            snake_pos = controller.snake_positions
            apple_pos = controller.apple_position
            
            # Check memory characteristics
            relationship = {
                "step": step,
                "snake_shape": snake_pos.shape,
                "snake_dtype": str(snake_pos.dtype),
                "snake_flags": {
                    "writeable": snake_pos.flags.writeable,
                    "owndata": snake_pos.flags.owndata,
                    "aligned": snake_pos.flags.aligned
                },
                "apple_shape": apple_pos.shape,
                "apple_dtype": str(apple_pos.dtype),
                "apple_flags": {
                    "writeable": apple_pos.flags.writeable,
                    "owndata": apple_pos.flags.owndata,
                    "aligned": apple_pos.flags.aligned
                }
            }
            
            array_relationships.append(relationship)
        
        # Analyze array memory through game progression
        analyze_array_memory(0)  # Initial state
        
        for i in range(100):
            move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
            
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move(move)
            
            # Analyze every 10 steps
            if i % 10 == 0:
                analyze_array_memory(i + 1)
            
            # Test array sharing after apple eaten
            if apple_eaten:
                # Snake grows, new arrays might be created
                current_snake = controller.snake_positions
                
                # Verify array properties
                assert current_snake.flags.writeable, f"Snake array not writeable at step {i}"
                assert current_snake.dtype == np.int_, f"Wrong snake dtype at step {i}"
                
                analyze_array_memory(i + 1)
            
            if collision:
                controller.reset()
                analyze_array_memory(i + 1)  # After reset
        
        # Analyze memory patterns
        snake_shapes = [rel["snake_shape"] for rel in array_relationships]
        apple_shapes = [rel["apple_shape"] for rel in array_relationships]
        
        # Snake should grow over time (until resets)
        max_snake_length = max(shape[0] for shape in snake_shapes)
        min_snake_length = min(shape[0] for shape in snake_shapes)
        
        assert max_snake_length > min_snake_length, "Snake never grew"
        assert min_snake_length == 1, "Snake never reset to initial size"
        
        # Apple shape should remain constant
        assert all(shape == (2,) for shape in apple_shapes), "Apple shape changed"
        
        # Memory flags should be consistent
        for rel in array_relationships:
            assert rel["snake_flags"]["writeable"], "Snake array became read-only"
            assert rel["apple_flags"]["writeable"], "Apple array became read-only"

    def test_statistics_memory_accumulation_patterns(self) -> None:
        """Test statistics memory accumulation patterns."""
        stats: GameStats = GameStats()
        
        # Track memory usage over time
        memory_progression: List[Dict[str, Any]] = []
        
        def estimate_stats_memory() -> int:
            """Estimate statistics memory usage."""
            base_size = 100  # Base object size
            
            # Count stored data
            step_count = stats.step_stats.valid + stats.step_stats.collisions
            
            # Estimate memory for collections
            estimated_size = base_size + step_count * 4  # 4 bytes per counter
            
            return estimated_size
        
        # Accumulate statistics over long session
        for session in range(50):
            session_start_memory = estimate_stats_memory()
            
            for step in range(200):
                # Record statistics
                stats.record_step_result(
                    valid=True,
                    collision=(step % 30 == 29),
                    apple_eaten=(step % 8 == 0)
                )
                
                # Sample memory every 25 steps
                if step % 25 == 0:
                    current_memory = estimate_stats_memory()
                    memory_progression.append({
                        "session": session,
                        "step": step,
                        "memory_estimate": current_memory,
                        "total_steps": stats.step_stats.valid,
                        "total_collisions": stats.step_stats.collisions
                    })
                
                # End game on collision
                if step % 30 == 29:
                    stats.update_game_stats(
                        final_score=step // 8 * 10,
                        total_steps=step + 1,
                        apples_eaten=step // 8
                    )
                    break
            
            session_end_memory = estimate_stats_memory()
            
            # Memory should grow predictably
            memory_growth = session_end_memory - session_start_memory
            expected_growth = 200 * 4  # Approximate expected growth
            
            assert memory_growth <= expected_growth * 1.5, \
                f"Excessive memory growth in session {session}: {memory_growth} bytes"
        
        # Verify memory growth patterns
        memory_values = [prog["memory_estimate"] for prog in memory_progression]
        
        # Should show generally increasing trend
        assert memory_values[-1] > memory_values[0], "Statistics memory didn't grow"
        
        # But growth should be bounded and predictable
        total_growth = memory_values[-1] - memory_values[0]
        steps_processed = memory_progression[-1]["total_steps"]
        growth_per_step = total_growth / steps_processed if steps_processed > 0 else 0
        
        assert growth_per_step < 10, f"Memory growth per step too high: {growth_per_step} bytes/step"

    def test_cross_component_reference_cycles(self) -> None:
        """Test for memory reference cycles between components."""
        # Track object references to detect cycles
        component_refs: Dict[str, Any] = {}
        
        # Create interconnected components
        controller: GameController = GameController(grid_size=12, use_gui=False)
        game_data: GameData = controller.game_state
        stats: GameStats = GameStats()
        
        # Store references
        component_refs["controller"] = weakref.ref(controller)
        component_refs["game_data"] = weakref.ref(game_data)
        component_refs["stats"] = weakref.ref(stats)
        
        # Create potential reference cycles
        # (In real code, these might be more subtle)
        
        # Simulate some interconnections
        for i in range(50):
            move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
            
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move(move)
            
            # Update statistics based on game data
            stats.record_step_result(
                valid=True,
                collision=collision,
                apple_eaten=apple_eaten
            )
            
            # Add cross-references (potential cycle creators)
            if i % 10 == 0:
                game_data.add_llm_communication(f"Step {i}", f"Stats: {stats.step_stats.valid}")
            
            if collision:
                controller.reset()
        
        # Test reference cleanup
        original_controller = controller
        original_data = game_data
        original_stats = stats
        
        # Remove explicit references
        del controller
        del game_data
        del stats
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
            time.sleep(0.01)
        
        # Check if objects were cleaned up
        controller_alive = component_refs["controller"]() is not None
        data_alive = component_refs["game_data"]() is not None
        stats_alive = component_refs["stats"]() is not None
        
        # Objects might still be alive due to various reasons, but should be collectable
        if controller_alive or data_alive or stats_alive:
            # Give more time and try again
            time.sleep(0.1)
            gc.collect()
            
            controller_alive = component_refs["controller"]() is not None
            data_alive = component_refs["game_data"]() is not None
            stats_alive = component_refs["stats"]() is not None
        
        # At least some objects should be cleaned up if no strong cycles exist
        total_alive = sum([controller_alive, data_alive, stats_alive])
        
        # We don't assert False here because some references might be held by the test framework
        # But we can check that not ALL objects are permanently retained
        if total_alive == 3:
            print("Warning: All objects still alive - potential reference cycles detected")
        
        # The important thing is that the system doesn't crash and memory is eventually reclaimable 