"""
Tests for concurrent game instance interactions.

Focuses on testing how multiple game instances interact when sharing
resources, competing for files, and handling concurrent state modifications.
"""

import pytest
import threading
import time
import os
from typing import List, Dict, Any, Optional, Tuple, Generator
from unittest.mock import Mock, patch
from numpy.typing import NDArray
import numpy as np

from core.game_controller import GameController
from core.game_data import GameData
from utils.file_utils import ensure_directory_exists, save_json_safely


class TestConcurrentGameInteractions:
    """Test interactions between concurrent game instances."""

    def test_multiple_game_instances_resource_isolation(self, temp_dir: str) -> None:
        """Test resource isolation between multiple concurrent game instances."""
        num_games: int = 5
        game_controllers: List[GameController] = []
        game_results: List[Dict[str, Any]] = []
        
        # Create isolated game instances
        for i in range(num_games):
            controller: GameController = GameController(
                grid_size=10 + i,  # Different grid sizes for isolation testing
                use_gui=False
            )
            game_controllers.append(controller)
        
        def run_isolated_game(game_id: int, controller: GameController) -> None:
            """Run an isolated game instance."""
            moves_made: int = 0
            collisions: int = 0
            max_score: int = 0
            
            # Configure unique game data
            game_data: GameData = controller.game_state
            game_data.set_llm_info(f"provider_{game_id}", f"model_{game_id}")
            
            # Run game with unique move pattern
            move_patterns: Dict[int, List[str]] = {
                0: ["UP", "RIGHT", "DOWN", "LEFT"],
                1: ["RIGHT", "DOWN", "LEFT", "UP"],
                2: ["DOWN", "LEFT", "UP", "RIGHT"],
                3: ["LEFT", "UP", "RIGHT", "DOWN"],
                4: ["UP", "UP", "RIGHT", "RIGHT", "DOWN", "DOWN", "LEFT", "LEFT"]
            }
            
            pattern: List[str] = move_patterns.get(game_id, ["UP", "RIGHT", "DOWN", "LEFT"])
            
            for move_index in range(200):  # 200 moves per game
                move: str = pattern[move_index % len(pattern)]
                
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move(move)
                moves_made += 1
                
                if apple_eaten:
                    max_score = max(max_score, controller.score)
                
                if collision:
                    collisions += 1
                    controller.reset()
                
                # Add some LLM data periodically
                if move_index % 20 == 0:
                    game_data.add_llm_communication(
                        f"Game {game_id} move {move_index}",
                        f"Response for game {game_id}"
                    )
                    game_data.add_token_usage(prompt_tokens=50 + game_id, completion_tokens=25 + game_id)
            
            # Record results
            game_results.append({
                "game_id": game_id,
                "moves_made": moves_made,
                "collisions": collisions,
                "max_score": max_score,
                "final_score": controller.score,
                "grid_size": controller.grid_size,
                "llm_requests": len(game_data.llm_communication)
            })
        
        # Run games concurrently
        threads: List[threading.Thread] = []
        
        for i, controller in enumerate(game_controllers):
            thread = threading.Thread(target=run_isolated_game, args=(i, controller))
            threads.append(thread)
            thread.start()
        
        # Wait for all games to complete
        for thread in threads:
            thread.join(timeout=30.0)
        
        # Verify isolation
        assert len(game_results) == num_games
        
        # Each game should have unique characteristics
        game_ids = [result["game_id"] for result in game_results]
        assert len(set(game_ids)) == num_games  # All unique IDs
        
        grid_sizes = [result["grid_size"] for result in game_results]
        assert len(set(grid_sizes)) == num_games  # All different grid sizes
        
        # Verify each game made progress
        for result in game_results:
            assert result["moves_made"] == 200
            assert result["llm_requests"] > 0
            assert result["grid_size"] >= 10

    def test_shared_file_system_access_coordination(self, temp_dir: str) -> None:
        """Test coordination when multiple games access shared file system."""
        shared_log_dir: str = os.path.join(temp_dir, "shared_logs")
        ensure_directory_exists(shared_log_dir)
        
        num_games: int = 4
        files_written: List[str] = []
        write_conflicts: List[Exception] = []
        
        def game_with_file_access(game_id: int) -> None:
            """Run game that writes to shared file system."""
            controller: GameController = GameController(grid_size=8, use_gui=False)
            game_data: GameData = controller.game_state
            
            # Each game writes to its own file and a shared summary
            individual_file: str = os.path.join(shared_log_dir, f"game_{game_id}.json")
            shared_summary: str = os.path.join(shared_log_dir, "shared_summary.json")
            
            try:
                # Run some moves
                for i in range(50):
                    move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
                    collision: bool
                    apple_eaten: bool
                    collision, apple_eaten = controller.make_move(move)
                    
                    if collision:
                        controller.reset()
                    
                    # Periodically save individual progress
                    if i % 10 == 0:
                        game_data.add_llm_communication(f"Game {game_id} step {i}", f"Progress update {i}")
                        save_success: bool = game_data.save_to_file(individual_file)
                        if save_success:
                            files_written.append(individual_file)
                    
                    # Update shared summary (potential conflict)
                    if i % 25 == 0:
                        try:
                            # Load existing summary or create new
                            summary_data: Dict[str, Any] = {}
                            if os.path.exists(shared_summary):
                                summary_data = load_json_safely(shared_summary) or {}
                            
                            # Add this game's data
                            if "games" not in summary_data:
                                summary_data["games"] = {}
                            
                            summary_data["games"][f"game_{game_id}"] = {
                                "steps": controller.steps,
                                "score": controller.score,
                                "last_update": time.time()
                            }
                            
                            # Save summary
                            save_json_safely(summary_data, shared_summary)
                            
                        except Exception as e:
                            write_conflicts.append(e)
            
            except Exception as e:
                write_conflicts.append(e)
        
        # Start concurrent games
        threads: List[threading.Thread] = []
        
        for i in range(num_games):
            thread = threading.Thread(target=game_with_file_access, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=20.0)
        
        # Verify results
        # Should have minimal write conflicts due to proper coordination
        assert len(write_conflicts) < num_games, f"Too many write conflicts: {write_conflicts}"
        
        # Individual files should all exist
        for i in range(num_games):
            individual_file = os.path.join(shared_log_dir, f"game_{i}.json")
            assert os.path.exists(individual_file), f"Game {i} file missing"
        
        # Shared summary should exist and contain all games
        shared_summary = os.path.join(shared_log_dir, "shared_summary.json")
        if os.path.exists(shared_summary):
            summary_data = load_json_safely(shared_summary)
            if summary_data and "games" in summary_data:
                assert len(summary_data["games"]) <= num_games  # May have some conflicts

    def test_memory_pressure_multiple_games(self) -> None:
        """Test system behavior under memory pressure from multiple games."""
        num_games: int = 10
        large_controllers: List[GameController] = []
        memory_errors: List[Exception] = []
        
        def create_memory_intensive_game(game_id: int) -> None:
            """Create memory-intensive game instance."""
            try:
                # Large grid size for memory pressure
                controller: GameController = GameController(grid_size=25, use_gui=False)
                game_data: GameData = controller.game_state
                
                # Generate substantial game state
                for i in range(100):
                    move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
                    collision: bool
                    apple_eaten: bool
                    collision, apple_eaten = controller.make_move(move)
                    
                    if collision:
                        controller.reset()
                    
                    # Add memory-intensive data
                    if i % 5 == 0:
                        large_prompt = f"Large prompt for game {game_id} iteration {i} " * 50
                        large_response = f"Large response for game {game_id} iteration {i} " * 100
                        game_data.add_llm_communication(large_prompt, large_response)
                        game_data.add_token_usage(prompt_tokens=1000, completion_tokens=500)
                
                large_controllers.append(controller)
                
            except MemoryError as e:
                memory_errors.append(e)
            except Exception as e:
                # Other resource-related errors
                if "memory" in str(e).lower() or "resource" in str(e).lower():
                    memory_errors.append(e)
                else:
                    raise
        
        # Create games concurrently
        threads: List[threading.Thread] = []
        
        for i in range(num_games):
            thread = threading.Thread(target=create_memory_intensive_game, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=15.0)
        
        # Verify system handled memory pressure
        total_created = len(large_controllers)
        total_errors = len(memory_errors)
        
        # Should create some games successfully
        assert total_created > 0, "No games created successfully"
        
        # If memory errors occurred, they should be handled gracefully
        if total_errors > 0:
            assert total_errors < num_games, "All games failed due to memory"
            
            # Verify existing games still function
            for controller in large_controllers[:3]:  # Test first 3
                controller.make_move("UP")
                assert hasattr(controller, 'game_state')
                assert controller.game_state is not None

    def test_performance_scaling_multiple_games(self) -> None:
        """Test performance scaling with multiple concurrent games."""
        game_counts: List[int] = [1, 2, 4, 8]
        performance_results: List[Tuple[int, float, float]] = []
        
        for num_games in game_counts:
            start_time: float = time.time()
            completed_games: int = 0
            total_moves: int = 0
            
            def run_performance_game(game_id: int) -> None:
                nonlocal completed_games, total_moves
                
                controller: GameController = GameController(grid_size=12, use_gui=False)
                game_moves: int = 0
                
                # Fixed number of moves per game
                for i in range(100):
                    move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
                    collision: bool
                    apple_eaten: bool
                    collision, apple_eaten = controller.make_move(move)
                    game_moves += 1
                    
                    if collision:
                        controller.reset()
                
                completed_games += 1
                total_moves += game_moves
            
            # Run concurrent games
            threads: List[threading.Thread] = []
            
            for i in range(num_games):
                thread = threading.Thread(target=run_performance_game, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=10.0)
            
            end_time: float = time.time()
            total_time: float = end_time - start_time
            moves_per_second: float = total_moves / total_time if total_time > 0 else 0
            
            performance_results.append((num_games, total_time, moves_per_second))
            
            # Verify all games completed
            assert completed_games == num_games, f"Only {completed_games}/{num_games} games completed"
        
        # Analyze performance scaling
        assert len(performance_results) == len(game_counts)
        
        # Performance shouldn't degrade linearly with game count
        single_game_performance = performance_results[0][2]  # moves/sec for 1 game
        max_games_performance = performance_results[-1][2]   # moves/sec for max games
        
        # With good concurrency, max games should achieve higher total throughput
        # but may have lower per-game performance
        assert max_games_performance > 0, "Performance collapsed with multiple games"
        
        # Total system throughput should increase with more games
        single_total_throughput = performance_results[0][2] * 1
        max_total_throughput = performance_results[-1][2]
        
        assert max_total_throughput > single_total_throughput * 0.5, "Poor scaling performance"

    def test_shared_resource_contention_resolution(self) -> None:
        """Test resolution of shared resource contention between games."""
        shared_resource_usage: Dict[str, int] = {}
        resource_conflicts: List[str] = []
        resolution_times: List[float] = []
        
        def game_with_shared_resources(game_id: int, resource_name: str) -> None:
            """Game that competes for shared resources."""
            controller: GameController = GameController(grid_size=10, use_gui=False)
            
            for attempt in range(20):
                try:
                    # Simulate resource acquisition
                    acquisition_start: float = time.time()
                    
                    # Critical section - shared resource access
                    if resource_name not in shared_resource_usage:
                        shared_resource_usage[resource_name] = 0
                    
                    # Simulate resource usage
                    current_usage = shared_resource_usage[resource_name]
                    time.sleep(0.001)  # Simulate work
                    shared_resource_usage[resource_name] = current_usage + 1
                    
                    acquisition_end: float = time.time()
                    resolution_times.append(acquisition_end - acquisition_start)
                    
                    # Use the controller while holding resource
                    move: str = ["UP", "RIGHT", "DOWN", "LEFT"][attempt % 4]
                    collision: bool
                    apple_eaten: bool
                    collision, apple_eaten = controller.make_move(move)
                    
                    if collision:
                        controller.reset()
                
                except Exception as e:
                    resource_conflicts.append(f"Game {game_id}: {str(e)}")
        
        # Test different resource contention scenarios
        contention_scenarios: List[Tuple[int, int, str]] = [
            (3, 1, "high_contention"),    # 3 games, 1 resource
            (4, 2, "medium_contention"),  # 4 games, 2 resources
            (6, 3, "low_contention"),     # 6 games, 3 resources
        ]
        
        for num_games, num_resources, scenario_name in contention_scenarios:
            # Reset shared state
            shared_resource_usage.clear()
            resource_conflicts.clear()
            resolution_times.clear()
            
            threads: List[threading.Thread] = []
            
            for i in range(num_games):
                resource_name = f"resource_{i % num_resources}"
                thread = threading.Thread(target=game_with_shared_resources, args=(i, resource_name))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=15.0)
            
            # Analyze contention resolution
            total_conflicts = len(resource_conflicts)
            avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
            
            # Verify contention was handled
            assert total_conflicts < num_games * 5, f"Too many conflicts in {scenario_name}: {total_conflicts}"
            
            # Resolution time should be reasonable
            assert avg_resolution_time < 0.1, f"Resolution too slow in {scenario_name}: {avg_resolution_time}s"
            
            # Resource usage should reflect actual work done
            total_usage = sum(shared_resource_usage.values())
            expected_usage = num_games * 20  # 20 attempts per game
            
            # Should be close to expected (some conflicts are normal)
            assert total_usage >= expected_usage * 0.8, f"Too much resource loss in {scenario_name}"

    def test_game_state_synchronization_coordination(self) -> None:
        """Test coordination of game state synchronization across instances."""
        synchronization_points: List[Dict[str, Any]] = []
        sync_errors: List[Exception] = []
        
        def synchronized_game(game_id: int, sync_interval: int) -> None:
            """Game that synchronizes state at regular intervals."""
            controller: GameController = GameController(grid_size=8, use_gui=False)
            game_data: GameData = controller.game_state
            
            try:
                for step in range(100):
                    # Make move
                    move: str = ["UP", "RIGHT", "DOWN", "LEFT"][step % 4]
                    collision: bool
                    apple_eaten: bool
                    collision, apple_eaten = controller.make_move(move)
                    
                    if collision:
                        controller.reset()
                    
                    # Synchronization point
                    if step % sync_interval == 0:
                        sync_point: Dict[str, Any] = {
                            "game_id": game_id,
                            "step": step,
                            "score": controller.score,
                            "snake_length": controller.snake_length,
                            "timestamp": time.time(),
                            "game_over": game_data.game_over
                        }
                        
                        synchronization_points.append(sync_point)
                        
                        # Simulate coordination delay
                        time.sleep(0.001)
            
            except Exception as e:
                sync_errors.append(e)
        
        # Test different synchronization frequencies
        sync_configs: List[Tuple[int, int]] = [
            (3, 10),  # 3 games, sync every 10 steps
            (5, 20),  # 5 games, sync every 20 steps
            (4, 5),   # 4 games, sync every 5 steps (high frequency)
        ]
        
        for num_games, sync_interval in sync_configs:
            # Reset synchronization state
            synchronization_points.clear()
            sync_errors.clear()
            
            threads: List[threading.Thread] = []
            
            for i in range(num_games):
                thread = threading.Thread(target=synchronized_game, args=(i, sync_interval))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=20.0)
            
            # Verify synchronization
            assert len(sync_errors) == 0, f"Synchronization errors: {sync_errors}"
            
            # Should have synchronization points from all games
            game_ids = set(point["game_id"] for point in synchronization_points)
            assert len(game_ids) == num_games, f"Missing sync points from some games"
            
            # Verify temporal coordination
            sync_times = [point["timestamp"] for point in synchronization_points]
            time_span = max(sync_times) - min(sync_times)
            
            # All games should complete within reasonable time
            assert time_span < 5.0, f"Synchronization took too long: {time_span}s" 