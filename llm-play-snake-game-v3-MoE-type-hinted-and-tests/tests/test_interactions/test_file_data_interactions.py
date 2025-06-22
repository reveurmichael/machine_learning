"""
Tests for File utilities ↔ GameData interactions.

Focuses on testing how file operations and GameData interact during
serialization/deserialization, corruption recovery, and concurrent access.
"""

import pytest
import os
import json
import threading
import time
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch, mock_open

from core.game_data import GameData
from utils.file_utils import ensure_directory_exists, save_json_safely, load_json_safely


class TestFileDataInteractions:
    """Test interactions between file utilities and GameData."""

    def test_data_serialization_file_save_consistency(self, temp_dir: str) -> None:
        """Test consistency between data serialization and file saving."""
        game_data: GameData = GameData()
        
        # Build complex game state
        game_data.set_llm_info("test_provider", "test_model")
        game_data.update_scores(score=150, steps=75, snake_length=8)
        
        # Add various data types
        for i in range(20):
            game_data.add_llm_communication(f"prompt_{i}", f"response_{i}")
            game_data.add_token_usage(prompt_tokens=100+i, completion_tokens=50+i)
            game_data.moves.append(["UP", "RIGHT", "DOWN", "LEFT"][i % 4])
        
        game_data.increment_round()
        
        # Test multiple serialization/save approaches
        save_approaches: List[Tuple[str, callable]] = [
            ("direct_json", lambda path: game_data.save_to_file(path)),
            ("to_dict_then_save", lambda path: save_json_safely(game_data.to_dict(), path)),
            ("to_json_then_write", lambda path: open(path, 'w').write(game_data.to_json()) and True),
        ]
        
        for approach_name, save_func in save_approaches:
            file_path: str = os.path.join(temp_dir, f"test_{approach_name}.json")
            
            # Save using this approach
            success: bool = save_func(file_path)
            assert success, f"Save failed for {approach_name}"
            assert os.path.exists(file_path), f"File not created for {approach_name}"
            
            # Load and verify consistency
            loaded_data: GameData = GameData()
            load_success: bool = loaded_data.load_from_file(file_path)
            assert load_success, f"Load failed for {approach_name}"
            
            # Verify data integrity
            assert loaded_data.score == game_data.score
            assert loaded_data.steps == game_data.steps
            assert loaded_data.snake_length == game_data.snake_length
            assert loaded_data.moves == game_data.moves
            assert loaded_data.llm_info == game_data.llm_info
            assert loaded_data.round_count == game_data.round_count
            assert len(loaded_data.llm_communication) == len(game_data.llm_communication)

    def test_concurrent_file_access_data_integrity(self, temp_dir: str) -> None:
        """Test data integrity during concurrent file access."""
        shared_file: str = os.path.join(temp_dir, "concurrent_access.json")
        
        # Initialize with base data
        base_data: GameData = GameData()
        base_data.update_scores(score=100, steps=50, snake_length=5)
        base_data.save_to_file(shared_file)
        
        modification_results: List[Dict[str, Any]] = []
        access_errors: List[Exception] = []
        
        def concurrent_modifier(thread_id: int, operation: str) -> None:
            """Perform concurrent modifications."""
            try:
                if operation == "read":
                    # Multiple readers
                    for i in range(10):
                        data: GameData = GameData()
                        success: bool = data.load_from_file(shared_file)
                        if success:
                            # Verify data consistency during read
                            assert data.score >= 0
                            assert data.steps >= 0
                            assert data.snake_length >= 1
                        time.sleep(0.001)  # Small delay
                
                elif operation == "write":
                    # Multiple writers
                    for i in range(5):
                        data: GameData = GameData()
                        data.load_from_file(shared_file)
                        
                        # Modify data
                        data.update_scores(
                            score=data.score + thread_id,
                            steps=data.steps + 1,
                            snake_length=data.snake_length
                        )
                        data.add_llm_communication(f"Thread {thread_id} iteration {i}", f"Response {i}")
                        
                        # Save back
                        success = data.save_to_file(shared_file)
                        assert success
                        
                        time.sleep(0.002)  # Small delay
                
                modification_results.append({
                    "thread_id": thread_id,
                    "operation": operation,
                    "success": True
                })
                
            except Exception as e:
                access_errors.append(e)
        
        # Start concurrent operations
        threads: List[threading.Thread] = []
        
        # 3 readers, 2 writers
        for i in range(3):
            thread = threading.Thread(target=concurrent_modifier, args=(i, "read"))
            threads.append(thread)
        
        for i in range(3, 5):
            thread = threading.Thread(target=concurrent_modifier, args=(i, "write"))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Verify results
        assert len(access_errors) == 0, f"Concurrent access errors: {access_errors}"
        assert len(modification_results) == 5
        
        # Final file should be valid and readable
        final_data: GameData = GameData()
        load_success: bool = final_data.load_from_file(shared_file)
        assert load_success
        assert final_data.score >= 100  # Should have increased
        assert final_data.steps >= 50   # Should have increased

    def test_file_corruption_recovery_mechanisms(self, temp_dir: str) -> None:
        """Test recovery from various file corruption scenarios."""
        # Create valid baseline file
        valid_data: GameData = GameData()
        valid_data.update_scores(score=200, steps=100, snake_length=10)
        valid_data.set_llm_info("provider", "model")
        
        baseline_file: str = os.path.join(temp_dir, "baseline.json")
        valid_data.save_to_file(baseline_file)
        
        # Test various corruption scenarios
        corruption_scenarios: List[Tuple[str, str, bool]] = [
            ("truncated_json", '{"score": 200, "steps":', False),
            ("invalid_syntax", '{"score": 200, "steps": 100,}', True),  # Trailing comma
            ("missing_braces", '"score": 200, "steps": 100', False),
            ("extra_data", '{"score": 200, "steps": 100}extra_content', True),
            ("null_values", '{"score": null, "steps": 100, "snake_length": 10}', True),
            ("wrong_types", '{"score": "200", "steps": "100", "snake_length": "10"}', True),
            ("empty_file", '', False),
            ("only_whitespace", '   \n\t   ', False),
        ]
        
        for scenario_name, corrupted_content, should_recover in corruption_scenarios:
            corrupted_file: str = os.path.join(temp_dir, f"corrupted_{scenario_name}.json")
            
            # Write corrupted content
            with open(corrupted_file, 'w') as f:
                f.write(corrupted_content)
            
            # Test recovery
            recovery_data: GameData = GameData()
            
            try:
                load_success: bool = recovery_data.load_from_file(corrupted_file)
                
                if should_recover:
                    # Should either succeed or gracefully handle corruption
                    if load_success:
                        # Verify recovered data is reasonable
                        assert hasattr(recovery_data, 'score')
                        assert hasattr(recovery_data, 'steps')
                        assert hasattr(recovery_data, 'snake_length')
                    else:
                        # Failed to load but didn't crash
                        assert recovery_data.score == 0  # Default values
                        assert recovery_data.steps == 0
                else:
                    # Expected to fail
                    assert not load_success
                    
            except Exception as e:
                if should_recover:
                    assert False, f"Should recover from {scenario_name} but raised {e}"
                else:
                    # Expected failure
                    assert isinstance(e, (ValueError, json.JSONDecodeError, FileNotFoundError))
            
            # Test fallback to backup if available
            if not should_recover:
                # Try loading baseline as fallback
                fallback_success: bool = recovery_data.load_from_file(baseline_file)
                assert fallback_success
                assert recovery_data.score == 200

    def test_large_file_handling_performance(self, temp_dir: str) -> None:
        """Test performance with large game data files."""
        # Create large game data
        large_data: GameData = GameData()
        large_data.update_scores(score=10000, steps=5000, snake_length=100)
        
        # Add substantial LLM communication history
        for i in range(1000):
            large_data.add_llm_communication(
                f"Large prompt {i} with substantial content " * 10,
                f"Large response {i} with detailed analysis " * 15
            )
            large_data.add_token_usage(prompt_tokens=500, completion_tokens=300)
        
        # Add many moves
        for i in range(5000):
            large_data.moves.append(["UP", "RIGHT", "DOWN", "LEFT"][i % 4])
        
        large_file: str = os.path.join(temp_dir, "large_data.json")
        
        # Test save performance
        start_time: float = time.time()
        save_success: bool = large_data.save_to_file(large_file)
        save_time: float = time.time() - start_time
        
        assert save_success
        assert save_time < 5.0  # Should save in under 5 seconds
        
        # Verify file size is reasonable
        file_size: int = os.path.getsize(large_file)
        assert file_size > 100000  # Should be substantial (>100KB)
        assert file_size < 50000000  # But not excessive (<50MB)
        
        # Test load performance
        start_time = time.time()
        loaded_data: GameData = GameData()
        load_success: bool = loaded_data.load_from_file(large_file)
        load_time: float = time.time() - start_time
        
        assert load_success
        assert load_time < 5.0  # Should load in under 5 seconds
        
        # Verify data integrity
        assert loaded_data.score == large_data.score
        assert loaded_data.steps == large_data.steps
        assert len(loaded_data.llm_communication) == len(large_data.llm_communication)
        assert len(loaded_data.moves) == len(large_data.moves)

    def test_directory_management_data_organization(self, temp_dir: str) -> None:
        """Test directory management for organized data storage."""
        # Test hierarchical data organization
        session_name: str = "test_session_123"
        game_numbers: List[int] = [1, 2, 3, 4, 5]
        
        # Create session directory structure
        session_dir: str = os.path.join(temp_dir, session_name)
        ensure_directory_exists(session_dir)
        
        games_dir: str = os.path.join(session_dir, "games")
        ensure_directory_exists(games_dir)
        
        prompts_dir: str = os.path.join(session_dir, "prompts")
        ensure_directory_exists(prompts_dir)
        
        # Create game data files with proper organization
        created_files: List[str] = []
        
        for game_num in game_numbers:
            # Create game data
            game_data: GameData = GameData()
            game_data.update_scores(score=game_num * 10, steps=game_num * 5, snake_length=game_num + 1)
            game_data.set_llm_info(f"provider_{game_num}", f"model_{game_num}")
            
            # Add game-specific data
            for i in range(game_num * 2):
                game_data.add_llm_communication(f"Game {game_num} prompt {i}", f"Response {i}")
                game_data.moves.append(["UP", "RIGHT", "DOWN", "LEFT"][i % 4])
            
            # Save in organized structure
            game_file: str = os.path.join(games_dir, f"game_{game_num}.json")
            save_success: bool = game_data.save_to_file(game_file)
            assert save_success
            created_files.append(game_file)
            
            # Save prompts separately (if implemented)
            prompt_file: str = os.path.join(prompts_dir, f"game_{game_num}_prompts.json")
            if hasattr(game_data, 'llm_communication') and game_data.llm_communication:
                prompt_data: Dict[str, Any] = {
                    "game_id": game_num,
                    "prompts": game_data.llm_communication
                }
                save_json_safely(prompt_data, prompt_file)
                created_files.append(prompt_file)
        
        # Verify directory structure
        assert os.path.exists(session_dir)
        assert os.path.exists(games_dir)
        assert os.path.exists(prompts_dir)
        
        # Verify all files were created
        for file_path in created_files:
            assert os.path.exists(file_path)
            assert os.path.getsize(file_path) > 0
        
        # Test bulk loading
        loaded_games: List[GameData] = []
        
        for game_num in game_numbers:
            game_file = os.path.join(games_dir, f"game_{game_num}.json")
            game_data = GameData()
            load_success = game_data.load_from_file(game_file)
            assert load_success
            
            # Verify game-specific data
            assert game_data.score == game_num * 10
            assert game_data.steps == game_num * 5
            assert game_data.snake_length == game_num + 1
            
            loaded_games.append(game_data)
        
        assert len(loaded_games) == len(game_numbers)

    def test_atomic_write_operations(self, temp_dir: str) -> None:
        """Test atomic write operations to prevent data corruption."""
        target_file: str = os.path.join(temp_dir, "atomic_test.json")
        
        # Create initial data
        initial_data: GameData = GameData()
        initial_data.update_scores(score=100, steps=50, snake_length=5)
        initial_data.save_to_file(target_file)
        
        # Simulate atomic write scenarios
        write_scenarios: List[Tuple[str, bool, GameData]] = []
        
        for i in range(5):
            scenario_data: GameData = GameData()
            scenario_data.update_scores(score=100 + i * 10, steps=50 + i * 5, snake_length=5 + i)
            scenario_data.add_llm_communication(f"Atomic test {i}", f"Response {i}")
            
            write_scenarios.append((f"update_{i}", True, scenario_data))
        
        # Add failure scenarios
        corrupted_data: GameData = GameData()
        # Simulate data that might cause write failure
        corrupted_data.moves = ["INVALID"] * 10000  # Large invalid data
        write_scenarios.append(("corrupted", False, corrupted_data))
        
        for scenario_name, should_succeed, test_data in write_scenarios:
            # Record state before write
            pre_write_data: GameData = GameData()
            pre_write_data.load_from_file(target_file)
            
            try:
                # Attempt atomic write
                write_success: bool = test_data.save_to_file(target_file)
                
                if should_succeed:
                    assert write_success, f"Write should succeed for {scenario_name}"
                    
                    # Verify file is readable and consistent
                    post_write_data: GameData = GameData()
                    load_success: bool = post_write_data.load_from_file(target_file)
                    assert load_success, f"File should be readable after {scenario_name}"
                    
                    # Data should match what we wrote
                    assert post_write_data.score == test_data.score
                    assert post_write_data.steps == test_data.steps
                    assert post_write_data.snake_length == test_data.snake_length
                    
                else:
                    # Write might fail, but file should remain in valid state
                    # Either old data or new data, but not corrupted
                    recovery_data: GameData = GameData()
                    load_success = recovery_data.load_from_file(target_file)
                    assert load_success, f"File should remain readable after failed {scenario_name}"
                    
                    # Should have either old or new data (atomic operation)
                    assert recovery_data.score in [pre_write_data.score, test_data.score]
                    
            except Exception as e:
                if should_succeed:
                    assert False, f"Unexpected error in {scenario_name}: {e}"
                else:
                    # Expected failure, verify file integrity
                    recovery_data = GameData()
                    load_success = recovery_data.load_from_file(target_file)
                    assert load_success, f"File corrupted after failed {scenario_name}"

    def test_backup_and_versioning_interactions(self, temp_dir: str) -> None:
        """Test backup and versioning interactions with GameData."""
        primary_file: str = os.path.join(temp_dir, "primary_data.json")
        backup_dir: str = os.path.join(temp_dir, "backups")
        ensure_directory_exists(backup_dir)
        
        # Create initial version
        v1_data: GameData = GameData()
        v1_data.update_scores(score=100, steps=50, snake_length=5)
        v1_data.set_llm_info("provider_v1", "model_v1")
        v1_data.save_to_file(primary_file)
        
        # Create backup
        backup_v1: str = os.path.join(backup_dir, "data_v1.json")
        v1_data.save_to_file(backup_v1)
        
        # Simulate version progression
        versions: List[Tuple[int, Dict[str, Any]]] = [
            (2, {"score": 200, "steps": 100, "snake_length": 10}),
            (3, {"score": 350, "steps": 175, "snake_length": 15}),
            (4, {"score": 500, "steps": 250, "snake_length": 20}),
        ]
        
        for version, updates in versions:
            # Load current data
            current_data: GameData = GameData()
            current_data.load_from_file(primary_file)
            
            # Create backup before modification
            backup_file: str = os.path.join(backup_dir, f"data_v{version-1}.json")
            current_data.save_to_file(backup_file)
            
            # Update data
            current_data.update_scores(
                score=updates["score"],
                steps=updates["steps"],
                snake_length=updates["snake_length"]
            )
            current_data.add_llm_communication(f"Version {version} update", f"Updated to version {version}")
            
            # Save new version
            save_success: bool = current_data.save_to_file(primary_file)
            assert save_success
        
        # Verify all versions are accessible
        all_versions: List[GameData] = []
        
        for version in [1, 2, 3, 4]:
            if version == max([v[0] for v in versions] + [1]):
                # Latest version in primary file
                version_data: GameData = GameData()
                version_data.load_from_file(primary_file)
            else:
                # Backup version
                backup_file = os.path.join(backup_dir, f"data_v{version}.json")
                version_data = GameData()
                version_data.load_from_file(backup_file)
            
            all_versions.append(version_data)
        
        # Verify version progression
        assert len(all_versions) >= 4
        for i in range(1, len(all_versions)):
            # Each version should have higher or equal scores
            assert all_versions[i].score >= all_versions[i-1].score
            assert all_versions[i].steps >= all_versions[i-1].steps 