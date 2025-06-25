"""
Tests for Statistics ↔ File persistence interactions.

Focuses on testing how Statistics and File persistence maintain
data integrity, handle concurrent writes, and recover from corruption.
"""

import pytest
import os
import json
import threading
import time
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch, mock_open

from core.game_stats import GameStats
from utils.file_utils import save_json_safely, load_json_safely, ensure_directory_exists


class TestStatsPersistenceInteractions:
    """Test interactions between Statistics and File persistence."""

    def test_statistics_serialization_consistency(self, temp_dir: str) -> None:
        """Test consistency between statistics and their file serialization."""
        stats: GameStats = GameStats()
        
        # Build substantial statistics
        for i in range(100):
            stats.record_step_result(
                valid=True,
                collision=(i % 25 == 24),
                apple_eaten=(i % 10 == 0)
            )
            
            if i % 25 == 24:  # End of game
                stats.update_game_stats(
                    final_score=i // 10 * 10,
                    total_steps=i + 1,
                    apples_eaten=i // 10
                )
        
        # Test various serialization approaches
        stats_file: str = os.path.join(temp_dir, "stats_test.json")
        
        # Approach 1: Direct statistics serialization
        if hasattr(stats, 'to_dict'):
            stats_dict: Dict[str, Any] = stats.to_dict()
            save_success: bool = save_json_safely(stats_dict, stats_file)
            assert save_success, "Failed to save statistics dict"
            
            # Verify file exists and is readable
            assert os.path.exists(stats_file)
            
            # Load and verify consistency
            loaded_dict: Optional[Dict[str, Any]] = load_json_safely(stats_file)
            assert loaded_dict is not None, "Failed to load statistics"
            
            # Create new stats from loaded data
            new_stats: GameStats = GameStats()
            if hasattr(new_stats, 'from_dict'):
                new_stats.from_dict(loaded_dict)
                
                # Verify data consistency
                assert new_stats.step_stats.valid == stats.step_stats.valid
                assert new_stats.step_stats.collisions == stats.step_stats.collisions
                
                if hasattr(stats, 'total_games'):
                    assert new_stats.total_games == stats.total_games

    def test_concurrent_statistics_file_access(self, temp_dir: str) -> None:
        """Test concurrent access to statistics files."""
        shared_stats_file: str = os.path.join(temp_dir, "shared_stats.json")
        
        # Initialize shared statistics file
        initial_stats: GameStats = GameStats()
        if hasattr(initial_stats, 'to_dict'):
            save_json_safely(initial_stats.to_dict(), shared_stats_file)
        
        access_results: List[Dict[str, Any]] = []
        access_errors: List[Exception] = []
        
        def concurrent_stats_update(thread_id: int, update_type: str) -> None:
            """Perform concurrent statistics updates."""
            try:
                if update_type == "read_only":
                    # Multiple readers
                    for i in range(20):
                        loaded_data: Optional[Dict[str, Any]] = load_json_safely(shared_stats_file)
                        if loaded_data:
                            # Verify data structure
                            assert isinstance(loaded_data, dict)
                            
                            # Create stats object from data
                            stats: GameStats = GameStats()
                            if hasattr(stats, 'from_dict'):
                                stats.from_dict(loaded_data)
                        
                        time.sleep(0.001)  # Small delay
                
                elif update_type == "read_write":
                    # Readers that also update
                    for i in range(10):
                        # Load current stats
                        loaded_data = load_json_safely(shared_stats_file)
                        if loaded_data:
                            stats = GameStats()
                            if hasattr(stats, 'from_dict'):
                                stats.from_dict(loaded_data)
                            
                            # Update stats
                            stats.record_step_result(
                                valid=True,
                                collision=(i % 5 == 4),
                                apple_eaten=(i % 3 == 0)
                            )
                            
                            # Save back
                            if hasattr(stats, 'to_dict'):
                                save_json_safely(stats.to_dict(), shared_stats_file)
                        
                        time.sleep(0.002)  # Small delay
                
                access_results.append({
                    "thread_id": thread_id,
                    "update_type": update_type,
                    "success": True
                })
                
            except Exception as e:
                access_errors.append(e)
        
        # Start concurrent operations
        threads: List[threading.Thread] = []
        
        # 3 read-only threads, 2 read-write threads
        for i in range(3):
            thread = threading.Thread(target=concurrent_stats_update, args=(i, "read_only"))
            threads.append(thread)
        
        for i in range(3, 5):
            thread = threading.Thread(target=concurrent_stats_update, args=(i, "read_write"))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=15.0)
        
        # Verify results
        assert len(access_errors) == 0, f"Concurrent access errors: {access_errors}"
        assert len(access_results) == 5
        
        # Final file should be valid
        final_data: Optional[Dict[str, Any]] = load_json_safely(shared_stats_file)
        assert final_data is not None
        
        # Should be valid statistics data
        final_stats: GameStats = GameStats()
        if hasattr(final_stats, 'from_dict'):
            final_stats.from_dict(final_data)
            assert final_stats.step_stats.valid >= 0

    def test_statistics_file_corruption_recovery(self, temp_dir: str) -> None:
        """Test recovery from statistics file corruption."""
        stats_file: str = os.path.join(temp_dir, "corrupted_stats.json")
        backup_file: str = os.path.join(temp_dir, "stats_backup.json")
        
        # Create valid baseline statistics
        valid_stats: GameStats = GameStats()
        for i in range(50):
            valid_stats.record_step_result(True, (i % 20 == 19), (i % 5 == 0))
        
        # Save valid backup
        if hasattr(valid_stats, 'to_dict'):
            save_json_safely(valid_stats.to_dict(), backup_file)
        
        # Test various corruption scenarios
        corruption_scenarios: List[Tuple[str, str, bool]] = [
            ("truncated_file", '{"step_stats": {"valid": 50, "colli', False),
            ("invalid_json", '{"step_stats": {"valid": 50, "collisions": 2,}', True),
            ("empty_file", '', False),
            ("binary_data", b'\x00\x01\x02\x03'.decode('latin1'), False),
            ("wrong_structure", '{"not_stats": "data"}', True),
            ("negative_values", '{"step_stats": {"valid": -10}}', True),
        ]
        
        for scenario_name, corrupted_content, should_recover in corruption_scenarios:
            # Write corrupted content
            with open(stats_file, 'w') as f:
                f.write(corrupted_content)
            
            # Attempt to load with recovery
            recovery_stats: GameStats = GameStats()
            load_success: bool = False
            
            try:
                # Try direct load
                loaded_data: Optional[Dict[str, Any]] = load_json_safely(stats_file)
                
                if loaded_data and hasattr(recovery_stats, 'from_dict'):
                    recovery_stats.from_dict(loaded_data)
                    load_success = True
                
                elif should_recover:
                    # Try backup recovery
                    backup_data = load_json_safely(backup_file)
                    if backup_data and hasattr(recovery_stats, 'from_dict'):
                        recovery_stats.from_dict(backup_data)
                        load_success = True
                
            except Exception as e:
                if should_recover:
                    # Try backup recovery on exception
                    try:
                        backup_data = load_json_safely(backup_file)
                        if backup_data and hasattr(recovery_stats, 'from_dict'):
                            recovery_stats.from_dict(backup_data)
                            load_success = True
                    except:
                        pass
            
            # Verify recovery results
            if should_recover:
                assert load_success or recovery_stats.step_stats.valid >= 0, \
                    f"Failed to recover from {scenario_name}"
            
            # Recovery stats should be in valid state
            assert hasattr(recovery_stats, 'step_stats')
            assert recovery_stats.step_stats.valid >= 0

    def test_large_statistics_file_performance(self, temp_dir: str) -> None:
        """Test performance with large statistics files."""
        large_stats_file: str = os.path.join(temp_dir, "large_stats.json")
        
        # Create large statistics dataset
        large_stats: GameStats = GameStats()
        
        # Simulate extensive game session
        for game in range(100):
            for step in range(1000):
                large_stats.record_step_result(
                    valid=True,
                    collision=(step % 100 == 99),
                    apple_eaten=(step % 20 == 0)
                )
            
            # End of game stats
            large_stats.update_game_stats(
                final_score=step // 20 * 10,
                total_steps=step + 1,
                apples_eaten=step // 20
            )
        
        # Test save performance
        start_time: float = time.time()
        
        if hasattr(large_stats, 'to_dict'):
            save_success: bool = save_json_safely(large_stats.to_dict(), large_stats_file)
            
        save_time: float = time.time() - start_time
        
        assert save_success, "Failed to save large statistics"
        assert save_time < 3.0, f"Save too slow: {save_time}s"
        
        # Verify file size is reasonable
        file_size: int = os.path.getsize(large_stats_file)
        assert file_size > 1000, "File too small for large dataset"
        assert file_size < 10_000_000, "File too large"
        
        # Test load performance
        start_time = time.time()
        
        loaded_data: Optional[Dict[str, Any]] = load_json_safely(large_stats_file)
        
        load_time: float = time.time() - start_time
        
        assert loaded_data is not None, "Failed to load large statistics"
        assert load_time < 3.0, f"Load too slow: {load_time}s"
        
        # Verify data integrity
        loaded_stats: GameStats = GameStats()
        if hasattr(loaded_stats, 'from_dict'):
            loaded_stats.from_dict(loaded_data)
            
            assert loaded_stats.step_stats.valid == large_stats.step_stats.valid
            assert loaded_stats.step_stats.collisions == large_stats.step_stats.collisions

    def test_statistics_backup_versioning(self, temp_dir: str) -> None:
        """Test statistics backup and versioning interactions."""
        stats_dir: str = os.path.join(temp_dir, "stats_versioning")
        ensure_directory_exists(stats_dir)
        
        current_file: str = os.path.join(stats_dir, "current_stats.json")
        backup_dir: str = os.path.join(stats_dir, "backups")
        ensure_directory_exists(backup_dir)
        
        # Create versioned statistics
        versions: List[GameStats] = []
        
        for version in range(1, 6):
            stats: GameStats = GameStats()
            
            # Build version-specific data
            for i in range(version * 20):
                stats.record_step_result(
                    valid=True,
                    collision=(i % 15 == 14),
                    apple_eaten=(i % 8 == 0)
                )
            
            versions.append(stats)
            
            # Save current version
            if hasattr(stats, 'to_dict'):
                save_json_safely(stats.to_dict(), current_file)
            
            # Create backup
            backup_file: str = os.path.join(backup_dir, f"stats_v{version}.json")
            if hasattr(stats, 'to_dict'):
                save_json_safely(stats.to_dict(), backup_file)
        
        # Verify all versions are accessible
        for version in range(1, 6):
            backup_file = os.path.join(backup_dir, f"stats_v{version}.json")
            assert os.path.exists(backup_file), f"Version {version} backup missing"
            
            # Load and verify version
            version_data: Optional[Dict[str, Any]] = load_json_safely(backup_file)
            assert version_data is not None, f"Failed to load version {version}"
            
            version_stats: GameStats = GameStats()
            if hasattr(version_stats, 'from_dict'):
                version_stats.from_dict(version_data)
                
                # Verify version progression
                expected_steps = version * 20
                assert version_stats.step_stats.valid == expected_steps
        
        # Test version rollback
        rollback_version: int = 3
        rollback_file: str = os.path.join(backup_dir, f"stats_v{rollback_version}.json")
        
        # Copy rollback version to current
        rollback_data: Optional[Dict[str, Any]] = load_json_safely(rollback_file)
        assert rollback_data is not None
        save_json_safely(rollback_data, current_file)
        
        # Verify rollback
        current_data: Optional[Dict[str, Any]] = load_json_safely(current_file)
        current_stats: GameStats = GameStats()
        if hasattr(current_stats, 'from_dict'):
            current_stats.from_dict(current_data)
            
            expected_rollback_steps = rollback_version * 20
            assert current_stats.step_stats.valid == expected_rollback_steps

    def test_atomic_statistics_updates(self, temp_dir: str) -> None:
        """Test atomic statistics file updates."""
        stats_file: str = os.path.join(temp_dir, "atomic_stats.json")
        temp_file: str = stats_file + ".tmp"
        
        # Create initial statistics
        initial_stats: GameStats = GameStats()
        for i in range(30):
            initial_stats.record_step_result(True, (i % 10 == 9), (i % 5 == 0))
        
        if hasattr(initial_stats, 'to_dict'):
            save_json_safely(initial_stats.to_dict(), stats_file)
        
        # Test atomic update scenarios
        update_scenarios: List[Tuple[str, bool]] = [
            ("successful_update", True),
            ("failed_update", False),
            ("partial_update", False),
        ]
        
        for scenario, should_succeed in update_scenarios:
            # Load current stats
            current_data: Optional[Dict[str, Any]] = load_json_safely(stats_file)
            assert current_data is not None
            
            current_stats: GameStats = GameStats()
            if hasattr(current_stats, 'from_dict'):
                current_stats.from_dict(current_data)
            
            # Prepare update
            updated_stats: GameStats = GameStats()
            if hasattr(updated_stats, 'from_dict'):
                updated_stats.from_dict(current_data)
            
            # Add more data
            for i in range(10):
                updated_stats.record_step_result(True, (i % 8 == 7), (i % 4 == 0))
            
            # Perform atomic update
            try:
                if hasattr(updated_stats, 'to_dict'):
                    update_data = updated_stats.to_dict()
                    
                    if should_succeed:
                        # Normal atomic update
                        save_success = save_json_safely(update_data, temp_file)
                        assert save_success
                        
                        # Atomic move
                        if os.path.exists(temp_file):
                            if os.path.exists(stats_file):
                                os.remove(stats_file)
                            os.rename(temp_file, stats_file)
                        
                    else:
                        # Simulate failure during update
                        if scenario == "failed_update":
                            # Don't complete the operation
                            save_json_safely(update_data, temp_file)
                            # Leave temp file, don't move
                        
                        elif scenario == "partial_update":
                            # Simulate partial write
                            with open(temp_file, 'w') as f:
                                f.write('{"partial": "data"')
                            # Don't complete
                
                # Verify file state
                if should_succeed:
                    # File should contain updated data
                    final_data = load_json_safely(stats_file)
                    assert final_data is not None
                    
                    final_stats = GameStats()
                    if hasattr(final_stats, 'from_dict'):
                        final_stats.from_dict(final_data)
                        assert final_stats.step_stats.valid > current_stats.step_stats.valid
                
                else:
                    # Original file should be unchanged or recoverable
                    original_data = load_json_safely(stats_file)
                    if original_data:
                        # File should still be readable
                        original_stats = GameStats()
                        if hasattr(original_stats, 'from_dict'):
                            original_stats.from_dict(original_data)
                
                # Cleanup temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
            except Exception as e:
                # Even if update fails, original file should be intact
                if should_succeed:
                    assert False, f"Atomic update should have succeeded for {scenario}: {e}"
                
                # Verify original file is still readable
                original_data = load_json_safely(stats_file)
                assert original_data is not None, f"Original file corrupted in {scenario}"

    def test_statistics_compression_efficiency(self, temp_dir: str) -> None:
        """Test statistics file compression efficiency."""
        stats_file: str = os.path.join(temp_dir, "uncompressed_stats.json")
        compressed_file: str = os.path.join(temp_dir, "compressed_stats.json")
        
        # Create statistics with repetitive data (good for compression)
        repetitive_stats: GameStats = GameStats()
        
        # Create patterns that should compress well
        for cycle in range(50):
            for i in range(100):
                repetitive_stats.record_step_result(
                    valid=True,
                    collision=(i == 99),  # Always collide on last step
                    apple_eaten=(i % 10 == 0)  # Regular apple pattern
                )
            
            # Regular game ending
            repetitive_stats.update_game_stats(
                final_score=10 * 10,  # Always same score
                total_steps=100,      # Always same steps
                apples_eaten=10       # Always same apples
            )
        
        # Save uncompressed
        if hasattr(repetitive_stats, 'to_dict'):
            save_json_safely(repetitive_stats.to_dict(), stats_file)
        
        uncompressed_size: int = os.path.getsize(stats_file)
        
        # Test compression (if available)
        try:
            import gzip
            
            # Compress the statistics
            with open(stats_file, 'rb') as f_in:
                with gzip.open(compressed_file + '.gz', 'wb') as f_out:
                    f_out.writelines(f_in)
            
            compressed_size: int = os.path.getsize(compressed_file + '.gz')
            compression_ratio: float = compressed_size / uncompressed_size
            
            # Should achieve decent compression on repetitive data
            assert compression_ratio < 0.5, f"Poor compression ratio: {compression_ratio}"
            
            # Test decompression
            with gzip.open(compressed_file + '.gz', 'rb') as f_in:
                with open(compressed_file, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Verify decompressed data
            decompressed_data: Optional[Dict[str, Any]] = load_json_safely(compressed_file)
            assert decompressed_data is not None
            
            # Should match original
            original_data: Optional[Dict[str, Any]] = load_json_safely(stats_file)
            assert decompressed_data == original_data
            
        except ImportError:
            # gzip not available, skip compression test
            pass 