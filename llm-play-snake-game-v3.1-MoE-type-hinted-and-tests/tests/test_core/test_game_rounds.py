"""Tests for core.game_rounds module."""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, List, Optional

from core.game_rounds import RoundManager
from core.game_stats import RoundBuffer


class TestRoundManager:
    """Test suite for RoundManager class."""

    def test_initialization(self) -> None:
        """Test RoundManager initialization."""
        manager = RoundManager()
        
        assert manager.round_count == 1
        assert manager.rounds_data == {}
        assert isinstance(manager.round_buffer, RoundBuffer)
        assert manager.round_buffer.number == 1

    def test_start_new_round(self) -> None:
        """Test starting a new round."""
        manager = RoundManager()
        
        # Add some data to the first round
        manager.round_buffer.add_move("UP")
        manager.round_buffer.planned_moves = ["UP", "DOWN"]
        
        # Start new round
        apple_pos = [5, 6]
        manager.start_new_round(apple_pos)
        
        assert manager.round_count == 2
        assert isinstance(manager.round_buffer, RoundBuffer)
        assert manager.round_buffer.number == 2
        assert manager.round_buffer.apple_position == [5, 6]
        
        # Previous round should be flushed to rounds_data
        assert 1 in manager.rounds_data
        assert manager.rounds_data[1]["moves"] == ["UP"]

    def test_start_new_round_with_none_apple(self) -> None:
        """Test starting a new round with None apple position."""
        manager = RoundManager()
        
        manager.start_new_round(None)
        
        assert manager.round_count == 2
        assert manager.round_buffer.apple_position is None

    def test_start_new_round_with_numpy_array(self) -> None:
        """Test starting a new round with numpy-like array."""
        manager = RoundManager()
        
        # Mock numpy array
        mock_array = Mock()
        mock_array.tolist.return_value = [3, 4]
        
        manager.start_new_round(mock_array)
        
        assert manager.round_count == 2
        assert manager.round_buffer.apple_position == [3, 4]

    def test_record_apple_position(self) -> None:
        """Test recording apple position."""
        manager = RoundManager()
        
        manager.record_apple_position([7, 8])
        
        assert manager.round_buffer.apple_position == [7, 8]
        assert manager.rounds_data[1]["apple_position"] == [7, 8]

    def test_record_apple_position_numpy_array(self) -> None:
        """Test recording apple position with numpy-like array."""
        manager = RoundManager()
        
        # Mock numpy array
        mock_array = Mock()
        mock_array.tolist.return_value = [9, 10]
        
        manager.record_apple_position(mock_array)
        
        assert manager.round_buffer.apple_position == [9, 10]

    def test_record_planned_moves(self) -> None:
        """Test recording planned moves."""
        manager = RoundManager()
        
        moves = ["UP", "DOWN", "LEFT"]
        manager.record_planned_moves(moves)
        
        assert manager.round_buffer.planned_moves == ["UP", "DOWN", "LEFT"]

    def test_record_planned_moves_replacement(self) -> None:
        """Test that planned moves are replaced, not extended."""
        manager = RoundManager()
        
        # Set initial moves
        manager.record_planned_moves(["UP", "DOWN"])
        assert manager.round_buffer.planned_moves == ["UP", "DOWN"]
        
        # Replace with new moves
        manager.record_planned_moves(["LEFT", "RIGHT", "UP"])
        assert manager.round_buffer.planned_moves == ["LEFT", "RIGHT", "UP"]

    def test_record_planned_moves_empty(self) -> None:
        """Test recording empty planned moves."""
        manager = RoundManager()
        
        manager.record_planned_moves([])
        
        # Should not update if moves list is empty
        assert manager.round_buffer.planned_moves == []

    def test_record_planned_moves_no_buffer(self) -> None:
        """Test recording planned moves when buffer is None."""
        manager = RoundManager()
        manager.round_buffer = None
        
        # Should not raise error
        manager.record_planned_moves(["UP"])

    def test_sync_round_data(self) -> None:
        """Test synchronizing round data."""
        manager = RoundManager()
        
        # Set up buffer data
        manager.round_buffer.apple_position = [1, 2]
        manager.round_buffer.planned_moves = ["UP", "DOWN"]
        manager.round_buffer.add_move("UP")
        manager.round_buffer.add_move("DOWN")
        
        manager.sync_round_data()
        
        expected_data = {
            "round": 1,
            "apple_position": [1, 2],
            "planned_moves": ["UP", "DOWN"],
            "moves": ["UP", "DOWN"]
        }
        assert manager.rounds_data[1] == expected_data

    def test_sync_round_data_multiple_calls(self) -> None:
        """Test multiple sync calls don't duplicate planned moves."""
        manager = RoundManager()
        
        manager.round_buffer.planned_moves = ["UP", "DOWN"]
        manager.round_buffer.add_move("UP")
        
        # First sync
        manager.sync_round_data()
        
        # Add more moves and sync again
        manager.round_buffer.add_move("DOWN")
        manager.sync_round_data()
        
        # Planned moves should not be duplicated
        round_data = manager.rounds_data[1]
        assert round_data["planned_moves"] == ["UP", "DOWN"]
        assert round_data["moves"] == ["UP", "DOWN"]

    def test_sync_round_data_no_buffer(self) -> None:
        """Test sync when buffer is None."""
        manager = RoundManager()
        manager.round_buffer = None
        
        # Should not raise error
        manager.sync_round_data()

    def test_flush_buffer(self) -> None:
        """Test flushing the round buffer."""
        manager = RoundManager()
        
        # Add data to buffer
        manager.round_buffer.add_move("UP")
        manager.round_buffer.planned_moves = ["UP", "DOWN"]
        
        manager.flush_buffer()
        
        assert manager.round_buffer is None
        assert 1 in manager.rounds_data

    def test_flush_buffer_empty(self) -> None:
        """Test flushing empty buffer."""
        manager = RoundManager()
        
        # Buffer is empty by default
        manager.flush_buffer()
        
        assert manager.round_buffer is None
        # No data should be synced for empty buffer
        assert manager.rounds_data == {}

    def test_flush_buffer_already_none(self) -> None:
        """Test flushing when buffer is already None."""
        manager = RoundManager()
        manager.round_buffer = None
        
        # Should not raise error
        manager.flush_buffer()

    def test_get_or_create_round_data(self) -> None:
        """Test getting or creating round data."""
        manager = RoundManager()
        
        # First call should create new data
        data1 = manager._get_or_create_round_data(5)
        assert data1 == {"round": 5}
        assert manager.rounds_data[5] == {"round": 5}
        
        # Second call should return existing data
        data1["score"] = 100
        data2 = manager._get_or_create_round_data(5)
        assert data2 is data1
        assert data2["score"] == 100

    def test_get_ordered_rounds_data(self) -> None:
        """Test getting ordered rounds data."""
        manager = RoundManager()
        
        # Add rounds in non-sequential order
        manager._get_or_create_round_data(3)
        manager._get_or_create_round_data(1)
        manager._get_or_create_round_data(5)
        manager._get_or_create_round_data(2)
        
        ordered = manager.get_ordered_rounds_data()
        
        assert list(ordered.keys()) == [1, 2, 3, 5]
        assert all(data["round"] == key for key, data in ordered.items())

    def test_record_parsed_llm_response(self) -> None:
        """Test recording parsed LLM response."""
        manager = RoundManager()
        
        # Should not raise error (currently a no-op)
        manager.record_parsed_llm_response({"moves": ["UP"]}, True)
        manager.record_parsed_llm_response("some response", False)

    def test_to_list_or_none_with_list(self) -> None:
        """Test _to_list_or_none with regular list."""
        result = RoundManager._to_list_or_none([1, 2, 3])
        assert result == [1, 2, 3]

    def test_to_list_or_none_with_tuple(self) -> None:
        """Test _to_list_or_none with tuple."""
        result = RoundManager._to_list_or_none((4, 5, 6))
        assert result == [4, 5, 6]

    def test_to_list_or_none_with_none(self) -> None:
        """Test _to_list_or_none with None."""
        result = RoundManager._to_list_or_none(None)
        assert result is None

    def test_to_list_or_none_with_numpy_array(self) -> None:
        """Test _to_list_or_none with numpy-like array."""
        mock_array = Mock()
        mock_array.tolist.return_value = [7, 8, 9]
        
        result = RoundManager._to_list_or_none(mock_array)
        assert result == [7, 8, 9]


class TestRoundManagerIntegration:
    """Integration tests for RoundManager."""

    def test_complete_round_cycle(self) -> None:
        """Test a complete round cycle."""
        manager = RoundManager()
        
        # Round 1
        manager.record_apple_position([1, 1])
        manager.record_planned_moves(["UP", "RIGHT"])
        manager.round_buffer.add_move("UP")
        manager.round_buffer.add_move("RIGHT")
        
        # Start round 2
        manager.start_new_round([2, 2])
        
        # Verify round 1 was properly saved
        assert 1 in manager.rounds_data
        round1_data = manager.rounds_data[1]
        assert round1_data["apple_position"] == [1, 1]
        assert round1_data["planned_moves"] == ["UP", "RIGHT"]
        assert round1_data["moves"] == ["UP", "RIGHT"]
        
        # Verify round 2 is properly initialized
        assert manager.round_count == 2
        assert manager.round_buffer.number == 2
        assert manager.round_buffer.apple_position == [2, 2]

    def test_multiple_rounds_with_data(self) -> None:
        """Test managing multiple rounds with different data."""
        manager = RoundManager()
        
        # Round 1
        manager.record_apple_position([1, 1])
        manager.record_planned_moves(["UP"])
        manager.round_buffer.add_move("UP")
        
        # Round 2
        manager.start_new_round([2, 2])
        manager.record_planned_moves(["DOWN", "LEFT"])
        manager.round_buffer.add_move("DOWN")
        
        # Round 3
        manager.start_new_round([3, 3])
        manager.record_planned_moves(["RIGHT"])
        
        # Final flush
        manager.flush_buffer()
        
        # Verify all rounds
        ordered = manager.get_ordered_rounds_data()
        assert len(ordered) == 3
        
        assert ordered[1]["moves"] == ["UP"]
        assert ordered[2]["moves"] == ["DOWN"]
        assert ordered[3]["planned_moves"] == ["RIGHT"]

    def test_buffer_state_management(self) -> None:
        """Test buffer state transitions."""
        manager = RoundManager()
        
        # Initial state
        assert manager.round_buffer is not None
        assert manager.round_buffer.number == 1
        
        # Add some data
        manager.round_buffer.add_move("UP")
        assert not manager.round_buffer.is_empty()
        
        # Flush
        manager.flush_buffer()
        assert manager.round_buffer is None
        
        # Start new round
        manager.start_new_round([5, 5])
        assert manager.round_buffer is not None
        assert manager.round_buffer.number == 2

    def test_data_persistence_across_rounds(self) -> None:
        """Test that data persists correctly across round transitions."""
        manager = RoundManager()
        
        # Build up data over multiple rounds
        for round_num in range(1, 6):
            if round_num > 1:
                manager.start_new_round([round_num, round_num])
            
            manager.record_apple_position([round_num, round_num])
            manager.record_planned_moves([f"MOVE_{round_num}"])
            manager.round_buffer.add_move(f"MOVE_{round_num}")
        
        # Final flush
        manager.flush_buffer()
        
        # Verify all data is preserved
        ordered = manager.get_ordered_rounds_data()
        assert len(ordered) == 5
        
        for i in range(1, 6):
            assert ordered[i]["apple_position"] == [i, i]
            assert ordered[i]["planned_moves"] == [f"MOVE_{i}"]
            assert ordered[i]["moves"] == [f"MOVE_{i}"]

    def test_comprehensive_round_lifecycle(self) -> None:
        """Test comprehensive round lifecycle management."""
        manager = RoundManager()
        
        # Test complete lifecycle for multiple rounds
        lifecycle_data = [
            {"apple": [1, 1], "planned": ["UP", "RIGHT"], "executed": ["UP", "RIGHT"]},
            {"apple": [2, 3], "planned": ["DOWN"], "executed": ["DOWN", "LEFT"]},
            {"apple": [4, 5], "planned": ["LEFT", "UP", "RIGHT"], "executed": ["LEFT", "UP"]},
            {"apple": [6, 7], "planned": ["DOWN", "DOWN"], "executed": ["DOWN", "DOWN", "RIGHT"]}
        ]
        
        for i, data in enumerate(lifecycle_data, 1):
            if i > 1:
                manager.start_new_round(data["apple"])
            else:
                manager.record_apple_position(data["apple"])
            
            # Record planned moves
            manager.record_planned_moves(data["planned"])
            
            # Execute moves
            for move in data["executed"]:
                manager.round_buffer.add_move(move)
            
            # Sync data
            manager.sync_round_data()
        
        # Final flush
        manager.flush_buffer()
        
        # Verify complete lifecycle data
        ordered = manager.get_ordered_rounds_data()
        assert len(ordered) == 4
        
        for i, expected in enumerate(lifecycle_data, 1):
            round_data = ordered[i]
            assert round_data["apple_position"] == expected["apple"]
            assert round_data["planned_moves"] == expected["planned"]
            assert round_data["moves"] == expected["executed"]

    def test_comprehensive_error_handling(self) -> None:
        """Test comprehensive error handling scenarios."""
        manager = RoundManager()
        
        # Test with None buffer operations
        manager.round_buffer = None
        
        # These should not crash
        manager.record_apple_position([1, 1])
        manager.record_planned_moves(["UP"])
        manager.sync_round_data()
        manager.flush_buffer()
        
        # Test with invalid data types
        manager.start_new_round([1, 1])
        
        # Test edge cases
        manager.record_planned_moves([])  # Empty moves
        manager.record_planned_moves(None)  # None moves
        
        # Test with various position formats
        test_positions = [
            [0, 0],           # Normal list
            (1, 1),           # Tuple
            None,             # None
        ]
        
        for pos in test_positions:
            try:
                manager.record_apple_position(pos)
            except Exception:
                pass  # Should handle gracefully

    def test_comprehensive_data_integrity(self) -> None:
        """Test comprehensive data integrity across operations."""
        manager = RoundManager()
        
        # Test data integrity with complex operations
        for round_num in range(1, 10):
            if round_num > 1:
                manager.start_new_round([round_num, round_num * 2])
            else:
                manager.record_apple_position([round_num, round_num * 2])
            
            # Add varying amounts of data
            planned_moves = [f"MOVE_{i}" for i in range(round_num)]
            manager.record_planned_moves(planned_moves)
            
            # Execute some moves (not all planned)
            executed_count = min(round_num, 3)
            for i in range(executed_count):
                manager.round_buffer.add_move(f"EXECUTED_{round_num}_{i}")
            
            # Multiple syncs (should not duplicate data)
            manager.sync_round_data()
            manager.sync_round_data()
            manager.sync_round_data()
        
        # Final flush
        manager.flush_buffer()
        
        # Verify data integrity
        ordered = manager.get_ordered_rounds_data()
        assert len(ordered) == 9
        
        for round_num in range(1, 10):
            round_data = ordered[round_num]
            assert round_data["apple_position"] == [round_num, round_num * 2]
            assert len(round_data["planned_moves"]) == round_num
            
            # Executed moves should not exceed 3
            executed_count = min(round_num, 3)
            assert len(round_data["moves"]) == executed_count

    def test_comprehensive_buffer_management(self) -> None:
        """Test comprehensive buffer management scenarios."""
        manager = RoundManager()
        
        # Test buffer state transitions
        initial_buffer = manager.round_buffer
        assert initial_buffer is not None
        assert initial_buffer.number == 1
        
        # Test buffer with data
        initial_buffer.add_move("TEST_MOVE")
        initial_buffer.planned_moves = ["PLANNED_MOVE"]
        assert not initial_buffer.is_empty()
        
        # Start new round (should flush and create new buffer)
        manager.start_new_round([2, 2])
        
        new_buffer = manager.round_buffer
        assert new_buffer is not None
        assert new_buffer.number == 2
        assert new_buffer != initial_buffer
        assert new_buffer.is_empty()  # New buffer should be empty
        
        # Test multiple buffer operations
        for i in range(5):
            new_buffer.add_move(f"BUFFER_MOVE_{i}")
        
        assert len(new_buffer.moves) == 5
        
        # Test buffer flush
        manager.flush_buffer()
        assert manager.round_buffer is None

    def test_comprehensive_performance_characteristics(self) -> None:
        """Test performance characteristics with large datasets."""
        import time
        
        manager = RoundManager()
        
        # Test performance with many rounds
        start_time = time.time()
        
        for round_num in range(100):
            if round_num > 0:
                manager.start_new_round([round_num, round_num])
            else:
                manager.record_apple_position([round_num, round_num])
            
            # Add substantial data to each round
            planned_moves = [f"MOVE_{i}" for i in range(20)]
            manager.record_planned_moves(planned_moves)
            
            for i in range(15):
                manager.round_buffer.add_move(f"EXECUTED_{round_num}_{i}")
        
        manager.flush_buffer()
        
        elapsed_time = time.time() - start_time
        
        # Should complete quickly (under 1 second for 100 rounds)
        assert elapsed_time < 1.0
        
        # Verify data integrity with large dataset
        ordered = manager.get_ordered_rounds_data()
        assert len(ordered) == 100
        
        # Verify memory efficiency
        import sys
        data_size = sys.getsizeof(manager.rounds_data)
        assert data_size < 100000  # Should be reasonable size

    def test_comprehensive_concurrency_simulation(self) -> None:
        """Test behavior under simulated concurrent operations."""
        import threading
        import time
        
        manager = RoundManager()
        errors = []
        
        def add_rounds():
            try:
                for i in range(10):
                    manager.start_new_round([i, i])
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def add_moves():
            try:
                for i in range(50):
                    if manager.round_buffer:
                        manager.round_buffer.add_move(f"CONCURRENT_MOVE_{i}")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def sync_data():
            try:
                for i in range(20):
                    manager.sync_round_data()
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent operations
        threads = [
            threading.Thread(target=add_rounds),
            threading.Thread(target=add_moves),
            threading.Thread(target=sync_data)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should not have errors
        assert len(errors) == 0
        
        # Final cleanup
        manager.flush_buffer()

    def test_comprehensive_edge_cases(self) -> None:
        """Test comprehensive edge cases and boundary conditions."""
        manager = RoundManager()
        
        # Test with extremely large round numbers
        manager.start_new_round([999999, 999999])
        assert manager.round_count == 999999 + 1  # Incremented from 1
        
        # Test with negative coordinates
        manager.record_apple_position([-1, -1])
        assert manager.round_buffer.apple_position == [-1, -1]
        
        # Test with very long move sequences
        long_moves = [f"MOVE_{i}" for i in range(1000)]
        manager.record_planned_moves(long_moves)
        assert len(manager.round_buffer.planned_moves) == 1000
        
        # Test with empty and None data
        manager.record_planned_moves([])
        assert manager.round_buffer.planned_moves == []
        
        manager.record_apple_position(None)
        assert manager.round_buffer.apple_position is None
        
        # Test buffer operations on empty buffer
        empty_manager = RoundManager()
        empty_manager.round_buffer = None
        
        # Should not crash
        empty_manager.record_apple_position([1, 1])
        empty_manager.record_planned_moves(["UP"])
        empty_manager.sync_round_data()

    def test_comprehensive_data_serialization(self) -> None:
        """Test data serialization compatibility."""
        manager = RoundManager()
        
        # Create complex data structure
        for round_num in range(5):
            if round_num > 0:
                manager.start_new_round([round_num, round_num * 2])
            else:
                manager.record_apple_position([round_num, round_num * 2])
            
            planned_moves = [f"PLANNED_{round_num}_{i}" for i in range(3)]
            manager.record_planned_moves(planned_moves)
            
            for i in range(2):
                manager.round_buffer.add_move(f"EXECUTED_{round_num}_{i}")
        
        manager.flush_buffer()
        
        # Test JSON serialization
        import json
        ordered_data = manager.get_ordered_rounds_data()
        
        json_str = json.dumps(ordered_data)
        restored_data = json.loads(json_str)
        
        # Verify data integrity after serialization
        assert len(restored_data) == 5
        for i in range(5):
            round_key = str(i + 1)  # JSON keys are strings
            assert round_key in restored_data
            assert restored_data[round_key]["apple_position"] == [i, i * 2]

    def test_comprehensive_memory_management(self) -> None:
        """Test memory management with large datasets."""
        manager = RoundManager()
        
        # Create large dataset
        for round_num in range(50):
            if round_num > 0:
                manager.start_new_round([round_num, round_num])
            else:
                manager.record_apple_position([round_num, round_num])
            
            # Large planned moves
            planned_moves = [f"MOVE_{i}" for i in range(100)]
            manager.record_planned_moves(planned_moves)
            
            # Large executed moves
            for i in range(80):
                manager.round_buffer.add_move(f"EXEC_{round_num}_{i}")
        
        manager.flush_buffer()
        
        # Test memory efficiency
        import sys
        total_size = sys.getsizeof(manager.rounds_data)
        
        # Should handle large datasets efficiently
        assert total_size < 500000  # Less than 500KB
        
        # Verify data completeness
        ordered = manager.get_ordered_rounds_data()
        assert len(ordered) == 50
        
        # Verify each round has complete data
        for round_num in range(1, 51):
            round_data = ordered[round_num]
            assert len(round_data["planned_moves"]) == 100
            assert len(round_data["moves"]) == 80

    def test_comprehensive_integration_scenarios(self) -> None:
        """Test comprehensive integration scenarios."""
        manager = RoundManager()
        
        # Simulate realistic game scenarios
        game_scenarios = [
            {
                "description": "Quick game - few moves",
                "apple": [1, 1],
                "planned": ["UP", "RIGHT"],
                "executed": ["UP", "RIGHT"]
            },
            {
                "description": "Long game - many moves",
                "apple": [5, 5],
                "planned": ["UP", "RIGHT", "DOWN", "LEFT", "UP", "RIGHT"],
                "executed": ["UP", "RIGHT", "DOWN", "LEFT", "UP"]
            },
            {
                "description": "Failed execution - fewer moves than planned",
                "apple": [3, 7],
                "planned": ["DOWN", "DOWN", "LEFT"],
                "executed": ["DOWN"]
            },
            {
                "description": "Over-execution - more moves than planned",
                "apple": [8, 2],
                "planned": ["LEFT"],
                "executed": ["LEFT", "UP", "RIGHT"]
            }
        ]
        
        for i, scenario in enumerate(game_scenarios):
            if i > 0:
                manager.start_new_round(scenario["apple"])
            else:
                manager.record_apple_position(scenario["apple"])
            
            manager.record_planned_moves(scenario["planned"])
            
            for move in scenario["executed"]:
                manager.round_buffer.add_move(move)
            
            # Record some LLM responses
            manager.record_parsed_llm_response(
                {"moves": scenario["planned"], "reasoning": scenario["description"]},
                is_primary=True
            )
        
        manager.flush_buffer()
        
        # Verify all scenarios were recorded correctly
        ordered = manager.get_ordered_rounds_data()
        assert len(ordered) == len(game_scenarios)
        
        for i, scenario in enumerate(game_scenarios, 1):
            round_data = ordered[i]
            assert round_data["apple_position"] == scenario["apple"]
            assert round_data["planned_moves"] == scenario["planned"]
            assert round_data["moves"] == scenario["executed"] 