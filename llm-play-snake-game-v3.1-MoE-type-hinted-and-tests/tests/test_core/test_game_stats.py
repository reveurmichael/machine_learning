"""Tests for core.game_stats module."""

import pytest
from unittest.mock import Mock, patch
import time
from datetime import datetime

from core.game_stats import (
    TimeStats,
    TokenStats,
    RoundData,
    GameStatistics,
    StepStats,
    RoundBuffer,
)


class TestTimeStats:
    """Test suite for TimeStats class."""

    def test_initialization(self) -> None:
        """Test TimeStats initialization."""
        start_time = time.time()
        stats = TimeStats(start_time=start_time)
        
        assert stats.start_time == start_time
        assert stats.llm_communication_time == 0.0
        assert stats.end_time is None

    def test_initialization_with_params(self) -> None:
        """Test TimeStats initialization with all parameters."""
        start_time = time.time()
        end_time = start_time + 100
        stats = TimeStats(
            start_time=start_time,
            llm_communication_time=50.5,
            end_time=end_time
        )
        
        assert stats.start_time == start_time
        assert stats.llm_communication_time == 50.5
        assert stats.end_time == end_time

    def test_add_llm_comm(self) -> None:
        """Test adding LLM communication time."""
        stats = TimeStats(start_time=time.time())
        
        stats.add_llm_comm(10.5)
        assert stats.llm_communication_time == 10.5
        
        stats.add_llm_comm(5.2)
        assert stats.llm_communication_time == 15.7

    @patch('time.time')
    def test_asdict_without_end_time(self, mock_time: Mock) -> None:
        """Test asdict when end_time is None."""
        start_time = 1000.0
        current_time = 1100.0
        mock_time.return_value = current_time
        
        stats = TimeStats(start_time=start_time, llm_communication_time=25.0)
        result = stats.asdict()
        
        assert result["total_duration_seconds"] == 100.0
        assert result["llm_communication_time"] == 25.0
        assert "start_time" in result
        assert "end_time" in result

    def test_asdict_with_end_time(self) -> None:
        """Test asdict when end_time is set."""
        start_time = 1000.0
        end_time = 1150.0
        
        stats = TimeStats(
            start_time=start_time,
            end_time=end_time,
            llm_communication_time=30.0
        )
        result = stats.asdict()
        
        assert result["total_duration_seconds"] == 150.0
        assert result["llm_communication_time"] == 30.0

    @patch('time.time')
    def test_record_end_time(self, mock_time: Mock) -> None:
        """Test recording end time."""
        mock_time.return_value = 2000.0
        stats = TimeStats(start_time=1000.0)
        
        stats.record_end_time()
        
        assert stats.end_time == 2000.0


class TestTokenStats:
    """Test suite for TokenStats class."""

    def test_initialization(self) -> None:
        """Test TokenStats initialization."""
        stats = TokenStats(prompt_tokens=100, completion_tokens=50)
        
        assert stats.prompt_tokens == 100
        assert stats.completion_tokens == 50

    def test_total_tokens_property(self) -> None:
        """Test total_tokens property calculation."""
        stats = TokenStats(prompt_tokens=100, completion_tokens=50)
        
        assert stats.total_tokens == 150

    def test_asdict(self) -> None:
        """Test asdict method."""
        stats = TokenStats(prompt_tokens=200, completion_tokens=75)
        result = stats.asdict()
        
        expected = {
            "prompt_tokens": 200,
            "completion_tokens": 75,
            "total_tokens": 275
        }
        assert result == expected


class TestRoundData:
    """Test suite for RoundData class."""

    def test_initialization_defaults(self) -> None:
        """Test RoundData initialization with defaults."""
        data = RoundData()
        
        assert data.apple_position is None
        assert data.moves == []
        assert data.primary_response_times == []
        assert data.secondary_response_times == []
        assert data.primary_token_stats == []
        assert data.secondary_token_stats == []
        assert data.planned_moves == []

    def test_initialization_with_data(self) -> None:
        """Test RoundData initialization with data."""
        data = RoundData(
            apple_position=[5, 6],
            moves=["UP", "DOWN"],
            planned_moves=["UP", "DOWN", "LEFT"]
        )
        
        assert data.apple_position == [5, 6]
        assert data.moves == ["UP", "DOWN"]
        assert data.planned_moves == ["UP", "DOWN", "LEFT"]

    def test_asdict(self) -> None:
        """Test asdict method."""
        data = RoundData(
            apple_position=[3, 4],
            moves=["LEFT"],
            primary_response_times=[1.5, 2.0],
            planned_moves=["LEFT", "RIGHT"]
        )
        
        result = data.asdict()
        
        expected = {
            "apple_position": [3, 4],
            "moves": ["LEFT"],
            "planned_moves": ["LEFT", "RIGHT"],
            "primary_response_times": [1.5, 2.0],
            "secondary_response_times": [],
            "primary_token_stats": [],
            "secondary_token_stats": []
        }
        assert result == expected


class TestStepStats:
    """Test suite for StepStats class."""

    def test_initialization(self) -> None:
        """Test StepStats initialization."""
        stats = StepStats()
        
        assert stats.valid == 0
        assert stats.empty == 0
        assert stats.something_wrong == 0
        assert stats.invalid_reversals == 0
        assert stats.no_path_found == 0

    def test_initialization_with_values(self) -> None:
        """Test StepStats initialization with custom values."""
        stats = StepStats(
            valid=10,
            empty=2,
            something_wrong=1,
            invalid_reversals=3,
            no_path_found=1
        )
        
        assert stats.valid == 10
        assert stats.empty == 2
        assert stats.something_wrong == 1
        assert stats.invalid_reversals == 3
        assert stats.no_path_found == 1

    def test_asdict(self) -> None:
        """Test asdict method."""
        stats = StepStats(
            valid=15,
            empty=3,
            something_wrong=2,
            invalid_reversals=1,
            no_path_found=2
        )
        
        result = stats.asdict()
        
        expected = {
            "valid_steps": 15,
            "empty_steps": 3,
            "something_is_wrong_steps": 2,
            "invalid_reversals": 1,
            "no_path_found_steps": 2
        }
        assert result == expected


class TestRoundBuffer:
    """Test suite for RoundBuffer class."""

    def test_initialization(self) -> None:
        """Test RoundBuffer initialization."""
        buffer = RoundBuffer(number=5)
        
        assert buffer.number == 5
        assert buffer.apple_position is None
        assert buffer.moves == []
        assert buffer.planned_moves == []
        assert buffer.primary_times == []
        assert buffer.secondary_times == []

    def test_initialization_with_data(self) -> None:
        """Test RoundBuffer initialization with data."""
        buffer = RoundBuffer(
            number=3,
            apple_position=[1, 2],
            moves=["UP"],
            planned_moves=["UP", "DOWN"]
        )
        
        assert buffer.number == 3
        assert buffer.apple_position == [1, 2]
        assert buffer.moves == ["UP"]
        assert buffer.planned_moves == ["UP", "DOWN"]

    def test_add_move(self) -> None:
        """Test adding moves to buffer."""
        buffer = RoundBuffer(number=1)
        
        buffer.add_move("UP")
        assert buffer.moves == ["UP"]
        
        buffer.add_move("DOWN")
        assert buffer.moves == ["UP", "DOWN"]

    def test_set_apple(self) -> None:
        """Test setting apple position."""
        buffer = RoundBuffer(number=1)
        
        buffer.set_apple([5, 6])
        assert buffer.apple_position == [5, 6]
        
        buffer.set_apple(None)
        assert buffer.apple_position is None

    def test_is_empty_true(self) -> None:
        """Test is_empty returns True for empty buffer."""
        buffer = RoundBuffer(number=1)
        
        assert buffer.is_empty() is True

    def test_is_empty_false_with_moves(self) -> None:
        """Test is_empty returns False when moves exist."""
        buffer = RoundBuffer(number=1)
        buffer.add_move("UP")
        
        assert buffer.is_empty() is False

    def test_is_empty_false_with_planned_moves(self) -> None:
        """Test is_empty returns False when planned moves exist."""
        buffer = RoundBuffer(number=1)
        buffer.planned_moves = ["DOWN"]
        
        assert buffer.is_empty() is False


class TestGameStatistics:
    """Test suite for GameStatistics class."""

    @patch('time.time')
    def test_initialization(self, mock_time: Mock) -> None:
        """Test GameStatistics initialization."""
        mock_time.return_value = 1000.0
        stats = GameStatistics()
        
        assert isinstance(stats.time_stats, TimeStats)
        assert stats.time_stats.start_time == 1000.0
        assert isinstance(stats.step_stats, StepStats)
        assert stats.primary_response_times == []
        assert stats.secondary_response_times == []
        assert stats.primary_token_stats == []
        assert stats.secondary_token_stats == []
        assert stats.primary_total_tokens == 0
        assert stats.secondary_total_tokens == 0
        assert stats.primary_llm_requests == 0
        assert stats.secondary_llm_requests == 0
        assert stats.last_action_time is None

    @patch('time.perf_counter')
    def test_record_llm_communication_start(self, mock_perf: Mock) -> None:
        """Test recording LLM communication start."""
        mock_perf.return_value = 100.0
        stats = GameStatistics()
        
        stats.record_llm_communication_start()
        
        assert stats.last_action_time == 100.0

    @patch('time.perf_counter')
    def test_record_llm_communication_end(self, mock_perf: Mock) -> None:
        """Test recording LLM communication end."""
        mock_perf.side_effect = [100.0, 105.5]  # start, end
        stats = GameStatistics()
        
        stats.record_llm_communication_start()
        stats.record_llm_communication_end()
        
        assert stats.last_action_time is None
        assert stats.time_stats.llm_communication_time == 5.5

    def test_record_llm_communication_end_no_start(self) -> None:
        """Test recording LLM communication end without start."""
        stats = GameStatistics()
        
        # Should not crash
        stats.record_llm_communication_end()
        
        assert stats.last_action_time is None
        assert stats.time_stats.llm_communication_time == 0.0

    def test_record_primary_response_time(self) -> None:
        """Test recording primary response time."""
        stats = GameStatistics()
        
        stats.record_primary_response_time(2.5)
        assert stats.primary_response_times == [2.5]
        
        stats.record_primary_response_time(3.0)
        assert stats.primary_response_times == [2.5, 3.0]

    def test_record_secondary_response_time(self) -> None:
        """Test recording secondary response time."""
        stats = GameStatistics()
        
        stats.record_secondary_response_time(1.5)
        assert stats.secondary_response_times == [1.5]
        
        stats.record_secondary_response_time(2.0)
        assert stats.secondary_response_times == [1.5, 2.0]

    def test_update_primary_averages(self) -> None:
        """Test updating primary averages."""
        stats = GameStatistics()
        
        # Add some token stats
        stats.record_primary_token_stats(100, 50)
        stats.record_primary_token_stats(200, 75)
        
        # Averages should be calculated
        assert stats.primary_avg_prompt_tokens == 150.0
        assert stats.primary_avg_completion_tokens == 62.5
        assert stats.primary_avg_total_tokens == 212.5

    def test_update_secondary_averages(self) -> None:
        """Test updating secondary averages."""
        stats = GameStatistics()
        
        # Add some token stats
        stats.record_secondary_token_stats(80, 40)
        stats.record_secondary_token_stats(120, 60)
        
        # Averages should be calculated
        assert stats.secondary_avg_prompt_tokens == 100.0
        assert stats.secondary_avg_completion_tokens == 50.0
        assert stats.secondary_avg_total_tokens == 150.0

    def test_record_primary_token_stats(self) -> None:
        """Test recording primary token statistics."""
        stats = GameStatistics()
        
        stats.record_primary_token_stats(100, 50)
        
        assert len(stats.primary_token_stats) == 1
        assert stats.primary_token_stats[0].prompt_tokens == 100
        assert stats.primary_token_stats[0].completion_tokens == 50
        assert stats.primary_llm_requests == 1
        assert stats.primary_total_prompt_tokens == 100
        assert stats.primary_total_completion_tokens == 50
        assert stats.primary_total_tokens == 150
        
        # Add another
        stats.record_primary_token_stats(200, 75)
        
        assert len(stats.primary_token_stats) == 2
        assert stats.primary_llm_requests == 2
        assert stats.primary_total_prompt_tokens == 300
        assert stats.primary_total_completion_tokens == 125
        assert stats.primary_total_tokens == 425

    def test_record_secondary_token_stats(self) -> None:
        """Test recording secondary token statistics."""
        stats = GameStatistics()
        
        stats.record_secondary_token_stats(80, 40)
        
        assert len(stats.secondary_token_stats) == 1
        assert stats.secondary_token_stats[0].prompt_tokens == 80
        assert stats.secondary_token_stats[0].completion_tokens == 40
        assert stats.secondary_llm_requests == 1
        assert stats.secondary_total_prompt_tokens == 80
        assert stats.secondary_total_completion_tokens == 40
        assert stats.secondary_total_tokens == 120

    def test_averages_with_zero_requests(self) -> None:
        """Test averages calculation with zero requests."""
        stats = GameStatistics()
        
        # Should not divide by zero
        stats._update_primary_averages()
        stats._update_secondary_averages()
        
        assert stats.primary_avg_prompt_tokens == 0
        assert stats.primary_avg_completion_tokens == 0
        assert stats.primary_avg_total_tokens == 0
        assert stats.secondary_avg_prompt_tokens == 0
        assert stats.secondary_avg_completion_tokens == 0
        assert stats.secondary_avg_total_tokens == 0


class TestGameStatisticsIntegration:
    """Integration tests for GameStatistics."""

    @patch('time.perf_counter')
    def test_complete_timing_cycle(self, mock_perf: Mock) -> None:
        """Test complete timing measurement cycle."""
        mock_perf.side_effect = [100.0, 102.5, 105.0, 108.0]
        stats = GameStatistics()
        
        # First communication
        stats.record_llm_communication_start()
        stats.record_llm_communication_end()
        
        # Second communication
        stats.record_llm_communication_start()
        stats.record_llm_communication_end()
        
        # Total communication time should be accumulated
        assert stats.time_stats.llm_communication_time == 5.5  # 2.5 + 3.0

    def test_complete_token_tracking(self) -> None:
        """Test complete token tracking for both primary and secondary."""
        stats = GameStatistics()
        
        # Record multiple requests
        stats.record_primary_token_stats(100, 50)
        stats.record_primary_token_stats(200, 75)
        stats.record_secondary_token_stats(80, 40)
        
        # Verify totals
        assert stats.primary_total_tokens == 425
        assert stats.secondary_total_tokens == 120
        
        # Verify averages
        assert stats.primary_avg_total_tokens == 212.5
        assert stats.secondary_avg_total_tokens == 120.0
        
        # Verify request counts
        assert stats.primary_llm_requests == 2
        assert stats.secondary_llm_requests == 1

    def test_mixed_statistics_tracking(self) -> None:
        """Test tracking mixed statistics simultaneously."""
        stats = GameStatistics()
        
        # Record response times
        stats.record_primary_response_time(1.5)
        stats.record_secondary_response_time(0.8)
        
        # Record token stats
        stats.record_primary_token_stats(150, 60)
        
        # Update step stats
        stats.step_stats.valid = 10
        stats.step_stats.empty = 2
        
        # Verify all data is maintained
        assert stats.primary_response_times == [1.5]
        assert stats.secondary_response_times == [0.8]
        assert stats.primary_total_tokens == 210
        assert stats.step_stats.valid == 10
        assert stats.step_stats.empty == 2

    def test_comprehensive_statistics_lifecycle(self) -> None:
        """Test comprehensive statistics lifecycle management."""
        stats = GameStatistics()
        
        # Simulate a complete game session
        session_data = [
            {"primary_tokens": (100, 50), "secondary_tokens": (80, 40), "primary_time": 1.5, "secondary_time": 0.8},
            {"primary_tokens": (200, 75), "secondary_tokens": (120, 60), "primary_time": 2.1, "secondary_time": 1.2},
            {"primary_tokens": (150, 60), "secondary_tokens": (90, 45), "primary_time": 1.8, "secondary_time": 0.9},
            {"primary_tokens": (180, 70), "secondary_tokens": (110, 55), "primary_time": 2.0, "secondary_time": 1.1}
        ]
        
        # Process all session data
        for data in session_data:
            # Record token statistics
            stats.record_primary_token_stats(*data["primary_tokens"])
            stats.record_secondary_token_stats(*data["secondary_tokens"])
            
            # Record response times
            stats.record_primary_response_time(data["primary_time"])
            stats.record_secondary_response_time(data["secondary_time"])
        
        # Verify comprehensive statistics
        assert stats.primary_llm_requests == 4
        assert stats.secondary_llm_requests == 4
        assert stats.primary_total_prompt_tokens == 630  # 100+200+150+180
        assert stats.primary_total_completion_tokens == 255  # 50+75+60+70
        assert stats.secondary_total_prompt_tokens == 400  # 80+120+90+110
        assert stats.secondary_total_completion_tokens == 200  # 40+60+45+55
        
        # Verify averages
        assert stats.primary_avg_prompt_tokens == 157.5
        assert stats.primary_avg_completion_tokens == 63.75
        assert stats.secondary_avg_prompt_tokens == 100.0
        assert stats.secondary_avg_completion_tokens == 50.0
        
        # Verify response times
        assert len(stats.primary_response_times) == 4
        assert len(stats.secondary_response_times) == 4

    def test_comprehensive_step_statistics(self) -> None:
        """Test comprehensive step statistics management."""
        stats = GameStatistics()
        
        # Simulate various step types
        step_scenarios = [
            {"valid": 45, "empty": 3, "wrong": 2, "reversals": 1, "no_path": 1},
            {"valid": 67, "empty": 5, "wrong": 3, "reversals": 2, "no_path": 0},
            {"valid": 23, "empty": 8, "wrong": 1, "reversals": 0, "no_path": 2},
            {"valid": 89, "empty": 2, "wrong": 4, "reversals": 3, "no_path": 1}
        ]
        
        for scenario in step_scenarios:
            # Create new step stats for each scenario
            step_stats = StepStats(
                valid=scenario["valid"],
                empty=scenario["empty"],
                something_wrong=scenario["wrong"],
                invalid_reversals=scenario["reversals"],
                no_path_found=scenario["no_path"]
            )
            
            # Verify step stats dictionary representation
            step_dict = step_stats.asdict()
            assert step_dict["valid_steps"] == scenario["valid"]
            assert step_dict["empty_steps"] == scenario["empty"]
            assert step_dict["something_is_wrong_steps"] == scenario["wrong"]
            assert step_dict["invalid_reversals"] == scenario["reversals"]
            assert step_dict["no_path_found_steps"] == scenario["no_path"]

    def test_comprehensive_timing_accuracy(self) -> None:
        """Test comprehensive timing accuracy and precision."""
        with patch('time.perf_counter') as mock_perf:
            stats = GameStatistics()
            
            # Simulate precise timing measurements
            timing_scenarios = [
                (100.0, 102.5),  # 2.5 seconds
                (105.0, 106.8),  # 1.8 seconds
                (110.0, 113.2),  # 3.2 seconds
                (115.0, 116.1)   # 1.1 seconds
            ]
            
            total_comm_time = 0.0
            
            for start_time, end_time in timing_scenarios:
                mock_perf.side_effect = [start_time, end_time]
                
                stats.record_llm_communication_start()
                stats.record_llm_communication_end()
                
                total_comm_time += (end_time - start_time)
            
            # Verify accumulated timing
            assert stats.time_stats.llm_communication_time == total_comm_time
            assert abs(stats.time_stats.llm_communication_time - 8.6) < 0.001

    def test_comprehensive_token_efficiency_analysis(self) -> None:
        """Test comprehensive token efficiency analysis."""
        stats = GameStatistics()
        
        # Simulate different efficiency scenarios
        efficiency_tests = [
            # High efficiency (low prompt, high completion)
            {"prompt": 50, "completion": 200, "efficiency": 4.0},
            # Medium efficiency
            {"prompt": 100, "completion": 150, "efficiency": 1.5},
            # Low efficiency (high prompt, low completion)
            {"prompt": 300, "completion": 100, "efficiency": 0.33},
            # Balanced efficiency
            {"prompt": 150, "completion": 150, "efficiency": 1.0}
        ]
        
        for test in efficiency_tests:
            stats.record_primary_token_stats(test["prompt"], test["completion"])
            
            # Calculate efficiency for the last request
            last_token_stat = stats.primary_token_stats[-1]
            efficiency = last_token_stat.completion_tokens / last_token_stat.prompt_tokens
            assert abs(efficiency - test["efficiency"]) < 0.1

    def test_comprehensive_memory_efficiency(self) -> None:
        """Test memory efficiency with large datasets."""
        stats = GameStatistics()
        
        # Generate large amounts of data
        for i in range(1000):
            # Vary token amounts
            prompt_tokens = 100 + (i % 200)
            completion_tokens = 50 + (i % 100)
            
            stats.record_primary_token_stats(prompt_tokens, completion_tokens)
            stats.record_primary_response_time(1.0 + (i % 5) * 0.1)
            
            # Every 10th request, add secondary stats
            if i % 10 == 0:
                stats.record_secondary_token_stats(80 + (i % 50), 40 + (i % 30))
                stats.record_secondary_response_time(0.5 + (i % 3) * 0.1)
        
        # Verify data integrity
        assert stats.primary_llm_requests == 1000
        assert stats.secondary_llm_requests == 100
        assert len(stats.primary_response_times) == 1000
        assert len(stats.secondary_response_times) == 100
        
        # Test memory usage is reasonable
        import sys
        primary_tokens_size = sys.getsizeof(stats.primary_token_stats)
        primary_times_size = sys.getsizeof(stats.primary_response_times)
        
        # Should not use excessive memory
        assert primary_tokens_size < 100000  # Less than 100KB
        assert primary_times_size < 50000   # Less than 50KB

    def test_comprehensive_concurrent_operations(self) -> None:
        """Test behavior under simulated concurrent operations."""
        import threading
        import time
        
        stats = GameStatistics()
        errors = []
        
        def record_primary_stats():
            try:
                for i in range(100):
                    stats.record_primary_token_stats(100 + i, 50 + i)
                    stats.record_primary_response_time(1.0 + i * 0.01)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def record_secondary_stats():
            try:
                for i in range(50):
                    stats.record_secondary_token_stats(80 + i, 40 + i)
                    stats.record_secondary_response_time(0.5 + i * 0.01)
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)
        
        def update_step_stats():
            try:
                for i in range(200):
                    stats.step_stats.valid += 1
                    if i % 10 == 0:
                        stats.step_stats.empty += 1
                    if i % 20 == 0:
                        stats.step_stats.something_wrong += 1
                    time.sleep(0.0005)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent operations
        threads = [
            threading.Thread(target=record_primary_stats),
            threading.Thread(target=record_secondary_stats),
            threading.Thread(target=update_step_stats)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should not have errors
        assert len(errors) == 0
        
        # Verify final state is reasonable
        assert stats.primary_llm_requests >= 100
        assert stats.secondary_llm_requests >= 50
        assert stats.step_stats.valid >= 200

    def test_comprehensive_edge_cases(self) -> None:
        """Test comprehensive edge cases and boundary conditions."""
        stats = GameStatistics()
        
        # Test with zero values
        stats.record_primary_token_stats(0, 0)
        stats.record_secondary_token_stats(0, 0)
        stats.record_primary_response_time(0.0)
        stats.record_secondary_response_time(0.0)
        
        assert stats.primary_total_tokens == 0
        assert stats.secondary_total_tokens == 0
        assert stats.primary_response_times == [0.0]
        assert stats.secondary_response_times == [0.0]
        
        # Test with very large values
        large_prompt = 999999
        large_completion = 888888
        large_time = 999.999
        
        stats.record_primary_token_stats(large_prompt, large_completion)
        stats.record_primary_response_time(large_time)
        
        assert stats.primary_total_prompt_tokens == large_prompt
        assert stats.primary_total_completion_tokens == large_completion
        assert large_time in stats.primary_response_times
        
        # Test step stats edge cases
        stats.step_stats.valid = 999999
        stats.step_stats.empty = 888888
        
        step_dict = stats.step_stats.asdict()
        assert step_dict["valid_steps"] == 999999
        assert step_dict["empty_steps"] == 888888

    def test_comprehensive_serialization_compatibility(self) -> None:
        """Test comprehensive serialization compatibility."""
        stats = GameStatistics()
        
        # Build complex statistics
        for i in range(10):
            stats.record_primary_token_stats(100 + i * 10, 50 + i * 5)
            stats.record_secondary_token_stats(80 + i * 8, 40 + i * 4)
            stats.record_primary_response_time(1.0 + i * 0.1)
            stats.record_secondary_response_time(0.5 + i * 0.05)
        
        # Update step stats
        stats.step_stats.valid = 95
        stats.step_stats.empty = 3
        stats.step_stats.something_wrong = 2
        
        # Test serialization of various components
        import json
        
        # Test step stats serialization
        step_dict = stats.step_stats.asdict()
        step_json = json.dumps(step_dict)
        step_restored = json.loads(step_json)
        assert step_restored["valid_steps"] == 95
        
        # Test token stats serialization
        token_dicts = [ts.asdict() for ts in stats.primary_token_stats]
        token_json = json.dumps(token_dicts)
        token_restored = json.loads(token_json)
        assert len(token_restored) == 10
        assert token_restored[0]["prompt_tokens"] == 100
        
        # Test time stats serialization
        time_dict = stats.time_stats.asdict()
        time_json = json.dumps(time_dict)
        time_restored = json.loads(time_json)
        assert "start_time" in time_restored
        assert "llm_communication_time" in time_restored

    def test_comprehensive_performance_benchmarks(self) -> None:
        """Test performance benchmarks for statistics operations."""
        import time
        
        stats = GameStatistics()
        
        # Benchmark token recording
        start_time = time.time()
        for i in range(1000):
            stats.record_primary_token_stats(100 + i, 50 + i)
        token_time = time.time() - start_time
        
        assert token_time < 0.5  # Should complete quickly
        
        # Benchmark response time recording
        start_time = time.time()
        for i in range(1000):
            stats.record_primary_response_time(1.0 + i * 0.001)
        response_time = time.time() - start_time
        
        assert response_time < 0.1  # Should be very fast
        
        # Benchmark step stats updates
        start_time = time.time()
        for i in range(10000):
            stats.step_stats.valid += 1
            if i % 100 == 0:
                stats.step_stats.empty += 1
        step_time = time.time() - start_time
        
        assert step_time < 0.1  # Should be very fast

    def test_comprehensive_data_integrity_validation(self) -> None:
        """Test comprehensive data integrity validation."""
        stats = GameStatistics()
        
        # Build statistics with known patterns
        expected_primary_total = 0
        expected_secondary_total = 0
        
        for i in range(20):
            prompt_tokens = (i + 1) * 10  # 10, 20, 30, ...
            completion_tokens = (i + 1) * 5   # 5, 10, 15, ...
            
            stats.record_primary_token_stats(prompt_tokens, completion_tokens)
            expected_primary_total += prompt_tokens + completion_tokens
            
            if i % 2 == 0:  # Every other request
                sec_prompt = (i + 1) * 8
                sec_completion = (i + 1) * 4
                stats.record_secondary_token_stats(sec_prompt, sec_completion)
                expected_secondary_total += sec_prompt + sec_completion
        
        # Verify data integrity
        assert stats.primary_total_tokens == expected_primary_total
        assert stats.secondary_total_tokens == expected_secondary_total
        assert stats.primary_llm_requests == 20
        assert stats.secondary_llm_requests == 10
        
        # Verify averages are calculated correctly
        expected_primary_avg = expected_primary_total / 20
        expected_secondary_avg = expected_secondary_total / 10
        
        assert abs(stats.primary_avg_total_tokens - expected_primary_avg) < 0.001
        assert abs(stats.secondary_avg_total_tokens - expected_secondary_avg) < 0.001

    def test_comprehensive_time_statistics_validation(self) -> None:
        """Test comprehensive time statistics validation."""
        with patch('time.time') as mock_time:
            # Test time stats with known values
            mock_time.return_value = 1000.0  # Fixed start time
            
            time_stats = TimeStats(start_time=1000.0)
            
            # Test communication time accumulation
            time_stats.add_llm_comm(2.5)
            time_stats.add_llm_comm(1.8)
            time_stats.add_llm_comm(3.2)
            
            assert time_stats.llm_communication_time == 7.5
            
            # Test end time recording
            mock_time.return_value = 1100.0  # 100 seconds later
            time_stats.record_end_time()
            
            # Test dictionary representation
            time_dict = time_stats.asdict()
            assert time_dict["total_duration_seconds"] == 100.0
            assert time_dict["llm_communication_time"] == 7.5
            assert "start_time" in time_dict
            assert "end_time" in time_dict

    def test_comprehensive_round_data_management(self) -> None:
        """Test comprehensive round data management."""
        # Test round data with various configurations
        round_configs = [
            {
                "apple": [1, 1],
                "moves": ["UP", "RIGHT"],
                "planned": ["UP", "RIGHT", "DOWN"],
                "primary_times": [1.5, 2.0],
                "secondary_times": [0.8],
                "primary_tokens": [{"prompt_tokens": 100, "completion_tokens": 50}],
                "secondary_tokens": [{"prompt_tokens": 80, "completion_tokens": 40}]
            },
            {
                "apple": [5, 5],
                "moves": ["DOWN", "LEFT", "UP"],
                "planned": ["DOWN", "LEFT"],
                "primary_times": [1.8, 2.2, 1.9],
                "secondary_times": [0.9, 1.1],
                "primary_tokens": [
                    {"prompt_tokens": 120, "completion_tokens": 60},
                    {"prompt_tokens": 110, "completion_tokens": 55}
                ],
                "secondary_tokens": [{"prompt_tokens": 90, "completion_tokens": 45}]
            }
        ]
        
        for config in round_configs:
            round_data = RoundData(
                apple_position=config["apple"],
                moves=config["moves"],
                planned_moves=config["planned"],
                primary_response_times=config["primary_times"],
                secondary_response_times=config["secondary_times"],
                primary_token_stats=config["primary_tokens"],
                secondary_token_stats=config["secondary_tokens"]
            )
            
            # Test dictionary representation
            round_dict = round_data.asdict()
            
            assert round_dict["apple_position"] == config["apple"]
            assert round_dict["moves"] == config["moves"]
            assert round_dict["planned_moves"] == config["planned"]
            assert round_dict["primary_response_times"] == config["primary_times"]
            assert round_dict["secondary_response_times"] == config["secondary_times"]
            assert round_dict["primary_token_stats"] == config["primary_tokens"]
            assert round_dict["secondary_token_stats"] == config["secondary_tokens"] 