# Replay Module Test Suite

This directory contains comprehensive tests for the replay package, which handles replaying of previously recorded Snake games.

## Overview

The replay module consists of:
- `replay_engine.py`: Core replay engine that extends GameController
- `replay_utils.py`: Utility functions for loading and parsing game data
- Package initialization and exports

## Test Structure

### Core Test Files

#### `test_replay_engine.py`
Comprehensive tests for the ReplayEngine class covering:
- **Initialization**: Default and custom parameter testing
- **GUI Integration**: GUI setting, state synchronization, drawing
- **State Management**: State building, consistency, updates
- **Game Data Loading**: JSON loading, parsing, error handling
- **Gameplay Control**: Move execution, timing, pause/resume
- **Event Handling**: Keyboard events, navigation, speed control
- **Main Loop**: Run loop, timing, without GUI operation

**Test Classes:**
- `TestReplayEngineInitialization`: Engine setup and configuration
- `TestReplayEngineGUIIntegration`: GUI interaction and state sync
- `TestReplayEngineStateBuilding`: State construction and consistency
- `TestReplayEngineDrawing`: Rendering and display functionality
- `TestReplayEngineGameDataLoading`: Game file loading and parsing
- `TestReplayEngineGameplayControl`: Move execution and timing
- `TestReplayEngineEventHandling`: User input and controls
- `TestReplayEngineMainLoop`: Core game loop functionality
- `TestReplayEngineIntegration`: End-to-end workflow testing

#### `test_replay_utils.py`
Comprehensive tests for replay utility functions:
- **JSON Loading**: File operations, error handling, path management
- **Data Parsing**: Game data validation, format conversion
- **Apple Positions**: Multiple format support (dict/list/tuple)
- **Move Sequences**: Validation and extraction
- **Planned Moves**: Round data processing and extraction
- **LLM Information**: Provider and model string building
- **Metadata Handling**: Timestamp and game statistics
- **Error Recovery**: Graceful handling of malformed data

**Test Classes:**
- `TestLoadGameJson`: File loading, error conditions, path handling
- `TestParseGameData`: Data parsing, validation, format conversion
- `TestReplayUtilsIntegration`: Complete workflow testing

#### `test_replay_integration.py`
Integration tests for the complete replay module:
- **Module Imports**: Package structure and exports
- **Multi-Game Sessions**: Sequential game loading and navigation
- **State Persistence**: Data consistency across game switches
- **GUI Integration**: Complete GUI workflow testing
- **Move Execution**: End-to-end move processing
- **Timing Control**: Pause/resume and speed control
- **Error Handling**: Graceful error recovery
- **Data Validation**: Consistency checking throughout workflow
- **Memory Management**: Proper cleanup and resource management

**Test Classes:**
- `TestReplayModuleIntegration`: Complete module functionality testing

#### `test_replay_performance.py`
Performance and stress tests:
- **Loading Performance**: Large file handling, multiple games
- **State Building**: High-frequency state access patterns
- **Move Execution**: Batch move processing performance
- **JSON Parsing**: Large data structure parsing efficiency
- **Memory Usage**: Memory efficiency and cleanup testing
- **Stress Testing**: Rapid game switching, concurrent operations
- **Error Recovery**: Performance under error conditions

**Test Classes:**
- `TestReplayPerformance`: Performance benchmarks and timing
- `TestReplayMemoryUsage`: Memory efficiency and cleanup
- `TestReplayStressTests`: High-load and error conditions

## Test Coverage

### Functionality Coverage
- ✅ **Initialization**: All constructor parameters and defaults
- ✅ **Game Loading**: JSON file loading, parsing, validation
- ✅ **State Management**: State building, consistency, updates
- ✅ **Move Execution**: Replay move processing and validation
- ✅ **GUI Integration**: Drawing, event handling, state sync
- ✅ **Navigation**: Game switching, speed control, pause/resume
- ✅ **Error Handling**: Malformed data, missing files, exceptions
- ✅ **Performance**: Large datasets, high-frequency operations
- ✅ **Memory Management**: Cleanup, resource efficiency

### Component Integration
- ✅ **Engine ↔ Utils**: Data loading and parsing integration
- ✅ **Engine ↔ GUI**: Display and interaction integration
- ✅ **Engine ↔ Core**: GameController inheritance and extension
- ✅ **Utils ↔ File System**: File operations and error handling
- ✅ **Package Structure**: Module imports and exports

### Error Conditions
- ✅ **Missing Files**: Game files not found
- ✅ **Malformed JSON**: Invalid JSON syntax
- ✅ **Invalid Data**: Missing required fields, empty arrays
- ✅ **Permission Errors**: File access restrictions
- ✅ **Memory Issues**: Large dataset handling
- ✅ **State Inconsistencies**: Data validation failures

## Running Tests

### Run All Replay Tests
```bash
pytest tests/test_replay/ -v
```

### Run Specific Test Files
```bash
# Engine tests
pytest tests/test_replay/test_replay_engine.py -v

# Utils tests
pytest tests/test_replay/test_replay_utils.py -v

# Integration tests
pytest tests/test_replay/test_replay_integration.py -v

# Performance tests
pytest tests/test_replay/test_replay_performance.py -v
```

### Run Specific Test Classes
```bash
# Engine initialization tests
pytest tests/test_replay/test_replay_engine.py::TestReplayEngineInitialization -v

# JSON loading tests
pytest tests/test_replay/test_replay_utils.py::TestLoadGameJson -v

# Performance tests
pytest tests/test_replay/test_replay_performance.py::TestReplayPerformance -v
```

### Run Performance Tests Only
```bash
pytest tests/test_replay/test_replay_performance.py -v -m "not slow"
```

## Test Data

Tests use dynamically created test data including:

### Game Data Structure
```json
{
  "game_number": 1,
  "score": 150,
  "metadata": {
    "timestamp": "2024-01-01 10:00:00",
    "round_count": 3
  },
  "detailed_history": {
    "apple_positions": [
      {"x": 5, "y": 6},
      {"x": 7, "y": 8}
    ],
    "moves": ["UP", "RIGHT", "DOWN", "LEFT"],
    "rounds_data": {
      "round_1": {
        "moves": ["PLANNED", "UP", "RIGHT"],
        "llm_response": "Move response"
      }
    }
  },
  "llm_info": {
    "primary_provider": "deepseek",
    "primary_model": "deepseek-reasoner",
    "parser_provider": "mistral",
    "parser_model": "mistral-7b"
  },
  "game_end_reason": "apple_eaten"
}
```

### Test Scenarios
- **Small Games**: 10-50 moves for basic functionality
- **Medium Games**: 500-1000 moves for normal operation
- **Large Games**: 5000-20000 moves for performance testing
- **Invalid Data**: Various malformed data patterns
- **Edge Cases**: Empty arrays, missing fields, format variations

## Mocking Strategy

### External Dependencies
- **File System**: `pathlib.Path`, file operations
- **Time Operations**: `time.time()` for timing control
- **Game Controller**: `move()`, `reset()`, `_update_board()`
- **GUI Components**: Drawing, event handling
- **Utility Functions**: File utilities, path operations

### Mock Patterns
```python
# File operations
with patch('pathlib.Path.exists', return_value=True):
    with patch('pathlib.Path.open', mock_open(read_data=json_data)):
        # Test file loading

# Time control
with patch('time.time', side_effect=[100.0, 101.0, 102.0]):
    # Test timing-dependent operations

# Game controller methods
with patch.object(engine, 'move', return_value=True):
    # Test move execution without actual game logic
```

## Performance Benchmarks

### Expected Performance Metrics
- **Small Game Loading** (< 100 moves): < 0.1 seconds
- **Medium Game Loading** (< 1000 moves): < 1.0 seconds
- **Large Game Loading** (< 10000 moves): < 10.0 seconds
- **State Building**: < 0.001 seconds per operation
- **Move Execution**: < 0.01 seconds per move
- **JSON Parsing**: < 0.5 seconds for 20K moves

### Memory Efficiency
- **Memory Growth**: < 10,000 objects for large datasets
- **Cleanup**: Proper resource deallocation between games
- **State Size**: Efficient handling of large snake positions

## Integration Points

### With Core Module
- **GameController**: Inheritance and method extension
- **Game Data**: State management and consistency
- **Move Processing**: Replay-specific move handling

### With Utils Module
- **File Operations**: Game data loading and parsing
- **Path Management**: Log directory handling
- **JSON Processing**: Data format conversion

### With Config Module
- **Constants**: Game constants and configuration
- **UI Settings**: Timing and display parameters

## Error Handling

### Graceful Degradation
- **Missing Files**: Continue with available games
- **Parse Errors**: Skip invalid games, continue session
- **State Errors**: Reset to safe state, log issues
- **GUI Errors**: Fallback to headless mode

### Error Recovery
- **Data Validation**: Multi-level validation with fallbacks
- **State Consistency**: Automatic state repair and validation
- **Resource Cleanup**: Proper cleanup on error conditions

## Future Enhancements

### Test Coverage Expansion
- **Network Replay**: Remote game data loading
- **Real-time Replay**: Live game replay functionality
- **Export Features**: Game state export and import
- **Replay Analysis**: Statistical analysis and reporting

### Performance Optimization
- **Lazy Loading**: On-demand data loading
- **Caching**: Intelligent data caching strategies
- **Streaming**: Large dataset streaming support
- **Compression**: Data compression for storage efficiency 