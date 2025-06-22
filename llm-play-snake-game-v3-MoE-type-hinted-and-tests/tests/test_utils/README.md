# Utility Module Tests

This directory contains comprehensive tests for all utility modules in the `utils/` folder. Each test file provides thorough coverage of its corresponding utility module with detailed type hints and realistic test scenarios.

## Test Files Overview

### Core Utility Tests

#### `test_json_utils.py` (29KB, 811 lines)
**Comprehensive JSON processing tests**
- JSON parsing with error recovery and validation
- Structured data extraction and transformation
- Complex nested object handling
- Performance testing for large JSON datasets
- Schema validation and data integrity checks
- Edge cases: malformed JSON, encoding issues, circular references

#### `test_moves_utils.py` (25KB, 658 lines)
**Game move processing and validation tests**
- Move command parsing and validation
- Coordinate transformation and boundary checking
- Snake movement calculation and collision detection
- Direction vector processing and path optimization
- Real-time move processing and queue management
- Integration with game state and grid systems

#### `test_file_utils.py` (28KB, 780 lines) 
**File system operations and I/O tests**
- File reading/writing with different formats
- Directory management and path validation
- Atomic file operations and transactional I/O
- File locking and concurrent access handling
- Backup and recovery mechanisms
- Cross-platform compatibility testing

### Network and Communication Tests

#### `test_network_utils.py` (26KB, 614 lines)
**Network communication and HTTP utilities tests**
- HTTP request handling with retry logic
- Connection management and timeout handling
- Network resilience and error recovery
- API rate limiting and throttling
- SSL/TLS connection validation
- Performance monitoring and metrics collection

#### `test_web_utils.py` (27KB, 646 lines)
**Web framework integration tests**
- HTTP request processing and validation
- Session management and cookie handling
- Response formatting and JSON serialization
- Request validation and sanitization
- Error handling and logging
- Web security and CSRF protection

### Advanced System Tests

#### `test_initialization_utils.py` (26KB, 637 lines)
**System initialization and component management tests**
- Component initialization sequencing
- Configuration validation and merging
- Dependency resolution and ordering
- Startup sequence execution
- Health checks and monitoring
- Graceful shutdown procedures

#### `test_continuation_utils.py` (23KB, 505 lines)
**Session continuation and recovery tests**
- Session state checkpoint creation
- State recovery and validation
- Continuation point identification
- Session merging and consolidation
- Integrity checking and corruption recovery
- Multi-fragment session reconstruction

#### `test_session_utils.py` (4.9KB, 125 lines)
**Session lifecycle management tests**
- Session creation and initialization
- State management and updates
- Persistence and recovery operations
- Validation and cleanup procedures
- Concurrent session handling
- Session security and isolation

### Game Management Tests

#### `test_game_manager_utils.py` (29KB, 680 lines)
**Game session coordination tests**
- Multi-game session management
- Player coordination and turn handling
- Game state synchronization
- Resource allocation and cleanup
- Performance monitoring and optimization
- Error handling and recovery workflows

#### `test_game_stats_utils.py` (21KB, 467 lines)
**Statistics and analytics tests**
- Basic game statistics calculation
- Performance trend analysis over time
- Comparative analysis between sessions
- Advanced statistical metrics computation
- Real-time statistics updates
- Data visualization and reporting

### Text Processing Tests

#### `test_text_utils.py` (24KB, 579 lines)
**Text processing and manipulation tests**
- String formatting and template processing
- Text parsing and data extraction
- Validation and sanitization functions
- Text analysis and metrics calculation
- Search and replacement operations
- Encoding and escaping utilities

## Test Design Principles

### Type Safety
All test files include comprehensive type hints following PEP 484 standards:
- **Function signatures**: Complete parameter and return type annotations
- **Variable declarations**: Explicit typing for complex data structures
- **Mock objects**: Properly typed mock specifications
- **Test data**: Typed test cases and expected results

### Test Coverage Patterns
Each test file follows consistent patterns:
- **Comprehensive scenarios**: Multiple test cases covering normal, edge, and error conditions
- **Real-world data**: Realistic test data reflecting actual usage patterns
- **Integration points**: Tests that verify interaction with other components
- **Performance considerations**: Tests that validate efficiency and scalability

### Testing Methodologies
- **Unit testing**: Isolated testing of individual functions and methods
- **Integration testing**: Testing of component interactions and data flow
- **Property-based testing**: Validation of invariants and behavioral properties
- **Performance testing**: Benchmarking and load testing of critical operations
- **Error condition testing**: Comprehensive error handling and recovery validation

## Key Features Tested

### Data Processing
- JSON parsing and serialization with error recovery
- Complex data transformation and validation
- Schema validation and integrity checking
- Performance optimization for large datasets

### Network Communication
- HTTP client/server communication
- Connection pooling and retry mechanisms
- Error handling and circuit breaker patterns
- Security validation and protection measures

### System Integration
- Component lifecycle management
- Configuration management and validation
- Dependency injection and resolution
- Inter-component communication protocols

### Performance and Scalability
- High-frequency operation handling
- Memory management and optimization
- Concurrent access and thread safety
- Resource allocation and cleanup

### Error Handling and Recovery
- Graceful degradation under error conditions
- Data corruption detection and recovery
- Network failure resilience
- State consistency maintenance

## Running Utility Tests

```bash
# Run all utility tests
python tests/run_tests.py --utils

# Run specific utility test files
python tests/run_tests.py --specific tests/test_utils/test_json_utils.py
python tests/run_tests.py --specific tests/test_utils/test_network_utils.py

# Run with coverage reporting
python tests/run_tests.py --utils --coverage --report-html

# Run in parallel for faster execution
python tests/run_tests.py --utils --parallel --workers 8

# Run with verbose output for debugging
python tests/run_tests.py --utils --verbose --show-locals
```

## Integration with Overall Test Suite

These utility tests integrate seamlessly with the broader test infrastructure:
- **Component interaction tests**: Verify how utilities work together in `tests/test_interactions/`
- **Integration tests**: End-to-end scenarios using utilities in `tests/test_integration/`
- **Performance tests**: Stress testing of utility performance in `tests/test_stress/`
- **Edge case tests**: Boundary condition testing in `tests/test_edge_cases/`

The comprehensive test coverage ensures that all utility functions work correctly in isolation and in combination, providing a solid foundation for the entire SnakeGTP system. 