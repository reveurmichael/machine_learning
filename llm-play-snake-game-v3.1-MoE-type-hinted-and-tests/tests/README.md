# Test Suite Documentation

This directory contains a comprehensive test suite for the mypy-v3-MoE project. The test suite provides extensive coverage of all major components, including type safety validation, functionality testing, integration testing, and performance validation.

## Test Structure

### Core Module Tests (`test_core/`)
Tests for the main game logic and control components:

- **`test_game_controller.py`** (337 lines) - Game controller functionality
- **`test_game_data.py`** (414 lines) - Game data management and persistence
- **`test_game_logic.py`** (320 lines) - LLM integration and game logic
- **`test_game_rounds.py`** (380+ lines) - Round management functionality
- **`test_game_stats.py`** (400+ lines) - Statistics tracking and analysis
- **`test_game_loop.py`** (400+ lines) - Main game loop functionality
- **`test_game_manager.py`** (350+ lines) - High-level game management

**Coverage**: Initialization, state management, LLM integration, round handling, statistics calculation, game flow control, error handling, and data persistence.

### Configuration Tests (`test_config/`)
Comprehensive tests for all configuration modules:

- **`test_game_constants.py`** - Game configuration constants and rules
- **`test_llm_constants.py`** - LLM configuration parameters
- **`test_network_constants.py`** - Network and host configuration
- **`test_ui_constants.py`** - UI colors, dimensions, and display settings
- **`test_prompt_templates.py`** - LLM prompt template validation

**Coverage**: Constants validation, type checking, boundary testing, integration compatibility, immutability verification, and configuration consistency.

### LLM Module Tests (`test_llm/`)
Complete testing of the language model integration:

- **`test_client.py`** - LLM client functionality with provider testing
- **`test_communication_utils.py`** - LLM communication and retry logic
- **`test_parsing_utils.py`** - Response parsing and validation
- **`test_prompt_utils.py`** - Prompt preparation and formatting
- **`test_setup_utils.py`** - Environment and provider setup
- **`test_providers.py`** - Individual provider implementations (Ollama, DeepSeek, Mistral, Hunyuan)

**Coverage**: Provider abstraction, API communication, response parsing, error handling, retry mechanisms, prompt formatting, environment validation, and provider-specific implementations.

### Utility Module Tests (`test_utils/`)
Tests for all utility functions (11 files):

- **`test_continuation_utils.py`** - Session continuation functionality
- **`test_file_utils.py`** - File operations and I/O
- **`test_game_manager_utils.py`** - Game management utilities
- **`test_game_stats_utils.py`** - Statistics utilities
- **`test_initialization_utils.py`** - System initialization
- **`test_json_utils.py`** - JSON processing
- **`test_moves_utils.py`** - Move validation and processing
- **`test_network_utils.py`** - Network operations
- **`test_session_utils.py`** - Session management
- **`test_text_utils.py`** - Text processing utilities
- **`test_web_utils.py`** - Web interface utilities

**Coverage**: File I/O, JSON handling, network operations, session management, move validation, text processing, and web utilities.

### Replay Module Tests (`test_replay/`)
Complete replay system testing (4 files, 92+ tests):

- **`test_replay_engine.py`** (35 test methods, 8 test classes) - Core replay engine
- **`test_replay_utils.py`** (26 test methods, 3 test classes) - Utility functions
- **`test_replay_integration.py`** (12 test methods) - Module integration
- **`test_replay_performance.py`** (14 test methods) - Performance and stress testing

**Coverage**: Replay initialization, data loading, state management, move execution, GUI integration, navigation controls, error handling, and performance with large datasets.

### Interaction Tests (`test_interactions/`)
Component interaction testing (26 files):

Extensive testing of component interactions including:
- Client-controller interactions
- Provider communication patterns  
- Concurrent game operations
- Data persistence workflows
- High-frequency operations
- Memory sharing between components
- Network-LLM interactions
- Error propagation mechanisms
- System integration workflows
- Session continuation flows
- Web-core integrations

### Integration Tests (`test_integration/`)
System-wide integration testing (3 files):

- **`test_game_flow.py`** - Complete game flow testing
- **`test_complex_interactions.py`** - Multi-component interaction scenarios
- **`test_full_game_workflow_interactions.py`** (51KB) - Complete game workflow testing
- **`test_replay_workflow_interactions.py`** (24KB) - Full replay system testing
- **`test_config_component_interactions.py`** (17KB) - Configuration integration testing

### Specialized Test Categories

#### Stress Tests (`test_stress/`)
Performance and load testing:
- High-frequency operations
- Large dataset handling
- Memory usage validation
- Concurrent operation testing
- Resource cleanup verification

#### Edge Case Tests (`test_edge_cases/`)
Corner case and boundary condition testing:
- Boundary value testing
- Error condition handling
- Invalid input processing
- Resource exhaustion scenarios
- Network failure simulation

## Test Statistics

### Overall Coverage
- **Total Test Files**: 60+ files
- **Total Test Methods**: 800+ individual test methods
- **Total Lines of Test Code**: 25,000+ lines
- **Coverage Areas**: All major components, utilities, configurations, and integrations

### Test Categories
- **Unit Tests**: 70% (individual function/class testing)
- **Integration Tests**: 20% (component interaction testing)
- **System Tests**: 10% (end-to-end workflow testing)

### Quality Metrics
- **Type Safety**: All tests include proper type hints and mypy compatibility
- **Error Handling**: Comprehensive error condition testing
- **Edge Cases**: Boundary condition and corner case coverage
- **Performance**: Load testing and resource usage validation
- **Concurrency**: Multi-threading and async operation testing

## Running Tests

### Run All Tests
```bash
# From project root
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Run Specific Test Categories
```bash
# Core functionality tests
python -m pytest tests/test_core/ -v

# Configuration tests
python -m pytest tests/test_config/ -v

# LLM integration tests  
python -m pytest tests/test_llm/ -v

# Utility function tests
python -m pytest tests/test_utils/ -v

# Replay system tests
python -m pytest tests/test_replay/ -v

# Integration tests
python -m pytest tests/test_integration/ -v

# Interaction tests
python -m pytest tests/test_interactions/ -v
```

### Run Performance Tests
```bash
# Stress tests
python -m pytest tests/test_stress/ -v

# Performance benchmarks
python -m pytest tests/test_replay/test_replay_performance.py -v
```

### Run Tests with Specific Markers
```bash
# Async tests only
python -m pytest tests/ -m asyncio -v

# Integration tests only
python -m pytest tests/ -k "integration" -v

# Error handling tests
python -m pytest tests/ -k "error" -v
```

## Test Configuration

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v 
    --strict-markers
    --tb=short
    --asyncio-mode=strict
markers =
    asyncio: marks tests as async
    integration: marks tests as integration tests
    slow: marks tests as slow running
    network: marks tests that require network access
```

### Test Dependencies
- **pytest**: Test framework
- **pytest-asyncio**: Async test support
- **pytest-mock**: Mocking utilities
- **pytest-cov**: Coverage reporting
- **unittest.mock**: Mocking for unit tests

## Test Quality Standards

### Code Quality
- All tests follow PEP 8 style guidelines
- Comprehensive docstrings for all test classes and methods
- Type hints for all test functions and fixtures
- Consistent naming conventions across all test files

### Test Design Principles
- **Isolation**: Each test is independent and can run in any order
- **Clarity**: Test names clearly describe what is being tested
- **Coverage**: Tests cover both happy path and error conditions
- **Performance**: Tests include performance benchmarks where appropriate
- **Maintainability**: Tests are easy to understand and modify

### Error Handling
- Comprehensive testing of error conditions
- Validation of error messages and types
- Recovery mechanism testing
- Resource cleanup verification

### Integration Testing
- Component interaction validation
- End-to-end workflow testing
- Data flow verification
- State consistency checking

## Continuous Integration

The test suite is designed to run in CI/CD environments with:
- Parallel test execution support
- Comprehensive coverage reporting
- Performance regression detection
- Type safety validation
- Cross-platform compatibility testing

## Contributing to Tests

When adding new features:
1. Write unit tests for new functions/classes
2. Add integration tests for component interactions
3. Include error condition testing
4. Add performance tests for critical paths
5. Update this documentation

### Test Naming Convention
- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<functionality>_<condition>`

Example:
```python
class TestGameController:
    def test_initialize_controller_with_valid_config(self):
        """Test controller initialization with valid configuration."""
        pass
    
    def test_initialize_controller_with_invalid_config_raises_error(self):
        """Test that invalid configuration raises appropriate error."""
        pass
```

This comprehensive test suite ensures high code quality, type safety, and robust functionality across all components of the mypy-v3-MoE project. 