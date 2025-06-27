"""
Testing Utilities for Snake Game AI Extensions

This module provides testing utilities for validating the common package
functionality and ensuring compatibility across all extensions.

Design Patterns Used:
- Builder Pattern: Test case construction with fluent interface
- Template Method Pattern: Standard test execution workflow
- Factory Pattern: Create appropriate test fixtures for different scenarios
- Strategy Pattern: Different testing strategies for different components

Educational Value:
Demonstrates how to build a comprehensive testing framework that validates
both individual components and their integration across the system.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import json
import traceback


class TestResult(Enum):
    """Test execution results"""
    PASS = "PASS"
    FAIL = "FAIL" 
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class TestCase:
    """
    Individual test case definition
    
    Design Pattern: Value Object Pattern
    Immutable container for test configuration that ensures
    consistent test execution across different environments.
    """
    name: str
    description: str
    test_function: Callable[[], bool]
    setup_function: Optional[Callable[[], None]] = None
    teardown_function: Optional[Callable[[], None]] = None
    skip_reason: Optional[str] = None
    expected_exception: Optional[type] = None


@dataclass
class TestResult:
    """Test execution result"""
    test_case: TestCase
    result: TestResult
    execution_time: float
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None


class TestRunner:
    """
    Comprehensive test runner for common utilities
    
    Design Pattern: Template Method Pattern
    Defines standard test execution workflow while allowing
    customization of specific test steps.
    
    Educational Note (SUPREME_RULE NO.4):
    This class is designed to be extensible through inheritance. Extensions
    can create specialized test runners by inheriting from this base class
    and overriding specific methods for algorithm-specific testing while
    maintaining the common testing workflow.
    
    SUPREME_RULE NO.4 Implementation:
    - Base class provides complete test execution functionality
    - Protected methods allow selective customization by subclasses
    - Virtual methods enable complete behavior replacement when needed
    - Algorithm-specific test runners can inherit and adapt as needed
    """
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.results: List[TestResult] = []
        self.setup_functions: List[Callable[[], None]] = []
        self.teardown_functions: List[Callable[[], None]] = []
        self._initialize_runner_specific_settings()
    
    def _initialize_runner_specific_settings(self) -> None:
        """
        Initialize runner-specific settings (SUPREME_RULE NO.4 Extension Point).
        
        This method can be overridden by subclasses to set up algorithm-specific
        test configurations, custom assertions, or specialized test environments.
        
        Example:
            class RLTestRunner(TestRunner):
                def _initialize_runner_specific_settings(self):
                    self.gpu_testing_enabled = True
                    self.tensorboard_validation = TensorboardValidator()
                    self.model_stability_tests = ModelStabilityTester()
        """
        pass
    
    def add_setup(self, setup_function: Callable[[], None]) -> None:
        """Add global setup function"""
        self.setup_functions.append(setup_function)
    
    def add_teardown(self, teardown_function: Callable[[], None]) -> None:
        """Add global teardown function"""
        self.teardown_functions.append(teardown_function)
    
    def add_test(self, test_case: TestCase) -> None:
        """Add test case to runner"""
        self.test_cases.append(test_case)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Execute all test cases and return summary
        
        Returns:
            Summary of test execution results
        """
        self.results = []
        
        # Global setup
        for setup_func in self.setup_functions:
            try:
                setup_func()
            except Exception as e:
                print(f"Global setup failed: {e}")
                return {"error": "Global setup failed", "exception": str(e)}
        
        # Execute tests
        for test_case in self.test_cases:
            result = self._execute_test(test_case)
            self.results.append(result)
        
        # Global teardown
        for teardown_func in self.teardown_functions:
            try:
                teardown_func()
            except Exception as e:
                print(f"Global teardown failed: {e}")
        
        return self._generate_summary()
    
    def _execute_test(self, test_case: TestCase) -> TestResult:
        """Execute individual test case"""
        import time
        start_time = time.time()
        
        # Check if test should be skipped
        if test_case.skip_reason:
            return TestResult(
                test_case=test_case,
                result=TestResult.SKIP,
                execution_time=0.0,
                error_message=test_case.skip_reason
            )
        
        try:
            # Setup
            if test_case.setup_function:
                test_case.setup_function()
            
            # Execute test
            if test_case.expected_exception:
                # Test should raise specific exception
                try:
                    test_case.test_function()
                    # If we get here, test failed (no exception raised)
                    return TestResult(
                        test_case=test_case,
                        result=TestResult.FAIL,
                        execution_time=time.time() - start_time,
                        error_message=f"Expected {test_case.expected_exception.__name__} but no exception was raised"
                    )
                except test_case.expected_exception:
                    # Expected exception was raised - test passes
                    result = TestResult.PASS
                except Exception as e:
                    # Wrong exception type
                    return TestResult(
                        test_case=test_case,
                        result=TestResult.FAIL,
                        execution_time=time.time() - start_time,
                        error_message=f"Expected {test_case.expected_exception.__name__} but got {type(e).__name__}: {e}"
                    )
            else:
                # Normal test execution
                success = test_case.test_function()
                result = TestResult.PASS if success else TestResult.FAIL
            
            # Teardown
            if test_case.teardown_function:
                test_case.teardown_function()
            
            return TestResult(
                test_case=test_case,
                result=result,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return TestResult(
                test_case=test_case,
                result=TestResult.ERROR,
                execution_time=time.time() - start_time,
                error_message=str(e),
                stack_trace=traceback.format_exc()
            )
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test execution summary"""
        total_tests = len(self.results)
        passed = sum(1 for r in self.results if r.result == TestResult.PASS)
        failed = sum(1 for r in self.results if r.result == TestResult.FAIL)
        errors = sum(1 for r in self.results if r.result == TestResult.ERROR)
        skipped = sum(1 for r in self.results if r.result == TestResult.SKIP)
        
        return {
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "success_rate": passed / max(total_tests - skipped, 1),
            "total_time": sum(r.execution_time for r in self.results),
            "results": [
                {
                    "name": r.test_case.name,
                    "result": r.result.value,
                    "time": r.execution_time,
                    "error": r.error_message
                }
                for r in self.results
            ]
        }


def create_test_environment() -> Path:
    """
    Create temporary test environment
    
    Returns:
        Path to temporary test directory
        
    Educational Value:
    Shows how to create isolated test environments that don't
    interfere with the main project or other tests.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="snake_test_"))
    
    # Create basic project structure
    (temp_dir / "config").mkdir()
    (temp_dir / "extensions" / "common").mkdir(parents=True)
    (temp_dir / "logs").mkdir()
    
    # Create minimal config files for testing
    (temp_dir / "config" / "__init__.py").touch()
    (temp_dir / "extensions" / "__init__.py").touch()
    (temp_dir / "extensions" / "common" / "__init__.py").touch()
    
    return temp_dir


def cleanup_test_environment(test_dir: Path) -> None:
    """Clean up temporary test environment"""
    if test_dir.exists() and "snake_test_" in str(test_dir):
        shutil.rmtree(test_dir)


# Test cases for common utilities
def test_path_utils():
    """Test path utilities functionality"""
    from .path_utils import ensure_project_root, get_dataset_path
    
    # Test path utilities work correctly
    try:
        # This should work in any environment
        root = ensure_project_root()
        assert root.exists(), "Project root should exist"
        
        # Test dataset path generation
        dataset_path = get_dataset_path(
            extension_type="test",
            version="0.01",
            grid_size=10,
            algorithm="test_algo",
            timestamp="20240101_120000"
        )
        
        expected_pattern = "logs/extensions/datasets/grid-size-10/test_v0.01_20240101_120000"
        assert expected_pattern in str(dataset_path), f"Dataset path should contain expected pattern"
        
        return True
    except Exception as e:
        print(f"Path utils test failed: {e}")
        return False


def test_csv_schema():
    """Test CSV schema functionality"""
    try:
        from .csv_schema_utils import generate_csv_schema, create_csv_row, FEATURE_COUNT
        
        # Test schema generation
        schema = generate_csv_schema(grid_size=10)
        assert schema is not None, "Schema should be generated"
        
        # Test feature count consistency
        assert FEATURE_COUNT == 16, "Should have exactly 16 features"
        
        # Test CSV row creation
        test_game_state = {
            'head_position': [5, 5],
            'apple_position': [7, 3],
            'snake_positions': [[5, 5], [4, 5], [3, 5]],
            'current_direction': 'RIGHT',
            'score': 2,
            'steps': 10,
            'snake_length': 3
        }
        
        csv_row = create_csv_row(
            game_state=test_game_state,
            target_move="UP",
            game_id=1,
            step_in_game=10,
            grid_size=10
        )
        
        assert len(csv_row) == 19, "CSV row should have 19 columns (2 metadata + 16 features + 1 target)"
        
        return True
    except Exception as e:
        print(f"CSV schema test failed: {e}")
        return False


def test_validation_system():
    """Test validation system functionality"""
    try:
        from ..validation import validate_dataset_format, ValidationResult
        
        # Create temporary test file
        test_dir = create_test_environment()
        test_file = test_dir / "test_dataset.csv"
        
        # Create minimal CSV content
        test_file.write_text("game_id,step_in_game,head_x,head_y,apple_x,apple_y,target_move\n1,1,5,5,7,3,UP\n")
        
        # Test validation (this might fail if validation is strict, but should not crash)
        result = validate_dataset_format(
            dataset_path=test_file,
            extension_type="test",
            version="0.01",
            expected_format="csv"
        )
        
        assert isinstance(result, ValidationResult), "Should return ValidationResult"
        
        # Cleanup
        cleanup_test_environment(test_dir)
        
        return True
    except Exception as e:
        print(f"Validation system test failed: {e}")
        return False


def test_configuration_access():
    """Test configuration access functionality"""
    try:
        from ..config.ml_constants import DEFAULT_LEARNING_RATE
        from ..config.validation_rules import MIN_GRID_SIZE, MAX_GRID_SIZE
        
        # Test that constants are accessible and reasonable
        assert isinstance(DEFAULT_LEARNING_RATE, float), "Learning rate should be float"
        assert 0.0001 <= DEFAULT_LEARNING_RATE <= 1.0, "Learning rate should be reasonable"
        
        assert isinstance(MIN_GRID_SIZE, int), "Min grid size should be int"
        assert isinstance(MAX_GRID_SIZE, int), "Max grid size should be int"
        assert MIN_GRID_SIZE < MAX_GRID_SIZE, "Min should be less than max"
        
        return True
    except Exception as e:
        print(f"Configuration access test failed: {e}")
        return False


def run_common_utilities_tests() -> Dict[str, Any]:
    """
    Run comprehensive tests for common utilities
    
    Returns:
        Test execution summary
        
    Educational Value:
    Demonstrates how to create a comprehensive test suite that validates
    all major components of a software system.
    """
    runner = TestRunner()
    
    # Add test cases
    runner.add_test(TestCase(
        name="test_path_utils",
        description="Test path management utilities",
        test_function=test_path_utils
    ))
    
    runner.add_test(TestCase(
        name="test_csv_schema", 
        description="Test CSV schema generation and processing",
        test_function=test_csv_schema
    ))
    
    runner.add_test(TestCase(
        name="test_validation_system",
        description="Test validation system functionality",
        test_function=test_validation_system
    ))
    
    runner.add_test(TestCase(
        name="test_configuration_access",
        description="Test configuration constants access",
        test_function=test_configuration_access
    ))
    
    # Execute tests
    summary = runner.run_all_tests()
    
    # Print results
    print("\n" + "="*60)
    print("COMMON UTILITIES TEST RESULTS")
    print("="*60)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Errors: {summary['errors']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Success Rate: {summary['success_rate']:.2%}")
    print(f"Total Time: {summary['total_time']:.3f}s")
    print()
    
    # Print individual results
    for result in summary['results']:
        status_symbol = "✓" if result['result'] == 'PASS' else "✗"
        print(f"{status_symbol} {result['name']} ({result['time']:.3f}s)")
        if result['error']:
            print(f"  Error: {result['error']}")
    
    print("="*60)
    
    return summary


if __name__ == "__main__":
    """Run tests when module is executed directly"""
    run_common_utilities_tests() 