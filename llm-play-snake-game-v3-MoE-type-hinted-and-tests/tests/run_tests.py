#!/usr/bin/env python3
"""
Comprehensive test runner for the SnakeGTP project.

This script provides a convenient interface for running the extensive test suite
including unit tests, integration tests, edge cases, performance tests, and more.
"""

import argparse
import subprocess
import sys
import os
from typing import List, Optional, Dict, Any


def run_command(cmd: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    
    if capture_output:
        return subprocess.run(cmd, capture_output=True, text=True)
    else:
        return subprocess.run(cmd)


def build_pytest_command(args: argparse.Namespace) -> List[str]:
    """Build the pytest command based on arguments."""
    cmd: List[str] = ["python", "-m", "pytest"]
    
    # Determine test paths
    test_paths: List[str] = []
    
    if args.all:
        test_paths.append("tests/")
    else:
        if args.core:
            test_paths.append("tests/test_core/")
        if args.utils:
            test_paths.append("tests/test_utils/")
        if args.llm:
            test_paths.append("tests/test_llm/")
        if args.integration:
            test_paths.append("tests/test_integration/")
        if args.interactions:
            test_paths.append("tests/test_interactions/")
        if args.edge_cases:
            test_paths.append("tests/test_edge_cases/")
        if args.stress:
            test_paths.append("tests/test_stress/")
        if args.specific:
            test_paths.extend(args.specific)
    
    # Default to all if no specific paths provided
    if not test_paths:
        test_paths.append("tests/")
    
    cmd.extend(test_paths)
    
    # Add coverage options
    if args.coverage:
        coverage_modules: List[str] = [
            "--cov=core",
            "--cov=utils", 
            "--cov=llm",
            "--cov=gui",
            "--cov=replay",
            "--cov=dashboard"
        ]
        cmd.extend(coverage_modules)
        
        # Coverage reporting options
        if args.report_html:
            cmd.append("--cov-report=html")
        if args.report_xml:
            cmd.append("--cov-report=xml")
        if args.report_term:
            cmd.append("--cov-report=term-missing")
        
        # Default to terminal report if no specific report requested
        if not any([args.report_html, args.report_xml, args.report_term]):
            cmd.append("--cov-report=term-missing")
    
    # Parallel execution
    if args.parallel:
        workers: int = args.workers or os.cpu_count() or 4
        cmd.extend(["-n", str(workers)])
    
    # Output options
    if args.verbose:
        cmd.append("-v")
    if args.quiet:
        cmd.append("-q")
    if args.show_locals:
        cmd.append("--tb=long")
        cmd.append("-l")
    
    # Test selection options
    if args.keyword:
        cmd.extend(["-k", args.keyword])
    if args.marker:
        cmd.extend(["-m", args.marker])
    
    # Performance options
    if args.benchmark:
        cmd.append("--benchmark-only")
    if args.profile:
        cmd.append("--profile")
    
    # Fail fast option
    if args.fail_fast:
        cmd.append("-x")
    
    # Output formats
    if args.junit_xml:
        cmd.extend(["--junit-xml", args.junit_xml])
    
    # Skip slow tests if requested
    if args.fast:
        cmd.extend(["-m", "not slow"])
    
    return cmd


def check_dependencies() -> bool:
    """Check if required test dependencies are installed."""
    required_packages: List[str] = [
        "pytest",
        "pytest-cov", 
        "numpy"
    ]
    
    optional_packages: List[str] = [
        "pytest-xdist",  # For parallel execution
        "psutil",        # For performance monitoring
        "pytest-benchmark"  # For benchmarking
    ]
    
    missing_required: List[str] = []
    missing_optional: List[str] = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        print(f"ERROR: Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"WARNING: Missing optional packages: {', '.join(missing_optional)}")
        print("Install with: pip install " + " ".join(missing_optional))
        print("Some features may not be available.\n")
    
    return True


def main() -> int:
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for SnakeGTP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                           # Run all tests
  %(prog)s --core --utils                  # Run core and utils tests
  %(prog)s --integration --coverage        # Run integration tests with coverage
  %(prog)s --interactions --verbose        # Run component interaction tests
  %(prog)s --stress --verbose              # Run performance tests with verbose output
  %(prog)s --edge-cases --parallel         # Run edge case tests in parallel
  %(prog)s --all --coverage --report-html  # All tests with HTML coverage report
  %(prog)s --fast --parallel --workers 8   # Fast tests with 8 parallel workers
  %(prog)s --specific tests/test_core/test_game_controller.py  # Specific test file
        """
    )
    
    # Test selection options
    test_group = parser.add_argument_group("Test Selection")
    test_group.add_argument("--all", action="store_true",
                           help="Run all tests")
    test_group.add_argument("--core", action="store_true",
                           help="Run core component tests")
    test_group.add_argument("--utils", action="store_true", 
                           help="Run utility tests")
    test_group.add_argument("--llm", action="store_true",
                           help="Run LLM integration tests")
    test_group.add_argument("--integration", action="store_true",
                           help="Run integration tests")
    test_group.add_argument("--interactions", action="store_true",
                           help="Run component interaction tests")
    test_group.add_argument("--edge-cases", dest="edge_cases", action="store_true",
                           help="Run edge case and corner case tests")
    test_group.add_argument("--stress", action="store_true",
                           help="Run stress and performance tests")
    test_group.add_argument("--specific", nargs="+", metavar="PATH",
                           help="Run specific test files or directories")
    
    # Coverage options
    coverage_group = parser.add_argument_group("Coverage Options")
    coverage_group.add_argument("--coverage", action="store_true",
                               help="Generate coverage reports")
    coverage_group.add_argument("--report-html", action="store_true",
                               help="Generate HTML coverage report")
    coverage_group.add_argument("--report-xml", action="store_true", 
                               help="Generate XML coverage report")
    coverage_group.add_argument("--report-term", action="store_true",
                               help="Show terminal coverage report")
    
    # Execution options
    exec_group = parser.add_argument_group("Execution Options")
    exec_group.add_argument("--parallel", action="store_true",
                           help="Run tests in parallel")
    exec_group.add_argument("--workers", type=int, metavar="N",
                           help="Number of parallel workers (default: CPU count)")
    exec_group.add_argument("--fast", action="store_true",
                           help="Skip slow tests")
    exec_group.add_argument("--fail-fast", action="store_true",
                           help="Stop on first failure")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--verbose", "-v", action="store_true",
                             help="Verbose output")
    output_group.add_argument("--quiet", "-q", action="store_true",
                             help="Quiet output")
    output_group.add_argument("--show-locals", action="store_true",
                             help="Show local variables in tracebacks")
    
    # Filtering options
    filter_group = parser.add_argument_group("Test Filtering")
    filter_group.add_argument("--keyword", "-k", metavar="EXPR",
                             help="Run tests matching keyword expression")
    filter_group.add_argument("--marker", "-m", metavar="MARKEXPR",
                             help="Run tests matching marker expression")
    
    # Performance options
    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument("--benchmark", action="store_true",
                           help="Run only benchmark tests")
    perf_group.add_argument("--profile", action="store_true",
                           help="Profile test execution")
    
    # Output format options
    format_group = parser.add_argument_group("Output Formats")
    format_group.add_argument("--junit-xml", metavar="FILE",
                             help="Generate JUnit XML report")
    
    # System options
    system_group = parser.add_argument_group("System Options")
    system_group.add_argument("--check-deps", action="store_true",
                             help="Check dependencies and exit")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    if args.check_deps:
        print("All required dependencies are available.")
        return 0
    
    # Validate arguments
    if args.quiet and args.verbose:
        print("ERROR: --quiet and --verbose are mutually exclusive")
        return 1
    
    # Build and run pytest command
    try:
        pytest_cmd = build_pytest_command(args)
        result = run_command(pytest_cmd)
        return result.returncode
        
    except FileNotFoundError:
        print("ERROR: pytest not found. Please install with: pip install pytest")
        return 1
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 