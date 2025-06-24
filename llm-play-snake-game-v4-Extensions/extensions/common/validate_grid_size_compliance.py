#!/usr/bin/env python3
"""
Grid-Size Directory Structure Compliance Validator
=================================================

This script validates that all extensions follow the mandatory grid-size directory structure:

DATASETS: logs/extensions/datasets/grid-size-N/...
MODELS:   logs/extensions/models/grid-size-N/...

Where N is the grid size used during training/generation.

Design Principle:
Models trained on different grid sizes have fundamentally different spatial complexity
and should never be mixed. This structure enforces clean separation and prevents
accidental contamination of experiments.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Ensure project root is on path so script can be called from anywhere
from extensions.common.path_utils import ensure_project_root_on_path

def find_hardcoded_grid_sizes(root_dir: Path) -> List[Tuple[str, int, str]]:
    """Find hardcoded grid-size references that should be dynamic."""
    violations = []
    
    # Pattern to match hardcoded grid-size-N paths
    pattern = re.compile(r'logs/extensions/(?:datasets|models)/grid-size-(\d+)')
    
    for py_file in root_dir.rglob("*.py"):
        if "/__pycache__/" in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Skip comments and docstrings  
                    stripped = line.strip()
                    if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                        continue
                        
                    matches = pattern.findall(line)
                    for grid_size in matches:
                        violations.append((str(py_file), line_num, line.strip()))
        except (UnicodeDecodeError, PermissionError):
            continue
    
    return violations

def find_non_grid_aware_paths(root_dir: Path) -> List[Tuple[str, int, str]]:
    """Find dataset/model paths that don't use grid-size structure."""
    violations = []
    
    # Patterns for dataset and model paths that should be grid-size aware
    dataset_pattern = re.compile(r'logs/extensions/datasets/(?!grid-size-)')
    model_pattern = re.compile(r'logs/extensions/models/(?!grid-size-)')
    
    for py_file in root_dir.rglob("*.py"):
        if "/__pycache__/" in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Skip comments and docstrings
                    stripped = line.strip()
                    if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                        continue
                    
                    # Check for non-grid-aware dataset paths
                    if dataset_pattern.search(line):
                        violations.append((str(py_file), line_num, f"DATASET: {line.strip()}"))
                    
                    # Check for non-grid-aware model paths  
                    if model_pattern.search(line):
                        violations.append((str(py_file), line_num, f"MODEL: {line.strip()}"))
                        
        except (UnicodeDecodeError, PermissionError):
            continue
    
    return violations

def check_training_scripts(root_dir: Path) -> List[Tuple[str, str]]:
    """Check that training scripts use grid-size aware model saving."""
    issues = []
    
    # Find training scripts
    training_files = []
    for pattern in ["**/train*.py", "**/training/*.py", "**/scripts/train*.py"]:
        training_files.extend(root_dir.glob(pattern))
    
    for script in training_files:
        if "/__pycache__/" in str(script):
            continue
            
        try:
            with open(script, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Check if it uses model saving
                if any(keyword in content for keyword in ["save_model", "torch.save", "model.save"]):
                    # Check if it's grid-size aware (uses standardized utils or explicit grid-size paths)
                    is_grid_aware = any([
                        "save_model_standardized" in content,
                        "get_model_directory" in content,
                        "grid-size-" in content,
                        "model_utils" in content
                    ])
                    
                    if not is_grid_aware and "grid_size" in content:
                        issues.append((str(script), "Uses model saving but may not be grid-size aware"))
                        
        except (UnicodeDecodeError, PermissionError):
            continue
    
    return issues

def validate_directory_structure() -> Dict[str, bool]:
    """Validate the actual directory structure exists correctly."""
    results = {}
    
    # Check if base directories exist
    datasets_root = Path("logs/extensions/datasets")
    models_root = Path("logs/extensions/models")
    
    results["datasets_root_exists"] = datasets_root.exists()
    results["models_root_exists"] = models_root.exists()
    
    # Check for proper grid-size subdirectories
    if datasets_root.exists():
        grid_dirs = list(datasets_root.glob("grid-size-*"))
        results["has_grid_size_dataset_dirs"] = len(grid_dirs) > 0
        results["dataset_grid_dirs"] = [d.name for d in grid_dirs]
    
    if models_root.exists():
        grid_dirs = list(models_root.glob("grid-size-*"))
        results["has_grid_size_model_dirs"] = len(grid_dirs) > 0  
        results["model_grid_dirs"] = [d.name for d in grid_dirs]
    
    return results

def main():
    """Main validation function."""
    print("🔍 Grid-Size Directory Structure Compliance Validator")
    print("=" * 60)
    
    # Ensure project root is on path and get the project root
    project_root = ensure_project_root_on_path()
    print(f"📁 Project root: {project_root}")
    
    # Use project root for all path operations
    root_dir = project_root
    
    # 1. Check for hardcoded grid sizes
    print("\n1. Checking for hardcoded grid-size paths...")
    hardcoded = find_hardcoded_grid_sizes(root_dir / "extensions")
    
    if hardcoded:
        print(f"❌ Found {len(hardcoded)} hardcoded grid-size references:")
        for file_path, line_num, line in hardcoded[:10]:  # Show first 10
            print(f"   {file_path}:{line_num} -> {line}")
        if len(hardcoded) > 10:
            print(f"   ... and {len(hardcoded) - 10} more")
    else:
        print("✅ No hardcoded grid-size paths found")
    
    # 2. Check for non-grid-aware paths
    print("\n2. Checking for non-grid-aware dataset/model paths...")
    non_grid_aware = find_non_grid_aware_paths(root_dir / "extensions")
    
    if non_grid_aware:
        print(f"❌ Found {len(non_grid_aware)} non-grid-aware paths:")
        for file_path, line_num, line in non_grid_aware[:10]:  # Show first 10
            print(f"   {file_path}:{line_num} -> {line}")
        if len(non_grid_aware) > 10:
            print(f"   ... and {len(non_grid_aware) - 10} more")
    else:
        print("✅ All dataset/model paths are grid-size aware")
    
    # 3. Check training scripts
    print("\n3. Checking training scripts...")
    training_issues = check_training_scripts(root_dir / "extensions")
    
    if training_issues:
        print(f"⚠️  Found {len(training_issues)} potential training script issues:")
        for script, issue in training_issues:
            print(f"   {script} -> {issue}")
    else:
        print("✅ Training scripts appear to be grid-size compliant")
    
    # 4. Validate directory structure
    print("\n4. Validating directory structure...")
    structure = validate_directory_structure()
    
    print(f"   Datasets root exists: {'✅' if structure.get('datasets_root_exists') else '❌'}")
    print(f"   Models root exists: {'✅' if structure.get('models_root_exists') else '❌'}")
    
    if structure.get('has_grid_size_dataset_dirs'):
        print(f"   ✅ Found dataset grid dirs: {structure.get('dataset_grid_dirs', [])}")
    else:
        print("   ⚠️  No grid-size dataset directories found")
        
    if structure.get('has_grid_size_model_dirs'):
        print(f"   ✅ Found model grid dirs: {structure.get('model_grid_dirs', [])}")
    else:
        print("   ⚠️  No grid-size model directories found")
    
    # 5. Summary
    print("\n📋 COMPLIANCE SUMMARY")
    print("=" * 30)
    
    total_violations = len(hardcoded) + len(non_grid_aware) + len(training_issues)
    
    if total_violations == 0:
        print("🎉 ALL CHECKS PASSED! Grid-size directory structure is properly enforced.")
        print("\n✅ Benefits achieved:")
        print("   • Clean separation of models by spatial complexity")
        print("   • No accidental mixing of different grid-size datasets")
        print("   • Scalable to new grid sizes without code changes")
        print("   • Clear experimental organization")
        return 0
    else:
        print(f"❌ Found {total_violations} total violations")
        print("\n🔧 Recommended fixes:")
        print("   • Replace hardcoded paths with dynamic grid-size logic")
        print("   • Use DatasetDirectoryManager.get_dataset_path()")
        print("   • Use model_utils.get_model_directory(grid_size=N)")
        print("   • Update training scripts to use grid-size aware saving")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 