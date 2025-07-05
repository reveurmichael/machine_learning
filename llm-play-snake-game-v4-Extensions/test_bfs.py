#!/usr/bin/env python3

import sys
from pathlib import Path
import importlib.util

# Add project root and heuristics-v0.04 to sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "extensions" / "heuristics-v0.04"))

# Dynamically import DatasetGenerator from the dash-named directory
module_path = project_root / "extensions" / "heuristics-v0.04" / "dataset_generator_core.py"
spec = importlib.util.spec_from_file_location("dataset_generator_core", str(module_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
DatasetGenerator = mod.DatasetGenerator

# Test BFS pathfinding
generator = DatasetGenerator('BFS-SAFE-GREEDY', Path('/tmp'))

# Test case 1: [0,7] to [4,6]
start = [0, 7]
goal = [4, 6]
obstacles = set()
grid_size = 10

print(f'Testing BFS from {start} to {goal}')
path = generator._bfs_pathfind(start, goal, obstacles, grid_size)
print(f'Path: {path}')
print(f'Length: {len(path) - 1 if path else None}')

# Manual calculation
manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
print(f'Manhattan distance: {manhattan}')

# Test case 2: [0,6] to [4,6]
start2 = [0, 6]
goal2 = [4, 6]
print(f'\nTesting BFS from {start2} to {goal2}')
path2 = generator._bfs_pathfind(start2, goal2, obstacles, grid_size)
print(f'Path: {path2}')
print(f'Length: {len(path2) - 1 if path2 else None}')

manhattan2 = abs(start2[0] - goal2[0]) + abs(start2[1] - goal2[1])
print(f'Manhattan distance: {manhattan2}')

# Test case 3: [3,7] to [3,9]
start3 = [3, 7]
goal3 = [3, 9]
print(f'\nTesting BFS from {start3} to {goal3}')
path3 = generator._bfs_pathfind(start3, goal3, obstacles, grid_size)
print(f'Path: {path3}')
print(f'Length: {len(path3) - 1 if path3 else None}')

manhattan3 = abs(start3[0] - goal3[0]) + abs(start3[1] - goal3[1])
print(f'Manhattan distance: {manhattan3}') 