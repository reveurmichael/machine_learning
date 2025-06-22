# Heuristics v0.01 Extension

A simple first extension for the Snake Game project that implements heuristic pathfinding algorithms using BFS (Breadth-First Search).

## How to run

```bash
python main.py
```

## Purpose of this v0.01 extension

This v0.01 extension is a simple proof of concept that things can go well with our abstraction prepared by Task0.

## BFS Algorithm

The BFS agent implements a simple breadth-first search pathfinding algorithm:

1. **State Representation**: Current snake head position and apple position
2. **Search Space**: All valid grid positions (not walls or snake body)
3. **Goal**: Find shortest path from head to apple
4. **Path Execution**: Return first move in optimal path

### Algorithm Characteristics

- **Optimality**: Always finds shortest path to apple
- **Completeness**: Will find solution if one exists
- **Time Complexity**: O(V + E) where V is grid cells, E is adjacency
- **Space Complexity**: O(V) for queue and visited set
