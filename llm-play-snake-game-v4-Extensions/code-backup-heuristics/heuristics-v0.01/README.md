# Heuristics v0.01 - Simple BFS Snake Agent

**The First Proof of Concept Extension**

This v0.01 extension demonstrates that the base class architecture from Task-0 can be successfully extended for non-LLM algorithms. It serves as the foundation that proves the concept works, paving the way for v0.02's multi-algorithm expansion.

## Purpose & Philosophy

This v0.01 extension is a **simple proof of concept** that validates our abstraction layer prepared by Task-0. It shows that:

- ✅ Base classes (`BaseGameManager`, `BaseGameLogic`, `BaseGameData`) work for any algorithm
- ✅ Clean inheritance without Task-0 pollution is possible
- ✅ Same JSON output format can be maintained across different approaches
- ✅ Extensions can be completely standalone

## How to Run

```bash
cd extensions/heuristics-v0.01
python main.py --max-games 3
python main.py --max-games 5 --max-steps 500
```

## BFS Algorithm Implementation

The BFS agent implements a simple breadth-first search pathfinding algorithm:

### Algorithm Characteristics
- **Optimality**: Always finds shortest path to apple
- **Completeness**: Will find solution if one exists  
- **Time Complexity**: O(V + E) where V is grid cells, E is adjacency
- **Space Complexity**: O(V) for queue and visited set

### Implementation Details
1. **State Representation**: Current snake head position and apple position
2. **Search Space**: All valid grid positions (not walls or snake body)
3. **Goal**: Find shortest path from head to apple
4. **Path Execution**: Return first move in optimal path

## Key Design Decisions

### Minimal Complexity
- **Single algorithm**: Focus on proving the concept works
- **Simple CLI**: Only essential arguments needed
- **Direct inheritance**: Extends base classes without over-engineering
- **Headless operation**: No GUI complexity for pure algorithm testing

### Architecture Validation
- **BaseGameManager**: Inherits all session management logic
- **BaseGameLogic**: Inherits game mechanics, adds BFS planning
- **BaseGameData**: Uses directly without custom extensions
- **Clean separation**: Zero Task-0 LLM code pollution


## Conclusion
The success of v0.01 validates that our base architecture is **future-ready** and can support the entire roadmap of Tasks 1-5.

*v0.01 serves as the essential stepping stone that proves the concept works, enabling confident progression to more sophisticated multi-algorithm implementations.*
