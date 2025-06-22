# Heuristics v0.02 - Multi-Algorithm Snake Agents

This extension demonstrates the natural evolution from v0.01 to v0.02, showcasing how software systems progress by expanding functionality while maintaining the same architectural foundation.

## Evolution from v0.01

**v0.01**: Simple proof-of-concept with single BFS algorithm
**v0.02**: Comprehensive multi-algorithm suite with 7 different heuristic approaches

## Available Algorithms

| Algorithm | Description | Performance (10x10 Grid) | Complexity |
|-----------|-------------|-------------|------------|
| `BFS` | Pure breadth-first search | Score: ~18 | Low |
| `BFS-SAFE-GREEDY` | Enhanced BFS with safety validation | Score: ~20 | Medium |
| `BFS-HAMILTONIAN` | BFS with Hamiltonian cycle fallback | Score: ~?? | High |
| `DFS` | Depth-first search (educational) | Score: ~0 | Low |
| `ASTAR` | A* with Manhattan heuristic | Score: ~?? | Medium |
| `ASTAR-HAMILTONIAN` | A* with Hamiltonian fallback | Score: ~?? | High |
| `HAMILTONIAN` | Pure Hamiltonian cycle | Score: ~?? | Medium |


## General Comparison
More generally, we have the following table:

| Algorithm   | Path Optimality    | Safety                  | Speed              | Snake Suitability |
| ----------- | ------------------ | ----------------------- | ------------------ | ----------------- |
| DFS         | ❌ Long, unsafe     | ❌ Poor                  | ⚠️ Slow on average | ❌ Bad for Snake   |
| BFS         | ✅ Shortest         | ✅ Safer with tail check | ⚠️ Slower than A\* | ✅ Good            |
| A\*         | ✅ Shortest         | ✅ Smart exploration     | ✅ Fast             | ✅ Best choice     |
| Hamiltonian | ❌ Longest but safe | ✅ Always                | ✅ Linear           | ✅ Great fallback  |


## Key Improvements in v0.02

1. **Multiple Algorithm Support**: 7 different heuristic approaches
2. **Factory Pattern**: Clean algorithm selection and instantiation
3. **Enhanced Safety**: Advanced pathfinding with trap avoidance
4. **Verbose Mode**: Detailed debugging output
5. **Simplified Logging**: Clean JSON output without Task-0 replay compatibility
6. **Utility Functions**: Shared code for direction calculation

## Usage

```bash
# Basic usage with different algorithms
python main.py --algorithm BFS --max-games 3
python main.py --algorithm BFS-SAFE-GREEDY --max-games 5
python main.py --algorithm ASTAR --max-games 3
python main.py --algorithm ASTAR-HAMILTONIAN --max-games 2

# With verbose output for debugging
python main.py --algorithm BFS-SAFE-GREEDY --max-games 1 --verbose

# Custom game parameters
python main.py --algorithm ASTAR --max-games 5 --max-steps 500 --grid-size 12
```

## Architecture

The extension follows the established pattern:
- **HeuristicGameManager**: Session management with algorithm factory
- **HeuristicGameLogic**: Game mechanics extension
- **Agent Classes**: 7 different heuristic implementations
- **Base Class Integration**: Seamless inheritance from core components

## Educational Value

This extension demonstrates:
- **Software Evolution**: Natural progression from simple to complex
- **Design Patterns**: Factory, Strategy, Template Method patterns
- **Algorithm Comparison**: Performance differences between approaches
- **Safety vs Efficiency**: Trade-offs in pathfinding strategies

## Performance Insights

1. **BFS-SAFE-GREEDY** performs best (score ~20) due to safety validation
2. **A*** is efficient but can get trapped without safety checks
3. **Hamiltonian** approaches guarantee safety but sacrifice efficiency
4. **DFS** is included for educational comparison (generally poor performance)

## Design Philosophy

The v0.02 extension maintains the same base class architecture as v0.01 while demonstrating how extensions can naturally evolve to support multiple algorithms through factory patterns and polymorphism.

This showcases the power of well-designed base classes that can accommodate diverse implementations without requiring architectural changes. 