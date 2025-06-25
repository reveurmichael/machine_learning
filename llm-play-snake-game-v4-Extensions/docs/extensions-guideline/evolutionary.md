**IMPORTANT**: Evolutionary algorithms are supported but use specialized state representations. The standard 16-feature CSV schema may be insufficient for evolutionary approaches, which often benefit from raw board states, spatial patterns, or graph structures.

**State Representation Decision Matrix**:

| Representation Type | Best For | Why Evolutionary Needs Different |
|-------------------|----------|--------------------------------|
| **16-Feature Tabular** | XGBoost, Random Forest, simple MLP | ❌ Too compressed for genetic operators |
| **Raw Board State** | **Evolutionary algorithms, GP** | ✅ Direct manipulation by genetic operators |
| **Sequential Data** | LSTM, GRU, temporal models | ❌ Fixed temporal structure limits evolution |
| **Spatial Arrays** | CNN, computer vision models | ✅ Good for spatial genetic operators |
| **Graph Structures** | GNN, relationship-based models | ✅ Excellent for graph evolution algorithms |

**Why Evolutionary Algorithms Need Special Representations:**
- **Genetic Operators**: Crossover and mutation work better on raw data structures
- **Search Space**: Full board state provides richer search landscape
- **Phenotype Mapping**: Direct representation enables clearer genotype-phenotype mapping
- **Population Diversity**: Raw representations support greater population diversity

# Evolutionary Algorithms for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ and extension guidelines. Evolutionary algorithms follow the same architectural patterns as other extensions.

## 🧬 **Core Philosophy: Population-Based Optimization**

Evolutionary algorithms represent a family of optimization techniques inspired by biological evolution. In the Snake Game AI context, these algorithms evolve populations of agents to discover optimal game-playing strategies through selection, crossover, and mutation operations.

### **Design Philosophy**
- **Population Diversity**: Maintain genetic diversity for robust exploration
- **Adaptive Fitness**: Evolve evaluation criteria alongside strategies
- **Emergent Behavior**: Allow complex strategies to emerge from simple rules
- **Educational Value**: Demonstrate bio-inspired optimization principles

## 🎯 **Integration with Extension Architecture**

### **Extension Evolution Integration**
Evolutionary algorithms follow the **same standardized evolution** as other algorithm types:

| Version | Evolutionary Extension Characteristics |
|---------|---------------------------------------|
| **v0.01** | Single GA implementation (proof of concept) |
| **v0.02** | Multiple evolutionary approaches + factory patterns |
| **v0.03** | Web interface + dataset generation capabilities |
| **v0.04** | ❌ Not supported (heuristics only) |

### **Following GOODFILES Patterns**
Evolutionary algorithms integrate seamlessly with the established architecture:

**Directory Structure (Final Decision 5)**:
```
extensions/evolutionary-v0.02/
├── agents/                     # 🔒 Core evolutionary algorithms
│   ├── __init__.py            # Factory pattern
│   ├── agent_ga.py            # Genetic Algorithm
│   ├── agent_es.py            # Evolution Strategies  
│   └── agent_gp.py            # Genetic Programming
```

### **Agent Naming Conventions**
Following Final Decision 4:
```python
# Standard agent naming pattern
agent_ga.py              → class GAAgent(BaseAgent)
agent_es.py              → class ESAgent(BaseAgent)
agent_gp.py              → class GPAgent(BaseAgent)
agent_ga_deap.py         → class GADeapAgent(BaseAgent)
```

### **Factory Pattern Integration**
Following Final Decision 7-8:
```python
class EvolutionaryAgentFactory:
    """Factory for creating evolutionary algorithm agents"""
    
    _agent_registry = {
        "GA": GAAgent,
        "ES": ESAgent,
        "GP": GPAgent,
        "GA_DEAP": GADeapAgent,
    }
    
    @classmethod
    def create_agent(cls, algorithm: str, **kwargs) -> BaseAgent:
        """Create evolutionary agent by algorithm name"""
        return cls._agent_registry[algorithm.upper()](**kwargs)
```

## 🔧 **Evolutionary Approaches**

### **Genetic Algorithms (GA)**
- **Classic Implementation**: Hand-coded genetic operators
- **DEAP Framework**: Leveraging established evolutionary framework
- **Hybrid Approaches**: Combining custom logic with framework benefits

### **Evolution Strategies (ES)**
- **Parameter Optimization**: Direct policy parameter evolution
- **Adaptive Mutation**: Self-adapting mutation parameters
- **Covariance Matrix Adaptation**: Advanced ES variants

### **Genetic Programming (GP)**
- **Tree-Based Programs**: Evolving decision trees
- **Graph-Based Networks**: Neural architecture search
- **Symbolic Regression**: Discovering mathematical relationships

## 🎓 **Educational and Research Value**

### **Design Pattern Demonstration**
Evolutionary algorithms showcase multiple design patterns:
- **Template Method**: Common evolutionary workflow
- **Strategy Pattern**: Different selection/crossover strategies
- **Observer Pattern**: Fitness tracking and visualization
- **Factory Pattern**: Algorithm creation and configuration

### **Comparative Studies**
- **vs. Heuristics**: Evolved strategies vs. hand-crafted algorithms
- **vs. ML Methods**: Population-based vs. gradient-based optimization
- **vs. RL**: Evolution vs. temporal difference learning
- **Hybrid Approaches**: Combining evolutionary with other methods

## 🧠 **State Representation Challenge**

### **Critical Design Decision**
The 16-feature CSV schema (from csv-schema-1.md) may be insufficient for evolutionary algorithms. Evolutionary approaches often benefit from:

- **Raw Board State**: Direct grid representation
- **Spatial Patterns**: 2D convolutional features
- **Temporal Sequences**: Historical state information
- **Graph Structures**: Snake body as connected components

### **Alternative Representations**
```python
# Example extended representation for evolutionary algorithms
class EvolutionaryGameState:
    """Extended state representation for evolutionary algorithms"""
    
    def __init__(self, game_state):
        self.raw_board = self.extract_board_matrix(game_state)
        self.spatial_features = self.compute_spatial_features(game_state)
        self.temporal_history = self.update_history(game_state)
        self.graph_representation = self.build_graph(game_state)
```

## 🚀 **Implementation Guidelines**

### **Path Management**
Following Final Decision 6:
```python
from extensions.common.path_utils import get_dataset_path

# Standardized evolutionary dataset paths
evolution_dataset_path = get_dataset_path(
    extension_type="evolutionary",
    version="0.02",
    grid_size=grid_size,
    algorithm="ga",
    timestamp=timestamp  # Format: YYYYMMDD_HHMMSS
)
# Result: logs/extensions/datasets/grid-size-{grid_size}/evolutionary_v0.02_{timestamp}/
```

### **Configuration Management**
Following Final Decision 2:
```python
from extensions.common.config.evolutionary_constants import (
    DEFAULT_POPULATION_SIZE,
    DEFAULT_MUTATION_RATE,
    DEFAULT_CROSSOVER_RATE,
    MAX_GENERATIONS
)
```

### **Multi-Framework Support**
- **DEAP Framework**: Mature, feature-rich evolutionary framework
- **Custom Implementation**: Educational, domain-specific optimizations
- **Hybrid Approaches**: Best of both worlds

## 🔮 **Future Directions**

### **Cross-Extension Integration**
- **Neural Evolution**: Evolving neural network architectures
- **Reward Evolution**: Evolutionary reward function design (Eureka integration)
- **Multi-Objective**: Optimizing multiple game performance metrics
- **Co-Evolution**: Competitive evolution of strategies

### **Educational Applications**
- **Algorithm Comparison**: Side-by-side evolutionary approach comparison
- **Parameter Studies**: Impact of population size, mutation rates, etc.
- **Visualization**: Real-time evolution progress and diversity metrics
- **Research Projects**: Framework for studying evolutionary computation

---

**Evolutionary algorithms provide a unique perspective on optimization, demonstrating how nature-inspired approaches can discover novel solutions. By following the established architectural patterns while addressing the unique challenges of evolutionary computation, these extensions maintain system coherence while exploring the fascinating world of population-based optimization.**