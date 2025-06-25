> **Important**: For authoritative data format decisions, see `data-format-decision-guide.md`.

**Data Format for Evolutionary Algorithms**: Evolutionary algorithms use **Raw Arrays (NPZ format)** for population-based optimization. The 16-feature CSV schema is not suitable for genetic operators.

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

### **Following GOOD_RULES Patterns**
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

## 🧠 **Specialized Data Format for Evolutionary Algorithms**

### **Evolutionary NPZ Format Specification**

Evolutionary algorithms require a **specialized data format** that supports population-based operations, genotype-phenotype mapping, and multi-objective optimization.

```python
# Evolutionary Algorithm Data Format (NPZ Raw Arrays)
evolutionary_data = {
    # Population Structure
    'population': np.array(shape=(population_size, individual_length)),
    'fitness_scores': np.array(shape=(population_size, num_objectives)),
    'generation_history': np.array(shape=(num_generations, population_size, individual_length)),
    
    # Genetic Operators Data
    'crossover_points': np.array(shape=(num_crossovers, 2)),  # Parent indices
    'mutation_mask': np.array(shape=(population_size, individual_length)),  # Boolean mask
    'selection_pressure': np.array(shape=(num_generations,)),  # Selection statistics
    
    # Fitness Landscape
    'fitness_landscape': np.array(shape=(grid_size, grid_size, num_objectives)),
    'pareto_front': np.array(shape=(pareto_size, num_objectives)),
    
    # Evolutionary Metadata
    'generation_metadata': {
        'best_fitness': np.array(shape=(num_generations,)),
        'average_fitness': np.array(shape=(num_generations,)),
        'diversity_metrics': np.array(shape=(num_generations,)),
        'convergence_rate': np.array(shape=(num_generations,))
    },
    
    # Game-Specific Evolutionary Data
    'game_performance': {
        'scores': np.array(shape=(population_size,)),
        'steps': np.array(shape=(population_size,)),
        'efficiency': np.array(shape=(population_size,)),
        'survival_rate': np.array(shape=(population_size,))
    }
}
```

### **Why This Format is Special for Evolutionary Algorithms**

#### **1. Population-Centric Structure**
- **Direct genetic representation**: Each individual is a raw array
- **Batch operations**: Support for population-wide genetic operators
- **Diversity tracking**: Built-in metrics for population health

#### **2. Multi-Objective Support**
- **Fitness vectors**: Multiple objectives per individual
- **Pareto front tracking**: Multi-objective optimization support
- **Trade-off analysis**: Objective correlation matrices

#### **3. Genetic Operator Efficiency**
- **Crossover tracking**: Record which individuals were crossed
- **Mutation history**: Track mutation patterns and success rates
- **Selection pressure**: Monitor selection algorithm performance

#### **4. Fitness Landscape Analysis**
- **Spatial representation**: Grid-based fitness mapping
- **Convergence tracking**: Monitor algorithm convergence
- **Diversity metrics**: Population diversity over generations

#### **5. Game-Specific Evolutionary Features**
- **Performance correlation**: Link genetic traits to game performance
- **Strategy evolution**: Track how strategies evolve over generations
- **Adaptation patterns**: Monitor adaptation to different game scenarios

### **Implementation Example**

```python
# extensions/evolutionary-v0.02/agents/agent_ga.py
class GAAgent(BaseAgent):
    """Genetic Algorithm Agent with specialized data format"""
    
    def __init__(self, population_size=100, individual_length=64):
        super().__init__()
        self.population_size = population_size
        self.individual_length = individual_length
        self.population = np.random.rand(population_size, individual_length)
        self.fitness_scores = np.zeros((population_size, 3))  # score, steps, efficiency
    
    def save_evolutionary_data(self, output_path):
        """Save evolutionary data in specialized NPZ format"""
        evolutionary_data = {
            'population': self.population,
            'fitness_scores': self.fitness_scores,
            'generation_history': self.generation_history,
            'crossover_points': self.crossover_history,
            'mutation_mask': self.mutation_history,
            'selection_pressure': self.selection_history,
            'fitness_landscape': self.compute_fitness_landscape(),
            'pareto_front': self.compute_pareto_front(),
            'generation_metadata': {
                'best_fitness': self.best_fitness_history,
                'average_fitness': self.avg_fitness_history,
                'diversity_metrics': self.diversity_history,
                'convergence_rate': self.convergence_history
            },
            'game_performance': {
                'scores': self.game_scores,
                'steps': self.game_steps,
                'efficiency': self.game_efficiency,
                'survival_rate': self.survival_rates
            }
        }
        
        np.savez(output_path, **evolutionary_data)
```

### **Benefits of This Evolutionary Format**

#### **1. Algorithm Efficiency**
- **Vectorized operations**: NumPy arrays enable fast genetic operators
- **Memory efficiency**: Compressed storage of large populations
- **Parallel processing**: Support for parallel fitness evaluation

#### **2. Research Value**
- **Reproducibility**: Complete evolutionary history preserved
- **Analysis capabilities**: Rich data for evolutionary analysis
- **Visualization support**: Data structure supports evolutionary visualization

#### **3. Educational Value**
- **Clear genotype-phenotype mapping**: Direct representation
- **Evolutionary process transparency**: Complete tracking of evolution
- **Multi-objective demonstration**: Shows trade-offs in optimization

#### **4. Cross-Extension Integration**
- **Fitness landscape sharing**: Other extensions can analyze fitness landscapes
- **Strategy transfer**: Evolved strategies can be analyzed by other algorithms
- **Benchmarking**: Provides benchmarks for other optimization approaches

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