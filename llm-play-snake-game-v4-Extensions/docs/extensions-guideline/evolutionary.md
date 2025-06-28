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

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines evolutionary algorithm patterns for extensions.

> **See also:** `agents.md`, `core.md`, `final-decision-10.md`, `factory-design-pattern.md`, `config.md`.

## 🎯 **Core Philosophy: Population-Based Optimization + SUPREME_RULES**

Evolutionary algorithms represent a family of optimization techniques inspired by biological evolution. **This extension strictly follows the SUPREME_RULES** established in `final-decision-10.md`, particularly the **canonical `create()` method patterns and simple logging requirements** for all population-based optimization systems.

### **Guidelines Alignment**
- **final-decision-10.md Guideline 1**: Follows all established GOOD_RULES patterns for evolutionary algorithm architectures
- **final-decision-10.md Guideline 2**: Uses precise `final-decision-N.md` format consistently throughout evolutionary implementations
- **simple logging**: Lightweight, OOP-based common utilities with simple logging (print() statements only)

### **Educational Value**
- **Population Diversity**: Learn bio-inspired optimization using canonical patterns
- **Adaptive Fitness**: Understand evolutionary optimization with simple logging throughout
- **Emergent Behavior**: Experience complex emergence following SUPREME_RULES compliance
- **Bio-Inspired AI**: See canonical patterns in nature-inspired algorithms

## 🏗️ **Evolutionary Algorithm Architecture (SUPREME_RULES Compliant)**

### **Factory Pattern Implementation (CANONICAL create() METHOD)**
**CRITICAL REQUIREMENT**: All Evolutionary Algorithm factories MUST use the canonical `create()` method exactly as specified in `final-decision-10.md` SUPREME_RULES:

```python
class EvolutionaryAgentFactory:
    """
    Factory Pattern for Evolutionary Algorithm agents following final-decision-10.md SUPREME_RULES
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for evolutionary AI agents
    Educational Value: Shows how SUPREME_RULES apply to advanced AI systems -
    canonical patterns work regardless of AI complexity.
    
    Reference: final-decision-10.md SUPREME_RULES for canonical method naming
    """
    
    _registry = {
        "GENETIC": GeneticAlgorithmAgent,
        "EVOLUTIONARY_STRATEGY": EvolutionaryStrategyAgent,
        "NEUROEVOLUTION": NeuroEvolutionAgent,
        "COEVOLUTION": CoEvolutionAgent,
    }
    
    @classmethod
    def create(cls, algorithm_type: str, **kwargs):  # CANONICAL create() method - SUPREME_RULES
        """Create Evolutionary agent using canonical create() method following final-decision-10.md"""
        agent_class = cls._registry.get(algorithm_type.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown Evolutionary algorithm: {algorithm_type}. Available: {available}")
        print(f"[EvolutionaryAgentFactory] Creating agent: {algorithm_type}")  # Simple logging - SUPREME_RULES
        return agent_class(**kwargs)

# ❌ FORBIDDEN: Non-canonical method names (violates SUPREME_RULES)
class EvolutionaryAgentFactory:
    def create_evolutionary_agent(self, algorithm_type: str):  # FORBIDDEN - not canonical
        pass
    
    def build_genetic_agent(self, algorithm_type: str):  # FORBIDDEN - not canonical
        pass
    
    def make_evolutionary_algorithm(self, algorithm_type: str):  # FORBIDDEN - not canonical
        pass
```

### **Population Management Factory (CANONICAL PATTERN)**
```python
class PopulationManagerFactory:
    """
    Factory for population management strategies following SUPREME_RULES.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Educational Value: Shows how canonical create() method enables
    consistent population management across different evolutionary algorithms.
    
    Reference: final-decision-10.md for canonical factory standards
    """
    
    _registry = {
        "SIMPLE": SimplePopulationManager,
        "ELITIST": ElitistPopulationManager,
        "TOURNAMENT": TournamentPopulationManager,
        "ROULETTE": RouletteWheelManager,
    }
    
    @classmethod
    def create(cls, manager_type: str, **kwargs):  # CANONICAL create() method
        """Create population manager using canonical create() method (SUPREME_RULES compliance)"""
        manager_class = cls._registry.get(manager_type.upper())
        if not manager_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown population manager: {manager_type}. Available: {available}")
        print(f"[PopulationManagerFactory] Creating population manager: {manager_type}")  # Simple logging
        return manager_class(**kwargs)
```

### **Genetic Operator Factory (CANONICAL PATTERN)**
```python
class GeneticOperatorFactory:
    """
    Factory for genetic operators following SUPREME_RULES.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Educational Value: Demonstrates canonical create() method for
    genetic operators across different evolutionary algorithm types.
    
    Reference: final-decision-10.md SUPREME_RULES for factory implementation
    """
    
    _registry = {
        "SINGLE_POINT": SinglePointCrossover,
        "TWO_POINT": TwoPointCrossover,
        "UNIFORM": UniformCrossover,
        "GAUSSIAN_MUTATION": GaussianMutation,
        "POLYNOMIAL_MUTATION": PolynomialMutation,
    }
    
    @classmethod
    def create(cls, operator_type: str, **kwargs):  # CANONICAL create() method
        """Create genetic operator using canonical create() method (SUPREME_RULES compliance)"""
        operator_class = cls._registry.get(operator_type.upper())
        if not operator_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown genetic operator: {operator_type}. Available: {available}")
        print(f"[GeneticOperatorFactory] Creating genetic operator: {operator_type}")  # Simple logging
        return operator_class(**kwargs)
```

## 🔧 **Implementation Patterns (SUPREME_RULES Compliant)**

### **Genetic Algorithm Agent (CANONICAL PATTERNS)**
```python
class GeneticAlgorithmAgent(BaseAgent):
    """
    Genetic Algorithm agent for Snake Game following SUPREME_RULES.
    
    Design Pattern: Template Method Pattern (Canonical Implementation)
    Purpose: Population-based optimization using canonical patterns
    Educational Value: Shows how canonical factory patterns work with
    bio-inspired optimization while maintaining simple logging.
    
    Reference: final-decision-10.md for canonical agent architecture
    """
    
    def __init__(self, name: str, grid_size: int,
                 population_size: int = 100,
                 crossover_type: str = "SINGLE_POINT",
                 mutation_type: str = "GAUSSIAN_MUTATION",
                 selection_type: str = "TOURNAMENT"):
        super().__init__(name, grid_size)
        
        # Use canonical factory patterns
        self.crossover_op = GeneticOperatorFactory.create(crossover_type)  # Canonical
        self.mutation_op = GeneticOperatorFactory.create(mutation_type)  # Canonical
        self.population_manager = PopulationManagerFactory.create(selection_type, size=population_size)  # Canonical
        
        self.population_size = population_size
        self.generation = 0
        self.population = self._initialize_population()
        
        print(f"[{name}] GA Agent initialized with {population_size} individuals")  # Simple logging
    
    def evolve_population(self, fitness_scores: list) -> None:
        """Evolve population for one generation with simple logging throughout"""
        print(f"[{self.name}] Starting evolution generation {self.generation}")  # Simple logging
        
        # Selection phase using canonical patterns
        parents = self.population_manager.select_parents(self.population, fitness_scores)
        print(f"[{self.name}] Selected {len(parents)} parents")  # Simple logging
        
        # Crossover phase using canonical patterns
        offspring = []
        for i in range(0, len(parents)-1, 2):
            child1, child2 = self.crossover_op.crossover(parents[i], parents[i+1])
            offspring.extend([child1, child2])
        print(f"[{self.name}] Generated {len(offspring)} offspring via crossover")  # Simple logging
        
        # Mutation phase using canonical patterns
        mutated_offspring = []
        for individual in offspring:
            mutated = self.mutation_op.mutate(individual)
            mutated_offspring.append(mutated)
        print(f"[{self.name}] Mutated {len(mutated_offspring)} offspring")  # Simple logging
        
        # Replacement using canonical population manager
        self.population = self.population_manager.replace_population(
            self.population, mutated_offspring, fitness_scores
        )
        
        self.generation += 1
        print(f"[{self.name}] Evolution generation {self.generation} completed")  # Simple logging
    
    def plan_move(self, game_state: dict) -> str:
        """Plan move using best individual with simple logging"""
        print(f"[{self.name}] Planning move using best individual")  # Simple logging
        
        # Get best individual from population
        best_individual = self.population_manager.get_best_individual(self.population)
        
        # Decode individual to game move
        move = self._decode_individual_to_move(best_individual, game_state)
        
        print(f"[{self.name}] GA decided: {move}")  # Simple logging
        return move
    
    def _initialize_population(self) -> list:
        """Initialize random population with simple logging"""
        print(f"[{self.name}] Initializing random population")  # Simple logging
        
        population = []
        for i in range(self.population_size):
            # Create random individual (strategy encoding)
            individual = np.random.rand(self.grid_size * self.grid_size)
            population.append(individual)
        
        print(f"[{self.name}] Population initialized: {len(population)} individuals")  # Simple logging
        return population
```

### **Evolution Strategy Agent (CANONICAL PATTERNS)**
```python
class EvolutionStrategyAgent(BaseAgent):
    """
    Evolution Strategy agent following canonical patterns.
    
    Design Pattern: Strategy Pattern (Canonical Implementation)
    Educational Value: Shows how canonical factory patterns enable
    consistent implementation across different evolutionary paradigms.
    
    Reference: final-decision-10.md for canonical evolutionary standards
    """
    
    def __init__(self, name: str, grid_size: int,
                 mu: int = 30,  # Parent population size
                 lambda_: int = 100,  # Offspring population size
                 mutation_type: str = "GAUSSIAN_MUTATION"):
        super().__init__(name, grid_size)
        
        # Use canonical factory patterns
        self.mutation_op = GeneticOperatorFactory.create(mutation_type)  # Canonical
        
        self.mu = mu
        self.lambda_ = lambda_
        self.generation = 0
        self.population = self._initialize_es_population()
        
        print(f"[{name}] ES Agent initialized: (μ={mu}, λ={lambda_})")  # Simple logging
    
    def evolve_step(self, fitness_scores: list) -> None:
        """Execute one evolution step with simple logging throughout"""
        print(f"[{self.name}] Starting ES evolution step {self.generation}")  # Simple logging
        
        # Select μ best individuals as parents
        parent_indices = np.argsort(fitness_scores)[-self.mu:]
        parents = [self.population[i] for i in parent_indices]
        print(f"[{self.name}] Selected {len(parents)} parents")  # Simple logging
        
        # Generate λ offspring using canonical mutation
        offspring = []
        for _ in range(self.lambda_):
            parent = random.choice(parents)
            child = self.mutation_op.mutate(parent.copy())
            offspring.append(child)
        print(f"[{self.name}] Generated {len(offspring)} offspring")  # Simple logging
        
        # Replace population with offspring (μ, λ)-ES
        self.population = offspring
        self.generation += 1
        
        print(f"[{self.name}] ES step {self.generation} completed")  # Simple logging
    
    def plan_move(self, game_state: dict) -> str:
        """Plan move using current best strategy with simple logging"""
        print(f"[{self.name}] Planning move with ES strategy")  # Simple logging
        
        # Use first individual as current strategy (could be best from last generation)
        strategy = self.population[0]
        move = self._strategy_to_move(strategy, game_state)
        
        print(f"[{self.name}] ES decided: {move}")  # Simple logging
        return move
```

### **Genetic Programming Agent (CANONICAL PATTERNS)**
```python
class GeneticProgrammingAgent(BaseAgent):
    """
    Genetic Programming agent following canonical patterns.
    
    Design Pattern: Composite Pattern (Canonical Implementation)
    Educational Value: Demonstrates canonical create() method for
    evolving program trees and complex decision structures.
    
    Reference: final-decision-10.md for canonical GP standards
    """
    
    def __init__(self, name: str, grid_size: int,
                 population_size: int = 50,
                 max_depth: int = 6,
                 crossover_type: str = "SUBTREE",
                 mutation_type: str = "POINT_MUTATION"):
        super().__init__(name, grid_size)
        
        # Use canonical factory patterns for GP operators
        self.tree_factory = TreeNodeFactory.create("STANDARD")  # Canonical
        self.crossover_op = GeneticOperatorFactory.create(crossover_type)  # Canonical
        self.mutation_op = GeneticOperatorFactory.create(mutation_type)  # Canonical
        
        self.population_size = population_size
        self.max_depth = max_depth
        self.generation = 0
        self.population = self._initialize_gp_population()
        
        print(f"[{name}] GP Agent initialized with {population_size} trees")  # Simple logging
    
    def evolve_trees(self, fitness_scores: list) -> None:
        """Evolve program trees with simple logging throughout"""
        print(f"[{self.name}] Starting GP evolution generation {self.generation}")  # Simple logging
        
        # Tournament selection
        parents = self._tournament_selection(self.population, fitness_scores)
        print(f"[{self.name}] Selected parents via tournament selection")  # Simple logging
        
        # Tree crossover using canonical patterns
        offspring = []
        for i in range(0, len(parents)-1, 2):
            child1, child2 = self.crossover_op.crossover_trees(parents[i], parents[i+1])
            offspring.extend([child1, child2])
        print(f"[{self.name}] Generated {len(offspring)} offspring trees")  # Simple logging
        
        # Tree mutation using canonical patterns
        for tree in offspring:
            if random.random() < 0.1:  # Mutation probability
                self.mutation_op.mutate_tree(tree)
        print(f"[{self.name}] Applied mutations to offspring trees")  # Simple logging
        
        # Replace population
        self.population = offspring[:self.population_size]
        self.generation += 1
        
        print(f"[{self.name}] GP generation {self.generation} completed")  # Simple logging
    
    def plan_move(self, game_state: dict) -> str:
        """Plan move using best evolved program with simple logging"""
        print(f"[{self.name}] Evaluating best program tree")  # Simple logging
        
        # Get best tree from population
        best_tree = self.population[0]  # Assuming sorted by fitness
        
        # Evaluate tree on current game state
        move = self._evaluate_tree(best_tree, game_state)
        
        print(f"[{self.name}] GP program decided: {move}")  # Simple logging
        return move
```

## 📊 **Simple Logging Standards for Evolutionary Operations**

### **Required Logging Pattern (SUPREME_RULES)**
All evolutionary operations MUST use simple print statements as established in `final-decision-10.md`:

```python
# ✅ CORRECT: Simple logging for evolutionary operations (SUPREME_RULES compliance)
def evolve_population(population: list, fitness_scores: list):
    print(f"[EvolutionEngine] Starting evolution cycle")  # Simple logging - REQUIRED
    
    # Selection phase
    parents = select_parents(population, fitness_scores)
    print(f"[EvolutionEngine] Selected {len(parents)} parents")  # Simple logging
    
    # Genetic operators
    offspring = apply_crossover(parents)
    print(f"[EvolutionEngine] Generated {len(offspring)} offspring")  # Simple logging
    
    mutated_offspring = apply_mutation(offspring)
    print(f"[EvolutionEngine] Applied mutations")  # Simple logging
    
    print(f"[EvolutionEngine] Evolution cycle completed")  # Simple logging
    return mutated_offspring

# ❌ FORBIDDEN: Complex logging frameworks (violates SUPREME_RULES)
import logging
logger = logging.getLogger(__name__)

def evolve_population(population: list, fitness_scores: list):
    logger.info(f"Starting evolution")  # FORBIDDEN - complex logging
    # This violates final-decision-10.md SUPREME_RULES
```

## 🎓 **Educational Applications with Canonical Patterns**

### **Bio-Inspired Optimization Understanding**
- **Population Dynamics**: Learn evolutionary dynamics using canonical patterns
- **Genetic Operators**: Understand crossover and mutation with simple logging throughout
- **Fitness Landscapes**: Experience optimization landscapes following SUPREME_RULES compliance
- **Emergent Intelligence**: See complex behavior emerge from canonical factory patterns

### **Pattern Consistency Across Optimization Methods**
- **Factory Patterns**: All evolutionary components use canonical `create()` method consistently
- **Simple Logging**: Print statements provide clear visibility into evolutionary processes
- **Educational Value**: Canonical patterns work identically across heuristics, ML, and evolutionary systems
- **SUPREME_RULES**: Bio-inspired algorithms follow same standards as other AI approaches

## 📋 **SUPREME_RULES Implementation Checklist for Evolutionary Algorithms**

### **Mandatory Requirements**
- [ ] **Canonical Method**: All factories use `create()` method exactly (SUPREME_RULES requirement)
- [ ] **Simple Logging**: Uses print() statements only for all evolutionary operations (final-decision-10.md compliance)
- [ ] **GOOD_RULES Reference**: References `final-decision-10.md` in all evolutionary documentation
- [ ] **Pattern Consistency**: Follows canonical patterns across all evolutionary implementations

### **Evolutionary-Specific Standards**
- [ ] **Population Management**: Canonical factory patterns for all population components
- [ ] **Genetic Operators**: Canonical factory patterns for all crossover and mutation systems
- [ ] **Selection Methods**: Canonical patterns for all selection and replacement strategies
- [ ] **Fitness Evaluation**: Simple logging for all fitness computation and evolution tracking

### **Educational Integration**
- [ ] **Clear Examples**: Simple examples using canonical `create()` method for evolutionary systems
- [ ] **Pattern Explanation**: Clear explanation of canonical patterns in bio-inspired context
- [ ] **Best Practices**: Demonstration of SUPREME_RULES in population-based optimization
- [ ] **Learning Value**: Easy to understand canonical patterns regardless of evolutionary complexity

---

**Evolutionary algorithms represent sophisticated bio-inspired optimization while maintaining strict compliance with `final-decision-10.md` SUPREME_RULES, proving that canonical patterns and simple logging provide robust foundations across all optimization paradigms.**

## 🔗 **See Also**

- **`agents.md`**: Authoritative reference for agent implementation with canonical patterns
- **`core.md`**: Base class architecture following canonical principles
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`factory-design-pattern.md`**: Canonical factory implementation for all systems