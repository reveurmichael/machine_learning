# Evolutionary Algorithms for Snake Game AI

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for evolutionary algorithms and supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`).

> **See also:** `data-format-decision-guide.md`, `final-decision-10.md`, `project-structure-plan.md`.

## 🎯 **Core Philosophy: Population-Based Optimization**

Evolutionary algorithms use **population-based optimization** to evolve solutions through natural selection, mutation, and crossover operations. These algorithms excel at finding complex, non-linear solutions that traditional algorithms might miss, strictly following `final-decision-10.md` SUPREME_RULES.

### **Educational Value**
- **Population Dynamics**: Understanding how populations evolve over time
- **Genetic Operations**: Learning mutation, crossover, and selection mechanisms
- **Fitness Landscapes**: Exploring how fitness functions guide evolution
- **Multi-Objective Optimization**: Balancing competing objectives

## 🏗️ **Factory Pattern: Canonical Method is create()**

All evolutionary algorithm factories must use the canonical method name `create()` for instantiation, not `create_individual()`, `create_population()`, or any other variant. This ensures consistency and aligns with the KISS principle and SUPREME_RULES from `final-decision-10.md`.

### **Evolutionary Factory Implementation (SUPREME_RULES Compliant)**
```python
from utils.factory_utils import SimpleFactory

class EvolutionaryFactory:
    """
    Factory Pattern for Evolutionary Algorithm components following SUPREME_RULES from final-decision-10.md
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Demonstrates canonical create() method for evolutionary AI components
    Educational Value: Shows how SUPREME_RULES apply to population-based AI systems -
    canonical patterns work regardless of AI complexity.
    
    Reference: SUPREME_RULES from final-decision-10.md for canonical method naming
    """
    
    _registry = {
        "INDIVIDUAL": SnakeIndividual,
        "POPULATION": EvolutionaryPopulation,
        "GENETIC_OPERATORS": GeneticOperators,
        "FITNESS_EVALUATOR": FitnessEvaluator,
        "EVOLUTIONARY_ALGORITHM": EvolutionaryAlgorithm,
    }
    
    @classmethod
    def create(cls, component_type: str, **kwargs):  # CANONICAL create() method per SUPREME_RULES
        """Create evolutionary component using canonical create() method following SUPREME_RULES from final-decision-10.md"""
        component_class = cls._registry.get(component_type.upper())
        if not component_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown evolutionary component: {component_type}. Available: {available}")
        print(f"[EvolutionaryFactory] Creating component: {component_type}")  # SUPREME_RULES compliant logging
        return component_class(**kwargs)

# ❌ FORBIDDEN: Non-canonical method names (violates SUPREME_RULES)
class EvolutionaryFactory:
    def create_individual(self, component_type: str):  # FORBIDDEN - not canonical
        pass
    
    def build_population(self, component_type: str):  # FORBIDDEN - not canonical
        pass
    
    def make_evolutionary_component(self, component_type: str):  # FORBIDDEN - not canonical
        pass
```

## 🧬 **Evolutionary Algorithm Components**

### **1. Individual Representation**
```python
@dataclass
class SnakeIndividual:
    """
    Represents a single snake strategy as a genetic individual.
    
    Design Pattern: Value Object Pattern
    - Immutable individual representation
    - Encapsulated genetic material
    - Clear fitness evaluation interface
    """
    
    # Genetic material: strategy parameters
    strategy_weights: List[float]  # Weights for different behaviors
    mutation_rate: float           # Individual mutation rate
    crossover_preference: float    # Crossover preference
    
    # Fitness and metadata
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate individual after initialization."""
        if len(self.strategy_weights) == 0:
            raise ValueError("Strategy weights cannot be empty")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("Mutation rate must be between 0.0 and 1.0")
        
        print(f"[SnakeIndividual] Created individual with {len(self.strategy_weights)} weights")  # SUPREME_RULES compliant logging
```

### **2. Population Management**
```python
class EvolutionaryPopulation:
    """
    Manages a population of snake individuals.
    
    Design Pattern: Collection Pattern
    - Encapsulates population operations
    - Provides iteration and access methods
    - Manages population statistics
    """
    
    def __init__(self, population_size: int, individual_length: int):
        self.population_size = population_size
        self.individual_length = individual_length
        self.individuals: List[SnakeIndividual] = []
        self.generation = 0
        
        print(f"[EvolutionaryPopulation] Initialized population of size {population_size}")  # SUPREME_RULES compliant logging
    
    def initialize_random_population(self) -> None:
        """Initialize population with random individuals."""
        self.individuals = []
        for i in range(self.population_size):
            individual = self._generate_random_individual()  # Updated method name
            individual.generation = self.generation
            individual.parent_ids = [i]  # Self as parent for initial generation
            self.individuals.append(individual)
        
        print(f"[EvolutionaryPopulation] Created random population of {len(self.individuals)} individuals")  # SUPREME_RULES compliant logging
    
    def _generate_random_individual(self) -> SnakeIndividual:  # Updated method name
        """Generate a random individual."""
        strategy_weights = [random.uniform(-1.0, 1.0) for _ in range(self.individual_length)]
        mutation_rate = random.uniform(0.01, 0.1)
        crossover_preference = random.uniform(0.0, 1.0)
        
        return SnakeIndividual(
            strategy_weights=strategy_weights,
            mutation_rate=mutation_rate,
            crossover_preference=crossover_preference
        )
```

### **3. Genetic Operations**
```python
class GeneticOperators:
    """
    Implements genetic operations for evolutionary algorithms.
    
    Design Pattern: Strategy Pattern
    - Pluggable genetic operations
    - Consistent interface for different operators
    - Easy to extend with new operators
    """
    
    def mutate(self, individual: SnakeIndividual, mutation_strength: float = 0.1) -> SnakeIndividual:
        """
        Apply mutation to an individual.
        
        Args:
            individual: Individual to mutate
            mutation_strength: Strength of mutation (0.0 to 1.0)
            
        Returns:
            New mutated individual
        """
        # Create new individual with mutated weights
        new_weights = []
        for weight in individual.strategy_weights:
            if random.random() < individual.mutation_rate:
                # Add Gaussian noise for mutation
                mutation = random.gauss(0, mutation_strength)
                new_weight = weight + mutation
                # Clamp to valid range
                new_weight = max(-1.0, min(1.0, new_weight))
                new_weights.append(new_weight)
            else:
                new_weights.append(weight)
        
        # Create new individual
        new_individual = SnakeIndividual(
            strategy_weights=new_weights,
            mutation_rate=individual.mutation_rate,
            crossover_preference=individual.crossover_preference,
            generation=individual.generation + 1,
            parent_ids=[id(individual)]  # Track parent for genealogy
        )
        
        print(f"[GeneticOperators] Mutated individual {id(individual)}")  # SUPREME_RULES compliant logging
        return new_individual
    
    def crossover(self, parent1: SnakeIndividual, parent2: SnakeIndividual) -> Tuple[SnakeIndividual, SnakeIndividual]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Tuple of two offspring individuals
        """
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1.strategy_weights) - 1)
        
        # Create offspring 1
        weights1 = parent1.strategy_weights[:crossover_point] + parent2.strategy_weights[crossover_point:]
        offspring1 = SnakeIndividual(
            strategy_weights=weights1,
            mutation_rate=(parent1.mutation_rate + parent2.mutation_rate) / 2,
            crossover_preference=(parent1.crossover_preference + parent2.crossover_preference) / 2,
            generation=parent1.generation + 1,
            parent_ids=[id(parent1), id(parent2)]
        )
        
        # Create offspring 2
        weights2 = parent2.strategy_weights[:crossover_point] + parent1.strategy_weights[crossover_point:]
        offspring2 = SnakeIndividual(
            strategy_weights=weights2,
            mutation_rate=(parent1.mutation_rate + parent2.mutation_rate) / 2,
            crossover_preference=(parent1.crossover_preference + parent2.crossover_preference) / 2,
            generation=parent1.generation + 1,
            parent_ids=[id(parent1), id(parent2)]
        )
        
        print(f"[GeneticOperators] Crossover between individuals {id(parent1)} and {id(parent2)}")  # SUPREME_RULES compliant logging
        return offspring1, offspring2
```

### **4. Fitness Evaluation**
```python
class FitnessEvaluator:
    """
    Evaluates fitness of individuals through game simulation.
    
    Design Pattern: Strategy Pattern
    - Pluggable fitness functions
    - Consistent evaluation interface
    - Support for multi-objective optimization
    """
    
    def __init__(self, grid_size: int, num_games: int = 10):
        self.grid_size = grid_size
        self.num_games = num_games
        self.game_simulator = GameSimulator(grid_size)
        
        print(f"[FitnessEvaluator] Initialized with grid size {grid_size}")  # SUPREME_RULES compliant logging
    
    def evaluate_fitness(self, individual: SnakeIndividual) -> float:
        """
        Evaluate fitness of an individual through game simulation.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        total_score = 0
        total_steps = 0
        games_won = 0
        
        for game_id in range(self.num_games):
            # Create agent with individual's strategy
            agent = self._generate_agent_from_individual(individual)  # Updated method name
            
            # Run game simulation
            game_result = self.game_simulator.run_game(agent)
            
            total_score += game_result.score
            total_steps += game_result.steps
            if game_result.won:
                games_won += 1
        
        # Calculate fitness as weighted combination of objectives
        avg_score = total_score / self.num_games
        avg_steps = total_steps / self.num_games
        win_rate = games_won / self.num_games
        
        # Multi-objective fitness function
        fitness = (
            0.5 * avg_score +      # Primary objective: score
            0.3 * win_rate +       # Secondary objective: win rate
            0.2 * (1.0 / avg_steps)  # Tertiary objective: efficiency
        )
        
        individual.fitness_score = fitness
        
        print(f"[FitnessEvaluator] Individual {id(individual)} fitness: {fitness:.3f}")  # SUPREME_RULES compliant logging
        return fitness
    
    def _generate_agent_from_individual(self, individual: SnakeIndividual) -> SnakeAgent:  # Updated method name
        """Generate a snake agent from individual's genetic material."""
        return SnakeAgent(
            strategy_weights=individual.strategy_weights,
            grid_size=self.grid_size
        )
```

## 🚀 **Evolutionary Algorithm Implementation**

### **Main Evolutionary Loop**
```python
class EvolutionaryAlgorithm:
    """
    Main evolutionary algorithm implementation.
    
    Design Pattern: Template Method Pattern
    - Defines evolutionary workflow
    - Allows customization of specific steps
    - Consistent algorithm structure
    """
    
    def __init__(self, population_size: int, individual_length: int, grid_size: int):
        self.population = EvolutionaryPopulation(population_size, individual_length)
        self.fitness_evaluator = FitnessEvaluator(grid_size)
        self.genetic_operators = GeneticOperators()
        self.generation = 0
        self.max_generations = 100
        
        print(f"[EvolutionaryAlgorithm] Initialized EA with population size {population_size}")  # SUPREME_RULES compliant logging
    
    def run_evolution(self) -> List[SnakeIndividual]:
        """
        Run the complete evolutionary process.
        
        Returns:
            List of best individuals from each generation
        """
        best_individuals = []
        
        # Initialize population
        self.population.initialize_random_population()
        
        for generation in range(self.max_generations):
            print(f"[EvolutionaryAlgorithm] Starting generation {generation + 1}")  # SUPREME_RULES compliant logging
            
            # Evaluate fitness of all individuals
            self._evaluate_population()
            
            # Select best individuals
            best_individuals_gen = self._select_best_individuals()
            best_individuals.append(best_individuals_gen[0])  # Keep best from each generation
            
            # Generate next generation
            self._generate_next_generation()  # Updated method name
            
            # Print generation statistics
            self._print_generation_stats(generation)
        
        return best_individuals
    
    def _evaluate_population(self) -> None:
        """Evaluate fitness of all individuals in population."""
        for individual in self.population.individuals:
            self.fitness_evaluator.evaluate_fitness(individual)
    
    def _select_best_individuals(self, selection_ratio: float = 0.2) -> List[SnakeIndividual]:
        """Select best individuals for reproduction."""
        # Sort by fitness (descending)
        sorted_individuals = sorted(
            self.population.individuals,
            key=lambda x: x.fitness_score,
            reverse=True
        )
        
        # Select top individuals
        num_selected = int(len(sorted_individuals) * selection_ratio)
        selected = sorted_individuals[:num_selected]
        
        print(f"[EvolutionaryAlgorithm] Selected {len(selected)} best individuals")  # SUPREME_RULES compliant logging
        return selected
    
    def _generate_next_generation(self) -> None:  # Updated method name
        """Generate next generation through crossover and mutation."""
        new_individuals = []
        
        # Elitism: keep best individual unchanged
        best_individual = max(self.population.individuals, key=lambda x: x.fitness_score)
        new_individuals.append(best_individual)
        
        # Generate offspring through crossover and mutation
        while len(new_individuals) < self.population.population_size:
            # Select parents for crossover
            parent1, parent2 = self._select_parents()
            
            # Perform crossover
            offspring1, offspring2 = self.genetic_operators.crossover(parent1, parent2)
            
            # Apply mutation
            offspring1 = self.genetic_operators.mutate(offspring1)
            offspring2 = self.genetic_operators.mutate(offspring2)
            
            new_individuals.extend([offspring1, offspring2])
        
        # Trim to population size
        new_individuals = new_individuals[:self.population.population_size]
        
        # Update population
        self.population.individuals = new_individuals
        self.population.generation += 1
        
        print(f"[EvolutionaryAlgorithm] Generated new generation with {len(new_individuals)} individuals")  # SUPREME_RULES compliant logging
    
    def _select_parents(self) -> Tuple[SnakeIndividual, SnakeIndividual]:
        """Select two parents for crossover using tournament selection."""
        tournament_size = 3
        
        # Tournament selection for parent 1
        tournament1 = random.sample(self.population.individuals, tournament_size)
        parent1 = max(tournament1, key=lambda x: x.fitness_score)
        
        # Tournament selection for parent 2
        tournament2 = random.sample(self.population.individuals, tournament_size)
        parent2 = max(tournament2, key=lambda x: x.fitness_score)
        
        return parent1, parent2
    
    def _print_generation_stats(self, generation: int) -> None:
        """Print statistics for current generation."""
        fitness_scores = [ind.fitness_score for ind in self.population.individuals]
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        max_fitness = max(fitness_scores)
        min_fitness = min(fitness_scores)
        
        print(f"[EvolutionaryAlgorithm] Generation {generation + 1} stats: "
              f"avg={avg_fitness:.3f}, max={max_fitness:.3f}, min={min_fitness:.3f}")  # SUPREME_RULES compliant logging
```

## 📊 **Data Format for Evolutionary Algorithms**

### **Specialized NPZ Format**
```python
def save_evolutionary_dataset(evolutionary_data: dict, output_path: Path):
    """
    Save evolutionary data in specialized NPZ format.
    
    This format is specifically designed for evolutionary algorithms
    and includes population data, genetic history, and fitness metrics.
    """
    np.savez(output_path, **evolutionary_data)
    print(f"[EvolutionaryUtils] Saved evolutionary dataset: {output_path}")  # SUPREME_RULES compliant logging

def create_evolutionary_dataset(
    population_history: List[List[SnakeIndividual]],
    fitness_history: List[List[float]],
    genetic_operations: List[dict]
) -> dict:
    """
    Create specialized evolutionary dataset.
    
    Returns:
        Dictionary containing all evolutionary data in NPZ format
    """
    evolutionary_data = {
        # Population Structure
        'population_history': np.array([
            [ind.strategy_weights for ind in gen] 
            for gen in population_history
        ]),
        'fitness_history': np.array(fitness_history),
        
        # Genetic Operations Data
        'crossover_points': np.array([
            op['crossover_point'] for op in genetic_operations 
            if op['type'] == 'crossover'
        ]),
        'mutation_masks': np.array([
            op['mutation_mask'] for op in genetic_operations 
            if op['type'] == 'mutation'
        ]),
        
        # Fitness Landscape
        'fitness_landscape': compute_fitness_landscape(population_history),
        'pareto_front': compute_pareto_front(fitness_history),
        
        # Evolutionary Metadata
        'generation_metadata': {
            'best_fitness': [max(gen) for gen in fitness_history],
            'average_fitness': [sum(gen)/len(gen) for gen in fitness_history],
            'diversity_metrics': compute_diversity_metrics(population_history),
            'convergence_rate': compute_convergence_rate(fitness_history)
        },
        
        # Game Performance Correlation
        'game_performance': {
            'scores': extract_game_scores(population_history),
            'steps': extract_game_steps(population_history),
            'efficiency': extract_efficiency_metrics(population_history),
            'survival_rate': extract_survival_rates(population_history)
        }
    }
    
    return evolutionary_data
```

## 🎯 **Implementation Checklist**

### **Core Components**
- [ ] **Individual Representation**: Proper genetic material encoding
- [ ] **Population Management**: Efficient population operations
- [ ] **Genetic Operations**: Mutation and crossover implementations
- [ ] **Fitness Evaluation**: Game-based fitness assessment

### **Algorithm Features**
- [ ] **Selection Methods**: Tournament, roulette wheel, or rank-based selection
- [ ] **Elitism**: Preserve best individuals across generations
- [ ] **Diversity Maintenance**: Prevent premature convergence
- [ ] **Multi-Objective Support**: Handle competing fitness objectives

### **Data Management**
- [ ] **NPZ Format**: Use specialized evolutionary NPZ format
- [ ] **Population History**: Track population evolution over time
- [ ] **Genetic Operations**: Record crossover and mutation history
- [ ] **Fitness Metrics**: Comprehensive fitness tracking and analysis

## 🎓 **Educational Benefits**

### **Learning Objectives**
- **Population Dynamics**: Understanding how populations evolve
- **Genetic Operations**: Learning mutation and crossover mechanisms
- **Fitness Landscapes**: Exploring optimization landscapes
- **Multi-Objective Optimization**: Balancing competing objectives

### **Best Practices**
- **Parameter Tuning**: Understanding evolutionary algorithm parameters
- **Convergence Analysis**: Monitoring algorithm convergence
- **Diversity Management**: Maintaining population diversity
- **Performance Evaluation**: Assessing evolutionary algorithm performance

---

**Evolutionary algorithms provide powerful population-based optimization for Snake Game AI, enabling discovery of complex strategies through natural selection and genetic operations.**

## 🔗 **See Also**

- **`data-format-decision-guide.md`**: Authoritative reference for data format decisions
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`project-structure-plan.md`**: Project structure and organization