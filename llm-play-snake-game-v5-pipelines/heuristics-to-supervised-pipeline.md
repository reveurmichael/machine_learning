# TODO: please keep this file VERY simple and concise.

# Heuristics to Supervised Learning Pipeline

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines the pipeline for converting heuristic data to supervised learning datasets.

> **See also:** `heuristics-as-foundation.md`, `supervised.md`, `final-decision-10.md`, `data-format-decision-guide.md`.

## 🎯 **Core Philosophy: Data Pipeline Architecture**

The heuristics-to-supervised learning pipeline demonstrates how to convert algorithmic decision-making data into training datasets for machine learning models. This pipeline showcases the educational value of understanding data transformation and model training workflows.

### **Educational Value**
- **Data Pipeline Design**: Understanding end-to-end data processing workflows
- **Algorithm Comparison**: Comparing heuristic and ML approaches
- **Feature Engineering**: Learning to extract meaningful features from game states
- **Model Training**: Understanding supervised learning training processes

## 🏗️ **Pipeline Architecture**

### **Factory Pattern: Canonical Method is create()**
All pipeline components must use the canonical method name `create()` for instantiation:

```python
class PipelineFactory:
    _registry = {
        "DATA_GENERATOR": DataGenerator,
        "FEATURE_EXTRACTOR": FeatureExtractor,
        "MODEL_TRAINER": ModelTrainer,
    }
    
    @classmethod
    def create(cls, component_type: str, **kwargs):
        component_class = cls._registry.get(component_type.upper())
        if not component_class:
            raise ValueError(f"Unknown component type: {component_type}")
        print(f"[PipelineFactory] Creating component: {component_type}")  # Simple logging
        return component_class(**kwargs)
```

### **Pipeline Components**
```python
class HeuristicsToSupervisedPipeline:
    """
    Pipeline for converting heuristic data to supervised learning datasets.
    
    Design Pattern: Pipeline Pattern
    - Sequential data processing stages
    - Clear input/output interfaces
    - Configurable processing parameters
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.data_generator = None
        self.feature_extractor = None
        self.model_trainer = None
        print(f"[Pipeline] Initialized heuristics-to-supervised pipeline")  # Simple logging
    
    def setup_pipeline(self):
        """Setup pipeline components"""
        # Create data generator
        self.data_generator = PipelineFactory.create("DATA_GENERATOR", 
                                                    algorithms=self.config.get('algorithms', ['BFS', 'ASTAR']),
                                                    num_games=self.config.get('num_games', 1000))
        
        # Create feature extractor
        self.feature_extractor = PipelineFactory.create("FEATURE_EXTRACTOR",
                                                       grid_size=self.config.get('grid_size', 10))
        
        # Create model trainer
        self.model_trainer = PipelineFactory.create("MODEL_TRAINER",
                                                   model_types=self.config.get('model_types', ['MLP', 'XGBOOST']))
        
        print(f"[Pipeline] Pipeline components setup completed")  # Simple logging
    
    def run_pipeline(self) -> dict:
        """Run the complete pipeline"""
        print(f"[Pipeline] Starting pipeline execution")  # Simple logging
        
        # Stage 1: Generate heuristic data
        heuristic_data = self.data_generator.generate_data()
        print(f"[Pipeline] Generated {len(heuristic_data)} heuristic games")  # Simple logging
        
        # Stage 2: Extract features
        training_data = self.feature_extractor.extract_features(heuristic_data)
        print(f"[Pipeline] Extracted features for {len(training_data)} samples")  # Simple logging
        
        # Stage 3: Train models
        model_results = self.model_trainer.train_models(training_data)
        print(f"[Pipeline] Trained {len(model_results)} models")  # Simple logging
        
        return {
            'heuristic_data': heuristic_data,
            'training_data': training_data,
            'model_results': model_results
        }
```

## 🚀 **Pipeline Implementation**

### **Data Generator Component**
```python
class DataGenerator:
    """
    Generates training data from heuristic algorithms.
    
    Design Pattern: Strategy Pattern
    - Different algorithms for data generation
    - Configurable generation parameters
    - Comprehensive data collection
    """
    
    def __init__(self, algorithms: list, num_games: int = 1000):
        self.algorithms = algorithms
        self.num_games = num_games
        self.game_manager = None
        print(f"[DataGenerator] Initialized for {algorithms}")  # Simple logging
    
    def generate_data(self) -> list:
        """Generate training data from heuristic algorithms"""
        all_data = []
        
        for algorithm in self.algorithms:
            print(f"[DataGenerator] Generating data for {algorithm}")  # Simple logging
            
            # Create game manager for algorithm
            self.game_manager = self._create_game_manager(algorithm)
            
            # Generate games
            algorithm_data = self._generate_algorithm_data(algorithm)
            all_data.extend(algorithm_data)
        
        print(f"[DataGenerator] Generated {len(all_data)} total games")  # Simple logging
        return all_data
    
    def _create_game_manager(self, algorithm: str):
        """Create game manager for specific algorithm"""
        from extensions.heuristics_v0_03.game_manager import HeuristicGameManager
        
        return HeuristicGameManager(
            algorithm=algorithm,
            grid_size=10,
            max_games=self.num_games // len(self.algorithms)
        )
    
    def _generate_algorithm_data(self, algorithm: str) -> list:
        """Generate data for specific algorithm"""
        algorithm_data = []
        
        for game_id in range(self.num_games // len(self.algorithms)):
            # Run game
            game_result = self.game_manager.run_single_game()
            
            # Collect game data
            game_data = {
                'algorithm': algorithm,
                'game_id': game_id,
                'moves': game_result['moves'],
                'states': game_result['states'],
                'scores': game_result['scores'],
                'final_score': game_result['final_score']
            }
            
            algorithm_data.append(game_data)
        
        return algorithm_data
```

### **Feature Extractor Component**
```python
class FeatureExtractor:
    """
    Extracts features from heuristic game data for supervised learning.
    
    Design Pattern: Strategy Pattern
    - Different feature extraction strategies
    - Configurable feature sets
    - Normalized feature representation
    """
    
    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        print(f"[FeatureExtractor] Initialized for {grid_size}x{grid_size} grid")  # Simple logging
    
    def extract_features(self, heuristic_data: list) -> dict:
        """Extract features from heuristic data"""
        X = []  # Features
        y = []  # Labels (moves)
        
        for game_data in heuristic_data:
            game_features, game_labels = self._extract_game_features(game_data)
            X.extend(game_features)
            y.extend(game_labels)
        
        print(f"[FeatureExtractor] Extracted {len(X)} feature samples")  # Simple logging
        
        return {
            'X': X,
            'y': y,
            'feature_names': self._get_feature_names(),
            'label_names': ['UP', 'DOWN', 'LEFT', 'RIGHT']
        }
    
    def _extract_game_features(self, game_data: dict) -> tuple:
        """Extract features from a single game"""
        features = []
        labels = []
        
        for i, (state, move) in enumerate(zip(game_data['states'], game_data['moves'])):
            # Extract features from game state
            feature_vector = self._state_to_features(state)
            features.append(feature_vector)
            labels.append(move)
        
        return features, labels
    
    def _state_to_features(self, state: dict) -> list:
        """Convert game state to feature vector"""
        snake_positions = state['snake_positions']
        apple_position = state['apple_position']
        head_position = snake_positions[0]
        
        # Basic features
        features = [
            # Head position (normalized)
            head_position[0] / (self.grid_size - 1),
            head_position[1] / (self.grid_size - 1),
            
            # Apple position (normalized)
            apple_position[0] / (self.grid_size - 1),
            apple_position[1] / (self.grid_size - 1),
            
            # Distance to apple
            self._manhattan_distance(head_position, apple_position) / (2 * self.grid_size),
            
            # Snake length (normalized)
            len(snake_positions) / (self.grid_size * self.grid_size),
            
            # Direction to apple
            self._direction_to_apple(head_position, apple_position)
        ]
        
        return features
    
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _direction_to_apple(self, head: tuple, apple: tuple) -> int:
        """Get primary direction to apple (0=horizontal, 1=vertical)"""
        dx = abs(apple[0] - head[0])
        dy = abs(apple[1] - head[1])
        return 0 if dx > dy else 1
    
    def _get_feature_names(self) -> list:
        """Get feature names for interpretability"""
        return [
            'head_x', 'head_y',
            'apple_x', 'apple_y',
            'distance_to_apple',
            'snake_length',
            'direction_to_apple'
        ]
```

### **Model Trainer Component**
```python
class ModelTrainer:
    """
    Trains supervised learning models on extracted features.
    
    Design Pattern: Strategy Pattern
    - Different model types and training strategies
    - Configurable training parameters
    - Comprehensive model evaluation
    """
    
    def __init__(self, model_types: list):
        self.model_types = model_types
        print(f"[ModelTrainer] Initialized for {model_types}")  # Simple logging
    
    def train_models(self, training_data: dict) -> dict:
        """Train multiple models on the training data"""
        X = training_data['X']
        y = training_data['y']
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_results = {}
        
        for model_type in self.model_types:
            print(f"[ModelTrainer] Training {model_type} model")  # Simple logging
            
            # Create and train model using canonical factory pattern
            model = ModelFactory.create(model_type)  # CANONICAL create() method
            model.fit(X_train, y_train)
            
            # Evaluate model
            accuracy = model.score(X_test, y_test)
            
            model_results[model_type] = {
                'model': model,
                'accuracy': accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            print(f"[ModelTrainer] {model_type} accuracy: {accuracy:.3f}")  # Simple logging
        
        return model_results
```

## 📊 **Pipeline Usage Example**

### **Complete Pipeline Execution**
```python
def run_heuristics_to_supervised_pipeline():
    """Run complete heuristics to supervised learning pipeline"""
    
    # Configuration
    config = {
        'algorithms': ['BFS', 'ASTAR', 'DFS'],
        'num_games': 1000,
        'grid_size': 10,
        'model_types': ['MLP', 'XGBOOST', 'RANDOMFOREST']
    }
    
    # Create and run pipeline
    pipeline = HeuristicsToSupervisedPipeline(config)
    pipeline.setup_pipeline()
    results = pipeline.run_pipeline()
    
    # Print results
    print(f"\nPipeline Results:")
    print(f"  Heuristic Games: {len(results['heuristic_data'])}")
    print(f"  Training Samples: {len(results['training_data']['X'])}")
    print(f"  Models Trained: {len(results['model_results'])}")
    
    for model_type, result in results['model_results'].items():
        print(f"  {model_type} Accuracy: {result['accuracy']:.3f}")
    
    return results
```

## 📋 **Implementation Checklist**

### **Pipeline Components**
- [ ] **Data Generator**: Generates heuristic game data
- [ ] **Feature Extractor**: Extracts features from game states
- [ ] **Model Trainer**: Trains supervised learning models
- [ ] **Factory Pattern**: Uses canonical `create()` method
- [ ] **Configuration**: Configurable pipeline parameters

### **Data Quality**
- [ ] **Data Validation**: Validates generated data
- [ ] **Feature Engineering**: Meaningful feature extraction
- [ ] **Data Splitting**: Proper train/test splits
- [ ] **Model Evaluation**: Comprehensive model evaluation

### **Educational Value**
- [ ] **Pipeline Understanding**: Clear pipeline workflow
- [ ] **Algorithm Comparison**: Comparing heuristic and ML approaches
- [ ] **Feature Engineering**: Learning feature extraction
- [ ] **Model Training**: Understanding supervised learning

## 🎓 **Educational Benefits**

### **Learning Objectives**
- **Data Pipeline Design**: Understanding end-to-end data processing
- **Feature Engineering**: Learning to extract meaningful features
- **Model Training**: Understanding supervised learning workflows
- **Algorithm Comparison**: Comparing different approaches

### **Best Practices**
- **Modular Design**: Clear separation of pipeline components
- **Configuration Management**: Configurable pipeline parameters
- **Data Validation**: Proper data validation and quality checks
- **Model Evaluation**: Comprehensive model evaluation and comparison

---

**The heuristics-to-supervised learning pipeline demonstrates the educational value of understanding data transformation and model training workflows, providing a clear example of how to convert algorithmic data into machine learning training datasets.**

## 🔗 **See Also**

- **`heuristics-as-foundation.md`**: Heuristics as foundation for ML
- **`supervised.md`**: Supervised learning implementation
- **`final-decision-10.md`**: final-decision-10.md governance system
- **`data-format-decision-guide.md`**: Data format standards
