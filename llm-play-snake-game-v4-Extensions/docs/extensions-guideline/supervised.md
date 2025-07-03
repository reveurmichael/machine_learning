# Supervised Learning Standards for Snake Game AI Extensions

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines supervised learning standards.

> **See also:** `agents.md`, `core.md`, `config.md`, `final-decision-10.md`, `factory-design-pattern.md`.

## 🎯 **Core Philosophy: Data-Driven Decision Making**

Supervised learning in the Snake Game AI project enables **data-driven decision making** through machine learning models trained on labeled game data. These models learn patterns from successful gameplay and apply them to new situations, strictly following `final-decision-10.md` SUPREME_RULES.

### **Educational Value**
- **Machine Learning**: Understanding supervised learning principles
- **Data Processing**: Learning how to prepare and process training data
- **Model Training**: Experience with different ML algorithms and frameworks
- **Performance Evaluation**: Understanding model evaluation and comparison

## 🏗️ **Factory Pattern: Canonical Method is create()**

All supervised learning factories must use the canonical method name `create()` for instantiation, not `create_supervised_agent()` or any other variant. This ensures consistency and aligns with the KISS principle.

### **Supervised Learning Factory Implementation**
```python
class SupervisedLearningFactory:
    """
    Factory for supervised learning agents following SUPREME_RULES.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create supervised learning agents with canonical patterns
    Educational Value: Shows how canonical factory patterns work with ML systems
    """
    
    _registry = {
        "MLP": MLPAgent,
        "CNN": CNNAgent,
        "LSTM": LSTMAgent,
        "XGBOOST": XGBoostAgent,
        "LIGHTGBM": LightGBMAgent,
        "RANDOMFOREST": RandomForestAgent,
    }
    
    @classmethod
    def create(cls, algorithm_type: str, **kwargs):  # CANONICAL create() method
        """Create supervised learning agent using canonical create() method (SUPREME_RULES compliance)"""
        agent_class = cls._registry.get(algorithm_type.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown algorithm type: {algorithm_type}. Available: {available}")
        print_info(f"[SupervisedLearningFactory] Creating agent: {algorithm_type}")  # Simple logging
        return agent_class(**kwargs)

# ❌ FORBIDDEN: Non-canonical method names (violates SUPREME_RULES)
class SupervisedLearningFactory:
    def create_supervised_agent(self, algorithm_type: str):  # FORBIDDEN - not canonical
        pass
    
    def build_ml_model(self, algorithm_type: str):  # FORBIDDEN - not canonical
        pass
    
    def make_supervised_algorithm(self, algorithm_type: str):  # FORBIDDEN - not canonical
        pass
```

## 🧠 **Supervised Learning Architecture Patterns**

### **Multi-Layer Perceptron (MLP) Agent**
```python
class MLPAgent(BaseAgent):
    """
    Multi-Layer Perceptron agent for supervised learning.
    
    Design Pattern: Strategy Pattern
    Purpose: Uses neural networks for pattern recognition
    Educational Value: Shows how to implement basic neural networks
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("MLP", config)
        self.model = None
        self.scaler = None
        self.load_model()
        print_info(f"[MLPAgent] Initialized MLP agent")  # Simple logging
    
    def load_model(self):
        """Load pre-trained MLP model"""
        model_path = self.config.get('model_path')
        if model_path and os.path.exists(model_path):
            self.model = torch.load(model_path)
            self.model.eval()
            print_info(f"[MLPAgent] Loaded model from {model_path}")  # Simple logging
        else:
            print_warning(f"[MLPAgent] No model found at {model_path}")  # Simple logging
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan move using MLP prediction"""
        # Convert game state to feature vector
        features = self._extract_features(game_state)
        
        # Normalize features
        if self.scaler:
            features = self.scaler.transform(features.reshape(1, -1))
        
        # Get model prediction
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features)
            prediction = self.model(features_tensor)
            move_probs = torch.softmax(prediction, dim=1)
            move_idx = torch.argmax(move_probs).item()
        
        # Convert to direction
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        move = directions[move_idx]
        
        print_info(f"[MLPAgent] Predicted move: {move}")  # Simple logging
        return move
    
    def _extract_features(self, game_state: Dict[str, Any]) -> np.ndarray:
        """Extract features from game state"""
        # Feature extraction implementation
        features = []
        head_pos = game_state['snake_positions'][0]
        apple_pos = game_state['apple_position']
        
        # Position features
        features.extend([head_pos[0], head_pos[1], apple_pos[0], apple_pos[1]])
        
        # Game state features
        features.append(len(game_state['snake_positions']))
        
        # Direction features
        features.extend(self._get_direction_features(head_pos, apple_pos))
        
        # Danger features
        features.extend(self._get_danger_features(game_state))
        
        # Free space features
        features.extend(self._get_free_space_features(game_state))
        
        return np.array(features)
    
    def _get_direction_features(self, head_pos: tuple, apple_pos: tuple) -> List[float]:
        """Get direction features"""
        dx = apple_pos[0] - head_pos[0]
        dy = apple_pos[1] - head_pos[1]
        
        return [
            1.0 if dy < 0 else 0.0,  # apple_dir_up
            1.0 if dy > 0 else 0.0,  # apple_dir_down
            1.0 if dx < 0 else 0.0,  # apple_dir_left
            1.0 if dx > 0 else 0.0,  # apple_dir_right
        ]
    
    def _get_danger_features(self, game_state: Dict[str, Any]) -> List[float]:
        """Get danger features"""
        head_pos = game_state['snake_positions'][0]
        grid_size = game_state['grid_size']
        
        # Check danger in each direction
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        danger_features = []
        
        for dx, dy in directions:
            new_pos = (head_pos[0] + dx, head_pos[1] + dy)
            is_danger = (
                new_pos in game_state['snake_positions'] or
                not (0 <= new_pos[0] < grid_size and 0 <= new_pos[1] < grid_size)
            )
            danger_features.append(1.0 if is_danger else 0.0)
        
        return danger_features[:3]  # Only straight, left, right
    
    def _get_free_space_features(self, game_state: Dict[str, Any]) -> List[float]:
        """Get free space features"""
        head_pos = game_state['snake_positions'][0]
        grid_size = game_state['grid_size']
        snake_positions = set(game_state['snake_positions'])
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        free_space_features = []
        
        for dx, dy in directions:
            free_count = 0
            current_pos = head_pos
            
            for _ in range(grid_size):
                current_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if (0 <= current_pos[0] < grid_size and 
                    0 <= current_pos[1] < grid_size and 
                    current_pos not in snake_positions):
                    free_count += 1
                else:
                    break
            
            free_space_features.append(free_count)
        
        return free_space_features
```

### **Convolutional Neural Network (CNN) Agent**
```python
class CNNAgent(BaseAgent):
    """
    Convolutional Neural Network agent for spatial pattern recognition.
    
    Design Pattern: Strategy Pattern
    Purpose: Uses CNNs for spatial pattern recognition
    Educational Value: Shows how to implement CNNs for game state analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("CNN", config)
        self.model = None
        self.load_model()
        print_info(f"[CNNAgent] Initialized CNN agent")  # Simple logging
    
    def load_model(self):
        """Load pre-trained CNN model"""
        model_path = self.config.get('model_path')
        if model_path and os.path.exists(model_path):
            self.model = torch.load(model_path)
            self.model.eval()
            print_info(f"[CNNAgent] Loaded model from {model_path}")  # Simple logging
        else:
            print_warning(f"[CNNAgent] No model found at {model_path}")  # Simple logging
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan move using CNN prediction"""
        # Convert game state to image representation
        game_image = self._state_to_image(game_state)
        
        # Get model prediction
        with torch.no_grad():
            image_tensor = torch.FloatTensor(game_image).unsqueeze(0)
            prediction = self.model(image_tensor)
            move_probs = torch.softmax(prediction, dim=1)
            move_idx = torch.argmax(move_probs).item()
        
        # Convert to direction
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        move = directions[move_idx]
        
        print_info(f"[CNNAgent] Predicted move: {move}")  # Simple logging
        return move
    
    def _state_to_image(self, game_state: Dict[str, Any]) -> np.ndarray:
        """Convert game state to image representation"""
        grid_size = game_state['grid_size']
        snake_positions = game_state['snake_positions']
        apple_position = game_state['apple_position']
        
        # Create 3-channel image: snake_body, snake_head, apple
        image = np.zeros((3, grid_size, grid_size))
        
        # Mark snake body
        for i, pos in enumerate(snake_positions):
            if i == 0:  # Head
                image[1, pos[1], pos[0]] = 1.0
            else:  # Body
                image[0, pos[1], pos[0]] = 1.0
        
        # Mark apple
        image[2, apple_position[1], apple_position[0]] = 1.0
        
        return image
```

### **Long Short-Term Memory (LSTM) Agent**
```python
class LSTMAgent(BaseAgent):
    """
    LSTM agent for sequential pattern recognition.
    
    Design Pattern: Strategy Pattern
    Purpose: Uses LSTMs for temporal pattern recognition
    Educational Value: Shows how to implement RNNs for game sequence analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("LSTM", config)
        self.model = None
        self.state_history = []
        self.sequence_length = self.config.get('sequence_length', 10)
        self.load_model()
        print_info(f"[LSTMAgent] Initialized LSTM agent")  # Simple logging
    
    def load_model(self):
        """Load pre-trained LSTM model"""
        model_path = self.config.get('model_path')
        if model_path and os.path.exists(model_path):
            self.model = torch.load(model_path)
            self.model.eval()
            print_info(f"[LSTMAgent] Loaded model from {model_path}")  # Simple logging
        else:
            print_warning(f"[LSTMAgent] No model found at {model_path}")  # Simple logging
    
    def plan_move(self, game_state: Dict[str, Any]) -> str:
        """Plan move using LSTM prediction"""
        # Update state history
        features = self._extract_features(game_state)
        self.state_history.append(features)
        
        # Keep only recent history
        if len(self.state_history) > self.sequence_length:
            self.state_history = self.state_history[-self.sequence_length:]
        
        # Pad sequence if needed
        if len(self.state_history) < self.sequence_length:
            padding = [np.zeros_like(features)] * (self.sequence_length - len(self.state_history))
            sequence = padding + self.state_history
        else:
            sequence = self.state_history
        
        # Get model prediction
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            prediction = self.model(sequence_tensor)
            move_probs = torch.softmax(prediction, dim=1)
            move_idx = torch.argmax(move_probs).item()
        
        # Convert to direction
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        move = directions[move_idx]
        
        print_info(f"[LSTMAgent] Predicted move: {move}")  # Simple logging
        return move
    
    def _extract_features(self, game_state: Dict[str, Any]) -> np.ndarray:
        """Extract features from game state"""
        # Similar to MLP agent but simplified for LSTM
        head_pos = game_state['snake_positions'][0]
        apple_pos = game_state['apple_position']
        
        features = [
            head_pos[0], head_pos[1],
            apple_pos[0], apple_pos[1],
            len(game_state['snake_positions']),
            game_state.get('score', 0)
        ]
        
        return np.array(features)
```

## 📊 **Training Pipeline Standards**

### **Supervised Training Pipeline**
```python
class SupervisedTrainingPipeline:
    """
    Training pipeline for supervised learning models.
    
    Design Pattern: Template Method Pattern
    Purpose: Provides consistent training workflow
    Educational Value: Shows how to train different supervised learning models
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        print_info(f"[SupervisedTrainingPipeline] Initialized training pipeline")  # Simple logging
    
    def prepare_data(self, dataset_path: str):
        """Prepare training data"""
        # Load dataset
        dataset = pd.read_csv(dataset_path)
        
        # Split features and target
        X = dataset.drop(['game_id', 'step_in_game', 'target_move'], axis=1)
        y = dataset['target_move']
        
        # Encode target
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Create data loaders
        self.train_loader = self._create_data_loader(X_train, y_train)
        self.val_loader = self._create_data_loader(X_val, y_val)
        self.test_loader = self._create_data_loader(X_test, y_test)
        
        print_success(f"[SupervisedTrainingPipeline] Data preparation complete")  # Simple logging
    
    def train_model(self, model_type: str):
        """Train the model"""
        # Create model
        self.model = self._build_model(model_type)
        
        # Training loop
        num_epochs = self.config.get('epochs', 100)
        for epoch in range(num_epochs):
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()
            
            if epoch % 10 == 0:
                print_info(f"[SupervisedTrainingPipeline] Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")  # Simple logging
    
    def _build_model(self, model_type: str):
        """Build model architecture"""
        if model_type == "MLP":
            return self._build_mlp_model()
        elif model_type == "CNN":
            return self._build_cnn_model()
        elif model_type == "LSTM":
            return self._build_lstm_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _build_mlp_model(self):
        """Build MLP model"""
        input_size = 16  # 16 features from CSV
        hidden_sizes = self.config.get('hidden_sizes', [64, 32])
        output_size = 4  # 4 directions
        
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
        
        model = nn.Sequential(*layers)
        print_info(f"[SupervisedTrainingPipeline] Built MLP model")  # Simple logging
        return model
    
    def _build_cnn_model(self):
        """Build CNN model"""
        # CNN architecture for 2D game state representation
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
        print_info(f"[SupervisedTrainingPipeline] Built CNN model")  # Simple logging
        return model
    
    def _build_lstm_model(self):
        """Build LSTM model"""
        input_size = 6  # Simplified features for LSTM
        hidden_size = self.config.get('hidden_size', 128)
        num_layers = self.config.get('num_layers', 2)
        
        model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Add final classification layer
        classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )
        
        print_info(f"[SupervisedTrainingPipeline] Built LSTM model")  # Simple logging
        return (model, classifier)
    
    def _create_data_loader(self, X, y):
        """Create data loader"""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.LongTensor(y)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Create data loader
        batch_size = self.config.get('batch_size', 32)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def _train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('learning_rate', 0.001))
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def _validate_epoch(self) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_model(self, model_path: str):
        """Save trained model"""
        torch.save(self.model.state_dict(), model_path)
        print_success(f"[SupervisedTrainingPipeline] Model saved to {model_path}")  # Simple logging
```

## 📋 **Implementation Checklist**

### **Required Components**
- [ ] **Factory Pattern**: Uses canonical `create()` method
- [ ] **Model Architecture**: Implements appropriate supervised learning model
- [ ] **Data Processing**: Proper feature extraction and preprocessing
- [ ] **Training Pipeline**: Standardized training workflow
- [ ] **Simple Logging**: Uses utils/print_utils.py functions for debugging

### **Quality Standards**
- [ ] **Model Performance**: Meets performance benchmarks
- [ ] **Data Quality**: Proper data validation and preprocessing
- [ ] **Training Efficiency**: Efficient training process
- [ ] **Documentation**: Clear documentation of model capabilities

### **Integration Requirements**
- [ ] **Data Compatibility**: Works with standard CSV datasets
- [ ] **Factory Integration**: Compatible with agent factory patterns
- [ ] **Configuration**: Supports standard configuration system
- [ ] **Evaluation**: Integrates with evaluation framework

---

**Supervised learning standards ensure consistent, high-quality ML implementations across all Snake Game AI extensions. By following these standards, developers can create robust, educational, and performant models that integrate seamlessly with the overall framework.**

## 🔗 **See Also**

- **`agents.md`**: Agent implementation standards
- **`core.md`**: Base class architecture and inheritance patterns
- **`config.md`**: Configuration management
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`factory-design-pattern.md`**: Factory pattern implementation