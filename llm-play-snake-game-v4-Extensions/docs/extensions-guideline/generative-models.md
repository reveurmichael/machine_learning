# Generative Models for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and follows established architectural patterns.

## 🎯 **Core Philosophy: AI-Driven Content Generation**

Generative models in the Snake Game AI context focus on creating new game content, strategies, and experiences through AI generation. These models can generate game levels, create training scenarios, synthesize gameplay data, and even generate novel game variants.

### **SUPREME_RULES Alignment**
- **SUPREME_RULE NO.1**: Follows all established GOOD_RULES patterns
- **SUPREME_RULE NO.2**: References `final-decision-N.md` format consistently  
- **SUPREME_RULE NO.3**: Uses lightweight, OOP-based common utilities with simple logging (print() statements)

### **Design Philosophy**
- **Content Creation**: Automated generation of game scenarios and levels
- **Data Synthesis**: Creating training data for other AI models
- **Strategy Generation**: Discovering new gameplay patterns and strategies
- **Game Variant Creation**: Generating new rule sets and game mechanics

## 🧠 **Generative Model Categories**

### **Level Generation Models**
- **Procedural Content Generation**: Create diverse game boards and obstacles
- **Difficulty Progression**: Generate levels with appropriate challenge curves
- **Scenario Creation**: Design specific training scenarios for other AI models
- **Multi-Objective Level Design**: Balance multiple design criteria

### **Gameplay Data Synthesis**
- **Synthetic Trajectories**: Generate realistic gameplay sequences
- **Edge Case Generation**: Create rare but important game situations
- **Data Augmentation**: Expand existing datasets with variations
- **Multi-Agent Scenarios**: Generate complex multi-player situations

### **Strategy Generation Models**
- **Novel Strategy Discovery**: Find unconventional but effective approaches
- **Counter-Strategy Generation**: Create strategies to counter existing ones
- **Adaptive Strategy Creation**: Generate strategies for different game variants
- **Human-Like Strategy Synthesis**: Create strategies that mimic human play patterns

### **Game Variant Generation**
- **Rule Set Generation**: Create new game mechanics and constraints
- **Objective Generation**: Design alternative win conditions
- **Mechanics Innovation**: Discover new game mechanics through AI
- **Balanced Variant Creation**: Ensure generated variants remain playable

## 🏗️ **Extension Structure**

### **Directory Layout**
```
extensions/generative-models-v0.02/
├── __init__.py
├── generators/
│   ├── __init__.py               # Generator factory
│   ├── level_generator.py        # Level/board generation
│   ├── trajectory_generator.py   # Gameplay sequence generation
│   ├── strategy_generator.py     # Strategy pattern generation
│   └── variant_generator.py      # Game variant generation
├── models/
│   ├── __init__.py
│   ├── vae_models.py            # Variational Autoencoders
│   ├── gan_models.py            # Generative Adversarial Networks
│   ├── diffusion_models.py     # Diffusion models for content
│   ├── transformer_models.py   # Transformer-based generators
│   └── flow_models.py           # Normalizing flows
├── training/
│   ├── __init__.py
│   ├── vae_trainer.py           # VAE training pipeline
│   ├── gan_trainer.py           # GAN training pipeline
│   ├── diffusion_trainer.py    # Diffusion model training
│   └── evaluation_trainer.py   # Generator evaluation
├── data/
│   ├── __init__.py
│   ├── dataset_collector.py     # Collect training data
│   ├── data_preprocessor.py     # Process game data for training
│   ├── synthetic_evaluator.py   # Evaluate synthetic data quality
│   └── augmentation_utils.py    # Data augmentation utilities
├── evaluation/
│   ├── __init__.py
│   ├── diversity_metrics.py     # Measure generation diversity
│   ├── quality_metrics.py       # Assess generation quality
│   ├── playability_tester.py    # Test generated content playability
│   └── human_evaluation.py     # Human assessment protocols
├── game_logic.py                # Generative model game logic
├── game_manager.py              # Multi-generator manager
└── main.py                      # CLI interface
```

## 🔧 **Implementation Examples**

### **Level Generator using VAE**
```python
class SnakeLevelVAE(nn.Module):
    """
    Variational Autoencoder for Snake Game Level Generation
    
    Design Pattern: Encoder-Decoder Architecture
    - Learns latent representations of game levels
    - Generates new levels by sampling from latent space
    - Enables controlled generation through latent manipulation
    
    Educational Value:
    Demonstrates how VAEs can learn structured representations
    of game content and generate novel but valid variations.
    """
    
    def __init__(self, grid_size: int = 10, latent_dim: int = 64):
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.input_dim = grid_size * grid_size
        print(f"[SnakeLevelVAE] Initialized for {grid_size}x{grid_size} grid")  # SUPREME_RULE NO.3
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # mean and log_var
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x.view(-1, self.input_dim))
        mean, log_var = h.chunk(2, dim=-1)
        return mean, log_var
    
    def reparameterize(self, mean, log_var):
        """Reparameterization trick for gradient flow"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        """Decode latent vector to game board"""
        return self.decoder(z).view(-1, self.grid_size, self.grid_size)
    
    def forward(self, x):
        """Forward pass through VAE"""
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var
    
    def generate_level(self, num_samples: int = 1, temperature: float = 1.0):
        """Generate new game levels"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim) * temperature
            generated_levels = self.decode(z)
            return self._post_process_levels(generated_levels)
    
    def _post_process_levels(self, raw_levels):
        """Convert continuous outputs to discrete game elements"""
        processed_levels = []
        for level in raw_levels:
            # Convert probabilities to discrete elements
            discrete_level = torch.zeros_like(level)
            
            # Apply thresholds for different game elements
            discrete_level[level > 0.8] = 1  # Walls
            discrete_level[(level > 0.4) & (level <= 0.8)] = 0.5  # Obstacles
            # 0 remains empty space
            
            processed_levels.append(discrete_level)
        
        return processed_levels
```

### **Trajectory Generator using Transformer**
```python
class GameplayTransformer(nn.Module):
    """
    Transformer model for generating realistic gameplay sequences
    
    Design Pattern: Sequence-to-Sequence Generation
    - Learns patterns from existing gameplay trajectories
    - Generates new gameplay sequences with similar characteristics
    - Enables conditional generation based on game state
    
    Educational Value:
    Shows how transformer architectures can model complex
    sequential patterns in game behavior and generate coherent trajectories.
    """
    
    def __init__(self, vocab_size: int, hidden_dim: int = 256, num_layers: int = 6):
        super().__init__()
        self.vocab_size = vocab_size  # Actions + game state tokens
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, mask=None):
        """Generate next tokens in sequence"""
        embedded = self.embedding(x) * math.sqrt(self.hidden_dim)
        embedded = self.pos_encoding(embedded)
        
        output = self.transformer(embedded, mask)
        return self.output_head(output)
    
    def generate_trajectory(self, 
                          start_state: torch.Tensor, 
                          max_length: int = 100,
                          temperature: float = 1.0,
                          top_k: int = 50):
        """Generate a complete gameplay trajectory"""
        self.eval()
        with torch.no_grad():
            sequence = start_state.clone()
            
            for _ in range(max_length):
                # Get predictions for next token
                logits = self.forward(sequence[-50:])  # Use last 50 tokens
                next_token_logits = logits[-1] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                sequence = torch.cat([sequence, next_token])
                
                # Check for end of game token
                if next_token.item() == self.end_token_id:
                    break
            
            return sequence
```

### **Strategy Generator using GAN**
```python
class StrategyGAN:
    """
    Generative Adversarial Network for novel strategy generation
    
    Design Pattern: Adversarial Training
    - Generator creates new strategies from noise
    - Discriminator distinguishes real from generated strategies
    - Adversarial training leads to realistic strategy generation
    
    Educational Value:
    Demonstrates how GANs can be applied to abstract concepts
    like game strategies, not just visual content.
    """
    
    def __init__(self, strategy_dim: int = 128, noise_dim: int = 100):
        self.generator = StrategyGenerator(noise_dim, strategy_dim)
        self.discriminator = StrategyDiscriminator(strategy_dim)
        self.strategy_dim = strategy_dim
        self.noise_dim = noise_dim
    
    def train(self, strategy_dataset, epochs: int = 100):
        """Train GAN on strategy representations"""
        g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        
        for epoch in range(epochs):
            for batch in strategy_dataset:
                # Train discriminator
                real_strategies = batch
                fake_strategies = self.generate_strategies(len(real_strategies))
                
                d_loss = self._discriminator_loss(real_strategies, fake_strategies)
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                
                # Train generator
                fake_strategies = self.generate_strategies(len(real_strategies))
                g_loss = self._generator_loss(fake_strategies)
                
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
    
    def generate_strategies(self, num_strategies: int = 1):
        """Generate new strategies"""
        with torch.no_grad():
            noise = torch.randn(num_strategies, self.noise_dim)
            generated_strategies = self.generator(noise)
            return generated_strategies
    
    def evaluate_strategy_quality(self, generated_strategies):
        """Evaluate quality of generated strategies"""
        # Test strategies in actual games
        scores = []
        for strategy in generated_strategies:
            agent = StrategyAgent(strategy)
            score = self._test_strategy_performance(agent)
            scores.append(score)
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'success_rate': len([s for s in scores if s > 10]) / len(scores)
        }
```

## 🚀 **Advanced Generation Techniques**

### **Conditional Generation**
- **Difficulty-Conditioned Levels**: Generate levels with specified difficulty
- **Strategy-Conditioned Trajectories**: Generate gameplay following specific strategies
- **Scenario-Conditioned Content**: Create content for specific training scenarios
- **Player-Style Conditioning**: Generate content matching player preferences

### **Multi-Modal Generation**
- **Text-to-Game**: Generate game content from natural language descriptions
- **Image-to-Level**: Convert visual concepts to playable levels
- **Audio-Driven Generation**: Create content synchronized with audio cues
- **Cross-Modal Translation**: Convert between different content representations

### **Evolutionary Generation**
- **Genetic Algorithm Content**: Evolve game content through genetic operators
- **Quality-Diversity Generation**: Generate diverse high-quality content
- **Interactive Evolution**: Human-guided evolution of game content
- **Multi-Objective Generation**: Balance multiple design objectives

## 🎓 **Evaluation Methodologies**

### **Quality Metrics**
```python
class GenerationEvaluator:
    """
    Comprehensive evaluation of generated game content
    
    Metrics:
    - Playability: Can the content be successfully played?
    - Diversity: How varied is the generated content?
    - Novelty: How different from training data?
    - Balance: Are generated elements well-balanced?
    """
    
    def evaluate_generated_content(self, generated_content, reference_content):
        """Comprehensive evaluation of generated content"""
        return {
            'playability_score': self._evaluate_playability(generated_content),
            'diversity_score': self._evaluate_diversity(generated_content),
            'novelty_score': self._evaluate_novelty(generated_content, reference_content),
            'balance_score': self._evaluate_balance(generated_content),
            'human_preference': self._human_evaluation(generated_content)
        }
    
    def _evaluate_playability(self, content):
        """Test if generated content is playable"""
        playable_count = 0
        for item in content:
            try:
                game_result = self._simulate_game(item)
                if game_result['completed']:
                    playable_count += 1
            except Exception:
                continue
        
        return playable_count / len(content)
    
    def _evaluate_diversity(self, content):
        """Measure diversity of generated content"""
        # Use various diversity metrics
        pairwise_distances = []
        for i, item1 in enumerate(content):
            for j, item2 in enumerate(content[i+1:], i+1):
                distance = self._content_distance(item1, item2)
                pairwise_distances.append(distance)
        
        return np.mean(pairwise_distances)
```

### **Human Evaluation Protocols**
- **Playability Assessment**: Human players test generated content
- **Preference Studies**: Compare generated vs. human-created content
- **Creativity Evaluation**: Assess novelty and creativity of generated content
- **Engagement Metrics**: Measure player engagement with generated content

## 📊 **Training and Configuration**

### **Model Training Commands**
```bash
# Train VAE for level generation
python main.py --model vae --task level_generation \
  --data-path ../../logs/extensions/datasets/grid-size-10/heuristics_v0.04_*/*/game_logs/ \
  --epochs 100 --latent-dim 64

# Train GAN for strategy generation
python main.py --model gan --task strategy_generation \
  --data-path ../../logs/extensions/datasets/grid-size-10/supervised_v0.02_*/*/strategy_data/ \
  --epochs 200 --noise-dim 100

# Train Transformer for trajectory generation
python main.py --model transformer --task trajectory_generation \
  --data-path ../../logs/extensions/datasets/grid-size-10/*/*/processed_data/sequential_data.npz \
  --epochs 50 --hidden-dim 256

# Generate new content
python main.py --mode generate --model-path ./models/level_vae_v1.pth \
  --num-samples 100 --output-dir ./generated_content/
```

### **Configuration Examples**
```python
GENERATION_CONFIGS = {
    'level_vae': {
        'latent_dim': 64,
        'hidden_dims': [256, 128],
        'learning_rate': 1e-3,
        'batch_size': 32
    },
    'strategy_gan': {
        'noise_dim': 100,
        'hidden_dim': 256,
        'learning_rate': 2e-4,
        'batch_size': 64
    },
    'trajectory_transformer': {
        'hidden_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'learning_rate': 1e-4
    }
}
```

## 🔗 **Integration with Extension Ecosystem**

### **Data Pipeline Integration**
- **Heuristics Extensions**: Use generated levels for algorithm testing
- **Supervised Learning**: Train on generated trajectories for data augmentation
- **Reinforcement Learning**: Use generated scenarios for diverse training environments
- **Human Players**: Provide generated content for human gameplay

### **Continuous Generation Pipeline**
- **Performance Feedback**: Use model performance to guide generation
- **Quality Filtering**: Automatically filter generated content for quality
- **Diversity Maintenance**: Ensure generated content remains diverse over time
- **Incremental Learning**: Update generators based on new data

## 🔮 **Future Research Directions**

### **Advanced Architectures**
- **Diffusion Models**: High-quality content generation through iterative refinement
- **Neural Cellular Automata**: Generate evolving game environments
- **Graph Neural Networks**: Generate graph-structured game content
- **Attention Mechanisms**: Fine-grained control over generation process

### **Interactive Generation**
- **Real-Time Generation**: Generate content during gameplay
- **Player-Adaptive Generation**: Adapt content to individual player preferences
- **Collaborative Generation**: Combine human creativity with AI generation
- **Iterative Refinement**: Allow human feedback to improve generated content

## 🔗 **GOOD_RULES Integration**

This document integrates with the following authoritative references from the **GOOD_RULES** system:

### **Core Architecture Integration**
- **`agents.md`**: Follows BaseAgent interface and factory patterns for generative model agents
- **`config.md`**: Uses authorized configuration hierarchies for generative model parameters
- **`core.md`**: Inherits from base classes and follows established inheritance patterns

### **Extension Development Standards**
- **`extensions-v0.02.md`** through **`extensions-v0.04.md`**: Follows version progression guidelines
- **`standalone.md`**: Maintains standalone principle (extension + common = self-contained)
- **`single-source-of-truth.md`**: Avoids duplication, uses centralized utilities

### **Data and Path Management**
- **`data-format-decision-guide.md`**: Follows format selection criteria for generated content and training data
- **`unified-path-management-guide.md`**: Uses centralized path utilities from extensions/common/
- **`datasets-folder.md`**: Follows standard directory structure for generated datasets
- **`models.md`**: Follows model management standards for trained generative models

### **UI and Interaction Standards**
- **`app.md`** and **`dashboard.md`**: Integrates with Streamlit architecture for generation monitoring
- **`unified-streamlit-architecture-guide.md`**: Follows OOP Streamlit patterns for interactive interfaces

### **Implementation Quality**
- **`documentation-as-first-class-citizen.md`**: Maintains rich docstrings and design pattern documentation
- **`elegance.md`**: Follows code quality and educational value standards
- **`naming_conventions.md`**: Uses consistent naming across all generative components

## 📝 **Simple Logging Examples (SUPREME_RULE NO.3)**

All code examples in this document follow **SUPREME_RULE NO.3** by using simple print() statements rather than complex logging mechanisms:

```python
# ✅ CORRECT: Simple logging as per SUPREME_RULE NO.3
def train_generative_model(self, model_type, epochs):
    print(f"[GenerativeTrainer] Starting {model_type} training for {epochs} epochs")
    
    for epoch in range(epochs):
        epoch_loss = 0
        print(f"[GenerativeTrainer] Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(self.dataloader):
            loss = self._train_batch(batch)
            epoch_loss += loss
            
            if batch_idx % 100 == 0:
                print(f"[GenerativeTrainer] Batch {batch_idx}, Loss: {loss:.4f}")
        
        avg_loss = epoch_loss / len(self.dataloader)
        print(f"[GenerativeTrainer] Epoch {epoch+1} average loss: {avg_loss:.4f}")

# ✅ CORRECT: Educational progress tracking
def generate_content(self, content_type, num_samples):
    print(f"[ContentGenerator] Generating {num_samples} {content_type} samples")
    
    generated_content = []
    for i in range(num_samples):
        sample = self._generate_single_sample(content_type)
        generated_content.append(sample)
        print(f"[ContentGenerator] Generated sample {i+1}/{num_samples}")
    
    quality_score = self._evaluate_quality(generated_content)
    print(f"[ContentGenerator] Generation complete. Quality score: {quality_score:.3f}")
    
    return generated_content
```

---

**Generative models for Snake Game AI represent a frontier in AI-driven content creation, enabling the automatic generation of diverse, high-quality game content that enhances training, testing, and player experience across the entire ecosystem while maintaining full compliance with established GOOD_RULES standards.**
