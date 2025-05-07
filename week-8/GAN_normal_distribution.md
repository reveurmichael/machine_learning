# Building a GAN to Generate Normal Distribution Data

In this comprehensive tutorial, we'll implement a Generative Adversarial Network (GAN) that learns to generate samples following a normal (Gaussian) distribution. This serves as an excellent introduction to GANs with a simple task before moving to more complex data like images.

## 1. Introduction to Generative Adversarial Networks

### 1.1 What are GANs?

Generative Adversarial Networks (GANs) were introduced by Ian Goodfellow and colleagues in 2014 and have revolutionized the field of generative modeling. GANs consist of two neural networks that are trained simultaneously in a competitive setting:

1. **Generator (G)**: Learns to create data that resembles the target distribution
2. **Discriminator (D)**: Learns to distinguish between real data and the generator's fake data

These networks are trained simultaneously in what is called a minimax game:
- The generator tries to fool the discriminator by producing increasingly realistic data
- The discriminator tries to correctly classify real vs. generated data

Mathematically, the GAN objective can be expressed as:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

Where:
- $p_{data}(x)$ is the real data distribution
- $p_z(z)$ is the noise distribution (usually Gaussian)
- $G(z)$ is the generator mapping from noise to data
- $D(x)$ is the discriminator's estimate of the probability that x is real

### 1.2 Why Start with Generating a Normal Distribution?

Before tackling complex data like images, learning to generate a simple normal distribution offers several advantages:

- Clear mathematical target (we know the exact distribution we want to learn)
- Easy to visualize and evaluate success quantitatively
- Simpler network architectures allow for faster experimentation
- Demonstrates core GAN concepts without additional complexity
- Provides intuition about the learning dynamics of adversarial training

Let's get started with implementing this simpler GAN before moving to more complex tasks.

## 2. Setting Up Our Environment

### 2.1 Import Required Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm  # For progress bars in Jupyter
import time
import math

# Set visual styling for plots
plt.style.use('seaborn-whitegrid')
sns.set(style="whitegrid", font_scale=1.2)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
```

### 2.2 Understanding the Target: Normal Distribution

The normal (or Gaussian) distribution is characterized by two parameters:
- **Mean (μ)**: The central tendency of the distribution
- **Standard Deviation (σ)**: The amount of dispersion in the data

Mathematically, the probability density function (PDF) of a normal distribution is:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$

Let's visualize the normal distribution we're trying to generate:

```python
def plot_normal_distribution(mean=0, std=1, samples=10000):
    """Plot the PDF and histogram of a normal distribution."""
    x = np.linspace(mean - 4*std, mean + 4*std, 1000)
    pdf = (1/(std * np.sqrt(2*np.pi))) * np.exp(-(x-mean)**2 / (2*std**2))
    
    # Generate random samples from this distribution
    samples = np.random.normal(mean, std, samples)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, pdf, 'r-', lw=2)
    plt.title(f'Normal Distribution PDF (μ={mean}, σ={std})')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    
    plt.subplot(1, 2, 2)
    plt.hist(samples, bins=50, density=True, alpha=0.7)
    plt.plot(x, pdf, 'r-', lw=2)
    plt.title(f'Normal Distribution Samples (n={len(samples)})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
# Visualize the target normal distribution
plot_normal_distribution(mean=0, std=1)
```

### 2.3 Define Parameters

```python
# Parameters
batch_size = 512     # Number of samples per batch
z_dim = 2            # Dimension of noise vector (latent space)
target_mean = 0      # Target mean of the normal distribution
target_std = 1       # Target standard deviation
lr = 0.0002          # Learning rate for Adam optimizer
beta1 = 0.5          # Beta1 hyperparameter for Adam optimizer
beta2 = 0.999        # Beta2 hyperparameter for Adam optimizer
num_epochs = 5000    # Total training epochs
display_step = 500   # How often to display results
hidden_size = [16, 32, 16]  # Sizes of hidden layers

# Advanced parameters (explained later)
label_smoothing = 0.1  # Label smoothing factor to help stabilize training

# Device configuration - will use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

## 3. Building the Generator Network

### 3.1 Understanding the Generator's Role

The generator in a GAN transforms random noise into samples that should follow our target distribution. For our normal distribution GAN:

- **Input**: Random noise vectors (typically from a uniform or normal distribution)
- **Output**: Scalar values that should follow a normal distribution with target mean and std
- **Goal**: Transform the input noise distribution into the target normal distribution

### 3.2 Implementation of the Generator

```python
class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dims=[16, 32, 16]):
        """
        Generator network that transforms noise to samples.
        
        Args:
            z_dim (int): Dimension of input noise vector
            hidden_dims (list): Dimensions of hidden layers
        """
        super(Generator, self).__init__()
        
        # Build layers dynamically based on hidden_dims
        layers = []
        input_dim = z_dim
        
        # Add hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.LeakyReLU(0.2))
            input_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        # Create sequential model
        self.gen = nn.Sequential(*layers)
        
    def forward(self, z):
        """
        Forward pass of the generator.
        
        Args:
            z (Tensor): Input noise, shape [batch_size, z_dim]
            
        Returns:
            Tensor: Generated samples, shape [batch_size, 1]
        """
        return self.gen(z)
    
    def generate_samples(self, num_samples=1000):
        """
        Generate samples from the generator.
        
        Args:
            num_samples (int): Number of samples to generate
            
        Returns:
            numpy array: Generated samples
        """
        # Generate random noise
        with torch.no_grad():
            z = torch.randn(num_samples, z_dim).to(device)
            samples = self.forward(z).cpu().numpy()
        return samples

# Initialize the generator
generator = Generator(z_dim, hidden_size).to(device)

# Print model architecture
print("Generator Architecture:")
print(generator)
print(f"Number of parameters: {sum(p.numel() for p in generator.parameters())}")

# Analyze a single forward pass with a small batch
test_noise = torch.randn(5, z_dim)
test_output = generator(test_noise)
print(f"\nTest noise shape: {test_noise.shape}")
print(f"Test output shape: {test_output.shape}")
print(f"Sample generator outputs: {test_output.flatten().tolist()}")
```

### 3.3 Visualizing Initial Generator Output

```python
def visualize_generator_output(generator, noise_dim=z_dim, n_samples=5000):
    """Visualize current generator output compared to target distribution."""
    # Generate samples
    with torch.no_grad():
        z = torch.randn(n_samples, noise_dim).to(device)
        generated = generator(z).cpu().numpy().flatten()
    
    # Generate true normal distribution samples
    true_samples = np.random.normal(target_mean, target_std, n_samples)
    
    # Calculate statistics
    gen_mean = generated.mean()
    gen_std = generated.std()
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram comparison
    bins = np.linspace(min(generated.min(), true_samples.min()) - 0.5, 
                       max(generated.max(), true_samples.max()) + 0.5, 50)
    
    ax[0].hist(true_samples, bins=bins, alpha=0.5, label=f'True (μ={target_mean}, σ={target_std})')
    ax[0].hist(generated, bins=bins, alpha=0.5, label=f'Generated (μ={gen_mean:.2f}, σ={gen_std:.2f})')
    ax[0].set_title('Distribution Comparison')
    ax[0].set_xlabel('Value')
    ax[0].set_ylabel('Frequency')
    ax[0].legend()
    
    # QQ plot
    ax[1].scatter(np.sort(true_samples), np.sort(generated), alpha=0.3)
    min_val = min(true_samples.min(), generated.min())
    max_val = max(true_samples.max(), generated.max())
    ax[1].plot([min_val, max_val], [min_val, max_val], 'r--')
    ax[1].set_title('Q-Q Plot (Perfect match would be on the line)')
    ax[1].set_xlabel('True Distribution Quantiles')
    ax[1].set_ylabel('Generated Distribution Quantiles')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Generated distribution statistics:")
    print(f"  Mean = {gen_mean:.4f} (Target: {target_mean})")
    print(f"  Std  = {gen_std:.4f} (Target: {target_std})")
    print(f"  Min  = {generated.min():.4f}")
    print(f"  Max  = {generated.max():.4f}")
    
    # Calculate Wasserstein distance
    from scipy.stats import wasserstein_distance
    wd = wasserstein_distance(generated, true_samples)
    print(f"  Wasserstein distance = {wd:.4f} (lower is better)")

# Visualize initial generator output (before training)
print("Generator output BEFORE training:")
visualize_generator_output(generator)
```

## 4. Building the Discriminator Network

### 4.1 Understanding the Discriminator's Role

The discriminator acts as a binary classifier:
- It takes a sample (either from the real data or from the generator)
- It outputs a probability indicating how likely the input is real (1) versus fake (0)
- During training, it tries to correctly classify real and fake samples

### 4.2 Implementation of the Discriminator

```python
class Discriminator(nn.Module):
    def __init__(self, hidden_dims=[16, 32, 16]):
        """
        Discriminator network that classifies samples as real or fake.
        
        Args:
            hidden_dims (list): Dimensions of hidden layers
        """
        super(Discriminator, self).__init__()
        
        # Build layers dynamically based on hidden_dims
        layers = []
        input_dim = 1  # We're classifying scalar values
        
        # Add hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.LeakyReLU(0.2))
            # Optional: Add dropout for regularization
            layers.append(nn.Dropout(0.3))
            input_dim = h_dim
        
        # Output layer with sigmoid activation for binary classification
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        # Create sequential model
        self.disc = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass of the discriminator.
        
        Args:
            x (Tensor): Input samples, shape [batch_size, 1]
            
        Returns:
            Tensor: Probability that each sample is real, shape [batch_size, 1]
        """
        return self.disc(x)

# Initialize the discriminator
discriminator = Discriminator(hidden_size).to(device)

# Print model architecture
print("Discriminator Architecture:")
print(discriminator)
print(f"Number of parameters: {sum(p.numel() for p in discriminator.parameters())}")

# Analyze a single forward pass with real and fake data
real_test = torch.tensor([[0.1], [0.5], [-0.2], [1.0], [-0.5]]).float()
real_scores = discriminator(real_test)
print(f"\nTest input shape: {real_test.shape}")
print(f"Test output shape: {real_scores.shape}")
print(f"Sample discriminator outputs (untrained):")
for sample, score in zip(real_test.flatten().tolist(), real_scores.flatten().tolist()):
    print(f"  Value: {sample:.4f}, D(x): {score:.4f}")
```

## 5. Setting Up Optimization

### 5.1 Loss Function Explained

For GANs, the traditional loss function is Binary Cross-Entropy (BCE):

$$\mathcal{L}_{BCE} = -y \log(p) - (1-y) \log(1-p)$$

Where:
- $y$ is the true label (1 for real, 0 for fake)
- $p$ is the predicted probability

In the GAN context:
- **Discriminator Loss** = BCE(D(real_data), 1) + BCE(D(G(noise)), 0)
- **Generator Loss** = BCE(D(G(noise)), 1)

The discriminator tries to maximize its accuracy, while the generator tries to maximize the discriminator's error on generated samples.

### 5.2 Setting Up Optimizers

```python
# Optimizers with beta parameters explained
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# Binary Cross Entropy Loss - appropriate for binary classification (real vs fake)
criterion = nn.BCELoss()

# Print optimization setup
print("Optimization Setup:")
print(f"Learning rate: {lr}")
print(f"Adam betas: ({beta1}, {beta2})")
print(f"Loss function: Binary Cross Entropy")
```

### 5.3 Preparing for Tracking Results

```python
# Lists to store metrics for plotting
g_losses = []  # Generator losses
d_losses = []  # Discriminator losses
mean_list = [] # Mean of generated distribution over time
std_list = []  # Standard deviation of generated distribution over time
w_distances = [] # Wasserstein distances over time

# Function to compute Wasserstein distance
def compute_wasserstein_distance(generated_samples, target_mean=0, target_std=1, n_samples=1000):
    """Compute Wasserstein distance between generated samples and target distribution."""
    from scipy.stats import wasserstein_distance
    true_samples = np.random.normal(target_mean, target_std, n_samples)
    return wasserstein_distance(generated_samples.flatten(), true_samples)

# Create a progress tracking time
start_time = time.time()
```

## 6. Understanding GAN Training Dynamics

Before we start training, it's important to understand the dynamics of GAN training:

### 6.1 The GAN Game

Training a GAN can be viewed as a two-player zero-sum game:
1. The discriminator tries to correctly classify real and fake samples
2. The generator tries to create samples that fool the discriminator

This adversarial process can be challenging to stabilize, as improvements in one network can cause the other to struggle.

### 6.2 Common GAN Training Issues

GANs often face challenges during training:

1. **Mode Collapse**: The generator produces limited varieties of samples
2. **Non-Convergence**: Models oscillate and fail to converge
3. **Vanishing Gradients**: Gradients become too small for effective learning
4. **Balance Issues**: One network overpowers the other

### 6.3 Techniques We'll Use to Stabilize Training

We'll implement several techniques to help stabilize GAN training:

1. **Label Smoothing**: Using 0.9 instead of 1.0 for real labels to prevent overconfidence
2. **Separate Batch Norms**: Different batches for real and fake data
3. **Proper Initialization**: Weights initialized to help with gradient flow
4. **Different Learning Rates**: Potentially different learning rates for G and D if needed
5. **Monitoring Both Networks**: Tracking losses to ensure balanced training

## 7. GAN Training Process in Detail

The GAN training involves a delicate balance between the generator and discriminator. Let's break down the training process into detailed steps.

### 7.1 Preparing the Training Loop

```python
# Tracking metrics
metrics = {
    'g_losses': [],    # Generator losses
    'd_losses': [],    # Discriminator losses
    'mean_values': [], # Generated distribution mean
    'std_values': [],  # Generated distribution std
    'w_distances': [], # Wasserstein distance
    'd_real_acc': [],  # Discriminator accuracy on real data
    'd_fake_acc': [],  # Discriminator accuracy on fake data
    'epochs': []       # Epoch markers
}

# Use TQDM for progress tracking
progress_bar = tqdm(range(num_epochs), desc="Training GAN")

# Label smoothing: use 0.9 for real labels instead of 1.0 to prevent overconfidence
real_label_value = 1.0 - label_smoothing
fake_label_value = 0.0
```

### 7.2 The Complete Training Loop

```python
for epoch in progress_bar:
    # =========================================================================
    # Step 1: Generate Real and Fake Samples
    # =========================================================================
    
    # Generate real samples from a true normal distribution
    real_samples = torch.randn(batch_size, 1) * target_std + target_mean
    real_samples = real_samples.to(device)
    
    # Create labels for real samples (with label smoothing)
    real_labels = torch.full((batch_size, 1), real_label_value, device=device)
    
    # Generate random noise for the generator input
    z = torch.randn(batch_size, z_dim).to(device)
    
    # Generate fake samples using the generator
    fake_samples = generator(z)
    
    # Create labels for fake samples
    fake_labels = torch.full((batch_size, 1), fake_label_value, device=device)
    
    # =========================================================================
    # Step 2: Train the Discriminator
    # =========================================================================
    
    # Reset discriminator gradients
    d_optimizer.zero_grad()
    
    # Train with real samples
    d_real_output = discriminator(real_samples)
    d_real_loss = criterion(d_real_output, real_labels)
    
    # Calculate discriminator accuracy on real samples
    d_real_accuracy = ((d_real_output > 0.5).float() == real_labels).float().mean().item()
    
    # Train with fake samples
    d_fake_output = discriminator(fake_samples.detach())  # detach to avoid updating generator
    d_fake_loss = criterion(d_fake_output, fake_labels)
    
    # Calculate discriminator accuracy on fake samples
    d_fake_accuracy = ((d_fake_output < 0.5).float() == (1 - fake_labels)).float().mean().item()
    
    # Combined discriminator loss
    d_loss = d_real_loss + d_fake_loss
    
    # Backpropagate and optimize
    d_loss.backward()
    d_optimizer.step()
    
    # =========================================================================
    # Step 3: Train the Generator
    # =========================================================================
    
    # Reset generator gradients
    g_optimizer.zero_grad()
    
    # The generator wants the discriminator to classify fake samples as real
    # Note: We recompute the discriminator output on fake samples
    # because we need the gradients to flow to the generator
    d_output_on_fake = discriminator(fake_samples)
    
    # Generator loss - make discriminator believe fakes are real
    g_loss = criterion(d_output_on_fake, real_labels)
    
    # Backpropagate and optimize
    g_loss.backward()
    g_optimizer.step()
    
    # =========================================================================
    # Step 4: Track Progress and Metrics
    # =========================================================================
    
    # Calculate discriminator overall accuracy
    d_accuracy = (d_real_accuracy + d_fake_accuracy) / 2
    
    # Update progress bar with current losses
    progress_bar.set_postfix({
        'D Loss': f'{d_loss.item():.4f}',
        'G Loss': f'{g_loss.item():.4f}',
        'D Acc': f'{d_accuracy:.2f}'
    })
    
    # Record detailed metrics at display intervals
    if (epoch + 1) % display_step == 0 or epoch == 0 or epoch == num_epochs - 1:
        # Generate samples for evaluation
        with torch.no_grad():
            eval_z = torch.randn(1000, z_dim).to(device)
            eval_samples = generator(eval_z).cpu().numpy()
            
            # Calculate metrics
            curr_mean = eval_samples.mean()
            curr_std = eval_samples.std()
            w_dist = compute_wasserstein_distance(eval_samples, target_mean, target_std)
            
            # Store metrics
            metrics['epochs'].append(epoch + 1)
            metrics['g_losses'].append(g_loss.item())
            metrics['d_losses'].append(d_loss.item())
            metrics['mean_values'].append(curr_mean)
            metrics['std_values'].append(curr_std)
            metrics['w_distances'].append(w_dist)
            metrics['d_real_acc'].append(d_real_accuracy)
            metrics['d_fake_acc'].append(d_fake_accuracy)
            
            # Display metrics
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print(f"  D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
            print(f"  D Accuracy - Real: {d_real_accuracy:.4f}, Fake: {d_fake_accuracy:.4f}")
            print(f"  Generated - Mean: {curr_mean:.4f}, Std: {curr_std:.4f}, W-dist: {w_dist:.4f}")
            
            # Visualize generator progress
            if (epoch + 1) % (display_step * 2) == 0:
                visualize_generator_output(generator)

# Display final results
print(f"\nTraining completed in {(time.time() - start_time)/60:.2f} minutes")

# Final evaluation
print("\nFinal Generator Evaluation:")
visualize_generator_output(generator, n_samples=10000)
```

### 7.3 Understanding the Training Loop in Detail

Let's break down the key components of the training process:

#### 7.3.1 Sample Generation

1. **Real Samples**: We generate real samples from a normal distribution with our target parameters:
   ```python
   real_samples = torch.randn(batch_size, 1) * target_std + target_mean
   ```
   
   This creates a batch of values that truly follow our target distribution.

2. **Fake Samples**: We generate random noise and pass it through our generator:
   ```python
   z = torch.randn(batch_size, z_dim).to(device)
   fake_samples = generator(z)
   ```
   
   Initially, these will be poor approximations of our target, but they should improve with training.

#### 7.3.2 The Two-Step Training Process

GANs train in two alternating steps:

1. **Training the Discriminator**:
   - Train on real samples with label 1 (or 0.9 with label smoothing)
   - Train on fake samples with label 0
   - Goal: Maximize ability to distinguish real from fake

2. **Training the Generator**:
   - Present fake samples to the discriminator
   - Goal: Make the discriminator classify fake samples as real (label 1)

The key to this step in the code is using `.detach()` when training the discriminator on fake samples:

```python
d_fake_output = discriminator(fake_samples.detach())
```

This prevents gradients from flowing back to the generator during discriminator training.

#### 7.3.3 Monitoring Training Balance

A balanced GAN training requires both networks to improve together:
- If the discriminator becomes too powerful, the generator receives little useful gradient information
- If the generator becomes too powerful too quickly, the discriminator fails to provide useful feedback

We monitor this balance through:
- Loss values for both networks
- Discriminator accuracy on real and fake samples
- Distribution statistics (mean, std, Wasserstein distance)

### 7.4 Visualization During Training

Periodically visualizing the generator's output helps us understand its progress:

```python
# Function to visualize training progress
def visualize_training_progress(metrics):
    """Visualize multiple training metrics in a single plot."""
    epochs = metrics['epochs']
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Plot losses
    axes[0, 0].plot(epochs, metrics['d_losses'], label='Discriminator')
    axes[0, 0].plot(epochs, metrics['g_losses'], label='Generator')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Plot discriminator accuracy
    axes[0, 1].plot(epochs, metrics['d_real_acc'], label='Real Accuracy')
    axes[0, 1].plot(epochs, metrics['d_fake_acc'], label='Fake Accuracy')
    axes[0, 1].set_title('Discriminator Accuracy')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Plot distribution mean
    axes[1, 0].plot(epochs, metrics['mean_values'])
    axes[1, 0].axhline(y=target_mean, color='r', linestyle='--')
    axes[1, 0].set_title(f'Generated Distribution Mean vs Target ({target_mean})')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Mean')
    
    # Plot distribution std
    axes[1, 1].plot(epochs, metrics['std_values'])
    axes[1, 1].axhline(y=target_std, color='r', linestyle='--')
    axes[1, 1].set_title(f'Generated Distribution Std vs Target ({target_std})')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Standard Deviation')
    
    # Plot Wasserstein distance
    axes[2, 0].plot(epochs, metrics['w_distances'])
    axes[2, 0].set_title('Wasserstein Distance (lower is better)')
    axes[2, 0].set_xlabel('Epochs')
    axes[2, 0].set_ylabel('Distance')
    
    # Plot final distribution comparison
    with torch.no_grad():
        z = torch.randn(5000, z_dim).to(device)
        gen_samples = generator(z).cpu().numpy().flatten()
    
    axes[2, 1].hist(np.random.normal(target_mean, target_std, 5000), 
                    bins=50, alpha=0.5, label='Target')
    axes[2, 1].hist(gen_samples, bins=50, alpha=0.5, label='Generated')
    axes[2, 1].set_title('Final Distribution Comparison')
    axes[2, 1].set_xlabel('Value')
    axes[2, 1].set_ylabel('Count')
    axes[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig('gan_training_progress.png')
    plt.show()

# Visualize all training metrics
visualize_training_progress(metrics)
```

## 8. Evaluating the Trained GAN

Once training is complete, we need to evaluate our generator's performance rigorously.

### 8.1 Generating Samples for Analysis

```python
# Generate a large number of samples for thorough analysis
with torch.no_grad():
    n_evaluation_samples = 50000
    z = torch.randn(n_evaluation_samples, z_dim).to(device)
    generated_samples = generator(z).cpu().numpy().flatten()

# Generate real samples for comparison
real_samples = np.random.normal(target_mean, target_std, n_evaluation_samples)

print(f"Generated {n_evaluation_samples} samples for evaluation")
```

### 8.2 Statistical Comparison

```python
# Calculate detailed statistics
from scipy import stats

def calculate_distribution_statistics(samples, distribution_name):
    """Calculate and print comprehensive statistics for a distribution."""
    mean = samples.mean()
    median = np.median(samples)
    std = samples.std()
    skewness = stats.skew(samples)
    kurtosis = stats.kurtosis(samples)
    min_val = samples.min()
    max_val = samples.max()
    
    # Calculate percentiles
    p5, p25, p75, p95 = np.percentile(samples, [5, 25, 75, 95])
    
    print(f"\n{distribution_name} Distribution Statistics:")
    print(f"  Mean: {mean:.6f}")
    print(f"  Median: {median:.6f}")
    print(f"  Standard Deviation: {std:.6f}")
    print(f"  Skewness: {skewness:.6f} (0 for perfect normal)")
    print(f"  Kurtosis: {kurtosis:.6f} (0 for perfect normal)")
    print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
    print(f"  Percentiles:")
    print(f"    5th: {p5:.6f}")
    print(f"    25th: {p25:.6f}")
    print(f"    75th: {p75:.6f}")
    print(f"    95th: {p95:.6f}")
    
    return {
        'mean': mean,
        'median': median,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'min': min_val,
        'max': max_val,
        'p5': p5,
        'p25': p25,
        'p75': p75,
        'p95': p95
    }

# Calculate statistics for both distributions
gen_stats = calculate_distribution_statistics(generated_samples, "Generated")
real_stats = calculate_distribution_statistics(real_samples, "Real (Target)")

# Calculate statistical tests to compare distributions
ks_stat, ks_pvalue = stats.ks_2samp(generated_samples, real_samples)
print(f"\nKolmogorov-Smirnov Test:")
print(f"  Statistic: {ks_stat:.6f}")
print(f"  p-value: {ks_pvalue:.6f}")
print(f"  Interpretation: {'Distributions are similar (fail to reject H0)' if ks_pvalue > 0.05 else 'Distributions are different (reject H0)'}")

# Wasserstein distance
w_distance = stats.wasserstein_distance(generated_samples, real_samples)
print(f"\nWasserstein Distance: {w_distance:.6f}")
```

### 8.3 Visual Evaluation

```python
def comprehensive_visual_evaluation(generated, real, gen_stats, real_stats):
    """Create comprehensive visualization comparing distributions."""
    fig = plt.figure(figsize=(20, 15))
    
    # Define common bins for consistent comparison
    all_samples = np.concatenate([generated, real])
    min_val, max_val = all_samples.min(), all_samples.max()
    bins = np.linspace(min_val, max_val, 100)
    
    # 1. Histograms with KDE
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    sns.histplot(real, bins=bins, kde=True, stat="density", label="Real", color="blue", alpha=0.5, ax=ax1)
    sns.histplot(generated, bins=bins, kde=True, stat="density", label="Generated", color="orange", alpha=0.5, ax=ax1)
    ax1.set_title("Distribution Comparison with KDE")
    ax1.legend()
    
    # 2. Box plots
    ax2 = plt.subplot2grid((3, 3), (0, 2))
    box_data = [real, generated]
    ax2.boxplot(box_data, labels=["Real", "Generated"])
    ax2.set_title("Box Plot Comparison")
    
    # 3. QQ plot
    ax3 = plt.subplot2grid((3, 3), (1, 0))
    qqplot = stats.probplot(generated, dist="norm", sparams=(gen_stats['mean'], gen_stats['std']), plot=ax3)
    ax3.set_title("Q-Q Plot (Generated vs Normal)")
    
    # 4. ECDFs
    ax4 = plt.subplot2grid((3, 3), (1, 1))
    x_real = np.sort(real)
    y_real = np.arange(1, len(x_real)+1) / len(x_real)
    x_gen = np.sort(generated)
    y_gen = np.arange(1, len(x_gen)+1) / len(x_gen)
    ax4.plot(x_real, y_real, label="Real", alpha=0.7)
    ax4.plot(x_gen, y_gen, label="Generated", alpha=0.7)
    ax4.set_title("Empirical Cumulative Distribution Functions")
    ax4.legend()
    
    # 5. Difference in histograms
    ax5 = plt.subplot2grid((3, 3), (1, 2))
    hist_real, _ = np.histogram(real, bins=bins, density=True)
    hist_gen, _ = np.histogram(generated, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax5.bar(bin_centers, hist_gen - hist_real, width=bin_centers[1]-bin_centers[0])
    ax5.set_title("Histogram Difference (Generated - Real)")
    ax5.axhline(0, color='black', linestyle='--', alpha=0.3)
    
    # 6. Statistical comparison table
    ax6 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create table data
    col_labels = ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis', 'Min', 'Max', '5th', '25th', '75th', '95th']
    row_labels = ['Real', 'Generated', 'Diff (Abs)', 'Diff (%)']
    
    # Row 1 - Real data
    row1 = [real_stats[key] for key in ['mean', 'median', 'std', 'skewness', 'kurtosis', 'min', 'max', 'p5', 'p25', 'p75', 'p95']]
    
    # Row 2 - Generated data
    row2 = [gen_stats[key] for key in ['mean', 'median', 'std', 'skewness', 'kurtosis', 'min', 'max', 'p5', 'p25', 'p75', 'p95']]
    
    # Row 3 - Absolute difference
    row3 = [abs(r - g) for r, g in zip(row1, row2)]
    
    # Row 4 - Percentage difference (with safe division)
    row4 = [abs(r - g) / (abs(r) + 1e-10) * 100 for r, g in zip(row1, row2)]
    
    # Format values for display
    def format_row(row):
        return [f"{x:.4f}" for x in row]
    
    table_data = [format_row(row1), format_row(row2), format_row(row3), [f"{x:.2f}%" for x in row4]]
    
    table = ax6.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Add KS test and Wasserstein distance as text
    txt = (f"Kolmogorov-Smirnov Test: Statistic={ks_stat:.6f}, p-value={ks_pvalue:.6f}\n"
           f"Wasserstein Distance: {w_distance:.6f}")
    plt.figtext(0.5, 0.01, txt, ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.suptitle("Comprehensive Distribution Comparison", fontsize=16)
    plt.savefig('gan_distribution_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run comprehensive visual evaluation
comprehensive_visual_evaluation(generated_samples, real_samples, gen_stats, real_stats)
```

## 9. Advanced GAN Techniques

While our implementation provides a solid foundation, several advanced techniques can improve GAN performance.

### 9.1 Implementing Wasserstein GAN (WGAN)

Wasserstein GAN uses a different approach to measure the distance between distributions:

```python
# Critic (WGAN's discriminator) - no sigmoid activation
class Critic(nn.Module):
    def __init__(self, hidden_dims=[16, 32, 16]):
        super(Critic, self).__init__()
        layers = []
        input_dim = 1
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.LeakyReLU(0.2))
            input_dim = h_dim
        
        # No sigmoid - outputs a scalar value
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# WGAN training pseudocode (simplified)
"""
for epoch in epochs:
    # Train critic multiple times
    for _ in range(n_critic):
        # Generate real and fake samples
        # Compute critic loss: mean(critic(fake)) - mean(critic(real))
        # Apply gradient penalty (WGAN-GP variant)
        # Update critic
    
    # Train generator
    # Generate fake samples
    # Compute generator loss: -mean(critic(fake))
    # Update generator
"""
```

### 9.2 Implementing Conditional GAN

Conditional GANs allow control over the generation process:

```python
# Conditional Generator pseudocode
"""
class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim, condition_dim):
        # Initialize layers
        
    def forward(self, z, condition):
        # Concatenate noise and condition
        # Process through layers
        # Return generated sample
"""

# Example: Conditioning on target mean and std
"""
# Create condition vector (e.g., desired mean and std)
condition = torch.tensor([[desired_mean, desired_std]])

# Generate with condition
fake_samples = conditional_generator(z, condition)
"""
```

### 9.3 The Importance of Batch Normalization

Batch normalization can significantly stabilize GAN training:

```python
# Generator with batch normalization
class GeneratorWithBN(nn.Module):
    def __init__(self, z_dim):
        super(GeneratorWithBN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 16),
            nn.BatchNorm1d(16),  # Stabilizes training
            nn.LeakyReLU(0.2),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1)
        )
        
    def forward(self, z):
        return self.model(z)
```

## 10. Generating Samples from the Trained GAN

Now that we have a trained GAN, let's demonstrate how to use it to generate samples with specific properties.

### 10.1 Basic Sample Generation

```python
def generate_samples_from_model(model, n_samples=1000, noise_dim=z_dim):
    """Generate samples from the trained model."""
    with torch.no_grad():
        z = torch.randn(n_samples, noise_dim).to(device)
        samples = model(z).cpu().numpy().flatten()
    return samples

# Generate 10,000 samples
samples = generate_samples_from_model(generator, n_samples=10000)

# Visualize histogram
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, alpha=0.7)
plt.title(f'10,000 Samples from Trained Generator\nMean={samples.mean():.4f}, Std={samples.std():.4f}')
plt.xlabel('Value')
plt.ylabel('Count')
plt.grid(alpha=0.3)
plt.savefig('gan_final_samples.png')
plt.show()
```

### 10.2 Saving and Loading the Model

To use your trained model later:

```python
# Save the model
model_save_path = 'gan_normal_dist_model.pth'
torch.save({
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'g_optimizer_state_dict': g_optimizer.state_dict(),
    'd_optimizer_state_dict': d_optimizer.state_dict(),
    'parameters': {
        'z_dim': z_dim,
        'target_mean': target_mean,
        'target_std': target_std,
        'hidden_size': hidden_size
    }
}, model_save_path)

print(f"Model saved to {model_save_path}")

# Load the model (for future use)
def load_gan_model(model_path):
    """Load a saved GAN model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract parameters
    params = checkpoint['parameters']
    z_dim = params['z_dim']
    hidden_size = params['hidden_size']
    
    # Initialize models
    generator = Generator(z_dim, hidden_size).to(device)
    discriminator = Discriminator(hidden_size).to(device)
    
    # Load state dictionaries
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    return generator, discriminator, params

# Example of loading
# loaded_generator, loaded_discriminator, params = load_gan_model(model_save_path)
```

## 11. Relationship with MNIST GAN and Diffusion Models

### 11.1 From Normal Distribution to MNIST Digit Generation

Our normal distribution GAN serves as an excellent foundation for understanding more complex GANs. Let's explore how we would extend this to generate MNIST digits:

#### 11.1.1 Key Differences in Architecture

```python
# MNIST Generator (Simplified)
class MNISTGenerator(nn.Module):
    def __init__(self, z_dim=100):
        super(MNISTGenerator, self).__init__()
        
        # Fully connected layer to reshape
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 7*7*256),
            nn.BatchNorm1d(7*7*256),
            nn.ReLU(True)
        )
        
        # Convolutional layers
        self.conv = nn.Sequential(
            # Input: 256 x 7 x 7
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 128 x 14 x 14
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 64 x 28 x 28
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1),              # 1 x 28 x 28
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 7, 7)  # Reshape to feed into ConvTranspose2d
        x = self.conv(x)
        return x

# MNIST Discriminator (Simplified)
class MNISTDiscriminator(nn.Module):
    def __init__(self):
        super(MNISTDiscriminator, self).__init__()
        
        self.conv = nn.Sequential(
            # Input: 1 x 28 x 28
            nn.Conv2d(1, 64, 4, 2, 1),  # 64 x 14 x 14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 128 x 7 x 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc(x)
        return x
```

#### 11.1.2 Key Differences in Training Process

When extending to MNIST training:

1. **Data Preparation**: Load MNIST dataset instead of generating normal samples
   ```python
   # MNIST data loading (pseudocode)
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
   ])
   mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
   dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
   ```

2. **Training Loop**: Iterate through dataloaders instead of generating samples
   ```python
   # Training loop (pseudocode)
   for epoch in range(num_epochs):
       for real_images, _ in dataloader:  # Discard labels
           # Train discriminator and generator
   ```

3. **Loss Functions**: Often the same BCE loss, but sometimes different losses are used

4. **Evaluation**: Use visual inspection or metrics like FID score instead of statistical measures

#### 11.1.3 Dimensional Complexity

1. **Normal Distribution GAN**:
   - Generates 1D scalar values
   - Simple distribution shape
   - Easily quantified success metrics (mean, std)

2. **MNIST GAN**:
   - Generates 2D images (28×28 pixels)
   - Complex multimodal distribution (10 digit classes)
   - Visual quality assessment is more subjective

#### 11.1.4 Code Sample: MNIST GAN Training Loop

```python
# Simplified MNIST GAN training loop (pseudocode)
"""
# Initialize networks and optimizers
generator = MNISTGenerator(z_dim)
discriminator = MNISTDiscriminator()
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for real_imgs, _ in dataloader:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        
        # Real images
        real_labels = torch.ones(batch_size, 1).to(device)
        real_output = discriminator(real_imgs)
        d_loss_real = criterion(real_output, real_labels)
        
        # Fake images
        z = torch.randn(batch_size, z_dim).to(device)
        fake_imgs = generator(z)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        fake_output = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(fake_output, fake_labels)
        
        # Combined loss and optimization
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        fake_output = discriminator(fake_imgs)
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        g_optimizer.step()
        
        # Print and save losses, images, etc.
"""
```

### 11.2 From GANs to Diffusion Models

Diffusion models represent a different approach to generative modeling, with several key differences from GANs.

#### 11.2.1 Conceptual Comparison

1. **GAN Approach**:
   - Two competing networks (generator vs. discriminator)
   - Adversarial training (generator tries to fool discriminator)
   - One-step generation process (noise → output)
   - Often struggles with mode collapse and training instability

2. **Diffusion Model Approach**:
   - Forward diffusion process: gradually adds noise to data
   - Reverse diffusion process: learns to denoise
   - Multi-step generation process (iteratively removes noise)
   - More stable training, better mode coverage

#### 11.2.2 Mathematical Foundation

GANs and diffusion models have different mathematical foundations:

1. **GANs**: Based on game theory and direct distribution transformation
   - Generator directly maps from latent space to data space
   - Discriminator provides feedback through adversarial loss

2. **Diffusion Models**: Based on stochastic differential equations (SDEs)
   - Forward process: x₀ → x₁ → ... → xₜ (adding noise)
   - Reverse process: xₜ → xₜ₋₁ → ... → x₀ (removing noise)
   - Learn to predict noise at each step

#### 11.2.3 Simplified Diffusion Model for Normal Distribution

```python
# Highly simplified diffusion model concept for normal distribution
class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        # Simple noise prediction network
        self.denoise_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, t):
        """Predict noise at diffusion step t."""
        # In real implementation, t would be used to condition the network
        return self.denoise_net(x)

# Simplified diffusion training process (pseudocode)
"""
diffusion_model = DiffusionModel().to(device)
optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-4)

for epoch in range(epochs):
    # Get real samples
    real_samples = torch.randn(batch_size, 1) * target_std + target_mean
    
    # Sample noise level
    t = torch.rand(batch_size, 1)
    
    # Add noise to samples according to schedule
    noise = torch.randn_like(real_samples)
    noisy_samples = real_samples * (1 - t) + noise * t
    
    # Predict noise
    predicted_noise = diffusion_model(noisy_samples, t)
    
    # Loss is mean squared error between predicted and true noise
    loss = F.mse_loss(predicted_noise, noise)
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""
```

#### 11.2.4 Key Advantages of Diffusion Models

1. **Training Stability**: Diffusion models typically train more stably than GANs
2. **Mode Coverage**: Better at capturing the full data distribution
3. **Sample Quality**: State-of-the-art results in many domains
4. **Controllability**: More controllable generation process

#### 11.2.5 Key Disadvantages of Diffusion Models

1. **Sampling Speed**: Require multiple denoising steps, making generation slower
2. **Computational Cost**: Can be more computationally expensive to train
3. **Complexity**: Generally more complex to implement correctly

## 12. Applications and Extensions

### 12.1 Practical Applications of GANs

While our normal distribution GAN seems simple, the principles apply to many real-world applications:

1. **Image Generation**: Generating realistic images (MNIST, faces, scenes)
2. **Data Augmentation**: Creating synthetic data for training other models
3. **Domain Transfer**: Converting images from one domain to another (e.g., photos to paintings)
4. **Super-Resolution**: Enhancing low-resolution images
5. **Text-to-Image Synthesis**: Generating images from text descriptions
6. **Synthetic Data for Privacy**: Creating realistic but synthetic datasets

### 12.2 Extension: Adding Noise Transformation as an Exercise

As an extension, let's create a more challenging version of our GAN:

```python
# Generate real samples with a more complex distribution
def generate_bimodal_samples(batch_size, mean1=-2, std1=0.5, mean2=2, std2=0.5, split=0.5):
    """Generate samples from a bimodal distribution."""
    # Determine how many samples from each mode
    n1 = int(batch_size * split)
    n2 = batch_size - n1
    
    # Generate samples from two normal distributions
    samples1 = np.random.normal(mean1, std1, (n1, 1))
    samples2 = np.random.normal(mean2, std2, (n2, 1))
    
    # Combine and shuffle
    samples = np.vstack([samples1, samples2])
    np.random.shuffle(samples)
    
    return torch.FloatTensor(samples)

# Visualize bimodal distribution
bimodal_samples = generate_bimodal_samples(10000).numpy()
plt.figure(figsize=(10, 6))
plt.hist(bimodal_samples, bins=50)
plt.title("Bimodal Distribution")
plt.xlabel("Value")
plt.ylabel("Count")
plt.show()

# Challenge: Modify the GAN to learn this bimodal distribution
# Hint: You may need to increase the capacity of your generator
```

### 12.3 Comparing VAEs, GANs, and Diffusion Models

Let's briefly compare three major generative model families:

| Feature | VAE | GAN | Diffusion Model |
|---------|-----|-----|----------------|
| Training stability | High | Low | High |
| Sample quality | Moderate | High | Very High |
| Mode coverage | High | Low-Moderate | High |
| Training speed | Fast | Moderate | Slow |
| Sampling speed | Fast | Fast | Slow |
| Likelihood estimation | Yes | No | Yes |
| Mathematical foundation | Variational inference | Game theory | SDEs |

### 12.4 Further Resources and References

To continue learning about GANs and other generative models:

1. **Papers**:
   - Original GAN paper: "Generative Adversarial Networks" (Goodfellow et al., 2014)
   - WGAN: "Wasserstein GAN" (Arjovsky et al., 2017)
   - Diffusion Models: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)

2. **Courses**:
   - Stanford CS231n: Convolutional Neural Networks for Visual Recognition
   - Coursera Deep Learning Specialization
   - Fast.ai Practical Deep Learning for Coders

3. **Books**:
   - "Deep Learning" by Goodfellow, Bengio, and Courville
   - "Generative Deep Learning" by David Foster

4. **GitHub Repositories**:
   - PyTorch-GAN: https://github.com/eriklindernoren/PyTorch-GAN
   - Hugging Face Diffusers: https://github.com/huggingface/diffusers

## 13. Conclusion and Next Steps

### 13.1 Summary of What We've Learned

In this tutorial, we've:
- Built a GAN to generate samples from a normal distribution
- Understood the dynamics between generator and discriminator
- Implemented and debugged the training process
- Evaluated our model using statistical and visual methods
- Compared our approach to other generative modeling techniques

### 13.2 Next Steps in Your Generative Modeling Journey

1. **Experiment with this GAN**:
   - Try changing hyperparameters (learning rate, network size)
   - Implement more complex target distributions (bimodal, multimodal)
   - Add conditioning to control the output distribution

2. **Move to 2D Problems**:
   - Generate 2D normal distributions
   - Create simple image patterns

3. **Scale to MNIST**:
   - Implement the MNIST GAN we discussed
   - Compare different architectures (DCGAN, WGAN)

4. **Explore Diffusion Models**:
   - Implement a simple diffusion model
   - Compare performance with GANs

5. **Apply to Real-World Problems**:
   - Data augmentation for a classification task
   - Generate synthetic data for privacy-preserving applications

Remember that generative modeling is a rapidly evolving field with new techniques and improvements being developed constantly. The foundational understanding you've gained from this tutorial will help you navigate and apply these advances to your own projects.
