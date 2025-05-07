# Understanding Jensen-Shannon Divergence in Deep Learning

## Introduction

The Jensen-Shannon (JS) divergence is a fundamental metric in information theory and probability that has found significant applications in modern deep learning architectures. This tutorial aims to provide a comprehensive understanding of JS divergence, from its mathematical foundations to practical applications in deep learning models such as Generative Adversarial Networks (GANs).

Unlike more complex information theory concepts, JS divergence is relatively intuitive and can be understood with basic mathematical knowledge. By the end of this tutorial, you'll be able to:
- Understand what JS divergence measures and why it's important
- Implement JS divergence from scratch in Python
- See how JS divergence is used in modern deep learning frameworks
- Apply JS divergence in practical deep learning scenarios

## Mathematical Foundations

### Probability Distributions

Before diving into JS divergence, let's establish some basic concepts. A probability distribution describes how likely different outcomes are in a random experiment. For discrete distributions, we can represent them as:

P = [p₁, p₂, ..., pₙ]

where pᵢ is the probability of outcome i, and Σpᵢ = 1.

### Kullback-Leibler (KL) Divergence

JS divergence is related to the Kullback-Leibler (KL) divergence, which measures how one probability distribution differs from another. For discrete probability distributions P and Q, the KL divergence from Q to P is defined as:

KL(P||Q) = Σ pᵢ * log(pᵢ/qᵢ)

where the sum is over all states i of the random variable.

KL divergence has an important limitation: it's asymmetric, meaning KL(P||Q) ≠ KL(Q||P). Additionally, if there's any i where qᵢ = 0 but pᵢ > 0, the KL divergence becomes undefined (division by zero).

### Jensen-Shannon Divergence Definition

JS divergence addresses these limitations by creating a symmetric measure based on KL divergence. The Jensen-Shannon divergence between two probability distributions P and Q is defined as:

JS(P||Q) = 1/2 * KL(P||M) + 1/2 * KL(Q||M)

where M = 1/2 * (P + Q) is the point-wise mean of distributions P and Q.

Key properties of JS divergence:
1. Symmetry: JS(P||Q) = JS(Q||P)
2. Always defined: Even when some probabilities are zero
3. Bounded: The values are always between 0 and 1 (or between 0 and log(2) depending on the log base)
4. Zero only when the distributions are identical: JS(P||P) = 0

The square root of JS divergence is a proper metric, satisfying the triangle inequality.

## Implementing JS Divergence in Python

Let's implement JS divergence from scratch to build intuition:

```python
import numpy as np
from scipy.special import rel_entr

def kl_divergence(p, q):
    """
    Compute KL divergence between distributions p and q.
    Uses scipy's rel_entr for numerical stability.
    """
    return np.sum(rel_entr(p, q))

def js_divergence(p, q):
    """
    Compute Jensen-Shannon divergence between distributions p and q.
    """
    # Convert inputs to numpy arrays and normalize if needed
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate the midpoint distribution
    m = 0.5 * (p + q)
    
    # Calculate JS divergence
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

# Example usage
p = np.array([0.4, 0.5, 0.1])
q = np.array([0.3, 0.4, 0.3])
js = js_divergence(p, q)
print(f"JS divergence between p and q: {js:.6f}")
```

For continuous distributions or when working with large models, we often use PyTorch or TensorFlow. Here's a PyTorch implementation:

```python
import torch
import torch.nn.functional as F

def js_divergence_torch(p, q, epsilon=1e-10):
    """
    Compute JS divergence between probability distributions p and q using PyTorch.
    
    Args:
        p, q: PyTorch tensors representing probability distributions
        epsilon: Small constant to avoid numerical instability
        
    Returns:
        JS divergence value
    """
    # Ensure proper probability distributions (normalize)
    p = p / torch.sum(p)
    q = q / torch.sum(q)
    
    # Add small epsilon to avoid log(0)
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    
    # Compute midpoint distribution
    m = 0.5 * (p + q)
    
    # Compute KL divergences
    kl_p_m = torch.sum(p * torch.log(p / m))
    kl_q_m = torch.sum(q * torch.log(q / m))
    
    # Compute JS divergence
    js = 0.5 * (kl_p_m + kl_q_m)
    
    return js
```

## JS Divergence in Deep Learning

### Application in Generative Adversarial Networks (GANs)

One of the most prominent uses of JS divergence in deep learning is in the original GAN (Generative Adversarial Network) formulation. GANs consist of two neural networks: a generator and a discriminator. The generator creates samples to fool the discriminator, while the discriminator tries to distinguish between real data and generated samples.

The objective function of the original GAN implicitly minimizes the JS divergence between the real data distribution and the generator's distribution. Here's the connection:

When the discriminator is optimal, the GAN's objective function can be rewritten as:

min_G max_D V(G,D) ≈ 2 * JS(P_data || P_G) - log(4)

where P_data is the real data distribution and P_G is the generator's distribution.

Let's implement a simple GAN in PyTorch to illustrate this concept:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
z_dim = 100  # Noise dimension
hidden_dim = 256
lr = 0.0002
beta1 = 0.5
num_epochs = 300

# Generate a simple 1D Gaussian distribution as real data
def create_data(n_samples=10000):
    data = np.random.normal(loc=4.0, scale=1.5, size=n_samples)
    data = data.reshape(n_samples, 1).astype(np.float32)
    return torch.from_numpy(data)

# Networks
class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim=1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=hidden_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)

# Initialize networks
G = Generator(z_dim, hidden_dim).to(device)
D = Discriminator().to(device)

# Optimizers
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss function
criterion = nn.BCELoss()

# Create real data and data loader
real_data = create_data()
dataset = TensorDataset(real_data)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Lists to store losses for plotting
G_losses = []
D_losses = []
JS_divergences = []

def estimate_js_divergence(real_samples, fake_samples, discriminator):
    """
    Estimate JS divergence based on discriminator outputs
    """
    real_outputs = discriminator(real_samples).mean().item()
    fake_outputs = discriminator(fake_samples).mean().item()
    
    # Avoid numerical issues
    real_outputs = max(min(real_outputs, 0.99), 0.01)
    fake_outputs = max(min(fake_outputs, 0.99), 0.01)
    
    # Discriminator should output D(x) = P(real) / (P(real) + P(fake))
    # We can use this to estimate the JS divergence
    p_ratio = real_outputs / (1 - real_outputs)
    q_ratio = (1 - fake_outputs) / fake_outputs
    
    kl1 = real_outputs * np.log(2 * real_outputs)
    kl2 = (1 - real_outputs) * np.log(2 * (1 - real_outputs))
    
    js_est = 0.5 * (kl1 + kl2)
    return js_est

# Training loop
for epoch in range(num_epochs):
    for i, (real_samples,) in enumerate(data_loader):
        batch_size = real_samples.size(0)
        real_samples = real_samples.to(device)
        
        # Create labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # ====== Train Discriminator ======
        d_optimizer.zero_grad()
        
        # Train with real data
        outputs = D(real_samples)
        d_loss_real = criterion(outputs, real_labels)
        
        # Train with fake data
        z = torch.randn(batch_size, z_dim).to(device)
        fake_samples = G(z)
        outputs = D(fake_samples.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # ====== Train Generator ======
        g_optimizer.zero_grad()
        
        # Generate fake samples
        z = torch.randn(batch_size, z_dim).to(device)
        fake_samples = G(z)
        
        # Try to fool the discriminator
        outputs = D(fake_samples)
        g_loss = criterion(outputs, real_labels)
        
        g_loss.backward()
        g_optimizer.step()
        
        # Estimate JS divergence
        if i == 0:
            js_div = estimate_js_divergence(real_samples, fake_samples.detach(), D)
            JS_divergences.append(js_div)
            
    # Save losses for plotting
    G_losses.append(g_loss.item())
    D_losses.append(d_loss.item())
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, JS Div: {js_div:.4f}')

# Visualization after training
def visualize_results():
    # Generate samples
    with torch.no_grad():
        z = torch.randn(1000, z_dim).to(device)
        fake_samples = G(z).cpu().numpy()
    
    # Plot real and generated distributions
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 2, 1)
    plt.hist(real_data.numpy(), bins=50, alpha=0.7, label='Real Data')
    plt.hist(fake_samples, bins=50, alpha=0.7, label='Generated Data')
    plt.legend()
    plt.title('Data Distributions')
    
    plt.subplot(2, 2, 2)
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.legend()
    plt.title('Training Losses')
    
    plt.subplot(2, 2, 3)
    plt.plot(JS_divergences)
    plt.title('Estimated JS Divergence')
    
    plt.tight_layout()
    plt.show()

visualize_results()
```

### JS Divergence in Variational Autoencoders (VAEs)

In Variational Autoencoders, while KL divergence is more commonly used, JS divergence can be employed as an alternative measure for distribution matching. Let's explore a simple VAE implementation where we can compare both divergences:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 20
latent_dim = 20
image_size = 28 * 28

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# VAE Model
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(0.2)
        )
        
        self.mu = nn.Linear(h_dim, z_dim)
        self.log_var = nn.Linear(h_dim, z_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        # Encode
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        reconstructed = self.decoder(z)
        
        return reconstructed, mu, log_var
    
    def sample(self, num_samples):
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = self.decoder(z)
        return samples

# KL Divergence calculation
def kl_divergence_loss(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

# JS Divergence approximation between two Gaussian distributions
def js_divergence_gaussian(mu1, log_var1, mu2, log_var2):
    """
    Approximate JS divergence between two multivariate Gaussian distributions
    """
    # Create midpoint distribution parameters
    mu_m = 0.5 * (mu1 + mu2)
    log_var_m = 0.5 * (log_var1 + log_var2)
    
    # KL(P||M)
    kl_p_m = 0.5 * torch.sum(
        log_var_m - log_var1 + 
        (torch.exp(log_var1) + (mu1 - mu_m).pow(2)) / torch.exp(log_var_m) - 1,
        dim=1
    ).mean()
    
    # KL(Q||M)
    kl_q_m = 0.5 * torch.sum(
        log_var_m - log_var2 + 
        (torch.exp(log_var2) + (mu2 - mu_m).pow(2)) / torch.exp(log_var_m) - 1,
        dim=1
    ).mean()
    
    # JS divergence
    js = 0.5 * (kl_p_m + kl_q_m)
    
    return js

# Initialize the VAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        # Flatten images
        images = images.reshape(-1, image_size).to(device)
        
        # Forward pass
        reconstructed, mu, log_var = model(images)
        
        # Calculate loss
        reconstruction_loss = F.binary_cross_entropy(reconstructed, images, reduction='sum') / images.size(0)
        
        # Standard VAE uses KL divergence
        kl_loss = kl_divergence_loss(mu, log_var)
        
        # Calculate JS divergence (between encoded distribution and prior N(0,1))
        prior_mu = torch.zeros_like(mu)
        prior_log_var = torch.zeros_like(log_var)
        js_loss = js_divergence_gaussian(mu, log_var, prior_mu, prior_log_var)
        
        # Total loss (can switch between KL and JS)
        # loss = reconstruction_loss + kl_loss  # Standard VAE with KL
        loss = reconstruction_loss + js_loss  # VAE with JS
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'Recon Loss: {reconstruction_loss.item():.4f}, '
                  f'KL Loss: {kl_loss.item():.4f}, '
                  f'JS Loss: {js_loss.item():.4f}')
```

## Practical Applications of JS Divergence

### 1. Domain Adaptation

JS divergence is useful in domain adaptation tasks, where we aim to adapt a model trained on one domain to perform well on another domain. Here's a simplified implementation:

```python
def domain_adaptation_loss(source_logits, target_logits):
    """
    Calculate domain adaptation loss using JS divergence
    """
    # Convert logits to probabilities
    source_probs = F.softmax(source_logits, dim=1)
    target_probs = F.softmax(target_logits, dim=1)
    
    # Calculate average class distributions
    source_dist = torch.mean(source_probs, dim=0)
    target_dist = torch.mean(target_probs, dim=0)
    
    # Calculate JS divergence
    m = 0.5 * (source_dist + target_dist)
    kl_source_m = F.kl_div(torch.log(m), source_dist, reduction='sum')
    kl_target_m = F.kl_div(torch.log(m), target_dist, reduction='sum')
    
    js_div = 0.5 * (kl_source_m + kl_target_m)
    return js_div
```

### 2. Model Comparison and Ensemble Learning

JS divergence can be used to measure disagreement between different models, which is useful for ensemble learning:

```python
def model_disagreement(model_outputs):
    """
    Calculate pairwise JS divergence between model outputs
    
    Args:
        model_outputs: List of model output probability distributions
                       Each with shape [batch_size, num_classes]
    
    Returns:
        Average JS divergence between all model pairs
    """
    n_models = len(model_outputs)
    total_js = 0.0
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            # Calculate JS divergence between model i and j
            probs_i = F.softmax(model_outputs[i], dim=1)
            probs_j = F.softmax(model_outputs[j], dim=1)
            
            m = 0.5 * (probs_i + probs_j)
            kl_i_m = torch.sum(probs_i * torch.log(probs_i / m + 1e-10), dim=1).mean()
            kl_j_m = torch.sum(probs_j * torch.log(probs_j / m + 1e-10), dim=1).mean()
            
            js = 0.5 * (kl_i_m + kl_j_m)
            total_js += js
    
    # Return average JS divergence
    return total_js / (n_models * (n_models - 1) / 2)
```

### 3. Anomaly Detection

JS divergence is effective for measuring how much a sample deviates from a reference distribution:

```python
def anomaly_detection(reference_distribution, sample_distribution):
    """
    Detect anomalies using JS divergence
    
    Args:
        reference_distribution: Normal data distribution (normalized histogram)
        sample_distribution: Distribution of sample to test
        
    Returns:
        JS divergence score (higher means more anomalous)
    """
    # Normalize distributions if they aren't already
    reference_distribution = reference_distribution / np.sum(reference_distribution)
    sample_distribution = sample_distribution / np.sum(sample_distribution)
    
    # Calculate midpoint
    m = 0.5 * (reference_distribution + sample_distribution)
    
    # Calculate KL divergences
    kl_ref_m = np.sum(reference_distribution * np.log(reference_distribution / m + 1e-10))
    kl_sample_m = np.sum(sample_distribution * np.log(sample_distribution / m + 1e-10))
    
    # Calculate JS divergence
    js = 0.5 * (kl_ref_m + kl_sample_m)
    
    return js
```

## Theoretical Implications for Deep Learning

### JS Divergence vs. Other Divergences in GANs

The original GAN formulation implicitly minimizes JS divergence, but this has some limitations:

1. Mode Collapse: When generator distributions don't overlap with the real data distribution, the JS divergence gradient can vanish, causing training instability.

2. Comparing with Alternatives: Other GAN variants use different divergence measures:
   - Wasserstein GAN (WGAN): Uses Wasserstein distance (Earth Mover's distance)
   - f-GAN: Generalizes to various f-divergences
   - Least Squares GAN: Uses a different objective that behaves differently

Let's compare these approaches:

```python
# Example comparison of different GAN losses

def original_gan_loss(real_outputs, fake_outputs):
    """Original GAN loss (implicitly minimizes JS divergence)"""
    real_loss = -torch.mean(torch.log(real_outputs + 1e-10))
    fake_loss = -torch.mean(torch.log(1 - fake_outputs + 1e-10))
    return real_loss + fake_loss

def wasserstein_loss(real_outputs, fake_outputs):
    """Wasserstein loss"""
    return -torch.mean(real_outputs) + torch.mean(fake_outputs)

def least_squares_loss(real_outputs, fake_outputs):
    """Least squares GAN loss"""
    real_loss = torch.mean((real_outputs - 1) ** 2)
    fake_loss = torch.mean(fake_outputs ** 2)
    return 0.5 * (real_loss + fake_loss)

# We can visualize how these losses behave with different levels of overlap
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_behaviors():
    # Create a range of discriminator outputs
    d_values = np.linspace(0.01, 0.99, 100)
    
    # Calculate losses for different discriminator values
    js_losses = -np.log(d_values) - np.log(1 - d_values)
    w_losses = -d_values + (1 - d_values)
    ls_losses = 0.5 * ((d_values - 1)**2 + (1 - d_values)**2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(d_values, js_losses, label='JS-based Loss')
    plt.plot(d_values, w_losses, label='Wasserstein Loss')
    plt.plot(d_values, ls_losses, label='Least Squares Loss')
    plt.xlabel('Discriminator Output D(x)')
    plt.ylabel('Loss Value')
    plt.title('Comparison of GAN Loss Functions')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss_behaviors()
```

## Advanced Topics: Jensen-Shannon Divergence Variants

### Multi-Distribution JS Divergence

The standard JS divergence compares two distributions, but we can extend it to multiple distributions:

```python
def js_divergence_multi(distributions, weights=None):
    """
    Calculate JS divergence among multiple distributions
    
    Args:
        distributions: List of probability distributions
        weights: Optional weights for each distribution (default: equal weights)
        
    Returns:
        JS divergence value
    """
    n_distributions = len(distributions)
    
    # Default to equal weights if not provided
    if weights is None:
        weights = np.ones(n_distributions) / n_distributions
    
    # Ensure weights sum to 1
    weights = np.array(weights) / np.sum(weights)
    
    # Calculate the weighted average distribution
    m = np.zeros_like(distributions[0])
    for i in range(n_distributions):
        m += weights[i] * distributions[i]
    
    # Calculate weighted sum of KL divergences
    js = 0
    for i in range(n_distributions):
        js += weights[i] * np.sum(distributions[i] * np.log(distributions[i] / m + 1e-10))
    
    return js
```

### Smooth JS Divergence for Training Stability

To address the gradient vanishing problem in JS divergence, we can use a smoothed version:

```python
def smooth_js_divergence(p, q, alpha=0.1):
    """
    Smooth JS divergence with interpolation parameter alpha
    
    Args:
        p, q: Probability distributions
        alpha: Smoothing parameter (0 < alpha < 1)
        
    Returns:
        Smoothed JS divergence
    """
    # Interpolate the distributions
    p_smooth = (1 - alpha) * p + alpha * q
    q_smooth = alpha * p + (1 - alpha) * q
    
    # Calculate the midpoint
    m = 0.5 * (p_smooth + q_smooth)
    
    # Calculate KL divergences
    kl_p_m = np.sum(p_smooth * np.log(p_smooth / m + 1e-10))
    kl_q_m = np.sum(q_smooth * np.log(q_smooth / m + 1e-10))
    
    # Calculate smoothed JS divergence
    js_smooth = 0.5 * (kl_p_m + kl_q_m)
    
    return js_smooth
```

## Conclusion

Jensen-Shannon divergence is a powerful metric for comparing probability distributions in deep learning. Its mathematical properties make it particularly suitable for generative modeling tasks like GANs, while its extensions and variants continue to inspire new approaches in machine learning.

Key takeaways:
1. JS divergence provides a symmetric, bounded measure of distribution similarity
2. It forms the theoretical foundation of the original GAN formulation
3. It can be implemented efficiently in modern deep learning frameworks
4. It has applications beyond GANs, including VAEs, domain adaptation, and anomaly detection

As deep learning continues to evolve, understanding measures like JS divergence becomes increasingly important for developing stable, effective models.

## Further Reading and Resources

1. Original GAN paper by Goodfellow et al.: "Generative Adversarial Networks" (2014)
2. "f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization" by Nowozin et al.
3. "Improved Training of Wasserstein GANs" by Gulrajani et al.
4. "Information Theory and Statistical Mechanics" by Jaynes (1957)
5. PyTorch documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
6. TensorFlow probability library: [https://www.tensorflow.org/probability](https://www.tensorflow.org/probability)

## Appendix: Mathematical Derivations

For interested readers, here's a derivation of why the JS divergence is bounded between 0 and log(2):

The KL divergence can be expressed in terms of entropy:
KL(P||Q) = H(P,Q) - H(P)

where H(P) is the entropy of P and H(P,Q) is the cross-entropy between P and Q.

The JS divergence can thus be written as:
JS(P||Q) = 1/2[KL(P||M) + KL(Q||M)]
         = 1/2[H(P,M) - H(P) + H(Q,M) - H(Q)]
         = 1/2[H(P,M) + H(Q,M)] - 1/2[H(P) + H(Q)]

Since M = 1/2(P+Q), and using the convexity of entropy:
H(M) ≥ 1/2[H(P) + H(Q)]

The maximum value occurs when P and Q have disjoint supports, giving:
JS(P||Q)_max = log(2)

When using base-2 logarithms, this maximum becomes 1, which is why JS divergence is often described as bounded between 0 and 1.
