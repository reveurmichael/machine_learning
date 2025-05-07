# Understanding Kullback-Leibler Divergence in Deep Learning

## Introduction

The Kullback-Leibler (KL) divergence is a foundational concept in information theory that has become essential in modern deep learning. It quantifies the difference between two probability distributions and serves as a key component in many machine learning algorithms, particularly in generative models and neural network training.

KL divergence (also known as relative entropy) was introduced by Solomon Kullback and Richard Leibler in 1951. Today, it underpins critical elements of deep learning, including:

- Variational Autoencoders (VAEs)
- Neural network regularization
- Information bottleneck methods
- Reinforcement learning algorithms
- Bayesian neural networks

In this comprehensive tutorial, we'll build a solid understanding of KL divergence from its mathematical foundations to practical implementations in deep learning. By the end, you'll be able to:

- Understand what KL divergence measures and why it's important
- Implement KL divergence from scratch in Python
- Apply KL divergence in various deep learning contexts
- Recognize when and how to use KL divergence in your own projects

## Mathematical Foundations

### Probability Distributions

To understand KL divergence, we first need to understand probability distributions. A probability distribution describes how likely different outcomes are in a random experiment.

For discrete distributions, we can represent them as:

P = [p₁, p₂, ..., pₙ]

where pᵢ is the probability of outcome i, and Σpᵢ = 1.

For continuous distributions, we deal with probability density functions (PDFs) rather than discrete probabilities.

### Definition of KL Divergence

The KL divergence from a distribution Q to a distribution P is defined as:

For discrete distributions:
KL(P||Q) = Σ pᵢ * log(pᵢ/qᵢ)

For continuous distributions:
KL(P||Q) = ∫ p(x) * log(p(x)/q(x)) dx

Where:
- P is often the "true" or target distribution
- Q is typically the approximating or model distribution
- log is the natural logarithm (base e), though sometimes base 2 is used in information theory contexts

### Key Properties of KL Divergence

KL divergence has several important properties to understand:

1. **Non-negativity**: KL(P||Q) ≥ 0 for all distributions P and Q, and KL(P||Q) = 0 if and only if P = Q (almost everywhere).

2. **Asymmetry**: Generally, KL(P||Q) ≠ KL(Q||P). This is a critical property that affects how KL divergence is used in practice.

3. **Not a proper distance metric**: Because of its asymmetry and failure to satisfy the triangle inequality, KL divergence is not a true distance metric.

4. **Undefined behavior**: If there's any outcome i where qᵢ = 0 but pᵢ > 0, the KL divergence becomes undefined (due to division by zero).

### Intuitive Understanding

Intuitively, KL(P||Q) measures the "information loss" when Q is used to approximate P. Another way to think about it:

- KL(P||Q) measures the amount of additional information (in bits or nats, depending on the logarithm base) needed to represent a random variable with distribution P using an optimal code designed for distribution Q.

### KL Divergence from Information Theory Perspective

From information theory, KL divergence can be expressed as:

KL(P||Q) = H(P,Q) - H(P)

Where:
- H(P) is the entropy of P: H(P) = -Σ pᵢ * log(pᵢ)
- H(P,Q) is the cross-entropy between P and Q: H(P,Q) = -Σ pᵢ * log(qᵢ)

This formulation highlights that KL divergence is the "extra" entropy (or uncertainty) introduced by using distribution Q when the true distribution is P.

## Implementing KL Divergence in Python

Let's start by implementing KL divergence from scratch:

```python
import numpy as np
from scipy.special import rel_entr

def kl_divergence_manual(p, q):
    """
    Calculate KL divergence KL(P||Q) between discrete probability distributions P and Q.
    
    Args:
        p: numpy array representing distribution P
        q: numpy array representing distribution Q
        
    Returns:
        KL divergence value
    """
    # Ensure proper probability distributions (normalize)
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Handle zeros in q by adding a small epsilon
    epsilon = 1e-10
    q = np.maximum(q, epsilon)
    
    # Calculate KL divergence
    kl_div = np.sum(p * np.log(p / q))
    return kl_div

# Example usage
p = np.array([0.4, 0.5, 0.1])
q = np.array([0.3, 0.4, 0.3])

kl_pq = kl_divergence_manual(p, q)
kl_qp = kl_divergence_manual(q, p)

print(f"KL(P||Q) = {kl_pq:.6f}")
print(f"KL(Q||P) = {kl_qp:.6f}")
print(f"Notice that KL(P||Q) ≠ KL(Q||P), demonstrating the asymmetry property")
```

For numerical stability, SciPy provides a specialized function:

```python
def kl_divergence_scipy(p, q):
    """
    Calculate KL divergence using SciPy's rel_entr for numerical stability.
    """
    # Ensure proper probability distributions
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return np.sum(rel_entr(p, q))

# Compare with our manual implementation
kl_scipy = kl_divergence_scipy(p, q)
print(f"SciPy KL(P||Q) = {kl_scipy:.6f}")
```

### KL Divergence in PyTorch

When working with neural networks, you'll typically use PyTorch or TensorFlow. Here's a PyTorch implementation:

```python
import torch
import torch.nn.functional as F

def kl_divergence_torch(p, q, reduction='sum'):
    """
    Calculate KL divergence KL(P||Q) between probability distributions P and Q using PyTorch.
    
    Args:
        p: PyTorch tensor representing distribution P
        q: PyTorch tensor representing distribution Q
        reduction: 'sum' or 'mean' (over batch dimension)
        
    Returns:
        KL divergence value
    """
    # Ensure proper probability distributions
    p = p / torch.sum(p, dim=-1, keepdim=True)
    q = q / torch.sum(q, dim=-1, keepdim=True)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    q = torch.clamp(q, min=epsilon)
    
    # Calculate KL divergence
    kl_div = p * (torch.log(p + epsilon) - torch.log(q))
    
    if reduction == 'sum':
        return torch.sum(kl_div)
    elif reduction == 'mean':
        return torch.mean(torch.sum(kl_div, dim=-1))
    else:
        return kl_div

# Example with PyTorch tensors
p_torch = torch.tensor([0.4, 0.5, 0.1])
q_torch = torch.tensor([0.3, 0.4, 0.3])

kl_torch = kl_divergence_torch(p_torch, q_torch)
print(f"PyTorch KL(P||Q) = {kl_torch.item():.6f}")
```

PyTorch also provides built-in functions for KL divergence:

```python
def kl_divergence_pytorch_builtin(p, q):
    """
    Using PyTorch's built-in KL divergence function.
    Note: This expects log-probabilities for the second argument.
    """
    # Ensure proper probability distributions
    p = p / torch.sum(p)
    q = q / torch.sum(q)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    q = torch.clamp(q, min=epsilon)
    
    # F.kl_div expects log-probabilities for the second argument
    return F.kl_div(torch.log(q), p, reduction='sum')

# Test with the same distributions
kl_builtin = kl_divergence_pytorch_builtin(p_torch, q_torch)
print(f"PyTorch built-in KL(P||Q) = {kl_builtin.item():.6f}")
```

## KL Divergence for Common Distributions

### KL Divergence Between Normal Distributions

For two univariate normal distributions P ~ N(μ₁, σ₁²) and Q ~ N(μ₂, σ₂²), the KL divergence has a closed-form solution:

KL(P||Q) = log(σ₂/σ₁) + (σ₁² + (μ₁ - μ₂)²)/(2σ₂²) - 1/2

Let's implement this:

```python
def kl_normal_distributions(mu1, sigma1, mu2, sigma2):
    """
    Calculate KL divergence between two univariate normal distributions.
    
    Args:
        mu1, sigma1: Mean and standard deviation of distribution P
        mu2, sigma2: Mean and standard deviation of distribution Q
        
    Returns:
        KL(P||Q): KL divergence from Q to P
    """
    return np.log(sigma2/sigma1) + (sigma1**2 + (mu1 - mu2)**2)/(2*sigma2**2) - 0.5

# Example
mu1, sigma1 = 0, 1  # Standard normal distribution
mu2, sigma2 = 1, 2  # Different normal distribution

kl_normal = kl_normal_distributions(mu1, sigma1, mu2, sigma2)
print(f"KL divergence between N({mu1}, {sigma1}²) and N({mu2}, {sigma2}²) = {kl_normal:.6f}")
```

For multivariate normal distributions, the formula is more complex but still analytically tractable:

```python
def kl_multivariate_normal(mu1, cov1, mu2, cov2):
    """
    KL divergence between multivariate normal distributions.
    
    Args:
        mu1, cov1: Mean vector and covariance matrix of distribution P
        mu2, cov2: Mean vector and covariance matrix of distribution Q
        
    Returns:
        KL(P||Q): KL divergence from Q to P
    """
    import numpy.linalg as la
    
    # Convert inputs to numpy arrays
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    cov1 = np.asarray(cov1)
    cov2 = np.asarray(cov2)
    
    # Calculate determinants and inverse
    k = mu1.shape[0]  # Dimension
    cov2_inv = la.inv(cov2)
    
    # Compute terms
    trace_term = np.trace(cov2_inv @ cov1)
    det_term = np.log(la.det(cov2) / la.det(cov1))
    quad_term = (mu2 - mu1).T @ cov2_inv @ (mu2 - mu1)
    
    # Final formula
    kl = 0.5 * (trace_term + quad_term - k + det_term)
    return kl

# Example with 2D normal distributions
mu1 = np.array([0, 0])
cov1 = np.array([[1, 0], [0, 1]])  # Identity covariance

mu2 = np.array([1, 2])
cov2 = np.array([[2, 0.5], [0.5, 3]])

kl_mv = kl_multivariate_normal(mu1, cov1, mu2, cov2)
print(f"KL divergence between multivariate normals = {kl_mv:.6f}")
```

### KL Divergence Between Bernoulli Distributions

For Bernoulli distributions P ~ Bern(p) and Q ~ Bern(q):

```python
def kl_bernoulli(p, q):
    """
    KL divergence between Bernoulli distributions with parameters p and q.
    """
    # Add epsilon for numerical stability
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1 - epsilon)
    q = np.clip(q, epsilon, 1 - epsilon)
    
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

# Example
p, q = 0.3, 0.7
kl_bern = kl_bernoulli(p, q)
print(f"KL divergence between Bern({p}) and Bern({q}) = {kl_bern:.6f}")
```

## KL Divergence in Deep Learning

Now that we understand the basics, let's explore how KL divergence is used in deep learning.

### Variational Autoencoders (VAEs)

One of the most prominent applications of KL divergence is in Variational Autoencoders. In VAEs, KL divergence is used to ensure that the learned latent representation follows a specific distribution (usually a standard normal).

The loss function for a VAE typically consists of:
1. A reconstruction term (how well the autoencoder reconstructs the input)
2. A KL divergence term (how close the encoded latent distribution is to the prior, usually N(0,1))

Here's a simplified implementation of a VAE using PyTorch:

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
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(h_dim, z_dim)
        self.log_var = nn.Linear(h_dim, z_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.log_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var

# KL Divergence calculation for VAE
def kl_divergence_normal(mu, log_var):
    """
    Compute KL divergence between N(mu, exp(log_var)) and N(0, 1)
    """
    # KL divergence between N(mu, exp(log_var)) and N(0, 1)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kl_loss

# Initialize the VAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for i, (images, _) in enumerate(train_loader):
        # Flatten images
        images = images.reshape(-1, image_size).to(device)
        
        # Forward pass
        reconstructed, mu, log_var = model(images)
        
        # Calculate losses
        reconstruction_loss = F.binary_cross_entropy(reconstructed, images, reduction='sum')
        kl_loss = kl_divergence_normal(mu, log_var)
        
        # Total loss = reconstruction loss + KL divergence
        loss = reconstruction_loss + kl_loss
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += reconstruction_loss.item()
        total_kl_loss += kl_loss.item()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'Loss: {loss.item()/len(images):.4f}, '
                  f'Recon Loss: {reconstruction_loss.item()/len(images):.4f}, '
                  f'KL Loss: {kl_loss.item()/len(images):.4f}')
    
    # Print epoch statistics
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Average Loss: {total_loss/len(train_dataset):.4f}, '
          f'Average Recon Loss: {total_recon_loss/len(train_dataset):.4f}, '
          f'Average KL Loss: {total_kl_loss/len(train_dataset):.4f}')

# Generate samples after training
def generate_samples(n_samples=10):
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        samples = model.decode(z).cpu()
        
    # Display samples
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    for i in range(n_samples):
        plt.subplot(2, 5, i+1)
        plt.imshow(samples[i].view(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()

generate_samples()
```

#### Understanding the KL Term in VAE

In a VAE, we're modeling our data with a latent variable z that follows a prior distribution p(z), typically N(0, I). However, computing the exact posterior p(z|x) is intractable, so we approximate it with a simpler distribution q(z|x).

The optimization objective of a VAE is the Evidence Lower Bound (ELBO):

ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))

Where:
- The first term is the reconstruction loss: how well can we reconstruct x given samples from q(z|x)?
- The second term is KL divergence: how much does our approximate posterior q(z|x) differ from the prior p(z)?

For Gaussian distributions, which are typically used in VAEs, this KL term has a nice analytical solution:

KL(N(μ, σ²) || N(0, 1)) = 0.5 * (μ² + σ² - log(σ²) - 1)

This explains the implementation in the `kl_divergence_normal` function above.

### Importance of KL Divergence in VAEs

The KL term in VAEs serves several important purposes:

1. **Regularization**: It prevents overfitting by ensuring the encoded representations don't become too complex.

2. **Distribution matching**: It ensures the latent space has a known, well-behaved distribution that we can sample from.

3. **Information bottleneck**: It limits the amount of information that flows through the latent variables, encouraging the model to find efficient representations.

4. **Disentanglement**: With proper modifications (like in β-VAE), it can encourage disentangled representations where different latent dimensions capture different semantic factors.

### The β-VAE: Controlling the KL Term

A popular variant of VAE is the β-VAE, which introduces a hyperparameter β to control the weight of the KL term:

ELBO = E[log p(x|z)] - β * KL(q(z|x) || p(z))

By adjusting β, we can control the trade-off between reconstruction quality and latent space properties:

```python
# In the training loop, modify the loss function:
loss = reconstruction_loss + beta * kl_loss
```

- When β < 1: Better reconstruction but potentially less structured latent space
- When β > 1: More regularized latent space but potentially worse reconstruction
- When β = 1: Standard VAE

### KL Annealing

Another technique is KL annealing, where β starts low and gradually increases during training:

```python
# KL annealing
def kl_weight_scheduler(epoch, num_epochs, final_beta=1.0):
    """
    Gradually increase the weight of KL term from 0 to final_beta
    """
    return min(final_beta, final_beta * epoch / (num_epochs // 2))

# In the training loop
beta = kl_weight_scheduler(epoch, num_epochs)
loss = reconstruction_loss + beta * kl_loss
```

This helps the model first focus on learning good reconstructions before enforcing the prior distribution constraint.

## KL Divergence for Model Regularization

Beyond VAEs, KL divergence is widely used for regularization in neural networks.

### Weight Regularization via Bayesian Neural Networks

In Bayesian neural networks, we treat the weights as random variables with prior distributions. KL divergence can be used to regularize the posterior distribution of weights:

```python
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Prior parameters
        self.prior_mu = 0
        self.prior_sigma = 1
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_var = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu)
        nn.init.constant_(self.weight_log_var, -10)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_log_var, -10)
    
    def kl_loss(self):
        """
        Calculate KL divergence between the weight posterior and prior
        """
        kl_weight = 0.5 * torch.sum(
            -self.weight_log_var + 
            (torch.exp(self.weight_log_var) + (self.weight_mu - self.prior_mu)**2) / self.prior_sigma**2 - 1
        )
        
        kl_bias = 0.5 * torch.sum(
            -self.bias_log_var + 
            (torch.exp(self.bias_log_var) + (self.bias_mu - self.prior_mu)**2) / self.prior_sigma**2 - 1
        )
        
        return kl_weight + kl_bias
    
    def forward(self, x):
        # Sample weights from posterior
        if self.training:
            weight = self.weight_mu + torch.exp(0.5 * self.weight_log_var) * torch.randn_like(self.weight_log_var)
            bias = self.bias_mu + torch.exp(0.5 * self.bias_log_var) * torch.randn_like(self.bias_log_var)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        # Linear transformation
        return F.linear(x, weight, bias)
```

### Kullback-Leibler Regularization for Neural Networks

KL divergence can also be used to regularize the output distributions of neural networks:

```python
def kl_regularized_loss(output_logits, target_labels, regularizer_distribution, beta=0.1):
    """
    Loss function with KL regularization to a target distribution
    
    Args:
        output_logits: Model output logits
        target_labels: True labels
        regularizer_distribution: Target distribution for regularization
        beta: Weight of the KL term
    """
    # Standard cross-entropy loss
    ce_loss = F.cross_entropy(output_logits, target_labels)
    
    # KL regularization
    output_probs = F.softmax(output_logits, dim=1)
    kl_reg = F.kl_div(
        torch.log(output_probs + 1e-10),
        regularizer_distribution,
        reduction='batchmean'
    )
    
    # Combined loss
    return ce_loss + beta * kl_reg
```

## KL Divergence in Policy Optimization for Reinforcement Learning

In reinforcement learning, KL divergence is often used to constrain policy updates, particularly in techniques like Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO).

Here's a simplified implementation showing how KL divergence is used in policy optimization:

```python
def policy_update_with_kl_constraint(old_policy, new_policy, states, max_kl=0.01):
    """
    Update policy with KL constraint to prevent large policy changes
    
    Args:
        old_policy: Previous policy network
        new_policy: Updated policy network
        states: Batch of states to evaluate KL over
        max_kl: Maximum allowed KL divergence
    
    Returns:
        True if the update satisfies KL constraint, False otherwise
    """
    # Compute action probabilities under old and new policies
    with torch.no_grad():
        old_action_probs = F.softmax(old_policy(states), dim=1)
    
    new_action_logits = new_policy(states)
    new_action_probs = F.softmax(new_action_logits, dim=1)
    
    # Compute KL divergence
    kl_div = torch.mean(torch.sum(
        old_action_probs * (torch.log(old_action_probs + 1e-10) - torch.log(new_action_probs + 1e-10)),
        dim=1
    ))
    
    # Check if KL constraint is satisfied
    return kl_div.item() <= max_kl, kl_div.item()
```

### Entropy and KL Divergence in Maximum Entropy RL

In maximum entropy reinforcement learning (like Soft Actor-Critic), we optimize a policy to maximize both expected return and policy entropy. KL divergence appears in various formulations of these algorithms:

```python
def soft_actor_critic_loss(q_values, action_probs, log_action_probs, alpha, target_entropy):
    """
    Soft Actor-Critic policy loss with entropy regularization
    
    Args:
        q_values: Q-values for state-action pairs
        action_probs: Probabilities of actions under current policy
        log_action_probs: Log probabilities of actions
        alpha: Temperature parameter
        target_entropy: Target entropy for the policy
    """
    # Policy loss: Expected Q-value - alpha * entropy
    policy_loss = torch.mean(action_probs * (alpha * log_action_probs - q_values))
    
    # Entropy term
    entropy = -torch.mean(torch.sum(action_probs * log_action_probs, dim=1))
    
    # Alpha loss (to automatically adjust temperature)
    alpha_loss = alpha * torch.mean(entropy - target_entropy)
    
    return policy_loss, entropy, alpha_loss
```

## Visualization and Intuition of KL Divergence

To build better intuition about KL divergence, let's visualize it for some common distributions:

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def plot_kl_normal_distributions():
    """
    Visualize KL divergence between normal distributions
    """
    # Create range of points
    x = np.linspace(-10, 10, 1000)
    
    # Fixed standard normal distribution
    std_normal = norm(0, 1)
    std_normal_pdf = std_normal.pdf(x)
    
    # Create a figure
    plt.figure(figsize=(15, 10))
    
    # Plot KL divergence for varying means
    means = [-3, -1, 0, 1, 3]
    plt.subplot(2, 1, 1)
    for mu in means:
        other_normal = norm(mu, 1)
        other_pdf = other_normal.pdf(x)
        
        # Compute KL divergence
        kl = kl_normal_distributions(0, 1, mu, 1)
        
        plt.plot(x, other_pdf, label=f'N({mu}, 1), KL = {kl:.4f}')
    
    plt.plot(x, std_normal_pdf, 'k--', label='N(0, 1)')
    plt.title('KL Divergence with Varying Means')
    plt.legend()
    plt.grid(True)
    
    # Plot KL divergence for varying standard deviations
    stds = [0.5, 0.8, 1, 1.5, 2]
    plt.subplot(2, 1, 2)
    for sigma in stds:
        other_normal = norm(0, sigma)
        other_pdf = other_normal.pdf(x)
        
        # Compute KL divergence
        kl = kl_normal_distributions(0, 1, 0, sigma)
        
        plt.plot(x, other_pdf, label=f'N(0, {sigma}), KL = {kl:.4f}')
    
    plt.plot(x, std_normal_pdf, 'k--', label='N(0, 1)')
    plt.title('KL Divergence with Varying Standard Deviations')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_kl_normal_distributions()
```

### Visualizing KL Divergence in 2D Space

For a deeper understanding, let's visualize KL divergence between 2D distributions:

```python
from matplotlib import cm
from scipy.stats import multivariate_normal

def plot_kl_2d():
    """
    Visualize KL divergence between 2D normal distributions
    """
    # Create a grid of points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Standard 2D normal
    mean1 = [0, 0]
    cov1 = [[1, 0], [0, 1]]
    rv1 = multivariate_normal(mean1, cov1)
    
    plt.figure(figsize=(18, 6))
    
    # Different means
    mean2 = [2, 1]
    cov2 = [[1, 0], [0, 1]]
    rv2 = multivariate_normal(mean2, cov2)
    kl12 = kl_multivariate_normal(mean1, cov1, mean2, cov2)
    kl21 = kl_multivariate_normal(mean2, cov2, mean1, cov1)
    
    plt.subplot(1, 3, 1)
    plt.contour(X, Y, rv1.pdf(pos), cmap=cm.Blues)
    plt.contour(X, Y, rv2.pdf(pos), cmap=cm.Reds)
    plt.scatter(mean1[0], mean1[1], c='blue', marker='x', s=100)
    plt.scatter(mean2[0], mean2[1], c='red', marker='x', s=100)
    plt.title(f'Different means\nKL(P1||P2) = {kl12:.4f}, KL(P2||P1) = {kl21:.4f}')
    plt.axis('equal')
    
    # Different covariances
    mean2 = [0, 0]
    cov2 = [[2, 0.5], [0.5, 1]]
    rv2 = multivariate_normal(mean2, cov2)
    kl12 = kl_multivariate_normal(mean1, cov1, mean2, cov2)
    kl21 = kl_multivariate_normal(mean2, cov2, mean1, cov1)
    
    plt.subplot(1, 3, 2)
    plt.contour(X, Y, rv1.pdf(pos), cmap=cm.Blues)
    plt.contour(X, Y, rv2.pdf(pos), cmap=cm.Reds)
    plt.scatter(mean1[0], mean1[1], c='blue', marker='x', s=100)
    plt.scatter(mean2[0], mean2[1], c='red', marker='x', s=100)
    plt.title(f'Different covariances\nKL(P1||P2) = {kl12:.4f}, KL(P2||P1) = {kl21:.4f}')
    plt.axis('equal')
    
    # Different means and covariances
    mean2 = [1.5, -1]
    cov2 = [[1.5, -0.7], [-0.7, 1.5]]
    rv2 = multivariate_normal(mean2, cov2)
    kl12 = kl_multivariate_normal(mean1, cov1, mean2, cov2)
    kl21 = kl_multivariate_normal(mean2, cov2, mean1, cov1)
    
    plt.subplot(1, 3, 3)
    plt.contour(X, Y, rv1.pdf(pos), cmap=cm.Blues)
    plt.contour(X, Y, rv2.pdf(pos), cmap=cm.Reds)
    plt.scatter(mean1[0], mean1[1], c='blue', marker='x', s=100)
    plt.scatter(mean2[0], mean2[1], c='red', marker='x', s=100)
    plt.title(f'Different means and covariances\nKL(P1||P2) = {kl12:.4f}, KL(P2||P1) = {kl21:.4f}')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

plot_kl_2d()