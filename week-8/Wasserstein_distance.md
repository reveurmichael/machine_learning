# Understanding Wasserstein Distance in Deep Learning

## Introduction

The Wasserstein distance, also known as the Earth Mover's Distance (EMD), is a powerful metric that has revolutionized several areas of deep learning, particularly in generative modeling. This tutorial aims to provide a comprehensive understanding of the Wasserstein distance, its mathematical foundations, and its applications in deep learning, especially in Generative Adversarial Networks (GANs).

## Table of Contents

1. [Mathematical Foundations](#mathematical-foundations)
   - [Defining Wasserstein Distance](#defining-wasserstein-distance)
   - [Intuitive Understanding: The Earth Mover's Analogy](#intuitive-understanding-the-earth-movers-analogy)
   - [Properties of Wasserstein Distance](#properties-of-wasserstein-distance)
2. [Wasserstein Distance in Deep Learning](#wasserstein-distance-in-deep-learning)
   - [Limitations of Traditional Metrics](#limitations-of-traditional-metrics)
   - [Advantages of Wasserstein Distance](#advantages-of-wasserstein-distance)
3. [Wasserstein GANs (WGANs)](#wasserstein-gans-wgans)
   - [From Standard GANs to WGANs](#from-standard-gans-to-wgans)
   - [The Critic Function and Lipschitz Constraints](#the-critic-function-and-lipschitz-constraints)
   - [Gradient Penalty](#gradient-penalty)
4. [Implementing Wasserstein Distance](#implementing-wasserstein-distance)
   - [Computing Wasserstein Distance in 1D](#computing-wasserstein-distance-in-1d)
   - [Implementing WGAN with PyTorch](#implementing-wgan-with-pytorch)
   - [Implementing WGAN-GP with TensorFlow](#implementing-wgan-gp-with-tensorflow)
5. [Advanced Topics](#advanced-topics)
   - [Sliced Wasserstein Distance](#sliced-wasserstein-distance)
   - [Other Applications in Deep Learning](#other-applications-in-deep-learning)
6. [Conclusion and Further Reading](#conclusion-and-further-reading)

## Mathematical Foundations

### Defining Wasserstein Distance

The Wasserstein distance is a measure of the distance between two probability distributions. Formally, the p-Wasserstein distance between two probability distributions P and Q on a metric space (X, d) is defined as:

$$W_p(P, Q) = \left( \inf_{\gamma \in \Gamma(P, Q)} \int_{X \times X} d(x, y)^p d\gamma(x, y) \right)^{1/p}$$

where:
- Γ(P, Q) is the set of all joint distributions γ(x, y) whose marginals are P and Q
- d(x, y) is the distance function in the metric space
- p ≥ 1 is the order of the Wasserstein distance

For most applications in deep learning, we typically use the 1-Wasserstein distance (p=1), often denoted as W₁.

### Intuitive Understanding: The Earth Mover's Analogy

To understand the Wasserstein distance intuitively, imagine you have two piles of earth (representing two probability distributions). The Wasserstein distance measures the minimum amount of work required to transform one pile into the other, where "work" is defined as the amount of earth moved multiplied by the distance it is moved.

This is why the Wasserstein distance is sometimes called the "Earth Mover's Distance" - it quantifies how much "earth" needs to be moved to transform one distribution into another.

Consider a simple example:
- Distribution P has all its mass at point x=0
- Distribution Q has all its mass at point x=5

The Wasserstein-1 distance between P and Q is simply 5, as we need to move all the mass a distance of 5 units.

### Properties of Wasserstein Distance

The Wasserstein distance has several important properties that make it valuable in deep learning:

1. **Metric Properties**: It satisfies the properties of a metric:
   - Non-negativity: W(P, Q) ≥ 0
   - Identity of indiscernibles: W(P, Q) = 0 if and only if P = Q
   - Symmetry: W(P, Q) = W(Q, P)
   - Triangle inequality: W(P, R) ≤ W(P, Q) + W(Q, R)

2. **Sensitivity to Distribution Geometry**: Unlike some other metrics (e.g., KL-divergence), the Wasserstein distance takes into account the underlying geometry of the distribution space.

3. **Well-defined Even for Non-overlapping Distributions**: The Wasserstein distance provides meaningful values even when distributions have non-overlapping supports, unlike metrics such as KL-divergence or Jensen-Shannon divergence.

## Wasserstein Distance in Deep Learning

### Limitations of Traditional Metrics

In deep learning, especially in generative modeling, we often need to measure the similarity between the generated data distribution and the real data distribution. Traditional metrics like KL-divergence and Jensen-Shannon divergence have significant limitations:

1. **Vanishing Gradients**: When the supports of two distributions do not overlap or have negligible overlap, these metrics may provide uninformative gradients for training.

2. **Mode Collapse**: In GANs, these metrics can lead to mode collapse, where the generator produces limited varieties of samples.

3. **Training Instability**: They often result in unstable training dynamics, making it difficult to achieve convergence.

### Advantages of Wasserstein Distance

The Wasserstein distance addresses these limitations:

1. **Meaningful Gradients**: It provides meaningful gradients even when distributions do not overlap, enabling more effective training.

2. **Correlation with Sample Quality**: The Wasserstein distance correlates well with the quality of generated samples, making it a useful metric for evaluating generative models.

3. **Stability in Training**: It leads to more stable training dynamics, especially in the context of GANs.

4. **Theoretical Guarantees**: It provides theoretical guarantees for convergence under certain conditions.

## Wasserstein GANs (WGANs)

Wasserstein GANs (WGANs), introduced by Arjovsky et al. in 2017, represent a significant advancement in generative adversarial networks by employing the Wasserstein distance as the training objective.

### From Standard GANs to WGANs

Standard GANs use a discriminator that distinguishes between real and fake samples, formulated as a binary classification problem. The objective function can be expressed as:

$$\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log(1 - D(G(z)))]$$

This objective implicitly minimizes the Jensen-Shannon divergence between the real and generated data distributions, which can lead to training instability and mode collapse.

WGANs, on the other hand, aim to minimize the Wasserstein distance between these distributions. The Kantorovich-Rubinstein duality allows us to reformulate the Wasserstein distance as:

$$W_1(P_r, P_g) = \sup_{||f||_L \leq 1} \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{x \sim P_g}[f(x)]$$

where the supremum is taken over all 1-Lipschitz functions f. This leads to the WGAN objective:

$$\min_G \max_D \mathbb{E}_{x \sim P_r}[D(x)] - \mathbb{E}_{z \sim P_z}[D(G(z))]$$

where D (often called the "critic" rather than the discriminator) approximates a 1-Lipschitz function.

### The Critic Function and Lipschitz Constraints

In WGANs, the discriminator is replaced by a critic function that estimates the Wasserstein distance between the real and generated distributions. Unlike the discriminator in standard GANs, the critic doesn't need to output probabilities, so no sigmoid activation is used in the final layer.

A key challenge in WGANs is enforcing the Lipschitz constraint on the critic function (||f||_L ≤ 1). The original WGAN paper proposed weight clipping to enforce this constraint:

```python
# Weight clipping in the critic
for param in critic.parameters():
    param.data.clamp_(-c, c)
```

where c is a small positive constant (e.g., 0.01). However, weight clipping can lead to capacity underuse and pathological behavior.

### Gradient Penalty

To address the limitations of weight clipping, Gulrajani et al. introduced WGAN with Gradient Penalty (WGAN-GP). Instead of explicit weight clipping, they add a penalty term to the objective function that constrains the gradient norm of the critic's output with respect to its input:

$$L_{\text{critic}} = \mathbb{E}_{\hat{x} \sim P_g}[D(\hat{x})] - \mathbb{E}_{x \sim P_r}[D(x)] + \lambda \mathbb{E}_{\tilde{x} \sim P_{\tilde{x}}}[(||\nabla_{\tilde{x}}D(\tilde{x})||_2 - 1)^2]$$

where λ is a hyperparameter (typically set to 10) and $\tilde{x}$ is sampled uniformly along straight lines between pairs of points from the real and generated distributions. This approach more effectively enforces the Lipschitz constraint and leads to more stable training.

## Implementing Wasserstein Distance

### Computing Wasserstein Distance in 1D

For one-dimensional distributions, the Wasserstein distance has a closed-form solution based on the inverse cumulative distribution functions (CDFs). Let's implement this for discrete samples:

```python
import numpy as np
from scipy import stats

def wasserstein_1d(x, y):
    """
    Compute the 1-Wasserstein distance between two 1D distributions.
    
    Args:
        x: Samples from the first distribution
        y: Samples from the second distribution
        
    Returns:
        The 1-Wasserstein distance
    """
    # Sort the samples
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    # Compute the absolute differences between sorted samples
    return np.mean(np.abs(x_sorted - y_sorted))

# Example usage
np.random.seed(42)
x = np.random.normal(0, 1, 1000)  # Samples from N(0, 1)
y = np.random.normal(2, 1.5, 1000)  # Samples from N(2, 1.5)

w_distance = wasserstein_1d(x, y)
print(f"Wasserstein distance between distributions: {w_distance:.4f}")
```

This implementation works well for one-dimensional distributions with equal sample sizes. For more general cases or higher dimensions, we can use optimal transport libraries like POT (Python Optimal Transport).

```python
import numpy as np
import matplotlib.pyplot as plt
import ot

# Generate 2D samples
np.random.seed(42)
n_samples = 100

# Distribution 1: samples from a 2D Gaussian
mean1 = [0, 0]
cov1 = [[1, 0], [0, 1]]
X = np.random.multivariate_normal(mean1, cov1, n_samples)

# Distribution 2: samples from another 2D Gaussian
mean2 = [3, 2]
cov2 = [[1.5, 0.5], [0.5, 1.5]]
Y = np.random.multivariate_normal(mean2, cov2, n_samples)

# Uniform weights for empirical distributions
a = np.ones(n_samples) / n_samples
b = np.ones(n_samples) / n_samples

# Compute cost matrix (squared Euclidean distance)
M = ot.dist(X, Y)
M /= M.max()  # Normalize for numerical stability

# Compute the Wasserstein distance
w_distance = ot.emd2(a, b, M)

# Compute the optimal transport plan
transport_plan = ot.emd(a, b, M)

# Visualize the distributions and transport plan
plt.figure(figsize=(10, 5))

# Plot distributions
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5, label='Distribution 1')
plt.scatter(Y[:, 0], Y[:, 1], c='red', alpha=0.5, label='Distribution 2')
plt.legend()
plt.title('Two 2D Distributions')
plt.grid(True)

# Plot a subset of the transport plan
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5)
plt.scatter(Y[:, 0], Y[:, 1], c='red', alpha=0.5)

# Plot transport plan (only show a subset for clarity)
for i in range(n_samples):
    for j in range(n_samples):
        if transport_plan[i, j] > 1e-3:  # Only show significant connections
            plt.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], 'k-', alpha=0.1*transport_plan[i, j]*n_samples)

plt.title(f'Transport Plan (W-distance: {w_distance:.4f})')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Wasserstein distance between 2D distributions: {w_distance:.4f}")
```

### Implementing WGAN with PyTorch

Let's implement a WGAN with weight clipping to train a generator on the MNIST dataset using PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Configuration
batch_size = 64
z_dim = 100
n_critic = 5  # Number of critic iterations per generator iteration
clip_value = 0.01  # Clipping parameter for weight clipping
lr = 0.00005  # Learning rate
n_epochs = 50

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

# Define Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),  # 28x28 = 784
            nn.Tanh()  # Output range: [-1, 1]
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# Define Critic (Discriminator)
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)  # No sigmoid, output unbounded value
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Initialize models
generator = Generator().to(device)
critic = Critic().to(device)

# Optimizers
optimizer_G = optim.RMSprop(generator.parameters(), lr=lr)
optimizer_C = optim.RMSprop(critic.parameters(), lr=lr)

# Training loop
generator_losses = []
critic_losses = []
img_list = []

for epoch in range(n_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        # Configure input
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        
        # ---------------------
        #  Train Critic
        # ---------------------
        optimizer_C.zero_grad()
        
        # Sample noise for generator
        z = torch.randn(batch_size, z_dim).to(device)
        
        # Generate fake images
        fake_imgs = generator(z).detach()
        
        # Compute critic loss
        real_validity = critic(real_imgs)
        fake_validity = critic(fake_imgs)
        critic_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
        
        # Backward and optimize
        critic_loss.backward()
        optimizer_C.step()
        
        # Clip weights of critic
        for p in critic.parameters():
            p.data.clamp_(-clip_value, clip_value)
        
        # Train generator every n_critic iterations
        if i % n_critic == 0:
            # ---------------------
            #  Train Generator
            # ---------------------
            optimizer_G.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = generator(z)
            
            # Compute generator loss
            fake_validity = critic(fake_imgs)
            generator_loss = -torch.mean(fake_validity)
            
            # Backward and optimize
            generator_loss.backward()
            optimizer_G.step()
            
            # Save losses for plotting
            generator_losses.append(generator_loss.item())
            critic_losses.append(critic_loss.item())
            
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[Critic loss: {critic_loss.item():.4f}] [Generator loss: {generator_loss.item():.4f}]")
                
                # Save generated images for visualization
                with torch.no_grad():
                    fake = generator(torch.randn(64, z_dim).to(device)).detach().cpu()
                img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True).numpy())

# Visualize results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(generator_losses, label='Generator')
plt.plot(critic_losses, label='Critic')
plt.legend()
plt.title('Losses')

plt.subplot(1, 2, 2)
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.title('Generated Images')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### Implementing WGAN-GP with TensorFlow

Now, let's implement a WGAN with gradient penalty (WGAN-GP) using TensorFlow and Keras:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration
batch_size = 64
z_dim = 100
n_critic = 5
lambda_gp = 10.0  # Gradient penalty lambda
lr = 0.0001
n_epochs = 30

# Load MNIST dataset
(x_train, _), (_, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 127.5 - 1.0  # Normalize to [-1, 1]
x_train = np.expand_dims(x_train, axis=3)

# Create dataset
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=60000).batch(batch_size)

# Define Generator
def build_generator():
    model = keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(z_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Reshape((7, 7, 256)),
        
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

# Define Critic
def build_critic():
    model = keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        
        layers.Flatten(),
        layers.Dense(1)  # No activation
    ])
    return model

# Initialize models
generator = build_generator()
critic = build_critic()

# Optimizers
generator_optimizer = keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
critic_optimizer = keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)

# Define gradient penalty function
def gradient_penalty(real_images, fake_images):
    batch_size = real_images.shape[0]
    
    # Create random interpolated points between real and fake images
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = real_images * alpha + fake_images * (1 - alpha)
    
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        critic_interpolated = critic(interpolated, training=True)
    
    # Calculate gradients w.r.t interpolated images
    grads = gp_tape.gradient(critic_interpolated, [interpolated])[0]
    
    # Calculate the norm of the gradients
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    
    return gp

# Training step for critic
@tf.function
def train_critic(real_images):
    batch_size = tf.shape(real_images)[0]
    
    # Generate random noise
    noise = tf.random.normal([batch_size, z_dim])
    
    with tf.GradientTape() as critic_tape:
        # Generate fake images
        fake_images = generator(noise, training=True)
        
        # Critic outputs
        real_output = critic(real_images, training=True)
        fake_output = critic(fake_images, training=True)
        
        # Wasserstein loss
        critic_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        
        # Gradient penalty
        gp = gradient_penalty(real_images, fake_images)
        
        # Total critic loss
        critic_loss = critic_loss + lambda_gp * gp
    
    # Calculate gradients and update critic
    critic_gradients = critic_tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))
    
    return critic_loss

# Training step for generator
@tf.function
def train_generator():
    batch_size = batch_size = 64
    
    # Generate random noise
    noise = tf.random.normal([batch_size, z_dim])
    
    with tf.GradientTape() as gen_tape:
        # Generate fake images
        fake_images = generator(noise, training=True)
        
        # Critic output on fake images
        fake_output = critic(fake_images, training=True)
        
        # Generator loss (negative of critic output)
        generator_loss = -tf.reduce_mean(fake_output)
    
    # Calculate gradients and update generator
    generator_gradients = gen_tape.gradient(generator_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    
    return generator_loss

# Training loop
generator_losses = []
critic_losses = []
generated_images = []

for epoch in range(n_epochs):
    start_time = time.time()
    
    for batch_idx, real_images in enumerate(train_dataset):
        # Train critic for n_critic iterations
        for _ in range(n_critic):
            c_loss = train_critic(real_images)
        
        # Train generator
        g_loss = train_generator()
        
        # Store losses
        if batch_idx % 10 == 0:
            critic_losses.append(c_loss.numpy())
            generator_losses.append(g_loss.numpy())
    
    # Generate and save images for visualization
    noise = tf.random.normal([16, z_dim])
    generated_batch = generator(noise, training=False)
    generated_images.append(generated_batch)
    
    # Print progress
    print(f"Epoch {epoch+1}/{n_epochs}, Time: {time.time()-start_time:.2f}s, "
          f"Critic Loss: {c_loss:.4f}, Generator Loss: {g_loss:.4f}")

# Visualize results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(generator_losses, label='Generator')
plt.plot(critic_losses, label='Critic')
plt.legend()
plt.title('Losses')

plt.subplot(1, 2, 2)
# Display the last generated images
img_grid = tf.concat([tf.concat([generated_images[-1][i*4+j] for j in range(4)], axis=1) 
                      for i in range(4)], axis=0)
img_grid = (img_grid + 1) / 2.0  # Rescale to [0, 1]
plt.imshow(img_grid.numpy().squeeze(), cmap='gray')
plt.axis('off')
plt.title('Generated Images')

plt.tight_layout()
plt.show()
```

## Advanced Topics

### Sliced Wasserstein Distance

The Sliced Wasserstein Distance (SWD) is a computationally efficient approximation of the Wasserstein distance. It works by projecting the high-dimensional distributions onto random one-dimensional lines and computing the average Wasserstein distance between these one-dimensional projections.

The SWD is particularly useful for high-dimensional data, as computing the exact Wasserstein distance in high dimensions can be computationally prohibitive. Here's a simple implementation:

```python
import numpy as np
from scipy import stats

def sliced_wasserstein_distance(X, Y, num_projections=50, seed=None):
    """
    Compute the approximate Sliced Wasserstein distance between two sets of samples.
    
    Args:
        X: Samples from the first distribution (n_samples x n_features)
        Y: Samples from the second distribution (n_samples x n_features)
        num_projections: Number of random projections to use
        seed: Random seed for reproducibility
        
    Returns:
        The approximate Sliced Wasserstein distance
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_features = X.shape[1]
    
    # Generate random projections
    projections = np.random.normal(size=(num_projections, n_features))
    projections = projections / np.sqrt(np.sum(projections**2, axis=1, keepdims=True))
    
    # Project the data
    X_projections = np.dot(X, projections.T)
    Y_projections = np.dot(Y, projections.T)
    
    # Compute Wasserstein distance for each projection
    wasserstein_distances = np.zeros(num_projections)
    for i in range(num_projections):
        wasserstein_distances[i] = stats.wasserstein_distance(
            X_projections[:, i], Y_projections[:, i])
    
    # Return the mean Wasserstein distance
    return np.mean(wasserstein_distances)

# Example usage
np.random.seed(42)
X = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=1000)
Y = np.random.multivariate_normal(mean=[2, 2], cov=[[1, 0.5], [0.5, 1]], size=1000)

swd = sliced_wasserstein_distance(X, Y, num_projections=100, seed=42)
print(f"Sliced Wasserstein distance: {swd:.4f}")
```

### Other Applications in Deep Learning

The Wasserstein distance has found applications in various areas of deep learning beyond GANs:

1. **Domain Adaptation**: The Wasserstein distance can be used to align feature distributions from source and target domains, enabling more effective transfer learning.

2. **Image-to-Image Translation**: Models like CycleGAN can benefit from Wasserstein distance to better preserve the structural properties of translated images.

3. **Reinforcement Learning**: In distributional reinforcement learning, the Wasserstein distance provides a natural metric for comparing value distributions.

4. **Variational Autoencoders (VAEs)**: Wasserstein VAEs use the Wasserstein distance to measure the discrepancy between the prior and the encoded distributions, leading to improved sample quality.

5. **Text Generation**: The Wasserstein distance has been used in text generation models to better capture the discrete nature of text data.

## Conclusion and Further Reading

This tutorial has provided a comprehensive overview of the Wasserstein distance and its applications in deep learning, particularly focusing on Wasserstein GANs. We covered the mathematical foundations, intuition, implementation details, and advanced topics related to the Wasserstein distance.

The Wasserstein distance offers significant advantages over traditional metrics in the context of generative modeling, including more stable training, meaningful gradients even for non-overlapping distributions, and better correlation with sample quality.

For further exploration, here are some key papers and resources:

1. Arjovsky, M., Chintala, S., & Bottou, L. (2017). [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
2. Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
3. Kolouri, S., Park, S. R., Thorpe, M., Slepcev, D., & Rohde, G. K. (2017). [Optimal Mass Transport: Signal Processing and Machine-Learning Applications](https://ieeexplore.ieee.org/document/7974883)
4. Peyré, G., & Cuturi, M. (2019). [Computational Optimal Transport](https://arxiv.org/abs/1803.00567)
5. [Python Optimal Transport Library](https://pythonot.github.io/)

By understanding and leveraging the Wasserstein distance, researchers and practitioners can develop more robust and effective deep learning models, particularly in the field of generative modeling.
