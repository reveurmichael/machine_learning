# Neural Networks from Scratch with Python and NumPy  
## Session 2: Backpropagation, SGD, and Adam with MNIST

### Introduction

In this session, we will extend our previous work on neural networks from scratch by focusing on how a network learns. We start with an in-depth exploration of **Backpropagation (BP)**, which is the fundamental algorithm used to compute gradients through the network, and explain intuitively why and how it works.

After exploring backpropagation, we will discuss **Stochastic Gradient Descent (SGD)**, an optimization method that updates the network's weights in small batches. Finally, we include an overview of the **Adam optimizer**, which is a more advanced optimization algorithm, first proposed in 2014, that builds on SGD but automates the learning rate adaptation for each parameter. Although we cover Adam less extensively, you will see how easy it is to swap it into our training loop.

We'll work with the classic **MNIST** dataset of handwritten digits and implement a neural network from scratch to classify these digits. To maintain our "from scratch" philosophy, we'll implement everything using only NumPy, with minimal dependencies.

Make sure you have watched the 3b1b videos before doing the code session!

Related videos:
- Video - Chinese version: https://www.bilibili.com/video/BV1uW4y1s7Ci
- Video - English version: https://www.bilibili.com/video/BV1nM41167j9

### Session Overview

In this session we will cover:
1. **Backpropagation in Detail**  
   - Intuitive explanation of how gradients flow backward.
   - A step-by-step breakdown of the chain rule in the context of neural networks.
   - Code-level implementation of BP using our custom layer classes.

2. **Optimization Techniques**  
   - **Stochastic Gradient Descent (SGD):**  
     How SGD works for mini-batch training and weight updates.
   - **Adam Optimizer:**  
     A brief explanation and implementation details that show how Adam extends SGD with adaptive learning rates.

3. **Dataset: MNIST**  
   - How to load and preprocess MNIST from sklearn.
   - Converting MNIST images into a format suitable for our neural network (flattened arrays and normalization).

4. **Training a Neural Network on MNIST**  
   - Integrating backpropagation with both SGD and Adam.
   - Visualizing training dynamics, losses, and accuracy.

### 1. Backpropagation: Intuition and Implementation

#### 1.1. Intuitive Overview

Backpropagation is the process by which gradients of a loss function are computed with respect to each parameter in the network by applying the chain rule of calculus. The core idea behind BP is:
- **Forward Pass:**  
  Compute the output of the network by propagating the input forward through each layer.
- **Loss Computation:**  
  Calculate the difference between the prediction and the true target.
- **Backward Pass:**  
  Propagate the gradient of the loss back through the network (in reverse order) to compute each parameter's gradient.
- **Parameter Update:**  
  Use the computed gradients to adjust the network's parameters (weights and biases) in order to minimize the loss.

The chain rule enables us to express the derivative of the loss with respect to any parameter as the product of derivatives from subsequent layers.

#### 1.2. Code Implementation of Backpropagation

Below is our implementation of the forward and backward passes. The code builds on the modular layer classes from session one (i.e., Dense, ReLU, etc.) and extends them for optimization.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from tqdm import tqdm  # For progress bars

# ---------------------------
# Layer Base Class
# ---------------------------
class Layer:
    """
    Base class for neural network layers.
    Defines the interface for forward and backward propagation.
    """
    def __init__(self):
        pass

    def forward(self, input):
        """
        Forward pass: Compute the output given the input.
        """
        return input

    def backward(self, input, grad_output):
        """
        Backward pass: Compute gradients and propagate them backward.
        """
        return grad_output

# ---------------------------
# ReLU Activation Layer
# ---------------------------
class ReLU(Layer):
    """
    Implements the Rectified Linear Unit (ReLU) activation function.
    Forward: output = max(0, input)
    Backward: gradient is 1 for input > 0, else 0.
    """
    def forward(self, input):
        self.input = input  # Store for backward pass
        return np.maximum(0, input)
    
    def backward(self, input, grad_output):
        relu_grad = (input > 0).astype(float)
        return grad_output * relu_grad

# ---------------------------
# Dense (Fully Connected) Layer
# ---------------------------
class Dense(Layer):
    """
    Implements a fully connected (dense) layer.
    Forward: output = input.dot(weights) + biases
    """
    def __init__(self, input_units, output_units, learning_rate=0.1, optimizer=None):
        self.learning_rate = learning_rate
        # Xavier/Glorot initialization for weights
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(2.0 / input_units)
        self.biases = np.zeros(output_units)
        self.optimizer = optimizer  # We'll update parameters using the optimizer
    
    def forward(self, input):
        self.input = input  # Cache input for backpropagation
        return np.dot(input, self.weights) + self.biases
    
    def backward(self, input, grad_output):
        # Compute gradients with respect to parameters
        grad_weights = np.dot(input.T, grad_output) / input.shape[0]
        grad_biases = np.mean(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weights.T)
        
        # If an optimizer is provided, use it to update parameters
        if self.optimizer is None:
            # Basic SGD update
            self.weights -= self.learning_rate * grad_weights
            self.biases -= self.learning_rate * grad_biases
        else:
            # Use optimizer's update method (more on this below)
            self.weights, self.biases = self.optimizer.update(self.weights, self.biases, grad_weights, grad_biases)
        
        return grad_input

# ---------------------------
# Loss Function: Softmax Cross-Entropy
# ---------------------------
def softmax_crossentropy_with_logits(logits, labels):
    """
    Computes softmax cross-entropy loss and gradient.
    Arguments:
      logits -- raw predictions from the network, shape (batch_size, num_classes)
      labels -- true labels, shape (batch_size,)
    Returns:
      loss -- scalar value for cross-entropy loss
      grad -- gradient of loss with respect to logits
    """
    batch_size = logits.shape[0]
    # One-hot encoding for labels
    one_hot_labels = np.zeros_like(logits)
    one_hot_labels[np.arange(batch_size), labels] = 1
    
    # Softmax computation with stability trick
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    loss = -np.sum(one_hot_labels * np.log(softmax_probs + 1e-9)) / batch_size
    grad = (softmax_probs - one_hot_labels) / batch_size
    
    return loss, grad

# ---------------------------
# Forward and Backward for the Network
# ---------------------------
def forward(network, X):
    """
    Computes forward pass through the entire network.
    Returns a list of activations for each layer.
    """
    activations = []
    input = X
    for layer in network:
        output = layer.forward(input)
        activations.append(output)
        input = output
    return activations

def predict(network, X):
    """
    Get class predictions for input X.
    """
    logits = forward(network, X)[-1]
    return np.argmax(logits, axis=-1)

def train_batch(network, X, y):
    """
    Train the network on a single batch using backpropagation.
    """
    # Forward pass
    activations = forward(network, X)
    layer_inputs = [X] + activations[:-1]
    logits = activations[-1]
    
    # Compute loss and initial gradient using our loss function
    loss, grad_logits = softmax_crossentropy_with_logits(logits, y)
    
    # Backward pass - traverse layers in reverse order
    grad = grad_logits
    for i in range(len(network) - 1, -1, -1):
        layer = network[i]
        grad = layer.backward(layer_inputs[i], grad)
    
    return loss
```

### 2. Optimization Techniques

Optimization algorithms determine how the network's parameters are updated based on the computed gradients. We describe and implement two optimizers: SGD and Adam.

#### 2.1. Stochastic Gradient Descent (SGD)

Traditional gradient descent computes the gradient of the loss function using the entire dataset, which can be computationally expensive and slow, especially for large datasets. In contrast, **Stochastic Gradient Descent (SGD)** updates the model parameters using only a small random subset (mini-batch) of the data at each step. 

**Benefits of SGD:**
- **Faster Updates:** By using mini-batches, SGD can update the model parameters more frequently, leading to faster convergence.
- **Noise Helps Escape Local Minima:** The inherent noise in the updates (due to using only a subset of data) can help the model escape local minima and explore the loss landscape more effectively.
- **Better Generalization:** The variability introduced by mini-batch updates can lead to better generalization on unseen data.

The update rule for SGD is:

$$\theta = \theta - \alpha \frac{\partial L}{\partial \theta}$$

where:  
- \(\theta\) represents a weight or bias,
- \(\alpha\) is the learning rate, and
- \(\frac{\partial L}{\partial \theta}\) is the gradient for that parameter.

#### 2.2. Adam Optimizer

**Adam** (Adaptive Moment Estimation) combines the advantages of two other extensions of SGD: momentum and RMSProp. It maintains two moving averages for each parameter: one for the gradients (first moment) and one for the squared gradients (second moment). 

**Benefits of Adam:**
- **Adaptive Learning Rates:** Adam adjusts the learning rate for each parameter individually based on the historical gradients, allowing for more efficient training.
- **Faster Convergence:** By using both momentum and adaptive learning rates, Adam often converges faster than SGD, especially in complex problems.
- **Less Tuning Required:** Adam typically requires less tuning of the learning rate compared to SGD, making it easier to use in practice.

The update rules for Adam are:

1. **First Moment Estimate (mean):**  
   \(m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t\)

2. **Second Moment Estimate (uncentered variance):**  
   \(v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2\)

3. **Bias-Corrected Estimates:**  
   \(\hat{m}_t = \frac{m_t}{1 - \beta_1^t}\)  
   \(\hat{v}_t = \frac{v_t}{1 - \beta_2^t}\)

4. **Parameter Update:**  
   \(\theta = \theta - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\)

Below is an implementation of a simple Adam optimizer class that can be attached to our Dense layers.

```python
# ---------------------------
# Optimizer: SGD and Adam Classes
# ---------------------------
class SGD:
    """
    Stochastic Gradient Descent optimizer.
    """
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update(self, weights, biases, grad_weights, grad_biases):
        weights_updated = weights - self.learning_rate * grad_weights
        biases_updated = biases - self.learning_rate * grad_biases
        return weights_updated, biases_updated

class Adam:
    """
    Adam optimizer implementation.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # State dictionaries for first and second moment estimates
        self.m_weights = {}
        self.v_weights = {}
        self.m_biases = {}
        self.v_biases = {}
        self.t = 0

    def update(self, weights, biases, grad_weights, grad_biases):
        self.t += 1
        
        # Identify parameters uniquely via id() (for multiple layers)
        key_w, key_b = id(weights), id(biases)
        
        if key_w not in self.m_weights:
            # Initialize moment estimates with zeros, same shape as parameters
            self.m_weights[key_w] = np.zeros_like(grad_weights)
            self.v_weights[key_w] = np.zeros_like(grad_weights)
            self.m_biases[key_b] = np.zeros_like(grad_biases)
            self.v_biases[key_b] = np.zeros_like(grad_biases)
        
        # Update biased first moment estimate for weights and biases
        self.m_weights[key_w] = self.beta1 * self.m_weights[key_w] + (1 - self.beta1) * grad_weights
        self.m_biases[key_b] = self.beta1 * self.m_biases[key_b] + (1 - self.beta1) * grad_biases
        
        # Update biased second moment estimate for weights and biases
        self.v_weights[key_w] = self.beta2 * self.v_weights[key_w] + (1 - self.beta2) * (grad_weights ** 2)
        self.v_biases[key_b] = self.beta2 * self.v_biases[key_b] + (1 - self.beta2) * (grad_biases ** 2)
        
        # Bias-corrected estimates
        m_hat_weights = self.m_weights[key_w] / (1 - self.beta1 ** self.t)
        v_hat_weights = self.v_weights[key_w] / (1 - self.beta2 ** self.t)
        m_hat_biases = self.m_biases[key_b] / (1 - self.beta1 ** self.t)
        v_hat_biases = self.v_biases[key_b] / (1 - self.beta2 ** self.t)
        
        # Update parameters
        weights_updated = weights - self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        biases_updated = biases - self.learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)
        
        return weights_updated, biases_updated
```

> **Key Point:**  
> Both SGD and Adam share the same interface – an `update` method that accepts current parameters and gradients, and returns updated values. This design makes it easy to swap optimizers in our training loop.


### 3. MNIST Dataset Preparation

In this session, we'll be using the MNIST dataset, a classic benchmark in machine learning. MNIST contains 70,000 grayscale images of handwritten digits (28x28 pixels), with 60,000 training examples and 10,000 test examples.

We'll load the dataset using sklearn's `fetch_openml` function and implement preprocessing steps to prepare it for our neural network:

```python
def load_mnist():
    """
    Load the MNIST dataset from sklearn
    
    Returns:
    X_train -- training images, shape (n_samples, 28, 28)
    y_train -- training labels, shape (n_samples,)
    X_test -- test images, shape (n_samples, 28, 28)
    y_test -- test labels, shape (n_samples,)
    """
    print("Loading MNIST dataset from sklearn...")
    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    
    # Convert pandas DataFrame to numpy arrays
    X = X.to_numpy().astype('float32')
    y = y.to_numpy().astype('int')
    
    # Reshape to (samples, 28, 28)
    X = X.reshape(-1, 28, 28)
    
    # Split the data into train and test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    # Normalize the data (scale pixel values to [0, 1])
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

def visualize_mnist_samples(X, y, num_samples=10):
    """Visualize sample images from the MNIST dataset"""
    indices = np.random.choice(len(X), size=num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        # MNIST images are grayscale
        axes[i].imshow(X[idx], cmap='gray')
        axes[i].set_title(f"Label: {y[idx]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
```

After loading the dataset, we need to preprocess it to suit our neural network. This involves:
- Flattening the 2D images (28×28) into 1D vectors (784)
- Normalizing the data to have zero mean and unit variance
- Creating validation and training splits

```python
def preprocess_mnist(X_train, y_train, X_test, y_test):
    """Preprocess MNIST data for our neural network"""
    # Flatten the images from (N, 28, 28) to (N, 784)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print(f"Flattened training data shape: {X_train_flat.shape}")
    print(f"Flattened test data shape: {X_test_flat.shape}")
    
    # Normalize data to have zero mean and unit variance
    # Compute mean and std on training data
    mean = np.mean(X_train_flat, axis=0)
    std = np.std(X_train_flat, axis=0) + 1e-9  # Add small constant to avoid division by zero
    
    X_train_normalized = (X_train_flat - mean) / std
    X_test_normalized = (X_test_flat - mean) / std
    
    # Create a validation set from training data
    val_size = 5000
    val_indices = np.random.choice(len(X_train_normalized), val_size, replace=False)
    
    X_val = X_train_normalized[val_indices]
    y_val = y_train[val_indices]
    
    # Remove validation samples from training set
    train_mask = np.ones(len(X_train_normalized), dtype=bool)
    train_mask[val_indices] = False
    X_train_final = X_train_normalized[train_mask]
    y_train_final = y_train[train_mask]
    
    return X_train_final, y_train_final, X_val, y_val, X_test_normalized, y_test
```

Let's load and preprocess the data:

```python
# Load MNIST dataset
X_train, y_train, X_test, y_test = load_mnist()

# Visualize some samples
visualize_mnist_samples(X_train, y_train)

# Preprocess the data
X_train, y_train, X_val, y_val, X_test, y_test = preprocess_mnist(X_train, y_train, X_test, y_test)
```


### 4. Building and Training the Neural Network with Backpropagation & Optimizers

Now, we integrate all pieces together to build a neural network that uses backpropagation and either SGD or Adam for weight updates. We'll construct a multi-layer network with three hidden layers and train it on MNIST.

#### 4.1. Creating the Network

We set up the network using our Dense and ReLU layers. This time, we pass an optimizer instance (either SGD or Adam) to the Dense layers so that parameter updates are performed through the chosen method.

```python
def create_network(optimizer_type='sgd'):
    """
    Creates a neural network with three hidden layers using the specified optimizer.
    Arguments:
        optimizer_type -- 'sgd' or 'adam'
    Returns:
        network -- list of layers forming the network.
    """
    if optimizer_type == 'sgd':
        optimizer = SGD(learning_rate=0.05)
    elif optimizer_type == 'adam':
        optimizer = Adam(learning_rate=0.0005, beta1=0.9, beta2=0.999)
    else:
        raise ValueError("Unsupported optimizer type. Use 'sgd' or 'adam'.")
    
    network = [
        Dense(input_units=784, output_units=256, learning_rate=0.03, optimizer=optimizer),
        ReLU(),
        Dense(input_units=256, output_units=128, learning_rate=0.03, optimizer=optimizer),
        ReLU(),
        Dense(input_units=128, output_units=64, learning_rate=0.03, optimizer=optimizer),
        ReLU(),
        Dense(input_units=64, output_units=10, learning_rate=0.03, optimizer=optimizer)
    ]
    return network
```

#### 4.2. Training Loop and Evaluation

Our training loop processes the data in mini-batches, calls `train_batch` for each batch, and reports the loss periodically. After each epoch, we evaluate accuracy on a validation subset.

```python
def train_network(network, X_train, y_train, X_val, y_val, num_epochs=10, batch_size=64):
    """
    Trains the neural network on training data and periodically evaluates on validation data.
    """
    num_samples = X_train.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    history = {'loss': [], 'val_accuracy': []}
    
    for epoch in range(num_epochs):
        # Shuffle training data at each epoch
        indices = np.random.permutation(num_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_losses = []
        for batch in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
            start = batch * batch_size
            end = min(start + batch_size, num_samples)
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]
            
            loss = train_batch(network, X_batch, y_batch)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        history['loss'].append(avg_loss)
        print(f"Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}")
        
        # Evaluate validation accuracy on a subset (for speed)
        val_indices = np.random.choice(X_val.shape[0], size=1000, replace=False)
        val_preds = predict(network, X_val[val_indices])
        val_acc = np.mean(val_preds == y_val[val_indices])
        history['val_accuracy'].append(val_acc)
        print(f"Validation Accuracy: {val_acc*100:.2f}%\n")
    
    return history

def evaluate_network(network, X_test, y_test):
    """
    Evaluate and print accuracy on the test set.
    """
    test_preds = predict(network, X_test)
    accuracy = np.mean(test_preds == y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Create a confusion matrix
    conf_matrix = np.zeros((10, 10), dtype=int)
    for i in range(len(y_test)):
        conf_matrix[y_test[i], test_preds[i]] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap='Blues')
    plt.colorbar()
    
    # Add labels
    plt.xticks(np.arange(10), np.arange(10))
    plt.yticks(np.arange(10), np.arange(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Add text annotations
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(conf_matrix[i, j]), 
                   ha="center", va="center", 
                   color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
    
    plt.tight_layout()
    plt.show()
    
    return accuracy
```

#### 4.3. Running Training with Both Optimizers

We'll train two models - one with Adam and one with SGD - to compare their performance:

```python
# Use a subset of data for quicker training during demo
subset_size = 20000
indices = np.random.choice(X_train.shape[0], subset_size, replace=False)
X_train_subset = X_train[indices]
y_train_subset = y_train[indices]

# Create a validation set from the training data
val_size = 1000
val_indices = np.random.choice(X_val.shape[0], val_size, replace=False)
X_val_subset = X_val[val_indices]
y_val_subset = y_val[val_indices]

# Create network using Adam optimizer
network_adam = create_network(optimizer_type='adam')
print("Training with Adam optimizer...")
history_adam = train_network(
    network_adam,
    X_train_subset,
    y_train_subset,
    X_val_subset,
    y_val_subset,
    num_epochs=20,
    batch_size=64
)

# Create network using SGD optimizer for comparison
network_sgd = create_network(optimizer_type='sgd')
print("\nTraining with SGD optimizer...")
history_sgd = train_network(
    network_sgd,
    X_train_subset,
    y_train_subset,
    X_val_subset,
    y_val_subset,
    num_epochs=20,
    batch_size=64
)

# Evaluate both models on test data
print("\nEvaluating Adam model...")
adam_accuracy = evaluate_network(network_adam, X_test, y_test)

print("\nEvaluating SGD model...")
sgd_accuracy = evaluate_network(network_sgd, X_test, y_test)

print(f"\nAdam accuracy: {adam_accuracy*100:.2f}%")
print(f"SGD accuracy: {sgd_accuracy*100:.2f}%")
```

#### 4.4. Visualizing Training History and Comparing Optimizers

Plotting the training loss and validation accuracy across epochs helps us compare the performance of SGD and Adam.

```python
def plot_training_comparison(history_adam, history_sgd):
    """Compare training performance between Adam and SGD optimizers"""
    epochs = range(1, len(history_adam['loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_adam['loss'], marker='o', label='Adam')
    plt.plot(epochs, history_sgd['loss'], marker='s', label='SGD')
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_adam['val_accuracy'], marker='o', label='Adam')
    plt.plot(epochs, history_sgd['val_accuracy'], marker='s', label='SGD')
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_comparison(history_adam, history_sgd)
```

#### 4.5. Visualizing Predictions

Finally, let's visualize some example images along with their predictions:

```python
def visualize_predictions(network, X, y, num_samples=10):
    """Visualize predictions on sample images"""
    # Select random samples
    indices = np.random.choice(len(X), size=num_samples, replace=False)
    X_samples = X[indices]
    y_samples = y[indices]
    
    # Get predictions
    predictions = predict(network, X_samples)
    
    # Plot
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i, idx in enumerate(range(num_samples)):
        # Reshape for visualization (to 28x28)
        img = X_samples[idx].reshape(28, 28)
        
        # Plot original image
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f"True: {y_samples[idx]}")
        axes[0, i].axis('off')
        
        # Add prediction with color coding (green for correct, red for incorrect)
        color = 'green' if predictions[idx] == y_samples[idx] else 'red'
        axes[1, i].text(0.5, 0.5, f"Pred: {predictions[idx]}", 
                      horizontalalignment='center',
                      verticalalignment='center',
                      color=color, fontsize=12)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize predictions using the better model
better_network = network_adam if adam_accuracy > sgd_accuracy else network_sgd
visualize_predictions(better_network, X_test, y_test)
```

### 5. Conclusion

In this session we have:

1. **Deep Dived into Backpropagation:**  
   - Explained the chain rule and its role in computing gradients.
   - Illustrated how gradients are propagated from the output back through each layer using our custom forward/backward interface.

2. **Explored Optimization Methods:**  
   - Implemented both basic SGD and a more sophisticated Adam optimizer.
   - Demonstrated how a consistent update interface in our layer classes allows effortless switching between optimizers.

3. **Worked with the MNIST Dataset:**  
   - Loaded MNIST from sklearn and preprocessed it for neural network training.
   - Visualized the dataset and model predictions.

4. **Integrated it All in a Training Pipeline:**  
   - Built a complete training loop, computed loss, updated parameters via backpropagation, and periodically evaluated accuracy.
   - Compared the performance of SGD and Adam optimizers on the same task.
   - Visualized training progress and model predictions.
   
5. **Results:**
   - Our architecture with three hidden layers (256→128→64→10) performs well on MNIST.
   - Adam optimizer typically achieves around 95-97% accuracy on the test set.
   - SGD optimizer usually reaches ~90-95% accuracy but converges more slowly.
   - With just 20 epochs of training on a subset of the data, we achieve good performance, showing the power of these optimization techniques.

