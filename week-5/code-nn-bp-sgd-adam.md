# Neural Networks: Backpropagation, SGD, and Adam Optimizer

## Introduction

In our previous session, we built a basic neural network from scratch using Python and NumPy. We implemented forward propagation and a simple update rule, but we didn't dive deeply into the inner workings of the backpropagation algorithm or explore different optimization techniques.

In this session, we'll take a closer look at:

1. **Backpropagation**: The mathematical foundation and intuition behind this critical algorithm
2. **Stochastic Gradient Descent (SGD)**: Why mini-batches matter and how SGD improves training
3. **Adam Optimizer**: A more advanced optimization algorithm that combines the benefits of momentum and adaptive learning rates

We'll also transition from the MNIST dataset to the more challenging CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 different classes.

Let's start by importing the necessary libraries:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import time
from tqdm.notebook import tqdm

# For dataset
import pickle
import os
import urllib.request
import tarfile
```

## Visualizing Neural Networks as Computational Graphs

To understand backpropagation, it's helpful to view neural networks as computational graphs where information flows forward during prediction and gradients flow backward during training.

```python
def plot_computational_graph():
    """Visualize a neural network as a computational graph"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up the axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Input layer (X)
    ax.add_patch(Rectangle((1, 5.5), 0.5, 0.5, fc='skyblue', ec='black'))
    ax.text(1.25, 5.75, "x₁", ha='center', va='center')
    
    ax.add_patch(Rectangle((1, 5), 0.5, 0.5, fc='skyblue', ec='black'))
    ax.text(1.25, 5.25, "x₂", ha='center', va='center')
    
    ax.add_patch(Rectangle((1, 4.5), 0.5, 0.5, fc='skyblue', ec='black'))
    ax.text(1.25, 4.75, "x₃", ha='center', va='center')
    
    # First linear transformation (W1·X + b1)
    ax.add_patch(Rectangle((3, 5.5), 0.5, 0.5, fc='lightgreen', ec='black'))
    ax.text(3.25, 5.75, "W₁·X + b₁", ha='center', va='center', fontsize=8)
    
    ax.add_patch(Rectangle((3, 5), 0.5, 0.5, fc='lightgreen', ec='black'))
    ax.text(3.25, 5.25, "⋮", ha='center', va='center', fontsize=10)
    
    ax.add_patch(Rectangle((3, 4.5), 0.5, 0.5, fc='lightgreen', ec='black'))
    ax.text(3.25, 4.75, "W₁·X + b₁", ha='center', va='center', fontsize=8)
    
    # First activation (ReLU)
    ax.add_patch(Rectangle((5, 5.5), 0.5, 0.5, fc='gold', ec='black'))
    ax.text(5.25, 5.75, "ReLU", ha='center', va='center', fontsize=8)
    
    ax.add_patch(Rectangle((5, 5), 0.5, 0.5, fc='gold', ec='black'))
    ax.text(5.25, 5.25, "⋮", ha='center', va='center', fontsize=10)
    
    ax.add_patch(Rectangle((5, 4.5), 0.5, 0.5, fc='gold', ec='black'))
    ax.text(5.25, 4.75, "ReLU", ha='center', va='center', fontsize=8)
    
    # Second linear transformation (W2·A1 + b2)
    ax.add_patch(Rectangle((7, 5), 0.5, 0.5, fc='lightgreen', ec='black'))
    ax.text(7.25, 5.25, "W₂·A₁ + b₂", ha='center', va='center', fontsize=8)
    
    # Loss function
    ax.add_patch(Rectangle((9, 5), 0.5, 0.5, fc='indianred', ec='black'))
    ax.text(9.25, 5.25, "Loss", ha='center', va='center', fontsize=8)
    
    # Arrows for forward pass
    arrow_style = "simple,head_width=0.15,head_length=0.2"
    
    # Input to first layer
    ax.add_patch(FancyArrowPatch((1.5, 5.75), (3, 5.75), 
                                 connectionstyle="arc3,rad=0", 
                                 arrowstyle=arrow_style, color='blue'))
    ax.add_patch(FancyArrowPatch((1.5, 5.25), (3, 5.25), 
                                 connectionstyle="arc3,rad=0", 
                                 arrowstyle=arrow_style, color='blue'))
    ax.add_patch(FancyArrowPatch((1.5, 4.75), (3, 4.75), 
                                 connectionstyle="arc3,rad=0", 
                                 arrowstyle=arrow_style, color='blue'))
    
    # First layer to activation
    ax.add_patch(FancyArrowPatch((3.5, 5.75), (5, 5.75), 
                                 connectionstyle="arc3,rad=0", 
                                 arrowstyle=arrow_style, color='blue'))
    ax.add_patch(FancyArrowPatch((3.5, 5.25), (5, 5.25), 
                                 connectionstyle="arc3,rad=0", 
                                 arrowstyle=arrow_style, color='blue'))
    ax.add_patch(FancyArrowPatch((3.5, 4.75), (5, 4.75), 
                                 connectionstyle="arc3,rad=0", 
                                 arrowstyle=arrow_style, color='blue'))
    
    # Activation to second layer
    ax.add_patch(FancyArrowPatch((5.5, 5.75), (7, 5.25), 
                                 connectionstyle="arc3,rad=0.2", 
                                 arrowstyle=arrow_style, color='blue'))
    ax.add_patch(FancyArrowPatch((5.5, 5.25), (7, 5.25), 
                                 connectionstyle="arc3,rad=0", 
                                 arrowstyle=arrow_style, color='blue'))
    ax.add_patch(FancyArrowPatch((5.5, 4.75), (7, 5.25), 
                                 connectionstyle="arc3,rad=-0.2", 
                                 arrowstyle=arrow_style, color='blue'))
    
    # Second layer to loss
    ax.add_patch(FancyArrowPatch((7.5, 5.25), (9, 5.25), 
                                 connectionstyle="arc3,rad=0", 
                                 arrowstyle=arrow_style, color='blue'))
    
    # Arrows for backward pass
    # Loss to second layer
    ax.add_patch(FancyArrowPatch((9, 5.15), (7.5, 5.15), 
                                 connectionstyle="arc3,rad=0", 
                                 arrowstyle=arrow_style, color='red', linestyle='--'))
    ax.text(8.25, 5, "∂L/∂z₂", color='red', fontsize=8)
    
    # Second layer to activation
    ax.add_patch(FancyArrowPatch((7, 5.15), (5.5, 5.65), 
                                 connectionstyle="arc3,rad=-0.2", 
                                 arrowstyle=arrow_style, color='red', linestyle='--'))
    ax.add_patch(FancyArrowPatch((7, 5.15), (5.5, 5.15), 
                                 connectionstyle="arc3,rad=0", 
                                 arrowstyle=arrow_style, color='red', linestyle='--'))
    ax.add_patch(FancyArrowPatch((7, 5.15), (5.5, 4.65), 
                                 connectionstyle="arc3,rad=0.2", 
                                 arrowstyle=arrow_style, color='red', linestyle='--'))
    ax.text(6.25, 5.5, "∂L/∂a₁", color='red', fontsize=8)
    
    # Activation to first layer
    ax.add_patch(FancyArrowPatch((5, 5.65), (3.5, 5.65), 
                                 connectionstyle="arc3,rad=0", 
                                 arrowstyle=arrow_style, color='red', linestyle='--'))
    ax.add_patch(FancyArrowPatch((5, 5.15), (3.5, 5.15), 
                                 connectionstyle="arc3,rad=0", 
                                 arrowstyle=arrow_style, color='red', linestyle='--'))
    ax.add_patch(FancyArrowPatch((5, 4.65), (3.5, 4.65), 
                                 connectionstyle="arc3,rad=0", 
                                 arrowstyle=arrow_style, color='red', linestyle='--'))
    ax.text(4.25, 5.5, "∂L/∂z₁", color='red', fontsize=8)
    
    # Legend
    ax.add_patch(Rectangle((1, 2), 0.5, 0.5, fc='skyblue', ec='black'))
    ax.text(1.6, 2.25, "Input Values", ha='left', va='center')
    
    ax.add_patch(Rectangle((1, 1.5), 0.5, 0.5, fc='lightgreen', ec='black'))
    ax.text(1.6, 1.75, "Linear Transformation", ha='left', va='center')
    
    ax.add_patch(Rectangle((1, 1), 0.5, 0.5, fc='gold', ec='black'))
    ax.text(1.6, 1.25, "Activation Function", ha='left', va='center')
    
    ax.add_patch(Rectangle((1, 0.5), 0.5, 0.5, fc='indianred', ec='black'))
    ax.text(1.6, 0.75, "Loss Function", ha='left', va='center')
    
    # Forward/backward arrows for legend
    ax.add_patch(FancyArrowPatch((5, 2.25), (6, 2.25), 
                                 connectionstyle="arc3,rad=0", 
                                 arrowstyle=arrow_style, color='blue'))
    ax.text(6.2, 2.25, "Forward Pass", ha='left', va='center', color='blue')
    
    ax.add_patch(FancyArrowPatch((5, 1.75), (6, 1.75), 
                                 connectionstyle="arc3,rad=0", 
                                 arrowstyle=arrow_style, color='red', linestyle='--'))
    ax.text(6.2, 1.75, "Backward Pass (Gradients)", ha='left', va='center', color='red')
    
    # Title
    ax.set_title('Neural Network as a Computational Graph', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.show()
```

Let's visualize this computational graph:

```python
plot_computational_graph()
```

## The Mathematics of Backpropagation

Backpropagation leverages the chain rule from calculus, which states that if we have a composite function like f(g(x)), then:

$\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$

In a neural network, we compute the derivative of the loss with respect to each parameter by working backward from the output. Let's illustrate this with a simple example:

Consider a 2-layer neural network:
- Input: $X$
- Hidden layer: $Z^{[1]} = W^{[1]}X + b^{[1]}$, $A^{[1]} = \text{ReLU}(Z^{[1]})$
- Output layer: $Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$, $\hat{Y} = \text{softmax}(Z^{[2]})$
- Loss: $L = \text{cross_entropy}(\hat{Y}, Y)$

To update $W^{[1]}$ using gradient descent, we need to compute $\frac{\partial L}{\partial W^{[1]}}$. Using the chain rule:

$\frac{\partial L}{\partial W^{[1]}} = \frac{\partial L}{\partial Z^{[2]}} \cdot \frac{\partial Z^{[2]}}{\partial A^{[1]}} \cdot \frac{\partial A^{[1]}}{\partial Z^{[1]}} \cdot \frac{\partial Z^{[1]}}{\partial W^{[1]}}$

Let's break this down:

1. $\frac{\partial L}{\partial Z^{[2]}} = \hat{Y} - Y$ (for cross-entropy loss with softmax)
2. $\frac{\partial Z^{[2]}}{\partial A^{[1]}} = W^{[2]}$
3. $\frac{\partial A^{[1]}}{\partial Z^{[1]}} = \mathbb{1}_{Z^{[1]} > 0}$ (derivative of ReLU)
4. $\frac{\partial Z^{[1]}}{\partial W^{[1]}} = X$

Combining these, we get:

$\frac{\partial L}{\partial W^{[1]}} = X \cdot ((\hat{Y} - Y) \cdot W^{[2]} \cdot \mathbb{1}_{Z^{[1]} > 0})^T$

This is exactly what we calculate in the backward pass of our neural network!

### Visualizing Gradient Flow in a Neural Network

Let's create a visual representation of how gradients flow backward through a neural network:

```python
def plot_gradient_flow():
    """Visualize gradient flow in a simple neural network"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors
    node_colors = ['#A0CBE2', '#FFBE7D', '#8CD17D', '#FF9F9B']
    
    # Set up axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Draw layers
    layer_positions = [1, 3.5, 6, 8.5]
    layer_sizes = [3, 4, 4, 2]
    layer_names = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Output\nLayer']
    nodes = {}
    
    for l, (pos, size, name, color) in enumerate(zip(layer_positions, layer_sizes, layer_names, node_colors)):
        ax.text(pos, 5.5, name, ha='center', fontsize=12, fontweight='bold')
        nodes[l] = []
        
        for i in range(size):
            y_pos = 4.5 - i * 0.8
            circle = plt.Circle((pos, y_pos), 0.3, fc=color, ec='black')
            ax.add_patch(circle)
            nodes[l].append((pos, y_pos))
            
            if l == 3:  # Add labels for output layer
                ax.text(pos + 0.5, y_pos, f"Output {i+1}", va='center')
    
    # Connect nodes with arrows for forward pass
    for l in range(3):
        for i, (x1, y1) in enumerate(nodes[l]):
            for j, (x2, y2) in enumerate(nodes[l+1]):
                ax.arrow(x1 + 0.3, y1, x2 - x1 - 0.6, y2 - y1, 
                         head_width=0.1, head_length=0.1, fc='blue', ec='blue', 
                         length_includes_head=True, alpha=0.3)
    
    # Draw the loss function
    loss_pos = (8.5, 6)
    loss_circle = plt.Circle(loss_pos, 0.3, fc='#FF5733', ec='black')
    ax.add_patch(loss_circle)
    ax.text(loss_pos[0], loss_pos[1], "L", ha='center', va='center', fontweight='bold')
    
    # Connect output to loss
    for i, (x, y) in enumerate(nodes[3]):
        ax.arrow(x, y + 0.3, loss_pos[0] - x, loss_pos[1] - y - 0.3, 
                 head_width=0.1, head_length=0.1, fc='blue', ec='blue', 
                 length_includes_head=True, alpha=0.3)
    
    # Gradient flow (backward pass)
    # From loss to output layer
    for i, (x, y) in enumerate(nodes[3]):
        ax.arrow(loss_pos[0], loss_pos[1] - 0.3, x - loss_pos[0], y + 0.3 - (loss_pos[1] - 0.3), 
                 head_width=0.1, head_length=0.1, fc='red', ec='red', 
                 length_includes_head=True, linestyle='--')
        ax.text((loss_pos[0] + x)/2, (loss_pos[1] - 0.3 + y + 0.3)/2, "∂L/∂y₁", 
                color='red', ha='center', va='bottom', fontsize=8)
    
    # Sample gradient annotations
    grad_annotations = [
        (nodes[2][1], nodes[3][0], "∂L/∂w₃₁"),
        (nodes[1][2], nodes[2][1], "∂L/∂w₂₃"),
        (nodes[0][1], nodes[1][0], "∂L/∂w₁₂")
    ]
    
    for (start, end, label) in grad_annotations:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, label, color='red', ha='center', va='bottom', fontsize=8)
    
    # Legend
    ax.add_patch(plt.Rectangle((1, 0.5), 0.5, 0.25, fc='blue', ec='blue', alpha=0.3))
    ax.text(1.6, 0.625, "Forward Pass", va='center')
    
    ax.add_patch(plt.Rectangle((1, 0.2), 0.5, 0.25, fc='red', ec='red', linestyle='--'))
    ax.text(1.6, 0.325, "Backward Pass (Gradients)", va='center')
    
    ax.set_title("Gradient Flow in Backpropagation", fontsize=14)
    plt.tight_layout()
    plt.show()

plot_gradient_flow()
```

### Implementing Backpropagation Manually: A Worked Example

To better understand the algorithm, let's manually compute backpropagation for a very small neural network. We'll create a 2-layer network with:
- 2 input features
- 3 hidden neurons with ReLU activation
- 2 output neurons with softmax activation

```python
def manual_backpropagation_example():
    """Manually compute backpropagation for a small network"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define a small dataset
    X = np.array([[0.5, -0.2]])  # 1 example, 2 features
    y = np.array([1])            # Class 1
    
    # Define network architecture
    input_size = 2
    hidden_size = 3
    output_size = 2
    
    # Initialize parameters
    W1 = np.array([
        [0.1, 0.2, -0.1],   # W1 shape: (input_size, hidden_size)
        [-0.2, 0.3, 0.4]
    ])
    b1 = np.array([0.1, 0.1, 0.1])
    
    W2 = np.array([
        [0.3, -0.4],   # W2 shape: (hidden_size, output_size)
        [0.2, 0.2],
        [0.1, 0.3]
    ])
    b2 = np.array([0.1, 0.1])
    
    # Forward pass (manually)
    print("== Forward Pass ==")
    
    # Hidden layer
    Z1 = np.dot(X, W1) + b1
    print(f"Z1 = X·W1 + b1 = {Z1}")
    
    A1 = np.maximum(0, Z1)  # ReLU
    print(f"A1 = ReLU(Z1) = {A1}")
    
    # Output layer
    Z2 = np.dot(A1, W2) + b2
    print(f"Z2 = A1·W2 + b2 = {Z2}")
    
    # Softmax
    exp_scores = np.exp(Z2 - np.max(Z2))
    A2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    print(f"A2 = softmax(Z2) = {A2}")
    
    # Compute loss
    correct_class_score = A2[0, y[0]]
    loss = -np.log(correct_class_score)
    print(f"Loss = -log(A2[0, {y[0]}]) = {loss:.6f}")
    
    # Backward pass (manually)
    print("\n== Backward Pass ==")
    
    # Gradient of loss with respect to softmax output
    dA2 = np.zeros_like(A2)
    dA2[0, y[0]] = -1 / A2[0, y[0]]
    print(f"dL/dA2 = {dA2}")
    
    # Gradient of softmax
    dZ2 = A2.copy()
    dZ2[0, y[0]] -= 1
    print(f"dL/dZ2 = {dZ2}")
    
    # Gradients for W2 and b2
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0)
    print(f"dL/dW2 = {dW2}")
    print(f"dL/db2 = {db2}")
    
    # Gradient for A1
    dA1 = np.dot(dZ2, W2.T)
    print(f"dL/dA1 = {dA1}")
    
    # Gradient for Z1
    dZ1 = dA1.copy()
    dZ1[Z1 <= 0] = 0  # ReLU gradient
    print(f"dL/dZ1 = {dZ1}")
    
    # Gradients for W1 and b1
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0)
    print(f"dL/dW1 = {dW1}")
    print(f"dL/db1 = {db1}")
    
    # Parameter updates (using learning rate = 0.1)
    learning_rate = 0.1
    W1_new = W1 - learning_rate * dW1
    b1_new = b1 - learning_rate * db1
    W2_new = W2 - learning_rate * dW2
    b2_new = b2 - learning_rate * db2
    
    print("\n== Parameter Updates (learning_rate = 0.1) ==")
    print(f"W1 (before): {W1}")
    print(f"W1 (after): {W1_new}")
    print(f"b1 (before): {b1}")
    print(f"b1 (after): {b1_new}")
    print(f"W2 (before): {W2}")
    print(f"W2 (after): {W2_new}")
    print(f"b2 (before): {b2}")
    print(f"b2 (after): {b2_new}")
    
    # Check improvement in loss
    # Recompute forward pass with new parameters
    Z1_new = np.dot(X, W1_new) + b1_new
    A1_new = np.maximum(0, Z1_new)
    Z2_new = np.dot(A1_new, W2_new) + b2_new
    exp_scores_new = np.exp(Z2_new - np.max(Z2_new))
    A2_new = exp_scores_new / np.sum(exp_scores_new, axis=1, keepdims=True)
    new_loss = -np.log(A2_new[0, y[0]])
    
    print(f"\nOriginal loss: {loss:.6f}")
    print(f"Loss after one update: {new_loss:.6f}")
    print(f"Improvement: {loss - new_loss:.6f}")

manual_backpropagation_example()
```

In this example, we've manually traced through every step of backpropagation. This is essentially what happens inside our neural network's backward pass, but vectorized for efficiency.

## Optimization Algorithms: Beyond Basic Gradient Descent

Now that we understand backpropagation, let's explore different optimization algorithms that can improve training speed and performance.

### The Challenges of Vanilla Gradient Descent

While basic gradient descent is theoretically sound, it has several practical limitations:

```python
def visualize_optimization_challenges():
    """Visualize some common challenges with basic gradient descent"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Ravines/Saddle Points - Zigzagging behavior
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = 0.1 * X**2 + 5 * Y**2
    
    axes[0, 0].contour(X, Y, Z, levels=20, cmap='viridis')
    
    # Simulate zigzagging path
    path_x = [-3]
    path_y = [3]
    
    for i in range(10):
        new_x = path_x[-1] - 0.2 * 0.2 * path_x[-1]
        new_y = path_y[-1] - 0.2 * 10 * path_y[-1]
        path_x.append(new_x)
        path_y.append(new_y)
    
    axes[0, 0].plot(path_x, path_y, 'ro-', markersize=6)
    axes[0, 0].set_title('Challenge 1: Zigzagging in Ravines', fontsize=14)
    axes[0, 0].set_xlabel('Parameter θ₁')
    axes[0, 0].set_ylabel('Parameter θ₂')
    
    # 2. Local minima and plateaus
    def f(x):
        return np.sin(0.5*x) + 0.2*np.sin(3*x) + 0.5
    
    x = np.linspace(-10, 10, 1000)
    axes[0, 1].plot(x, f(x), 'b-')
    
    # Mark local minima and plateaus
    axes[0, 1].plot([-4.5], [f(-4.5)], 'ro', markersize=6)
    axes[0, 1].plot([4.5], [f(4.5)], 'ro', markersize=6)
    axes[0, 1].plot([0], [f(0)], 'ro', markersize=6)
    
    axes[0, 1].annotate('Local minimum', xy=(-4.5, f(-4.5)), xytext=(-6, 1),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    axes[0, 1].annotate('Global minimum', xy=(4.5, f(4.5)), xytext=(6, 0.8),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    axes[0, 1].annotate('Plateau', xy=(0, f(0)), xytext=(-2, 0.2),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    
    axes[0, 1].set_title('Challenge 2: Local Minima and Plateaus', fontsize=14)
    axes[0, 1].set_xlabel('Parameter θ')
    axes[0, 1].set_ylabel('Loss')
    
    # 3. Learning rate issues
    x = np.linspace(-3, 3, 100)
    y = x**2
    
    axes[1, 0].plot(x, y, 'b-')
    
    # Too small learning rate
    slow_path_x = [-2]
    for i in range(7):
        new_x = slow_path_x[-1] - 0.1 * 2 * slow_path_x[-1]
        slow_path_x.append(new_x)
    
    slow_path_y = [xi**2 for xi in slow_path_x]
    axes[1, 0].plot(slow_path_x, slow_path_y, 'g.-', label='Too small (slow convergence)')
    
    # Too large learning rate
    fast_path_x = [1]
    for i in range(5):
        new_x = fast_path_x[-1] - 0.9 * 2 * fast_path_x[-1]
        fast_path_x.append(new_x)
    
    fast_path_y = [xi**2 for xi in fast_path_x]
    axes[1, 0].plot(fast_path_x, fast_path_y, 'r.-', label='Too large (divergence)')
    
    # Just right learning rate
    right_path_x = [1.5]
    for i in range(5):
        new_x = right_path_x[-1] - 0.5 * 2 * right_path_x[-1]
        right_path_x.append(new_x)
    
    right_path_y = [xi**2 for xi in right_path_x]
    axes[1, 0].plot(right_path_x, right_path_y, 'k.-', label='Good learning rate')
    
    axes[1, 0].legend()
    axes[1, 0].set_title('Challenge 3: Learning Rate Sensitivity', fontsize=14)
    axes[1, 0].set_xlabel('Parameter θ')
    axes[1, 0].set_ylabel('Loss')
    
    # 4. Noisy gradients
    x = np.linspace(-2, 2, 100)
    y = x**2
    
    axes[1, 1].plot(x, y, 'b-')
    
    # Full batch gradient descent
    path_x = [1.5]
    for i in range(5):
        new_x = path_x[-1] - 0.4 * 2 * path_x[-1]
        path_x.append(new_x)
    
    path_y = [xi**2 for xi in path_x]
    axes[1, 1].plot(path_x, path_y, 'k.-', label='Full batch (smooth)')
    
    # Stochastic gradient descent
    np.random.seed(42)
    path_x = [1.5]
    for i in range(10):
        # Add noise to the gradient
        noise = np.random.normal(0, 0.3)
        new_x = path_x[-1] - 0.3 * (2 * path_x[-1] + noise)
        if new_x > -2 and new_x < 2:  # Keep within plot range
            path_x.append(new_x)
    
    path_y = [xi**2 for xi in path_x]
    axes[1, 1].plot(path_x, path_y, 'r.-', label='SGD (noisy)')
    
    axes[1, 1].legend()
    axes[1, 1].set_title('Challenge 4: Noisy Gradients with SGD', fontsize=14)
    axes[1, 1].set_xlabel('Parameter θ')
    axes[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

visualize_optimization_challenges()
```

### Stochastic Gradient Descent (SGD): A Practical Approach

The key insight behind SGD is that we don't need to compute gradients over the entire dataset to make progress. Instead, we can update parameters using just a small batch of examples. This leads to more frequent updates and often faster convergence.

#### The SGD Algorithm

1. Randomly shuffle the data
2. Divide the data into mini-batches
3. For each mini-batch:
   - Compute the gradient of the loss on the mini-batch
   - Update parameters using the mini-batch gradient

#### Benefits of SGD:

1. **Computational Efficiency**: Processing smaller batches requires less memory
2. **Faster Convergence**: Parameters update more frequently
3. **Regularization Effect**: The noise in the gradients can help escape local minima
4. **Online Learning**: Can handle streaming data and adapt to changing distributions

#### The Impact of Batch Size:

```python
def visualize_batch_size_impact():
    """Visualize the impact of different batch sizes"""
    np.random.seed(42)
    
    # Generate a simple dataset with noise
    n_samples = 1000
    X = np.random.randn(n_samples, 1)
    true_w, true_b = 2.0, 1.0
    y = true_w * X + true_b + 0.1 * np.random.randn(n_samples, 1)
    
    # Define different batch sizes to compare
    batch_sizes = [1, 10, 100, n_samples]  # Last one is full batch (BGD)
    labels = ['SGD (batch=1)', 'Mini-batch (batch=10)', 'Mini-batch (batch=100)', 'Full Batch GD']
    colors = ['red', 'green', 'blue', 'purple']
    
    # Training parameters
    learning_rate = 0.01
    n_iterations = 100
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Store loss history for each approach
    all_losses = []
    all_weights = []
    all_biases = []
    
    for batch_size, label, color in zip(batch_sizes, labels, colors):
        # Initialize parameters
        w = 0.0
        b = 0.0
        losses = []
        weights = []
        biases = []
        
        for iteration in range(n_iterations):
            # Track current parameters
            weights.append(w)
            biases.append(b)
            
            # Randomly select batch indices
            if batch_size < n_samples:
                batch_indices = np.random.choice(n_samples, batch_size, replace=False)
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
            else:
                X_batch = X
                y_batch = y
            
            # Forward pass
            y_pred = w * X_batch + b
            
            # Compute loss
            loss = np.mean((y_pred - y_batch) ** 2)
            losses.append(loss)
            
            # Compute gradients
            dw = np.mean(2 * X_batch * (y_pred - y_batch))
            db = np.mean(2 * (y_pred - y_batch))
            
            # Update parameters
            w = w - learning_rate * dw
            b = b - learning_rate * db
        
        all_losses.append(losses)
        all_weights.append(weights)
        all_biases.append(biases)
        
        # Plot loss curves
        ax[0].plot(losses, color=color, label=label)
    
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Loss (MSE)')
    ax[0].set_title('Loss vs. Iterations for Different Batch Sizes')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # Plot parameter trajectories
    for i, (weights, biases, label, color) in enumerate(zip(all_weights, all_biases, labels, colors)):
        ax[1].plot(weights, biases, color=color, label=label, alpha=0.7)
        ax[1].scatter(weights[-1], biases[-1], color=color, s=100, marker='*')
        
    # Mark true parameters and origin
    ax[1].scatter(true_w, true_b, color='black', s=200, marker='X', label='True Parameters')
    ax[1].scatter(0, 0, color='black', s=100, marker='o', label='Starting Point')
    
    ax[1].set_xlabel('Weight (w)')
    ax[1].set_ylabel('Bias (b)')
    ax[1].set_title('Parameter Trajectories')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_batch_size_impact()
```

As you can see from the visualization:

1. **Full Batch GD (purple)**: Follows a smooth, direct path but updates parameters infrequently
2. **SGD with batch=1 (red)**: Very noisy path but updates parameters most frequently
3. **Mini-batch GD (green, blue)**: Strike a balance between update frequency and noise

### Implementing SGD for Neural Networks

Let's update our neural network implementation to use mini-batch SGD:

```python
class Layer:
    """Base class for neural network layers"""
    def forward(self, input):
        pass
    
    def backward(self, input, grad_output):
        pass

class Dense(Layer):
    """Fully connected layer"""
    def __init__(self, input_units, output_units):
        # Xavier/Glorot initialization
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(2.0 / input_units)
        self.biases = np.zeros(output_units)
        
        # Initialize parameter gradients to zero
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)
    
    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases
    
    def backward(self, input, grad_output):
        # Compute gradients with respect to weights and biases
        self.grad_weights = np.dot(input.T, grad_output) / input.shape[0]
        self.grad_biases = np.mean(grad_output, axis=0)
        
        # Compute gradient with respect to input
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input
    
    def update(self, learning_rate):
        """Update parameters using gradients"""
        self.weights = self.weights - learning_rate * self.grad_weights
        self.biases = self.biases - learning_rate * self.grad_biases

class ReLU(Layer):
    """ReLU activation function"""
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)
    
    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad
    
    def update(self, learning_rate):
        """ReLU has no parameters to update"""
        pass

def softmax_crossentropy_with_logits(logits, labels):
    """Compute softmax cross-entropy loss and its gradient"""
    # Create one-hot vectors from labels
    batch_size = logits.shape[0]
    one_hot_labels = np.zeros_like(logits)
    one_hot_labels[np.arange(batch_size), labels] = 1
    
    # Compute softmax (with numeric stability)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Compute cross-entropy loss
    loss = -np.sum(one_hot_labels * np.log(softmax_probs + 1e-9)) / batch_size
    
    # Gradient of cross-entropy loss w.r.t. logits
    grad = (softmax_probs - one_hot_labels) / batch_size
    
    return loss, grad, softmax_probs

def forward(network, X):
    """Perform forward propagation through the network"""
    activations = []
    input = X
    
    # Pass input through each layer
    for layer in network:
        activations.append(layer.forward(input))
        input = activations[-1]
    
    return activations

def predict(network, X):
    """Get predictions from the network"""
    # Get the output of the last layer
    logits = forward(network, X)[-1]
    # Return the class with highest score
    return np.argmax(logits, axis=1)

def train_on_batch(network, X, y, learning_rate):
    """Train the network on a single batch"""
    # Forward pass
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations[:-1]
    logits = layer_activations[-1]
    
    # Compute loss and initial gradient
    loss, grad_logits, probabilities = softmax_crossentropy_with_logits(logits, y)
    
    # Backward pass (backpropagation)
    grad_output = grad_logits
    for i in range(len(network))[::-1]:  # Reversed order
        layer = network[i]
        layer_input = layer_inputs[i]
        grad_output = layer.backward(layer_input, grad_output)
    
    # Update parameters
    for layer in network:
        layer.update(learning_rate)
    
    return loss, probabilities
```

With our updated implementation, we can now implement SGD with mini-batches efficiently.

## Advanced Optimization: The Adam Algorithm

While SGD works well in practice, it still has limitations. To address these, researchers have developed more sophisticated optimization algorithms. One of the most popular is Adam (Adaptive Moment Estimation), which combines the benefits of two other extensions to SGD:

1. **Momentum**: Helps accelerate SGD by accumulating a moving average of past gradients
2. **RMSProp**: Adapts learning rates based on the average of recent gradient magnitudes

### The Intuition Behind Adam

1. **First Moment (Momentum)**: Keep track of an exponentially decaying average of past gradients
2. **Second Moment (RMSProp)**: Keep track of an exponentially decaying average of past squared gradients
3. **Bias Correction**: Correct the bias in the estimates of the first and second moments

Let's implement Adam for our neural network:

```python
class DenseWithAdam(Layer):
    """Dense layer with Adam optimizer"""
    def __init__(self, input_units, output_units, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Layer parameters
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(2.0 / input_units)
        self.biases = np.zeros(output_units)
        
        # Adam hyperparameters
        self.learning_rate = learning_rate
        self.beta1 = beta1  # Momentum parameter
        self.beta2 = beta2  # RMSProp parameter
        self.epsilon = epsilon  # Small constant for numerical stability
        
        # Adam accumulators
        self.m_weights = np.zeros_like(self.weights)  # First moment (momentum)
        self.v_weights = np.zeros_like(self.weights)  # Second moment (RMSProp)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)
        
        # Time step for bias correction
        self.t = 0
    
    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases
    
    def backward(self, input, grad_output):
        # Compute gradients with respect to weights and biases
        self.grad_weights = np.dot(input.T, grad_output) / input.shape[0]
        self.grad_biases = np.mean(grad_output, axis=0)
        
        # Compute gradient with respect to input
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input
    
    def update(self):
        """Update parameters using Adam optimizer"""
        self.t += 1
        
        # Update first and second moments for weights
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * self.grad_weights
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (self.grad_weights ** 2)
        
        # Bias-corrected first and second moments
        m_weights_corrected = self.m_weights / (1 - self.beta1 ** self.t)
        v_weights_corrected = self.v_weights / (1 - self.beta2 ** self.t)
        
        # Update weights
        self.weights -= self.learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + self.epsilon)
        
        # Update first and second moments for biases
        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * self.grad_biases
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (self.grad_biases ** 2)
        
        # Bias-corrected first and second moments
        m_biases_corrected = self.m_biases / (1 - self.beta1 ** self.t)
        v_biases_corrected = self.v_biases / (1 - self.beta2 ** self.t)
        
        # Update biases
        self.biases -= self.learning_rate * m_biases_corrected / (np.sqrt(v_biases_corrected) + self.epsilon)
```

### Why Adam Works Well

1. **Adaptive Learning Rates**: Each parameter has its own learning rate, which adapts during training
2. **Momentum**: Helps navigate ravines and saddle points more effectively
3. **RMSProp**: Handles different scales of gradients automatically
4. **Bias Correction**: Ensures that estimates aren't biased toward zero, especially at the beginning of training

### Comparing Optimizers: SGD vs Adam

Let's visualize how SGD and Adam perform on a challenging loss landscape:

```python
def compare_optimizers_on_landscape():
    """Compare SGD and Adam on a challenging loss landscape"""
    # Create a challenging 2D loss function
    def loss_function(x, y):
        return 0.1 * x**2 + 5 * y**2 + 0.1 * np.sin(5 * x) * np.cos(5 * y)
    
    def gradient(x, y):
        dx = 0.2 * x + 0.5 * np.cos(5 * x) * np.cos(5 * y)
        dy = 10 * y - 0.5 * np.sin(5 * x) * np.sin(5 * y)
        return np.array([dx, dy])
    
    # Parameter space
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_function(X, Y)
    
    # Starting point
    start_x, start_y = -4.0, 0.8
    
    # SGD parameters
    sgd_lr = 0.05
    
    # Adam parameters
    adam_lr = 0.1
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    # Run optimizers for n steps
    n_steps = 100
    sgd_path = [(start_x, start_y)]
    adam_path = [(start_x, start_y)]
    
    # SGD with momentum
    sgd_x, sgd_y = start_x, start_y
    momentum_x, momentum_y = 0, 0
    momentum = 0.9
    
    for _ in range(n_steps):
        grad_x, grad_y = gradient(sgd_x, sgd_y)
        
        # Update with momentum
        momentum_x = momentum * momentum_x - sgd_lr * grad_x
        momentum_y = momentum * momentum_y - sgd_lr * grad_y
        
        sgd_x += momentum_x
        sgd_y += momentum_y
        
        sgd_path.append((sgd_x, sgd_y))
    
    # Adam
    adam_x, adam_y = start_x, start_y
    m_x, m_y = 0, 0  # First moment
    v_x, v_y = 0, 0  # Second moment
    t = 0
    
    for _ in range(n_steps):
        t += 1
        grad_x, grad_y = gradient(adam_x, adam_y)
        
        # Update first and second moments
        m_x = beta1 * m_x + (1 - beta1) * grad_x
        m_y = beta1 * m_y + (1 - beta1) * grad_y
        
        v_x = beta2 * v_x + (1 - beta2) * (grad_x ** 2)
        v_y = beta2 * v_y + (1 - beta2) * (grad_y ** 2)
        
        # Bias correction
        m_x_corrected = m_x / (1 - beta1 ** t)
        m_y_corrected = m_y / (1 - beta1 ** t)
        
        v_x_corrected = v_x / (1 - beta2 ** t)
        v_y_corrected = v_y / (1 - beta2 ** t)
        
        # Update parameters
        adam_x -= adam_lr * m_x_corrected / (np.sqrt(v_x_corrected) + epsilon)
        adam_y -= adam_lr * m_y_corrected / (np.sqrt(v_y_corrected) + epsilon)
        
        adam_path.append((adam_x, adam_y))
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot contour
    contour = plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Loss')
    
    # Plot SGD path
    sgd_path = np.array(sgd_path)
    plt.plot(sgd_path[:, 0], sgd_path[:, 1], 'r.-', label='SGD with Momentum', alpha=0.7)
    
    # Plot Adam path
    adam_path = np.array(adam_path)
    plt.plot(adam_path[:, 0], adam_path[:, 1], 'b.-', label='Adam', alpha=0.7)
    
    # Mark start and endpoints
    plt.plot(start_x, start_y, 'ko', markersize=10, label='Start')
    plt.plot(sgd_path[-1, 0], sgd_path[-1, 1], 'ro', markersize=8, label='SGD End')
    plt.plot(adam_path[-1, 0], adam_path[-1, 1], 'bo', markersize=8, label='Adam End')
    
    plt.xlabel('Parameter x')
    plt.ylabel('Parameter y')
    plt.title('Comparison of Optimizers on a Challenging Loss Landscape', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

compare_optimizers_on_landscape()
```

As you can see from the visualization, Adam often converges faster and more directly to the minimum compared to SGD with momentum. This is one of the reasons Adam has become a default choice for many deep learning applications.

### When to Use Which Optimizer

- **SGD**: Still widely used and can achieve excellent results with proper tuning
  - Good for convex problems
  - May generalize better in some cases
  - Needs careful learning rate scheduling
  
- **SGD with Momentum**: Faster convergence than plain SGD
  - Helps overcome ravines and local minima
  - Less sensitive to the exact learning rate
  
- **Adam**: Robust, works well out of the box for most problems
  - Adapts learning rates automatically
  - Converges faster in many cases
  - Less sensitive to hyperparameter choices
  - Great for non-stationary objectives and noisy gradients

For our CIFAR-10 implementation, we'll use both SGD and Adam, and compare their performance.

## Applying Our Knowledge: Training a Neural Network on CIFAR-10

Now that we understand backpropagation and various optimization techniques, let's apply our knowledge to train a neural network on the CIFAR-10 dataset.

### The CIFAR-10 Dataset

CIFAR-10 consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. The classes are:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

Let's load the dataset:

```python
def load_cifar10():
    """
    Download and extract CIFAR-10 if needed, then load the dataset
    
    Returns:
    X_train -- training images, shape (50000, 3, 32, 32)
    y_train -- training labels, shape (50000,)
    X_test -- test images, shape (10000, 3, 32, 32)
    y_test -- test labels, shape (10000,)
    """
    # URL for CIFAR-10 dataset
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    
    # Create a directory for the dataset if it doesn't exist
    data_dir = 'cifar-10-batches-py'
    if not os.path.exists(data_dir):
        os.makedirs('data', exist_ok=True)
        
        # Download the dataset
        print("Downloading CIFAR-10 dataset...")
        file_path, _ = urllib.request.urlretrieve(url, "data/cifar-10-python.tar.gz")
        
        # Extract the dataset
        print("Extracting CIFAR-10 dataset...")
        with tarfile.open(file_path) as tar:
            tar.extractall(path='data')
        
        print("Dataset downloaded and extracted successfully.")
    
    # Load training data
    X_train = []
    y_train = []
    
    for batch_id in range(1, 6):
        batch_file = f'data/cifar-10-batches-py/data_batch_{batch_id}'
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f, encoding='bytes')
        
        X_train.append(batch_data[b'data'].reshape(-1, 3, 32, 32))
        y_train.append(batch_data[b'labels'])
    
    X_train = np.concatenate(X_train) / 255.0  # Normalize to [0, 1]
    y_train = np.concatenate(y_train)
    
    # Load test data
    with open('data/cifar-10-batches-py/test_batch', 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')
    
    X_test = test_data[b'data'].reshape(-1, 3, 32, 32) / 255.0
    y_test = np.array(test_data[b'labels'])
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

def visualize_cifar10_samples(X, y, class_names, num_samples=10):
    """Visualize sample images from the CIFAR-10 dataset"""
    indices = np.random.choice(len(X), size=num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        # CIFAR-10 images are stored as (channels, height, width)
        # We need to transpose to (height, width, channels) for matplotlib
        img = np.transpose(X[idx], (1, 2, 0))
        axes[i].imshow(img)
        axes[i].set_title(class_names[y[idx]])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Load CIFAR-10 dataset
X_train, y_train, X_test, y_test = load_cifar10()

# Visualize some samples
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
visualize_cifar10_samples(X_train, y_train, class_names)
```

### Preprocessing CIFAR-10 for Our Neural Network

Since our neural network is designed for vector inputs, we need to reshape the CIFAR-10 images into vectors:

```python
def preprocess_cifar10(X_train, y_train, X_test, y_test):
    """Preprocess CIFAR-10 data for our neural network"""
    # Flatten the images from (N, 3, 32, 32) to (N, 3*32*32)
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

# Preprocess the data
X_train, y_train, X_val, y_val, X_test, y_test = preprocess_cifar10(X_train, y_train, X_test, y_test)
```

### Creating a Neural Network for CIFAR-10

Let's create a neural network architecture suitable for CIFAR-10:

```python
def create_network(input_size, hidden_sizes, output_size, optimizer='sgd'):
    """Create a neural network with specified architecture and optimizer
    
    Arguments:
    input_size -- size of input features
    hidden_sizes -- list of sizes for hidden layers
    output_size -- number of output classes
    optimizer -- 'sgd' or 'adam'
    
    Returns:
    network -- list of layers in the network
    """
    network = []
    
    # Input layer to first hidden layer
    if optimizer == 'adam':
        network.append(DenseWithAdam(input_size, hidden_sizes[0]))
    else:
        network.append(Dense(input_size, hidden_sizes[0]))
    network.append(ReLU())
    
    # Additional hidden layers
    for i in range(1, len(hidden_sizes)):
        if optimizer == 'adam':
            network.append(DenseWithAdam(hidden_sizes[i-1], hidden_sizes[i]))
        else:
            network.append(Dense(hidden_sizes[i-1], hidden_sizes[i]))
        network.append(ReLU())
    
    # Last hidden layer to output layer
    if optimizer == 'adam':
        network.append(DenseWithAdam(hidden_sizes[-1], output_size))
    else:
        network.append(Dense(hidden_sizes[-1], output_size))
    
    return network

# Network parameters
input_size = X_train.shape[1]  # 3072 (3*32*32)
hidden_sizes = [1024, 512]
output_size = 10  # 10 classes in CIFAR-10

# Create networks with different optimizers
sgd_network = create_network(input_size, hidden_sizes, output_size, optimizer='sgd')
adam_network = create_network(input_size, hidden_sizes, output_size, optimizer='adam')

print("Neural network architecture:")
for i, layer in enumerate(sgd_network):
    if isinstance(layer, Dense):
        print(f"Layer {i}: Dense ({layer.weights.shape[0]} -> {layer.weights.shape[1]})")
    elif isinstance(layer, ReLU):
        print(f"Layer {i}: ReLU")
```

### Training the Network with Different Optimizers

Now let's train our neural network using both SGD and Adam optimizers, and compare their performance:

```python
def train_network(network, X_train, y_train, X_val, y_val, 
                  batch_size=128, epochs=10, learning_rate=0.001,
                  optimizer='sgd'):
    """Train a neural network with specified optimizer
    
    Arguments:
    network -- list of layers in the network
    X_train, y_train -- training data and labels
    X_val, y_val -- validation data and labels
    batch_size -- size of mini-batches
    epochs -- number of training epochs
    learning_rate -- learning rate for SGD
    optimizer -- 'sgd' or 'adam'
    
    Returns:
    history -- dictionary containing training and validation metrics
    """
    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # History to track metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Initialize metrics for this epoch
        epoch_loss = 0
        epoch_acc = 0
        
        # Mini-batch training
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Train on batch
            if optimizer == 'sgd':
                batch_loss, batch_probs = train_on_batch(network, X_batch, y_batch, learning_rate)
            else:  # Adam
                batch_loss, batch_probs = train_on_batch_adam(network, X_batch, y_batch)
            
            # Compute accuracy
            batch_preds = np.argmax(batch_probs, axis=1)
            batch_acc = np.mean(batch_preds == y_batch)
            
            # Update epoch metrics
            epoch_loss += batch_loss * (end_idx - start_idx) / n_samples
            epoch_acc += batch_acc * (end_idx - start_idx) / n_samples
        
        # Evaluate on validation set
        val_logits = forward(network, X_val)[-1]
        val_loss, _, val_probs = softmax_crossentropy_with_logits(val_logits, y_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = np.mean(val_preds == y_val)
        
        # Record metrics
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    return history

def train_on_batch_adam(network, X, y):
    """Train the network on a single batch using Adam optimizer"""
    # Forward pass
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations[:-1]
    logits = layer_activations[-1]
    
    # Compute loss and initial gradient
    loss, grad_logits, probabilities = softmax_crossentropy_with_logits(logits, y)
    
    # Backward pass (backpropagation)
    grad_output = grad_logits
    for i in range(len(network))[::-1]:  # Reversed order
        layer = network[i]
        layer_input = layer_inputs[i]
        grad_output = layer.backward(layer_input, grad_output)
    
    # Update parameters using Adam
    for layer in network:
        if hasattr(layer, 'update'):
            layer.update()
    
    return loss, probabilities

def visualize_training_history(sgd_history, adam_history):
    """Visualize training and validation metrics for different optimizers"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(sgd_history['train_loss'], 'r-', label='SGD')
    axes[0, 0].plot(adam_history['train_loss'], 'b-', label='Adam')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation loss
    axes[0, 1].plot(sgd_history['val_loss'], 'r-', label='SGD')
    axes[0, 1].plot(adam_history['val_loss'], 'b-', label='Adam')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training accuracy
    axes[1, 0].plot(sgd_history['train_acc'], 'r-', label='SGD')
    axes[1, 0].plot(adam_history['train_acc'], 'b-', label='Adam')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation accuracy
    axes[1, 1].plot(sgd_history['val_acc'], 'r-', label='SGD')
    axes[1, 1].plot(adam_history['val_acc'], 'b-', label='Adam')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Train with SGD
print("Training with SGD...")
sgd_history = train_network(sgd_network, X_train, y_train, X_val, y_val, 
                           batch_size=128, epochs=5, learning_rate=0.01, 
                           optimizer='sgd')

# Train with Adam
print("Training with Adam...")
adam_history = train_network(adam_network, X_train, y_train, X_val, y_val, 
                            batch_size=128, epochs=5, learning_rate=0.001, 
                            optimizer='adam')

# Visualize training history
visualize_training_history(sgd_history, adam_history)
```

### Evaluating the Models on Test Data

Let's evaluate our trained models on the test dataset:

```python
def evaluate_model(network, X, y):
    """Evaluate a trained model on data"""
    # Forward pass
    logits = forward(network, X)[-1]
    
    # Compute loss and probabilities
    loss, _, probabilities = softmax_crossentropy_with_logits(logits, y)
    
    # Get predictions
    predictions = np.argmax(probabilities, axis=1)
    
    # Compute accuracy
    accuracy = np.mean(predictions == y)
    
    return loss, accuracy, predictions

def visualize_confusion_matrix(y_true, y_pred, class_names):
    """Visualize a confusion matrix"""
    # Compute confusion matrix
    conf_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        conf_matrix[true, pred] += 1
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(conf_matrix, cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate the x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add numbers to cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black" if conf_matrix[i, j] < conf_matrix.max() / 2 else "white")
    
    plt.tight_layout()
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    
    plt.show()

# Evaluate on test data
print("Evaluating models on test data...")

sgd_loss, sgd_accuracy, sgd_predictions = evaluate_model(sgd_network, X_test, y_test)
print(f"SGD - Test Loss: {sgd_loss:.4f}, Test Accuracy: {sgd_accuracy:.4f}")

adam_loss, adam_accuracy, adam_predictions = evaluate_model(adam_network, X_test, y_test)
print(f"Adam - Test Loss: {adam_loss:.4f}, Test Accuracy: {adam_accuracy:.4f}")

# Visualize confusion matrix for the better model
best_predictions = adam_predictions if adam_accuracy > sgd_accuracy else sgd_predictions
best_optimizer = "Adam" if adam_accuracy > sgd_accuracy else "SGD"

print(f"Visualizing confusion matrix for {best_optimizer} model...")
visualize_confusion_matrix(y_test, best_predictions, class_names)
```

### Visualizing Model Predictions

Let's visualize some examples of correct and incorrect predictions:

```python
def visualize_predictions(X, y, predictions, class_names, correct=True, num_samples=10):
    """Visualize model predictions (either correct or incorrect)"""
    # Find indices of correct or incorrect predictions
    if correct:
        indices = np.where(predictions == y)[0]
        title = "Correct Predictions"
    else:
        indices = np.where(predictions != y)[0]
        title = "Incorrect Predictions"
    
    # Choose random samples
    if len(indices) > num_samples:
        indices = np.random.choice(indices, num_samples, replace=False)
    else:
        num_samples = len(indices)
    
    # Plot samples
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        # Reshape back to image format
        img = X[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        img = img * np.std(img) + np.mean(img)  # Unnormalize
        img = np.clip(img, 0, 1)  # Ensure values are in [0, 1]
        
        axes[i].imshow(img)
        axes[i].set_title(f"True: {class_names[y[idx]]}\nPred: {class_names[predictions[idx]]}")
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

# Visualize correct and incorrect predictions
best_predictions = adam_predictions if adam_accuracy > sgd_accuracy else sgd_predictions

print("Visualizing correct predictions...")
visualize_predictions(X_test, y_test, best_predictions, class_names, correct=True)

print("Visualizing incorrect predictions...")
visualize_predictions(X_test, y_test, best_predictions, class_names, correct=False)
```

## Conclusion

In this session, we've explored the inner workings of backpropagation and implemented different optimization algorithms for training neural networks. We've seen how:

1. **Backpropagation** efficiently computes gradients using the chain rule
2. **Stochastic Gradient Descent (SGD)** uses mini-batches to make optimization more efficient
3. **Adam optimizer** combines momentum and adaptive learning rates for faster convergence

We've also applied these techniques to the CIFAR-10 dataset, building a neural network capable of classifying images into 10 different categories.

In the next session, we'll explore regularization techniques to prevent overfitting and further improve our neural network's performance.