# Practice 1: Test-Driven Programming with Neural Networks from Scratch

In this practice session, you will learn how to apply test-driven development principles to build a neural network from scratch using Python and NumPy. By the end of this session, you will have:

1. Created a GitHub repository with continuous integration through GitHub Actions
2. Implemented a simple neural network from scratch that can classify MNIST digits
3. Written tests to verify each component of your neural network
4. Used GitHub Actions to automatically test your code when you push changes


## Step 1: Setting Up Your GitHub Repository

1. **Create a new repository on GitHub**
   - Go to [GitHub](https://github.com) and log in to your account
   - Click the "+" icon in the top-right corner and select "New repository"
   - Name your repository: `test-driven-programming-with-nn-from-scratch`
   - Add a description: "Implementing a neural network from scratch using test-driven development"
   - Initialize with a README file
   - Click "Create repository"

2. **Clone the repository to your local machine**
   ```bash
   git clone https://github.com/YOUR-USERNAME/test-driven-programming-with-nn-from-scratch.git
   cd test-driven-programming-with-nn-from-scratch
   ```

## Step 2: Setting Up the Project Structure

1. **Create the necessary directories in your repository**
   ```bash
   mkdir -p .github/workflows
   ```

2. **Create a requirements.txt file**
   Create a file named `requirements.txt` with the following content:
   ```
   numpy>=1.20.0
   matplotlib>=3.4.0
   pandas>=1.3.0
   pytest>=6.2.5
   ```

## Step 3: Copy MNIST Data

Copy `mnist_train.csv` and  `mnist_test.csv`  in the repo.


## Step 4: Creating the Neural Network Implementation

Now, let's create the main neural network implementation. This file will have some parts intentionally left blank that you'll need to complete.

Create a file named `nn_from_scratch.py` with the following content:

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_mnist_from_csv(train_csv_path, test_csv_path, val_split=0.1):
    """
    Load MNIST dataset from CSV files and split into train, validation, and test sets

    Arguments:
    train_csv_path -- path to the training CSV file
    test_csv_path -- path to the test CSV file
    val_split -- fraction of training data to use for validation

    Returns:
    X_train -- training features
    y_train -- training labels
    X_val -- validation features
    y_val -- validation labels
    X_test -- test features
    y_test -- test labels
    """
    print("Loading training data...")
    train_data = pd.read_csv(train_csv_path)

    print("Loading test data...")
    test_data = pd.read_csv(test_csv_path)

    # Extract labels and features
    y_train_full = train_data.iloc[:, 0].values
    X_train_full = (
        train_data.iloc[:, 1:].values / 255.0
    )  # Normalize pixel values to [0,1]

    y_test = test_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values / 255.0

    # Create validation set from training data
    n_val = int(len(X_train_full) * val_split)

    # Random indices for validation set
    np.random.seed(42)
    val_indices = np.random.choice(len(X_train_full), n_val, replace=False)

    # Create a mask for training indices
    train_mask = np.ones(len(X_train_full), dtype=bool)
    train_mask[val_indices] = False

    # Split into train and validation sets
    X_val = X_train_full[val_indices]
    y_val = y_train_full[val_indices]

    X_train = X_train_full[train_mask]
    y_train = y_train_full[train_mask]

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_from_csv(
    "./mnist_train.csv", "./mnist_test.csv", val_split=0.1
)


class Layer:
    """
    Base class for neural network layers

    Each layer performs two key operations:
    - Forward pass: Compute outputs from inputs
    - Backward pass: Compute gradients and update parameters
    """

    def __init__(self):
        """Initialize layer parameters (if any)"""
        pass

    def forward(self, input):
        """
        Forward pass: Compute outputs from inputs

        Arguments:
        input -- Input data, shape (batch_size, input_dimension)

        Returns:
        output -- Layer output, shape (batch_size, output_dimension)
        """
        # Base implementation (identity function)
        return input

    def backward(self, input, grad_output):
        """
        Backward pass: Compute gradient of loss with respect to inputs

        Arguments:
        input -- Input data from forward pass
        grad_output -- Gradient of loss with respect to layer output

        Returns:
        grad_input -- Gradient of loss with respect to layer input
        """
        # Base implementation (pass gradient unchanged)
        return grad_output


class ReLU(Layer):
    """
    ReLU activation layer

    Forward: f(x) = max(0, x)
    Backward gradient: f'(x) = 1 if x > 0 else 0
    """

    def forward(self, input):
        """Apply ReLU activation function"""
        # TODO: Implement the forward pass for ReLU activation
        # Hint: Use np.maximum to apply the ReLU function element-wise
        return  ______YOUR_CODE_HERE_________ 

    def backward(self, input, grad_output):
        """Compute gradient of loss w.r.t. ReLU input"""
        # TODO: Implement the backward pass for ReLU activation
        # Hint: ReLU gradient is 1 where input > 0, and 0 elsewhere
        relu_grad =  ______YOUR_CODE_HERE_________ 
        return grad_output * relu_grad


class Dense(Layer):
    """
    Fully connected (dense) layer

    Forward: output = input · weights + bias
    """

    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
        Initialize dense layer

        Arguments:
        input_units -- dimension of input
        output_units -- dimension of output
        learning_rate -- learning rate for gradient descent
        """
        self.learning_rate = learning_rate

        # Initialize weights with small random values (Xavier/Glorot initialization)
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(
            2.0 / input_units
        )
        self.biases = np.zeros(output_units)

    def forward(self, input):
        """
        Forward pass: compute output = input · weights + bias
        """
        # Store input for backward pass
        self.input = input

        # TODO: Implement the forward pass for a dense layer
        # Hint: Use np.dot for matrix multiplication
        return  ______YOUR_CODE_HERE_________ 

    def backward(self, input, grad_output):
        """
        Backward pass: compute gradients and update parameters
        """
        # Gradient of loss w.r.t. weights: input^T · grad_output
        grad_weights = np.dot(input.T, grad_output) / input.shape[0]

        # Gradient of loss w.r.t. biases: sum grad_output over batch dimension
        grad_biases = np.mean(grad_output, axis=0)

        # Gradient of loss w.r.t. input: grad_output · weights^T
        grad_input = np.dot(grad_output, self.weights.T)

        # Update parameters using gradient descent
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input


def softmax_crossentropy_with_logits(logits, labels):
    """
    Compute softmax cross-entropy loss and its gradient

    Arguments:
    logits -- raw model outputs, shape (batch_size, num_classes)
    labels -- true labels, shape (batch_size,)

    Returns:
    loss -- scalar value, cross-entropy loss
    grad -- gradient of loss w.r.t. logits, shape (batch_size, num_classes)
    """
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

    return loss, grad


def forward(network, X):
    """
    Perform forward propagation through the network

    Arguments:
    network -- list of layers
    X -- input data

    Returns:
    layer_activations -- list of activations from each layer
    """
    activations = []
    input = X

    # Pass input through each layer
    for layer in network:
        activations.append(layer.forward(input))
        input = ______YOUR_CODE_HERE_________  # Output of current layer becomes input to next layer

    return activations


def predict(network, X):
    """Get predictions from the network"""
    # Get the output of the last layer
    logits = forward(network, X)[-1]
    # Return the class with highest score
    return np.argmax(logits, axis=-1)


def train(network, X, y):
    """Train the network on a batch of examples"""
    # Forward pass
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations[:-1]  # Inputs to each layer
    logits = layer_activations[-1]

    # Compute loss and initial gradient
    loss, grad_logits = softmax_crossentropy_with_logits(logits, y)

    # Backward pass (backpropagation)
    grad_output = grad_logits
    for i in range(len(network))[::-1]:  # Reversed order
        layer = network[i]
        layer_input = layer_inputs[i]
        grad_output = layer.backward(layer_input, grad_output)

    return loss


def train_mnist_network(
    X_train, y_train, X_val, y_val, num_epochs=5, batch_size=32, subset_size=10000
):
    """Train a neural network on MNIST dataset"""
    # Initialize the network
    network = [
        Dense(784, 64),  # Input layer -> Hidden layer 1
        ReLU(),  # Activation function
        Dense(64, 32),  # Hidden layer 1 -> Hidden layer 2
        ReLU(),  # Activation function
        Dense(32, 10),  # Hidden layer 2 -> Output layer
    ]

    print("Network architecture:")
    for i, layer in enumerate(network):
        if isinstance(layer, Dense):
            print(
                f"Layer {i}: Dense ({layer.weights.shape[0]} -> {layer.weights.shape[1]})"
            )
        else:
            print(f"Layer {i}: {layer.__class__.__name__}")

    # Training history
    train_loss_history = []
    val_accuracy_history = []

    # Use a subset for faster training during demo
    if subset_size and subset_size < len(X_train):
        subset_indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_subset = X_train[subset_indices]
        y_train_subset = y_train[subset_indices]
    else:
        X_train_subset = X_train
        y_train_subset = y_train

    print(f"\nTraining on {len(X_train_subset)} examples")

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Shuffle the training data
        indices = np.random.permutation(len(X_train_subset))
        X_shuffled = X_train_subset[indices]
        y_shuffled = y_train_subset[indices]

        # Mini-batch training
        num_batches = (len(X_train_subset) + batch_size - 1) // batch_size
        epoch_losses = []

        for batch in range(num_batches):
            # Extract batch
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, len(X_train_subset))

            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            # Train on batch
            loss = train(network, X_batch, y_batch)
            epoch_losses.append(loss)
            train_loss_history.append(loss)

            # Print progress
            if batch % 20 == 0:
                print(f"  Batch {batch+1}/{num_batches}, Loss: {loss:.4f}")

        # Evaluate on validation set
        val_subset_size = min(
            1000, len(X_val)
        )  # Use a smaller subset for faster evaluation
        val_indices = np.random.choice(len(X_val), val_subset_size, replace=False)

        val_predictions = predict(network, X_val[val_indices])
        val_accuracy = np.mean(val_predictions == y_val[val_indices])
        val_accuracy_history.append(val_accuracy)

        avg_loss = np.mean(epoch_losses)
        print(
            f"  Epoch complete - Avg Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )

    return network, val_accuracy


network, val_accuracy = train_mnist_network(X_train, y_train, X_val, y_val)
assert val_accuracy > 0.65
```

## Step 5: Creating the Test File

Create a file in the folder "tests" named `test_all.py` with the following test functions:

```python
import sys
import os
import pytest
import numpy as np

# Add parent directory to path to import nn-from-scratch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the nn-from-scratch module
from nn_from_scratch import (
    Layer, 
    ReLU, 
    Dense, 
    softmax_crossentropy_with_logits,
    forward,
    predict,
    train,
    train_mnist_network,
    load_mnist_from_csv
)

def test_relu_layer():
    """Test the ReLU layer implementation"""
    # Create a ReLU layer
    relu = ReLU()

    # Test input with positive and negative values
    input = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])

    # Forward pass
    output = relu.forward(input)
    
    # Expected output: [0, 0, 0, 1, 2]
    expected = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])
    assert np.array_equal(output, expected), f"ReLU forward pass failed. Expected {expected}, got {output}"

    # Backward pass
    grad_output = np.ones_like(output)
    grad_input = relu.backward(input, grad_output)
    
    # Expected gradient: [0, 0, 0, 1, 1]
    expected_grad = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]])
    assert np.array_equal(grad_input, expected_grad), f"ReLU backward pass failed. Expected {expected_grad}, got {grad_input}"


def test_dense_layer():
    """Test the Dense layer implementation"""
    # Create a small dense layer: 2 inputs, 3 outputs
    dense = Dense(2, 3, learning_rate=0.1)

    # Set weights and biases for predictable testing
    dense.weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    dense.biases = np.array([0.1, 0.2, 0.3])

    # Test input: batch of 2 examples
    input = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Forward pass
    output = dense.forward(input)

    # Verify output with manual calculation
    expected_output = np.array(
        [
            [
                1.0 * 0.1 + 2.0 * 0.4 + 0.1,
                1.0 * 0.2 + 2.0 * 0.5 + 0.2,
                1.0 * 0.3 + 2.0 * 0.6 + 0.3,
            ],
            [
                3.0 * 0.1 + 4.0 * 0.4 + 0.1,
                3.0 * 0.2 + 4.0 * 0.5 + 0.2,
                3.0 * 0.3 + 4.0 * 0.6 + 0.3,
            ],
        ]
    )
    assert np.allclose(output, expected_output), "Dense forward pass failed"

    # Test backward pass
    grad_output = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    # Save original weights
    original_weights = dense.weights.copy()
    original_biases = dense.biases.copy()

    grad_input = dense.backward(input, grad_output)

    # Check parameter updates (weights and biases should be updated)
    assert not np.array_equal(original_weights, dense.weights), "Weights were not updated"
    assert not np.array_equal(original_biases, dense.biases), "Biases were not updated"


@pytest.mark.slow
def test_model_accuracy():
    """Test that the model achieves at least 70% validation accuracy"""
    try:
        # Load a small subset of data for testing
        X_train, y_train, X_val, y_val, _, _ = load_mnist_from_csv(
            "./mnist_train.csv", "./mnist_test.csv", val_split=0.1
        )

        # Train model with minimal configuration
        _, val_accuracy = train_mnist_network(
            X_train, y_train, X_val, y_val, num_epochs=5
        )

        # Assert that validation accuracy is at least 50%
        assert val_accuracy >= 0.5, f"Model accuracy {val_accuracy:.4f} is below the required 50%"
    except FileNotFoundError:
        pytest.skip("MNIST dataset files not found. Skipping accuracy test.")


if __name__ == "__main__":
    # Run tests
    print("Running ReLU layer test...")
    test_relu_layer()
    print("ReLU layer test passed!")
    
    print("\nRunning Dense layer test...")
    test_dense_layer()
    print("Dense layer test passed!")
    
    try:
        print("\nRunning model accuracy test...")
        test_model_accuracy()
        print("Model accuracy test passed!")
    except Exception as e:
        print(f"Model accuracy test skipped or failed: {e}") 
```

## Step 6: Setting Up GitHub Actions for Automated Testing

Create a file named `.github/workflows/test.yml` with the following content:

```yaml
name: Test Neural Network Implementation

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python 3.9
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install pytest
    
    - name: Test model accuracy
      run: |
        python -m pytest tests/test_all.py -v 
```

## Step 7: Completing the Exercises

Now, it's time to fill in the blanks in `nn_from_scratch.py` to make the neural network work.


## Step 8: Pushing Your Changes and Checking GitHub Actions

1. **Commit and push your completed work**
   ```bash
   git add .
   git commit -m "Implement neural network from scratch with tests"
   git push origin main
   ```

2. **Check GitHub Actions**
   - Go to your GitHub repository page
   - Click on the "Actions" tab
   - You should see your workflow running or completed
   - If tests pass, you'll see a green checkmark
   - If tests fail, you'll see a red X. Click on the workflow to see error details

## Conclusion

Congratulations! You've successfully:

1. Set up a GitHub repository with continuous integration
2. Implemented a neural network from scratch using NumPy
3. Created tests to verify your implementation
4. Used GitHub Actions to automate testing
