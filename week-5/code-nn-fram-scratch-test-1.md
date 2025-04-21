# Exam Preparation

Our nn-from-scratch code is not complicated.

This tutorial helps you understand the code and prepare for the test/exam.

## Network Architecture

Our network consists of a sequence of layers. Each layer implements two key methods:
- `forward(input)`: Processes the input and returns an output.
- `backward(input, grad_output)`: Computes gradients and updates parameters (in trainable layers).

The network we build for MNIST has this architecture:
```python
network = [
    Dense(784, 64),  # Input layer -> Hidden layer 1
    ReLU(),          # Activation function
    Dense(64, 32),   # Hidden layer 1 -> Hidden layer 2
    ReLU(),          # Activation function
    Dense(32, 10),   # Hidden layer 2 -> Output layer
]
```

This creates a 3-layer neural network (784→64→32→10) with ReLU activations between the Dense layers.

## Forward Pass

The forward pass is the process of passing input data through the network to obtain an output.

#### ReLU Layer

In ReLU, the forward pass is simply applying the function max(0, x) to each input element:

```python
def forward(self, input):
    return np.maximum(0, input)
```

This function ensures all negative values become zero while positive values remain unchanged.

#### Dense Layer

In the Dense layer, the forward pass involves a dot product between the weights and the input, plus the bias term:

```python
def forward(self, input):
    return np.dot(input, self.weights) + self.biases
```

This implements the fundamental neural network operation: \(Wx + b\), where \(W\) represents weights, \(x\) is the input, and \(b\) is the bias vector.


#### Forward Function

The `forward` function processes input data through each layer of the network:

```python
def forward(network, X):
    activations = []
    input = X

    # Pass input through each layer
    for layer in network:
        activations.append(layer.forward(input))
        input = activations[-1]  # Output of current layer becomes input to next layer

    return activations
```

This function returns a list of activations from each layer of the network after processing the input data. For a network with \(n\) layers, this list will contain \(n\) elements.

Note that the last layer doesn't have a softmax transformation yet. The softmax is applied separately in the loss function simplify the backward pass calculation.


You can safely write the forward function as follows, if it helps you understand the code better (in your exam, both ways of writing are correct):

```python
def forward(network, X):
    nn_after_forward = []
    input = X

    for layer in network:
        next = layer.forward(input)
        nn_after_forward.append(next)
        input = next

    return nn_after_forward
```

Explanation for the above modified version of forward function: This version is functionally identical to the original but uses more descriptive variable names. `nn_after_forward` stores the outputs of each layer, and we're clearly showing how each layer's output becomes the input to the next layer in the network.



## Backward Pass

The backward pass involves computing gradients and updating parameters based on the loss.

#### ReLU Layer

In ReLU, the derivative is 0 if \(x \leq 0\), and 1 if \(x > 0\). This is implemented in a vectorized way:

```python
def backward(self, input, grad_output):
    relu_grad = input > 0  # Creates a boolean mask (True where input > 0)
    return grad_output * relu_grad  # Element-wise multiplication
```

Where input is negative, gradients become zero - effectively "turning off" those neurons during the backward pass.

#### Dense Layer

In the Dense layer, the implementation computes gradients with respect to weights, biases, and inputs, then updates the parameters using gradient descent:

```python
def backward(self, input, grad_output):
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
```

In Dense, the implementation computes gradients with respect to weights, biases, and inputs, then updates the parameters. No need to understand how those grad are calculated for this moment, but you should know from the code how the parameters are updated with gradient descent (this latter will be in our test).


#### Backward Function

While we could conceptually write a separate backward function for the entire network as follows:

```python
def backward(network, layer_inputs, grad_output):
    # Now do the backward pass with correct inputs for each layer
    for i in range(len(network))[::-1]:  # Reversed order
        layer = network[i]
        layer_input = layer_inputs[i]
        grad_output = layer.backward(layer_input, grad_output)
        
    return grad_output
```

In the actual implementation, the backward pass is integrated directly in the `train` function.

## Training

### Train for One Batch

The main idea is to apply the forward pass to get the output of the network and hence the loss and the output gradient of the last layer, then the backward pass to update the weights and biases of the network.

Conceptually, it could be implemented as:

```python
def train_one_batch(network, X, y):
    # Forward pass
    layer_activations = forward(network, X) # Then A little bit of other preparatory stuffs
    
    # Compute loss and **initial** gradient (Pression coming from the President of France)
    loss, grad_output = softmax_crossentropy_with_logits(logits, y)
    
    # Backward pass (from Ministers to Mayers then to Citizens)
    backward(network, layer_inputs, grad_output)
    
    return loss
```

This is exactly what our `train` function does.

```python
def train(network, X, y):
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations[:-1] 
    logits = layer_activations[-1]

    loss, grad_logits = softmax_crossentropy_with_logits(logits, y)

    grad_output = grad_logits
    for i in range(len(network))[::-1]: 
        layer = network[i]
        layer_input = layer_inputs[i]
        grad_output = layer.backward(layer_input, grad_output)

    return loss
```

### Training for Multiple Epochs

A simplified version could be:

```python
def train_mnist_network(network, X, y, num_epochs=10):
    for epoch in range(num_epochs):
        train(network, X, y)
```

Our full implementation expands on this with (not to be included in the first test):
1. Network initialization
2. Mini-batch training
3. Data shuffling
4. Validation accuracy measurement
5. Progress tracking

Here's the detailed implementation:

```python
def train_mnist_network(
    X_train, y_train, X_val, y_val, num_epochs=10, batch_size=32, subset_size=10000
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

    # Training loop
    for epoch in range(num_epochs):
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

        # Evaluate on validation set
        val_predictions = predict(network, X_val)
        val_accuracy = np.mean(val_predictions == y_val)
        val_accuracy_history.append(val_accuracy)
```

## Prediction

Predict is just one forward pass.

It runs the input through the network and returns the class with the highest score:

```python
def predict(network, X):
    """Get predictions from the network"""
    # Get the output of the last layer
    logits = forward(network, X)[-1]
    # Return the class with highest score
    return np.argmax(logits, axis=-1)
```

Since `argmax` selects the index of the maximum value, and softmax preserves the ordering of values (softmax is a monotonic function), we can skip the softmax computation here for efficiency.

## Softmax and Cross-entropy

For this first test, we won't test on the softmax transformation and the cross-entropy loss details/code. But we should know the most important concepts and the general idea of the code.

The softmax function converts raw logits into probability distributions, ensuring outputs sum to 1:

```python
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
```

We subtract the maximum value (`np.max(logits, axis=1, keepdims=True)`) before applying `exp()` to prevent numerical overflow issues with large logits.

The combined softmax and cross-entropy function:

```python
def softmax_crossentropy_with_logits(logits, labels):
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
```

Note that the `grad` returned by `softmax_crossentropy_with_logits` function will serve as the input for the backward pass (like Pression Coming From The French President).

The gradient of softmax cross-entropy loss has an elegant form: \((\text{softmax\_probs} - \text{one\_hot\_labels}) / \text{batch\_size}\). This simplicity is one reason why this loss function is widely used for classification tasks.

## Other stuffs

#### Data Preprocessing and Handling

Not important for the exam.

#### Visualization

Not important for the exam.
