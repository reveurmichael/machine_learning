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
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(
            2.0 / input_units
        )
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
            self.weights, self.biases = self.optimizer.update(
                self.weights, self.biases, grad_weights, grad_biases
            )

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
        self.m_weights[key_w] = (
            self.beta1 * self.m_weights[key_w] + (1 - self.beta1) * grad_weights
        )
        self.m_biases[key_b] = (
            self.beta1 * self.m_biases[key_b] + (1 - self.beta1) * grad_biases
        )

        # Update biased second moment estimate for weights and biases
        self.v_weights[key_w] = self.beta2 * self.v_weights[key_w] + (
            1 - self.beta2
        ) * (grad_weights**2)
        self.v_biases[key_b] = self.beta2 * self.v_biases[key_b] + (1 - self.beta2) * (
            grad_biases**2
        )

        # Bias-corrected estimates
        m_hat_weights = self.m_weights[key_w] / (1 - self.beta1**self.t)
        v_hat_weights = self.v_weights[key_w] / (1 - self.beta2**self.t)
        m_hat_biases = self.m_biases[key_b] / (1 - self.beta1**self.t)
        v_hat_biases = self.v_biases[key_b] / (1 - self.beta2**self.t)

        # Update parameters
        weights_updated = weights - self.learning_rate * m_hat_weights / (
            np.sqrt(v_hat_weights) + self.epsilon
        )
        biases_updated = biases - self.learning_rate * m_hat_biases / (
            np.sqrt(v_hat_biases) + self.epsilon
        )

        return weights_updated, biases_updated


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
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, parser="auto")

    # Convert pandas DataFrame to numpy arrays
    X = X.to_numpy().astype("float32")
    y = y.to_numpy().astype("int")

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
    std = (
        np.std(X_train_flat, axis=0) + 1e-9
    )  # Add small constant to avoid division by zero

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


# Load MNIST dataset
X_train, y_train, X_test, y_test = load_mnist()

# Preprocess the data
X_train, y_train, X_val, y_val, X_test, y_test = preprocess_mnist(
    X_train, y_train, X_test, y_test
)


def create_network(optimizer_type="sgd"):
    """
    Creates a neural network with three hidden layers using the specified optimizer.
    Arguments:
        optimizer_type -- 'sgd' or 'adam'
    Returns:
        network -- list of layers forming the network.
    """
    if optimizer_type == "sgd":
        optimizer = SGD(learning_rate=0.05)
    elif optimizer_type == "adam":
        optimizer = Adam(learning_rate=0.0005, beta1=0.9, beta2=0.999)
    else:
        raise ValueError("Unsupported optimizer type. Use 'sgd' or 'adam'.")

    network = [
        Dense(
            input_units=784, output_units=256, learning_rate=0.03, optimizer=optimizer
        ),
        ReLU(),
        Dense(
            input_units=256, output_units=128, learning_rate=0.03, optimizer=optimizer
        ),
        ReLU(),
        Dense(
            input_units=128, output_units=64, learning_rate=0.03, optimizer=optimizer
        ),
        ReLU(),
        Dense(input_units=64, output_units=10, learning_rate=0.03, optimizer=optimizer),
    ]
    return network


def train_network(
    network, X_train, y_train, X_val, y_val, num_epochs=10, batch_size=64
):
    """
    Trains the neural network on training data and periodically evaluates on validation data.
    """
    num_samples = X_train.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    history = {"loss": [], "val_accuracy": []}

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
        history["loss"].append(avg_loss)
        print(f"Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}")

        # Evaluate validation accuracy on a subset (for speed)
        val_indices = np.random.choice(X_val.shape[0], size=1000, replace=False)
        val_preds = predict(network, X_val[val_indices])
        val_acc = np.mean(val_preds == y_val[val_indices])
        history["val_accuracy"].append(val_acc)
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
    plt.imshow(conf_matrix, cmap="Blues")
    plt.colorbar()

    # Add labels
    plt.xticks(np.arange(10), np.arange(10))
    plt.yticks(np.arange(10), np.arange(10))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Add text annotations
    for i in range(10):
        for j in range(10):
            plt.text(
                j,
                i,
                str(conf_matrix[i, j]),
                ha="center",
                va="center",
                color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black",
            )

    plt.tight_layout()
    plt.show()

    return accuracy


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
network_adam = create_network(optimizer_type="adam")
print("Training with Adam optimizer...")
history_adam = train_network(
    network_adam,
    X_train_subset,
    y_train_subset,
    X_val_subset,
    y_val_subset,
    num_epochs=20,
    batch_size=64,
)

# Create network using SGD optimizer for comparison
network_sgd = create_network(optimizer_type="sgd")
print("\nTraining with SGD optimizer...")
history_sgd = train_network(
    network_sgd,
    X_train_subset,
    y_train_subset,
    X_val_subset,
    y_val_subset,
    num_epochs=20,
    batch_size=64,
)

# Evaluate both models on test data
print("\nEvaluating Adam model...")
adam_accuracy = evaluate_network(network_adam, X_test, y_test)

print("\nEvaluating SGD model...")
sgd_accuracy = evaluate_network(network_sgd, X_test, y_test)

print(f"\nAdam accuracy: {adam_accuracy*100:.2f}%")
print(f"SGD accuracy: {sgd_accuracy*100:.2f}%")


def plot_training_comparison(history_adam, history_sgd):
    """Compare training performance between Adam and SGD optimizers"""
    epochs = range(1, len(history_adam["loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_adam["loss"], marker="o", label="Adam")
    plt.plot(epochs, history_sgd["loss"], marker="s", label="SGD")
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_adam["val_accuracy"], marker="o", label="Adam")
    plt.plot(epochs, history_sgd["val_accuracy"], marker="s", label="SGD")
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_training_comparison(history_adam, history_sgd)


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
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"True: {y_samples[idx]}")
        axes[0, i].axis("off")

        # Add prediction with color coding (green for correct, red for incorrect)
        color = "green" if predictions[idx] == y_samples[idx] else "red"
        axes[1, i].text(
            0.5,
            0.5,
            f"Pred: {predictions[idx]}",
            horizontalalignment="center",
            verticalalignment="center",
            color=color,
            fontsize=12,
        )
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


# Visualize predictions using the better model
better_network = network_adam if adam_accuracy > sgd_accuracy else network_sgd
visualize_predictions(better_network, X_test, y_test)


