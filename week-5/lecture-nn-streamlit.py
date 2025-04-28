import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_classification
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

st.set_page_config(page_title="Neural Network Playground", layout="wide")

st.title("Neural Network Playground")

st.markdown("""
## Introduction to Neural Networks

Neural Networks are powerful machine learning models that can learn complex patterns in data.
This playground lets you experiment with different architectures and datasets to build intuition.

### Key Concepts:

1. **Neurons**: Basic computational units that apply weights and biases to inputs
2. **Layers**: Groups of neurons that transform data sequentially
3. **Activation Functions**: Non-linear functions that allow networks to learn complex patterns
4. **Training**: Process of adjusting weights and biases to minimize error
""")

# Sidebar controls
st.sidebar.header("Network Configuration")

# Dataset selection
dataset_type = st.sidebar.selectbox(
    "Select Dataset",
    ["Circles", "Moons", "Binary Classification"]
)

# Network architecture
n_hidden = st.sidebar.slider("Number of Hidden Layers", 1, 5, 2)
neurons_per_layer = st.sidebar.slider("Neurons per Hidden Layer", 2, 32, 16)
activation = st.sidebar.selectbox(
    "Activation Function",
    ["ReLU", "Tanh", "Sigmoid"]
)

# Training parameters
learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 1.0, 0.01)
epochs = st.sidebar.slider("Number of Epochs", 10, 1000, 100)
batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)

# Generate dataset
n_samples = 1000
if dataset_type == "Circles":
    X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.3)
elif dataset_type == "Moons":
    X, y = make_moons(n_samples=n_samples, noise=0.1)
else:
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_clusters_per_class=1)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y.reshape(-1, 1))

# Create dataset and dataloader
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, neurons_per_layer, activation):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, neurons_per_layer))
        
        # Activation function
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        else:  # Sigmoid
            self.activation = nn.Sigmoid()
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
        
        # Output layer
        self.output = nn.Linear(neurons_per_layer, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.sigmoid(self.output(x))
        return x

# Initialize model
model = NeuralNetwork(input_size=2, hidden_layers=n_hidden, 
                     neurons_per_layer=neurons_per_layer, activation=activation)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()
    
    train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

# Plot training history
col1, col2 = st.columns(2)

with col1:
    st.subheader("Training Loss")
    fig_loss = plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(fig_loss)

with col2:
    st.subheader("Training Accuracy")
    fig_acc = plt.figure()
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    st.pyplot(fig_acc)

# Plot decision boundary
st.subheader("Decision Boundary")

model.eval()
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
with torch.no_grad():
    Z = model(grid_tensor).numpy()
Z = Z.reshape(xx.shape)

fig_boundary = plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"Decision Boundary - {dataset_type} Dataset")
st.pyplot(fig_boundary)

st.markdown("""
### Implementation Details

1. **Forward Pass**:
   - Input features are processed through multiple layers
   - Each layer applies weights, biases, and activation functions
   - Final layer produces classification probability

2. **Training Process**:
   - Uses backpropagation to compute gradients
   - Updates weights using Adam optimizer
   - Minimizes binary cross-entropy loss

### Tips for Better Results:

- More complex datasets may require more layers/neurons
- Adjust learning rate if training is unstable
- Try different activation functions for different problems
- Watch for overfitting in validation metrics
""")
