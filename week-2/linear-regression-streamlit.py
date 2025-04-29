import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf

st.set_page_config(page_title="Linear Regression Playground", layout="wide")

st.title("Linear Regression Playground")

st.markdown("""
## Linear Regression Implementation Comparison

This app demonstrates linear regression implemented using three popular libraries:
- scikit-learn
- PyTorch
- TensorFlow

You can experiment with different parameters to compare the implementations.
""")

# Sidebar controls
st.sidebar.header("Dataset Configuration")

# Dataset options
n_samples = st.sidebar.slider("Number of Samples", 50, 1000, 200)
noise = st.sidebar.slider("Noise", 0.0, 50.0, 10.0)

# Generate regression dataset with exactly 1 feature
X, y = make_regression(n_samples=n_samples, n_features=1, noise=noise, random_state=42)
X_display = X

# Create DataFrame for display
df = pd.DataFrame({'X': X.flatten(), 'y': y})

# Implementation options
st.sidebar.header("Implementation Options")
implementation = st.sidebar.selectbox(
    "Select Implementation",
    ["scikit-learn", "PyTorch", "TensorFlow"]
)

learning_rate = st.sidebar.number_input("Learning Rate (for PyTorch/TensorFlow)", 0.0001, 1.0, 0.01)
epochs = st.sidebar.slider("Epochs (for PyTorch/TensorFlow)", 1, 100, 30, step=10)

# Main content area
# Split data into train/test
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
train_indices, test_indices = indices[:train_size], indices[train_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize results containers
train_mse, test_mse, r2 = None, None, None
code_to_display = ""

# Implement linear regression based on selected library
if implementation == "scikit-learn":
    # scikit-learn implementation
    code_to_display = """
# scikit-learn Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize model
model = LinearRegression()

# Train model
model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluate model
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
"""
    
    # Execute scikit-learn code
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    
    # For visualization on original scale
    X_vis = np.linspace(min(X_display), max(X_display), 100).reshape(-1, 1)
    X_vis_scaled = scaler.transform(X_vis)
    y_vis = model.predict(X_vis_scaled)
    
elif implementation == "PyTorch":
    # PyTorch implementation
    code_to_display = """
# PyTorch Linear Regression
import torch
import torch.nn as nn
import torch.optim as optim

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))

# Define model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

# Initialize model, loss, and optimizer
model = LinearRegressionModel(X_train_scaled.shape[1])
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop with history tracking
train_losses = []
val_losses = []
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    train_losses.append(loss.item())
    
    # Validation loss
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
        val_losses.append(val_loss.item())
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate model
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train_tensor).numpy().flatten()
    y_test_pred = model(X_test_tensor).numpy().flatten()

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
"""
    
    # Execute PyTorch code
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))
    
    class LinearRegressionModel(nn.Module):
        def __init__(self, input_dim):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, 1)
            
        def forward(self, x):
            return self.linear(x)
    
    model = LinearRegressionModel(X_train_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop with history tracking
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        train_losses.append(loss.item())
        
        # Validation loss
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            val_losses.append(val_loss.item())
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_tensor).numpy().flatten()
        y_test_pred = model(X_test_tensor).numpy().flatten()
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    
    # For visualization on original scale
    X_vis = np.linspace(min(X_display), max(X_display), 100).reshape(-1, 1)
    X_vis_scaled = scaler.transform(X_vis)
    X_vis_tensor = torch.FloatTensor(X_vis_scaled)
    with torch.no_grad():
        y_vis = model(X_vis_tensor).numpy().flatten()
    
else:  # TensorFlow implementation
    # TensorFlow implementation
    code_to_display = """
# TensorFlow Linear Regression
import tensorflow as tf

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(X_train_scaled.shape[1],))
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              loss='mse',
              metrics=['mae'])

# Train model
history = model.fit(
    X_train_scaled, y_train,
    epochs=epochs,
    verbose=0,
    validation_split=0.2
)

# Evaluate model
y_train_pred = model.predict(X_train_scaled).flatten()
y_test_pred = model.predict(X_test_scaled).flatten()

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
"""
    
    # Execute TensorFlow code
    tf.random.set_seed(42)
    
    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(X_train_scaled.shape[1],))
    ])
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae'])
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        verbose=0,
        validation_split=0.2
    )
    
    # Evaluate model
    y_train_pred = model.predict(X_train_scaled).flatten()
    y_test_pred = model.predict(X_test_scaled).flatten()
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    
    # For visualization on original scale
    X_vis = np.linspace(min(X_display), max(X_display), 100).reshape(-1, 1)
    X_vis_scaled = scaler.transform(X_vis)
    y_vis = model.predict(X_vis_scaled).flatten()

# Display results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Performance")
    st.write(f"Training MSE: {train_mse:.4f}")
    st.write(f"Test MSE: {test_mse:.4f}")
    st.write(f"R² Score: {r2:.4f}")
    
    st.subheader("Data and Prediction")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_display[train_indices], y_train, color='blue', alpha=0.5, label='Training Data')
    ax.scatter(X_display[test_indices], y_test, color='green', alpha=0.5, label='Test Data')
    
    # Plot the prediction line
    if 'X_vis' in locals() and 'y_vis' in locals():
        ax.plot(X_vis, y_vis, color='red', linewidth=2, label='Prediction')
    
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Implementation Code")
    st.code(code_to_display, language="python")
    
    st.subheader("Training History")
    if implementation == "TensorFlow":
        # Plot TensorFlow training history
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.legend()
        st.pyplot(fig)
    elif implementation == "PyTorch":
        # Plot PyTorch training history
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='Training Loss')
        ax.plot(val_losses, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.legend()
        st.pyplot(fig)

# Display model information
st.subheader("Model Information")
if implementation == "scikit-learn":
    st.write(f"Coefficient: {model.coef_[0]:.4f}")
    st.write(f"Intercept: {model.intercept_:.4f}")
    st.write(f"Equation: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}")
elif implementation == "PyTorch":
    with torch.no_grad():
        weights = model.linear.weight.numpy()
        bias = model.linear.bias.item()
        st.write(f"Weight: {weights[0][0]:.4f}")
        st.write(f"Bias: {bias:.4f}")
        st.write(f"Equation: y = {weights[0][0]:.4f}x + {bias:.4f}")
else:  # TensorFlow
    weights = model.get_weights()[0]
    bias = model.get_weights()[1][0]
    st.write(f"Weight: {weights[0][0]:.4f}")
    st.write(f"Bias: {bias:.4f}")
    st.write(f"Equation: y = {weights[0][0]:.4f}x + {bias:.4f}")

st.markdown("""
## How Linear Regression Works

Linear regression finds the best-fitting line (or hyperplane for multiple features) through data points by minimizing the
sum of squared differences between observed and predicted values.

### Implementations Comparison:

1. **scikit-learn**: 
   - Simplest implementation using a high-level API
   - Solves using closed-form solution (normal equation)
   - Fast for small to medium datasets

2. **PyTorch**:
   - Uses gradient descent optimization
   - Provides more flexibility in model customization
   - Allows easy tracking of gradients

3. **TensorFlow**:
   - Similar to PyTorch but with different API
   - Good for production deployment
   - Built-in training history tracking

Try different parameters to see how each implementation performs!
""")
