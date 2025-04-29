import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import make_classification, make_moons, make_circles, make_blobs
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import seaborn as sns

st.set_page_config(page_title="Logistic Regression Playground", layout="wide")

st.title("Logistic Regression Playground")

st.markdown("""
## Logistic Regression Implementation Comparison

This app demonstrates logistic regression implemented using three popular libraries:
- scikit-learn
- PyTorch
- TensorFlow

You can experiment with different parameters to compare the implementations.
""")

# Sidebar controls
st.sidebar.header("Dataset Configuration")

# Dataset options
n_samples = st.sidebar.slider("Number of Samples", 10, 300, 100)

# Data shape/pattern selection
data_pattern = st.sidebar.selectbox(
    "Data Pattern",
    ["Linearly Separable", "Moons", "Circles", "Blobs"]
)

# Class options
n_classes = st.sidebar.radio("Number of Classes", [2, 3], index=0)

# Pattern-specific parameters
if data_pattern == "Linearly Separable":
    class_sep = st.sidebar.slider("Class Separation", 0.1, 3.0, 1.0)
    n_clusters = st.sidebar.slider("Clusters per Class", 1, 3, 1)
elif data_pattern == "Moons":
    noise = st.sidebar.slider("Noise", 0.0, 0.5, 0.1)
elif data_pattern == "Circles":
    noise = st.sidebar.slider("Noise", 0.0, 0.5, 0.1)
    factor = st.sidebar.slider("Scale Factor", 0.1, 1.0, 0.8, step=0.1)
elif data_pattern == "Blobs":
    cluster_std = st.sidebar.slider("Cluster Standard Deviation", 0.5, 3.0, 1.0)
    centers = st.sidebar.slider("Number of Centers", 2, 5, 3)

# Generate synthetic dataset based on selected pattern
if data_pattern == "Linearly Separable":
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=2, 
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=n_clusters,
        class_sep=class_sep,
        n_classes=n_classes,
        random_state=42
    )
elif data_pattern == "Moons":
    if n_classes == 2:
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    else:
        # For 3 classes, create modified moons
        X1, y1 = make_moons(n_samples=n_samples//3, noise=noise, random_state=42)
        X2, y2 = make_moons(n_samples=n_samples//3, noise=noise, random_state=43)
        X2[:, 0] += 1.5
        X2[:, 1] += 0.5
        X3, y3 = make_moons(n_samples=n_samples//3, noise=noise, random_state=44)
        X3[:, 0] += 0.5
        X3[:, 1] += 1.5
        
        X = np.vstack([X1, X2, X3])
        y = np.concatenate([y1, y2+1, y3+2])
        
        # Ensure we have exactly 3 classes (0, 1, 2)
        if len(np.unique(y)) > 3:
            y = y % 3
            
elif data_pattern == "Circles":
    if n_classes == 2:
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
    else:
        # For 3 classes, create concentric circles
        inner_samples = n_samples // 3
        middle_samples = n_samples // 3
        outer_samples = n_samples - inner_samples - middle_samples
        
        # Inner circle
        inner_radius = 0.3
        theta = np.linspace(0, 2*np.pi, inner_samples)
        X1 = np.column_stack([
            inner_radius * np.cos(theta),
            inner_radius * np.sin(theta)
        ])
        X1 += np.random.normal(0, noise, X1.shape)
        y1 = np.zeros(inner_samples)
        
        # Middle circle
        middle_radius = 0.6
        theta = np.linspace(0, 2*np.pi, middle_samples)
        X2 = np.column_stack([
            middle_radius * np.cos(theta),
            middle_radius * np.sin(theta)
        ])
        X2 += np.random.normal(0, noise, X2.shape)
        y2 = np.ones(middle_samples)
        
        # Outer circle
        outer_radius = 0.9
        theta = np.linspace(0, 2*np.pi, outer_samples)
        X3 = np.column_stack([
            outer_radius * np.cos(theta),
            outer_radius * np.sin(theta)
        ])
        X3 += np.random.normal(0, noise, X3.shape)
        y3 = np.ones(outer_samples) * 2
        
        X = np.vstack([X1, X2, X3])
        y = np.concatenate([y1, y2, y3])
        
elif data_pattern == "Blobs":
    if n_classes == 2:
        actual_centers = min(centers, 2)
        X, y = make_blobs(n_samples=n_samples, centers=actual_centers, 
                          cluster_std=cluster_std, n_features=2, random_state=42)
    else:
        actual_centers = min(centers, 3)
        X, y = make_blobs(n_samples=n_samples, centers=actual_centers, 
                          cluster_std=cluster_std, n_features=2, random_state=42)
        
# Create DataFrame for display
df = pd.DataFrame({
    'X1': X[:, 0], 
    'X2': X[:, 1], 
    'Class': y
})

# Implementation options
st.sidebar.header("Implementation Options")
implementation = st.sidebar.selectbox(
    "Select Implementation",
    ["scikit-learn", "PyTorch", "TensorFlow"]
)

# Add max_iter for scikit-learn
if implementation == "scikit-learn":
    max_iter = st.sidebar.slider("Max Iterations", 100, 2000, 1000, step=100)
else:
    max_iter = 1000  # Default value

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
accuracy, precision, recall, f1 = None, None, None, None
code_to_display = ""

# Implement logistic regression based on selected library
if implementation == "scikit-learn":
    # scikit-learn implementation
    code_to_display = f"""
# scikit-learn Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize model
model = LogisticRegression(max_iter={max_iter})

# Train model
model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')
"""
    
    # Execute scikit-learn code
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # For visualization - decision boundaries
    def plot_decision_boundary(ax, X, y, model, scaler):
        # Create a mesh grid of points
        h = .02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Scale the mesh grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = scaler.transform(mesh_points)
        
        # Predict class for each point in the mesh
        Z = model.predict(mesh_points_scaled)
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3)
        
        # Plot the data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', alpha=0.8)
        
        return scatter
    
elif implementation == "PyTorch":
    # PyTorch implementation
    code_to_display = """
# PyTorch Logistic Regression
import torch
import torch.nn as nn
import torch.optim as optim

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

# Define model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

# Initialize model, loss, and optimizer
model = LogisticRegressionModel(X_train_scaled.shape[1], n_classes)
criterion = nn.CrossEntropyLoss()
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
    _, y_train_pred = torch.max(model(X_train_tensor), 1)
    _, y_test_pred = torch.max(model(X_test_tensor), 1)
    y_train_pred = y_train_pred.numpy()
    y_test_pred = y_test_pred.numpy()

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')
"""
    
    # Execute PyTorch code
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    class LogisticRegressionModel(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(LogisticRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, num_classes)
            
        def forward(self, x):
            return self.linear(x)
    
    model = LogisticRegressionModel(X_train_scaled.shape[1], n_classes)
    criterion = nn.CrossEntropyLoss()
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
        _, y_train_pred = torch.max(model(X_train_tensor), 1)
        _, y_test_pred = torch.max(model(X_test_tensor), 1)
        y_train_pred = y_train_pred.numpy()
        y_test_pred = y_test_pred.numpy()
    
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # For visualization - decision boundaries
    def plot_decision_boundary(ax, X, y, model, scaler):
        # Create a mesh grid of points
        h = .02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Scale the mesh grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = scaler.transform(mesh_points)
        mesh_tensor = torch.FloatTensor(mesh_points_scaled)
        
        # Predict class for each point in the mesh
        with torch.no_grad():
            _, Z = torch.max(model(mesh_tensor), 1)
        Z = Z.numpy().reshape(xx.shape)
        
        # Plot the decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3)
        
        # Plot the data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', alpha=0.8)
        
        return scatter
    
else:  # TensorFlow implementation
    # TensorFlow implementation
    code_to_display = """
# TensorFlow Logistic Regression
import tensorflow as tf

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(n_classes, input_shape=(X_train_scaled.shape[1],), activation='softmax')
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(
    X_train_scaled, y_train,
    epochs=epochs,
    verbose=0,
    validation_split=0.2
)

# Evaluate model
y_train_pred = np.argmax(model.predict(X_train_scaled), axis=1)
y_test_pred = np.argmax(model.predict(X_test_scaled), axis=1)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')
"""
    
    # Execute TensorFlow code
    tf.random.set_seed(42)
    
    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(n_classes, input_shape=(X_train_scaled.shape[1],), activation='softmax')
    ])
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        verbose=0,
        validation_split=0.2
    )
    
    # Evaluate model
    y_train_pred = np.argmax(model.predict(X_train_scaled), axis=1)
    y_test_pred = np.argmax(model.predict(X_test_scaled), axis=1)
    
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # For visualization - decision boundaries
    def plot_decision_boundary(ax, X, y, model, scaler):
        # Create a mesh grid of points
        h = .02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Scale the mesh grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = scaler.transform(mesh_points)
        
        # Predict class for each point in the mesh
        Z = np.argmax(model.predict(mesh_points_scaled), axis=1)
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3)
        
        # Plot the data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', alpha=0.8)
        
        return scatter

# Function for visualizing decision boundary for all implementations
if 'plot_decision_boundary' not in locals():
    def plot_decision_boundary(ax, X, y, model, scaler):
        # Create a mesh grid of points
        h = .02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Scale the mesh grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = scaler.transform(mesh_points)
        
        # Predict class for each point in the mesh (implementation specific)
        if implementation == "scikit-learn":
            Z = model.predict(mesh_points_scaled)
        elif implementation == "PyTorch":
            mesh_tensor = torch.FloatTensor(mesh_points_scaled)
            with torch.no_grad():
                _, Z = torch.max(model(mesh_tensor), 1)
            Z = Z.numpy()
        else:  # TensorFlow
            Z = np.argmax(model.predict(mesh_points_scaled), axis=1)
        
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3)
        
        # Plot the data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', alpha=0.8)
        
        return scatter

# Display results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    
    # Data and Decision Boundary - Moved before Confusion Matrix
    st.subheader("Data and Decision Boundary")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = plot_decision_boundary(ax, X, y, model, scaler)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax, label='Class')
    st.pyplot(fig)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_test_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    st.pyplot(fig_cm)

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
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        st.pyplot(fig2)
    elif implementation == "PyTorch":
        # Plot PyTorch training history
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='Training Loss')
        ax.plot(val_losses, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)

# Display model information with mathematical formulation
st.subheader("Model Information")
if implementation == "scikit-learn":
    # Combine coefficients and intercepts into a single table
    if n_classes == 2:
        # For binary classification
        coef = model.coef_[0]
        intercept = model.intercept_[0]
        
        # Create combined DataFrame
        model_df = pd.DataFrame({
            "Value": np.append(coef, intercept)
        }, index=[f"Weight (Feature {i+1})" for i in range(X.shape[1])] + ["Bias"])
        
        st.dataframe(model_df)
        
        # Mathematical formulation for binary classification
        w1 = coef[0]
        w2 = coef[1]
        b = intercept
        
        st.markdown(f"""
        ### Mathematical Formula:
        
        For binary logistic regression, the probability of class 1 is given by:
        
        $P(y=1|x) = \\frac{{1}}{{1 + e^{{-(w_1 x_1 + w_2 x_2 + b)}}}}$
        
        where:
        - $w_1 = {w1:.4f}$
        - $w_2 = {w2:.4f}$
        - $b = {b:.4f}$
        
        The decision boundary is where $P(y=1|x) = 0.5$, which occurs when:
        
        ${w1:.4f} x_1 + {w2:.4f} x_2 + {b:.4f} = 0$
        """)
    else:
        # For multi-class classification
        # Create combined DataFrame with weights and bias for each class
        model_df = pd.DataFrame(
            index=[f"Weight (Feature {i+1})" for i in range(X.shape[1])] + ["Bias"]
        )
        
        for i in range(n_classes):
            class_weights = model.coef_[i]
            class_bias = model.intercept_[i]
            model_df[f"Class {i}"] = np.append(class_weights, class_bias)
        
        st.dataframe(model_df)
        
        # Mathematical formulation for multi-class
        st.markdown("""
        ### Mathematical Formula:
        
        For multi-class logistic regression, the probability of class k is given by the softmax function:
        
        $P(y=k|x) = \\frac{{e^{{w_k^T x + b_k}}}}{{\\sum_{j=0}^{K-1} e^{{w_j^T x + b_j}}}}$
        
        where:
        - $w_k$ is the weight vector for class k
        - $b_k$ is the bias term for class k
        - $K$ is the number of classes
        """)
        
elif implementation == "PyTorch":
    with torch.no_grad():
        weights = model.linear.weight.numpy()
        bias = model.linear.bias.numpy()
        
        # Combine weights and bias into a single table
        model_df = pd.DataFrame(
            index=[f"Weight (Feature {i+1})" for i in range(X.shape[1])] + ["Bias"]
        )
        
        for i in range(n_classes):
            if n_classes == 2 and i == 0:
                # For binary classification, only show one class
                continue
                
            class_weights = weights[i]
            class_bias = bias[i]
            model_df[f"Class {i}"] = np.append(class_weights, class_bias)
        
        st.dataframe(model_df)
        
        # Mathematical formulation based on PyTorch model
        if n_classes == 2:
            w1 = weights[1][0]  # Class 1 weights
            w2 = weights[1][1]
            b = bias[1]
            
            st.markdown(f"""
            ### Mathematical Formula:
            
            For binary classification (using PyTorch's CrossEntropyLoss), the model computes:
            
            $z = {w1:.4f} x_1 + {w2:.4f} x_2 + {b:.4f}$
            
            Then applies softmax to get probabilities.
            
            The decision boundary is where the logit equals 0:
            
            ${w1:.4f} x_1 + {w2:.4f} x_2 + {b:.4f} = 0$
            """)
        else:
            st.markdown("""
            ### Mathematical Formula:
            
            For multi-class classification, for each class k:
            
            $z_k = w_{k,1} x_1 + w_{k,2} x_2 + b_k$
            
            Then, the probability of each class is calculated using the softmax function:
            
            $P(y=k|x) = \\frac{{e^{{z_k}}}}{{\\sum_{j=0}^{K-1} e^{{z_j}}}}$
            
            The class with the highest probability is the predicted class.
            """)
else:  # TensorFlow
    weights, bias = model.get_weights()
    
    # Combine weights and bias into a single table
    model_df = pd.DataFrame(
        index=[f"Weight (Feature {i+1})" for i in range(X.shape[1])] + ["Bias"]
    )
    
    for i in range(n_classes):
        class_weights = [weights[0][i], weights[1][i]]  # Extract weights for each feature for this class
        class_bias = bias[i]
        model_df[f"Class {i}"] = np.append(class_weights, class_bias)
    
    st.dataframe(model_df)
    
    # Mathematical formulation based on TensorFlow model
    if n_classes == 2:
        w1 = weights[0][0]
        w2 = weights[1][0]
        b = bias[0]
        
        st.markdown(f"""
        ### Mathematical Formula:
        
        For binary classification, the TensorFlow model computes:
        
        $z = {w1:.4f} x_1 + {w2:.4f} x_2 + {b:.4f}$
        
        Then applies softmax to get probability.
        
        The decision boundary is where the softmax probability equals 0.5:
        
        ${w1:.4f} x_1 + {w2:.4f} x_2 + {b:.4f} = 0$
        """)
    else:
        st.markdown("""
        ### Mathematical Formula:
        
        For multi-class classification, the TensorFlow model computes for each class k:
        
        $z_k = \\sum_{i=1}^{n} w_{i,k} x_i + b_k$
        
        Then applies softmax to get the probability distribution:
        
        $P(y=k|x) = \\frac{{e^{{z_k}}}}{{\\sum_{j=0}^{K-1} e^{{z_j}}}}$
        
        The class with the highest probability is the predicted class.
        """)

st.markdown("""
## How Logistic Regression Works

Logistic regression is a classification algorithm that predicts the probability of an observation belonging to a particular class. 
Unlike linear regression which outputs continuous values, logistic regression applies the sigmoid function (for binary classification) 
or softmax function (for multi-class classification) to transform the output to a probability between 0 and 1.

### Implementations Comparison:

1. **scikit-learn**: 
   - Simplest implementation using a high-level API
   - Uses optimization algorithms like LBFGS or SAG by default
   - Fast for small to medium datasets

2. **PyTorch**:
   - Uses gradient descent optimization
   - Provides more flexibility in model customization
   - Easy tracking of gradients and training progress

3. **TensorFlow**:
   - Similar to PyTorch but with different API
   - Good for production deployment
   - Built-in training history tracking

Try different parameters to see how each implementation performs on classification tasks!
""")
