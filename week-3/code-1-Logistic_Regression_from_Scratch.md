# Logistic Regression from Scratch

## Introduction

Logistic Regression is a fundamental classification algorithm in machine learning. Unlike Linear Regression which predicts continuous values, Logistic Regression predicts the probability of an instance belonging to a particular class. This tutorial will guide you through implementing Logistic Regression from scratch using different optimization methods.

## References

- [Towards Data Science: Logistic Regression from Scratch in Python](https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2)
- [Python Engineer: ML From Scratch - Logistic Regression](https://www.python-engineer.com/courses/mlfromscratch/03_logisticregression/)
- [Philipp Muens: Logistic Regression from Scratch](https://philippmuens.com/logistic-regression-from-scratch)

## From Linear Regression to Logistic Regression

To understand Logistic Regression, it's helpful to first recap Linear Regression. In Week 2, we implemented Linear Regression from scratch:

```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


class MyOwnLinearRegression:
    def __init__(self, learning_rate=0.0001, n_iters=30000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            # Predict the target values
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage
regressor = MyOwnLinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

The key difference between Linear Regression and Logistic Regression is that Logistic Regression applies a sigmoid function to the linear model output to transform the predictions into probabilities between 0 and 1.

## Implementation 1: Logistic Regression with Gradient Descent

Let's implement Logistic Regression using Gradient Descent:

```python
class LogisticRegressionGD:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iters):
            # Calculate linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
```

### The Math Behind Gradient Descent for Logistic Regression

Logistic Regression uses the sigmoid function to map the linear output to a probability:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

where $z = X \cdot w + b$

The cost function for Logistic Regression is the Binary Cross Entropy:

$$J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})]$$

where $\hat{y}^{(i)} = \sigma(z^{(i)})$

The gradient of this cost function with respect to $w$ and $b$ is:

$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T \cdot (\hat{y} - y)$$
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

## Testing the Gradient Descent Implementation

Let's test our implementation on a simple classification dataset:

```python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a simple classification dataset
X, y = datasets.make_classification(
    n_samples=100, n_features=2, n_redundant=0, 
    n_informative=2, random_state=1, n_clusters_per_class=1
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the logistic regression model
model = LogisticRegressionGD(learning_rate=0.01, n_iters=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Visualize the results
def plot_decision_boundary(X, y, model):
    # Define the bounds of the plot
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Create a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Make predictions on the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create a contour plot
    plt.contourf(xx, yy, Z, alpha=0.3)
    
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50)
    plt.title('Logistic Regression Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot the decision boundary
plot_decision_boundary(X, y, model)
```

## Implementation 2: Newton-Raphson Method

The Newton-Raphson method is an alternative optimization technique that often converges faster than gradient descent for logistic regression:

```python
class LogisticRegressionNewtonRaphson:
    def __init__(self, n_iters=10):
        self.n_iters = n_iters
        self.beta = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Add a column of ones for the intercept
        X_with_intercept = np.concatenate((np.ones((n_samples, 1)), X), axis=1)

        # Initialize parameters
        self.beta = np.zeros(n_features + 1)

        # Newton Raphson Method
        for _ in range(self.n_iters):
            # Compute the predicted probabilities
            z = np.dot(X_with_intercept, self.beta)
            h = self._sigmoid(z)
            
            # Compute the gradient (first derivative)
            gradient = np.dot(X_with_intercept.T, (h - y)) / n_samples
            
            # Compute the Hessian (second derivative)
            W = np.diag(h * (1 - h))
            hessian = (1 / n_samples) * np.dot(np.dot(X_with_intercept.T, W), X_with_intercept)
            
            # Update parameters using Newton's method
            self.beta = self.beta - np.dot(np.linalg.inv(hessian), gradient)

    def predict(self, X):
        n_samples = X.shape[0]
        X_with_intercept = np.concatenate((np.ones((n_samples, 1)), X), axis=1)

        z = np.dot(X_with_intercept, self.beta)
        y_predicted = self._sigmoid(z)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
```

### The Math Behind the Newton-Raphson Method

The Newton-Raphson method uses both the gradient and the Hessian (matrix of second derivatives) to optimize the parameters:

$$\beta_{new} = \beta_{old} - H^{-1} \nabla J$$

where:
- $\nabla J$ is the gradient of the cost function
- $H$ is the Hessian matrix

For logistic regression:
- $\nabla J = X^T (h - y) / n$
- $H = X^T W X / n$, where $W$ is a diagonal matrix with elements $h_i(1-h_i)$

The Newton-Raphson method converges much faster than gradient descent (typically in fewer iterations), but each iteration is more computationally expensive, especially for high-dimensional data.

## Testing the Newton-Raphson Implementation

```python
# Create and train the Newton-Raphson model
nr_model = LogisticRegressionNewtonRaphson(n_iters=10)
nr_model.fit(X_train, y_train)

# Make predictions
y_pred_nr = nr_model.predict(X_test)

# Calculate accuracy
accuracy_nr = accuracy_score(y_test, y_pred_nr)
print(f"Newton-Raphson Accuracy: {accuracy_nr:.4f}")

# Plot the decision boundary
plot_decision_boundary(X, y, nr_model)
```

## Implementation 3: Logistic Regression with Binary Cross-Entropy Loss

This implementation explicitly calculates the loss function during training:

```python
class LogisticRegressionLoss:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
         
    # Sigmoid method
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
     
    # Method to calculate the Loss (Binary Cross Entropy)
    def loss(self, y, y_pred):
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
     
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        losses = []
        
        # Gradient descent
        for i in range(self.iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Calculate loss
            current_loss = self.loss(y, y_pred)
            losses.append(current_loss)
            
            # Backward pass (gradient calculation)
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print loss every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {current_loss:.4f}")
        
        return losses
     
    # Method to predict the class label
    def predict(self, X, threshold=0.5):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        class_predictions = (y_pred >= threshold).astype(int)
        return class_predictions
```

### The Math Behind Binary Cross-Entropy Loss

Binary Cross-Entropy is the standard loss function for binary classification problems:

$$L = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$$

where:
- $y_i$ is the true label (0 or 1)
- $p_i$ is the predicted probability of class 1
- $N$ is the number of samples

This loss function penalizes confident incorrect predictions more heavily than less confident ones.

## Testing the Binary Cross-Entropy Implementation

```python
# Create and train the model with explicit loss calculation
bce_model = LogisticRegressionLoss(learning_rate=0.1, iterations=1000)
losses = bce_model.fit(X_train, y_train)

# Make predictions
y_pred_bce = bce_model.predict(X_test)

# Calculate accuracy
accuracy_bce = accuracy_score(y_test, y_pred_bce)
print(f"Binary Cross-Entropy Accuracy: {accuracy_bce:.4f}")

# Plot the loss over iterations
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Loss vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Binary Cross-Entropy Loss')
plt.grid(True)
plt.show()

# Plot the decision boundary
plot_decision_boundary(X, y, bce_model)
```

## Real-World Example: Social Network Ads Dataset

Let's apply our implementations to a real-world dataset:

```python
# Load the Social Network Ads dataset
social_network = pd.read_csv('Social_Network_Ads.csv')

# Extract features and target
X = social_network[['Age', 'EstimatedSalary']].values
y = social_network['Purchased'].values

# Feature scaling (important for logistic regression)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Train models
gd_model = LogisticRegressionGD(learning_rate=0.01, n_iters=1000)
gd_model.fit(X_train, y_train)

nr_model = LogisticRegressionNewtonRaphson(n_iters=10)
nr_model.fit(X_train, y_train)

bce_model = LogisticRegressionLoss(learning_rate=0.1, iterations=1000)
bce_model.fit(X_train, y_train)

# Make predictions
y_pred_gd = gd_model.predict(X_test)
y_pred_nr = nr_model.predict(X_test)
y_pred_bce = bce_model.predict(X_test)

# Calculate accuracies
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Gradient Descent Model:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gd):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_gd)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_gd)}")

print("\nNewton-Raphson Model:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nr):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_nr)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_nr)}")

print("\nBinary Cross-Entropy Model:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_bce):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_bce)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_bce)}")

# Visualize the results for the Gradient Descent model
from matplotlib.colors import ListedColormap

plt.figure(figsize=(12, 10))

# Create a mesh grid
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
                     np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01))

# Plot the decision boundary
plt.contourf(X1, X2, gd_model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

# Plot the test data points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Logistic Regression (Test set)')
plt.xlabel('Age (Standardized)')
plt.ylabel('Estimated Salary (Standardized)')
plt.legend()
plt.show()
```

## Comparison of Methods

Here's a summary of the three implementations we covered:

1. **Gradient Descent**:
   - Pros: Simple to implement, works for large datasets
   - Cons: Can be slow to converge, requires careful learning rate tuning
   
2. **Newton-Raphson Method**:
   - Pros: Faster convergence (fewer iterations needed)
   - Cons: Computationally expensive for large datasets, requires matrix inversion
   
3. **Explicit Loss Calculation**:
   - Pros: Easy to track progress through loss values
   - Cons: Similar to gradient descent in terms of performance

## Conclusion

In this tutorial, we explored three different implementations of Logistic Regression from scratch. We learned that:

1. Logistic Regression is essentially Linear Regression with a sigmoid function applied to the output
2. The optimization objective is to minimize the Binary Cross-Entropy loss
3. There are multiple optimization methods, each with its own trade-offs

By implementing these algorithms ourselves, we gain a deeper understanding of the mathematics and principles behind logistic regression, which forms the foundation for more complex classification algorithms.

## Exercises

Compare the performance of your implementations with scikit-learn's LogisticRegression


