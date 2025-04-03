# Simple Linear Regression from Scratch

## Introduction

Linear regression is one of the most fundamental algorithms in machine learning. It models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data. Understanding how linear regression works under the hood provides valuable insights into more complex machine learning models.

In this tutorial, we will explore the mathematical foundations of simple linear regression, implement it from scratch using Python and NumPy, and enhance the basic implementation for improved performance and usability.

## Mathematical Foundations

### The Linear Model

At the core of linear regression is the linear model:

$$
y = \mathbf{w}^\top \mathbf{x} + b
$$

- $y$: Predicted value
- $\mathbf{x}$: Input feature vector
- $\mathbf{w}$: Weight vector
- $b$: Bias term

The objective is to find the optimal weights $\mathbf{w}$ and bias $b$ that minimize the difference between the predicted values and the actual target values.

### Cost Function

To quantify the difference between predicted and actual values, we use the Mean Squared Error (MSE) as the cost function:

$$
J(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^{N} (y^{(i)} - (\mathbf{w}^\top \mathbf{x}^{(i)} + b))^2
$$

Where:
- $N$: Number of samples
- $y^{(i)}$: Actual target value for the $i$-th sample
- $\mathbf{x}^{(i)}$: Input feature vector for the $i$-th sample

### Gradient Descent

To minimize the cost function $J(\mathbf{w}, b)$, we employ the Gradient Descent optimization algorithm. Gradient Descent iteratively updates the weights and bias in the direction that reduces the cost.

The gradients of the cost function with respect to the weights and bias are:

$$
\frac{\partial J}{\partial \mathbf{w}} = -\frac{2}{N} \sum_{i=1}^{N} \mathbf{x}^{(i)} (y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b)
$$

$$
\frac{\partial J}{\partial b} = -\frac{2}{N} \sum_{i=1}^{N} (y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b)
$$

Using these gradients, the update rules for weights and bias are:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \times \frac{\partial J}{\partial \mathbf{w}}
$$

$$
b \leftarrow b - \eta \times \frac{\partial J}{\partial b}
$$

Where $\eta$ is the learning rate, a hyperparameter that controls the step size in each update.

## Implementing Linear Regression in Python

Let's implement the linear regression model from scratch using Python and NumPy. We will start with a basic implementation and then enhance it for better performance and usability.

### Initial Implementation

Below is the initial implementation of the `MyOwnLinearRegression` class, which includes methods for fitting the model to data and making predictions.

```python
%matplotlib inline
import numpy as np

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
            # Questions to students: 1 / n_samples or 2 / n_samples, does that matter? 
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Example usage
regressor = MyOwnLinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Visualize the results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

### Code Explanation

#### Initialization 

The `__init__` method initializes the learning rate, number of iterations, and placeholders for weights and bias.

#### Fit Method (`fit`)

The `fit` method trains the model using gradient descent:

1. **Parameter Initialization:**
   - Weights are initialized to zeros.
   - Bias is initialized to zero.

2. **Gradient Descent Loop:**
   - For a specified number of iterations:
     - **Prediction:** Calculate the predicted values using the current weights and bias.
     - **Gradient Calculation:** Compute the gradients of the cost function with respect to weights and bias.
     - **Parameter Update:** Update the weights and bias by moving them in the opposite direction of the gradients.

#### Predict Method (`predict`)

The `predict` method generates predictions using the trained weights and bias:

$$
\hat{y} = \mathbf{w}^\top \mathbf{x} + b
$$

## Improving the Model

While the initial implementation works, there are several enhancements we can make to improve performance and usability:

1. **Tracking the Cost Function:** Monitor the cost function over iterations to observe convergence.
2. **Early Stopping:** Halt training when the improvement in the cost function becomes negligible.
3. **Feature Scaling:** Normalize the features to ensure better convergence.
4. **Verbose Mode:** Provide insights into the training process by printing periodic updates.
5. **Better Initialization:** Use smarter weight initialization techniques.

### Enhanced Implementation

Below is the improved version of the `MyOwnLinearRegression` class, incorporating the aforementioned enhancements.

```python
import numpy as np

class MyOwnLinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000, tolerance=1e-7, verbose=False):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.tolerance = tolerance
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            # Prediction
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute cost (Mean Squared Error)
            cost = (1 / (2 * n_samples)) * np.sum((y_predicted - y) ** 2)
            self.cost_history.append(cost)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Check for convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {i}")
                break

            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

### Enhancements Explained

1. **Cost Function Tracking:**
   - The `cost_history` list stores the cost at each iteration, allowing us to visualize the convergence.
   
2. **Early Stopping:**
   - The `tolerance` parameter defines the minimum improvement required to continue training. If the improvement between consecutive iterations is less than `tolerance`, training stops early.
   
3. **Verbose Mode:**
   - When `verbose=True`, the model prints updates every 100 iterations and notifies when convergence is achieved.

4. **Adjusted Learning Rate and Iterations:**
   - Increased the default learning rate to 0.001 and reduced the number of iterations to 1000 for faster convergence.

## Complete Implementation with Example Usage

Below is the complete implementation of the `MyOwnLinearRegression` class along with an example of how to use it on a synthetic dataset.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

class MyOwnLinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000, tolerance=1e-7, verbose=False):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.tolerance = tolerance
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            # Prediction
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute cost (Mean Squared Error)
            cost = (1 / (2 * n_samples)) * np.sum((y_predicted - y) ** 2)
            self.cost_history.append(cost)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Check for convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {i}")
                break

            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    # Generate a synthetic dataset
    X, y = make_regression(n_samples=1000, n_features=1, noise=3, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling (Standardization)
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    # Initialize and train the model
    regressor = MyOwnLinearRegression(learning_rate=0.01, n_iters=1000, tolerance=1e-8, verbose=True)
    regressor.fit(X_train, y_train)

    # Make predictions
    predictions = regressor.predict(X_test)

    # Plot the results
    plt.figure(figsize=(10,6))
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, predictions, color='red', label='Predicted')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Simple Linear Regression: Actual vs Predicted')
    plt.legend()
    plt.show()
```

### Explanation of Example Usage

1. **Dataset Generation:**
   - We use `make_regression` from scikit-learn to create a synthetic dataset with 1000 samples, one feature, and added noise for realism.

2. **Data Splitting:**
   - The dataset is split into training and testing sets with an 80-20 ratio using `train_test_split`.

3. **Feature Scaling:**
   - Features are standardized to have a mean of 0 and a standard deviation of 1. This step ensures that gradient descent converges more efficiently.

4. **Model Initialization and Training:**
   - An instance of `MyOwnLinearRegression` is created with a learning rate of 0.01, 1000 iterations, a tolerance of $1 \times 10^{-8}$, and verbose mode enabled.
   - The model is trained using the `fit` method on the training data.

5. **Making Predictions:**
   - The trained model is used to predict target values for the test set.

6. **Visualization:**
   - A scatter plot visualizes the actual vs. predicted values, showcasing the performance of the regression model.

### Running the Example

To run the example, ensure you have the necessary libraries installed:

```bash
pip install numpy scikit-learn matplotlib
```

Save the complete code above in a Python file `linear_regression_from_scratch.py` and execute it using Python:

```bash
python linear_regression_from_scratch.py
```

You should see iterative cost updates in the console and a plot displaying the actual versus predicted values after training.

## Evaluation Metrics

To evaluate the performance of our linear regression model, we can use metrics such as Mean Squared Error (MSE) and R-squared ($R^2$).

### Mean Squared Error (MSE)

The MSE measures the average of the squares of the errors between predicted and actual values:

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_{\text{true}}^{(i)} - y_{\text{pred}}^{(i)})^2
$$

### R-squared ($R^2$)

The $R^2$ score indicates the proportion of the variance in the dependent variable that is predictable from the independent variable(s):

$$
R^2 = 1 - \frac{\sum_{i=1}^{N} (y_{\text{true}}^{(i)} - y_{\text{pred}}^{(i)})^2}{\sum_{i=1}^{N} (y_{\text{true}}^{(i)} - \bar{y})^2}
$$

Where $\bar{y}$ is the mean of the actual target values.

### Implementing Evaluation Metrics

We can implement and display these metrics as follows:

```python
from sklearn.metrics import mean_squared_error, r2_score

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
```

Add these lines to the example usage section to evaluate the model's performance after making predictions.



### Visualizing Cost Function Convergence

Plotting the cost function over iterations provides a visual representation of the model's learning process, which can help diagnose potential issues:

```python
# Plotting the Cost Function
plt.figure(figsize=(10,6))
plt.plot(regressor.cost_history, color='purple')
plt.title('Cost Function Convergence')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.grid(True, alpha=0.3)
plt.show()
```

Add this snippet after training to visualize how the cost decreases over iterations.

## Conclusion

In this tutorial, we delved into the fundamentals of simple linear regression, exploring both the mathematical foundations and a practical implementation using Python and NumPy. By implementing linear regression from scratch, you gain a deeper understanding of how machine learning models learn from data, optimize their parameters, and make predictions. This foundational knowledge is crucial as you progress to more complex models and algorithms in machine learning.
