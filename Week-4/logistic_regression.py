# -*- coding: utf-8 -*-

import numpy as np

class LogisticRegressionGradientDescent:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class LogisticRegressionNewtonRaphson:
    def __init__(self, n_iters=1000):
        self.n_iters = n_iters
        self.beta = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X = np.concatenate((np.ones((n_samples, 1)), X.to_numpy()), axis=1)
        # init parameters
        self.beta = np.zeros(n_features + 1)
        XT = X.transpose()

        def gradient():
            sig = self._sigmoid(np.dot(X, self.beta))
            grad = np.dot(XT, (sig - y))
            grad = grad / len(X)
            return grad

        def hessian():
            sig = self._sigmoid(np.dot(X, self.beta))
            result = (1.0 / len(X) * np.dot(XT, X) * np.diag(sig) * np.diag(1 - sig))
            return result

        # Newton Raphson Method
        for _ in range(self.n_iters):
            hessianInv = np.linalg.inv(hessian())
            grad = gradient()
            self.beta = self.beta - np.dot(hessianInv, grad)
            # if(np.abs(np.dot(hessianInv, grad)) <= 0.00000001):
            #     break

    def predict(self, X):
        linear_model = np.dot(X, self.beta)
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))



# https://thelaziestprogrammer.com/sharrington/math-of-machine-learning/solving-logreg-newtons-method

def sigmoid(x, Θ_1, Θ_2):
    z = (Θ_1*x + Θ_2).astype("float_")
    return 1.0 / (1.0 + np.exp(-z))

def log_likelihood(x, y, Θ_1, Θ_2):
    sigmoid_probs = sigmoid(x, Θ_1, Θ_2)
    return np.sum(y * np.log(sigmoid_probs)
                  + (1 - y) * np.log(1 - sigmoid_probs))

def gradient(x, y, Θ_1, Θ_2):
    sigmoid_probs = sigmoid(x, Θ_1, Θ_2)
    return np.array([[np.sum((y - sigmoid_probs) * x),
                     np.sum((y - sigmoid_probs) * 1)]])

def hessian(x, y, Θ_1, Θ_2):
    sigmoid_probs = sigmoid(x, Θ_1, Θ_2)
    d1 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x * x)
    d2 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x * 1)
    d3 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * 1 * 1)
    H = np.array([[d1, d2],[d2, d3]])
    return H

def newtons_method(x, y):
    """
    :param x (np.array(float)): Vector of Boston House Values in dollars
    :param y (np.array(boolean)): Vector of Bools indicting if house has > 2 bedrooms:
    :returns: np.array of logreg's parameters after convergence, [Θ_1, Θ_2]
    """
    # Initialize log_likelihood & parameters
    Θ_1 = 15.1
    Θ_2 = -.4  # The intercept term
    Δl = np.Infinity
    l = log_likelihood(x, y, Θ_1, Θ_2)
    # Convergence Conditions
    δ = .0000000001
    max_iterations = 15
    i = 0
    while abs(Δl) > δ and i < max_iterations:
        i += 1
        g = gradient(x, y, Θ_1, Θ_2)
        hess = hessian(x, y, Θ_1, Θ_2)
        H_inv = np.linalg.inv(hess)
        # @ is syntactic sugar for np.dot(H_inv, g.T)¹
        Δ = H_inv @ g.T
        ΔΘ_1 = Δ[0][0]
        ΔΘ_2 = Δ[1][0]

        # Perform our update step
        Θ_1 += ΔΘ_1
        Θ_2 += ΔΘ_2

        # Update the log-likelihood at each iteration
        l_new = log_likelihood(x, y, Θ_1, Θ_2)
        Δl = l - l_new
        l = l_new
    return np.array([Θ_1, Θ_2])

