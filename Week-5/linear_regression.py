# -*- coding: utf-8 -*-

import numpy as np

'''
Different approaches to solve linear regression models
 
There are many different methods that we can apply to our 
linear regression model in order to make it more efficient. 
But we will discuss the most common of them here.

1. Gradient Descent
2. Least Square Method / Normal Equation Method
3. Adams Method
4. Singular Value Decomposition (SVD)
https://www.kdnuggets.com/2020/09/solving-linear-regression.html


SVD:
https://austingwalters.com/using-svd-to-obtain-regression-lines/
https://stats.stackexchange.com/questions/69605/svd-in-linear-regression
https://sthalles.github.io/svd-for-regression/


Related:
Logistic Regression and Decision Boundary
https://towardsdatascience.com/logistic-regression-and-decision-boundary-eab6e00c1e8

'''


class LinearRegressionGradientDescent:
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
            y_predicted = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            # y_predicted = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        # y_predicted = self._sigmoid(linear_model)
        return y_predicted

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


'''
Link:
https://www.quora.com/Does-it-make-sense-to-use-the-Newton-method-for-linear-regression-Does-it-make-sense-to-use-the-curvature-information-in-this-context

Question:
Does it make sense to use the Newton method for linear regression? 
Does it make sense to use the curvature information in this context?

Answer: 
The Newton Method for Linear Regression just ends up becoming the 
Least Square fit solution (you can work it out and prove it to yourself). 
So technically, it makes sense to use Newton’s method for 
linear regression, since Least Square solutions are used to solve 
linear regression problems all the time. But I think it would be 
better to just start with the Least Square framework than overthinking 
it using the Newton method.

Another link:
https://stats.stackexchange.com/questions/207710/newtons-method-for-regression-analysis-without-second-derivative
'''
class LogisticRegressionNewtonRaphson:
    def __init__(self, n_iters=1000):
        self.n_iters = n_iters
        self.beta = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # or: X = np.hstack((np.ones((y.n_samples, 1)), X))
        X = np.concatenate((np.ones((n_samples, 1)), X.to_numpy()), axis=1)

        # init parameters
        self.beta = np.zeros(n_features + 1)

        # Newton Raphson Method
        for _ in range(self.n_iters):
            h = self._sigmoid(np.dot(X, self.beta))
            gradient = np.dot(X.T, (h - y)) / y.size
            diag = np.multiply(h, (1 - h)) * np.identity(n_samples)
            hessian = (1 / n_samples) * np.dot(np.dot(X.T, diag), X)
            self.beta = self.beta - np.dot(np.linalg.inv(hessian), gradient)

    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X.to_numpy()), axis=1)

        linear_model = np.dot(X, self.beta)
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


