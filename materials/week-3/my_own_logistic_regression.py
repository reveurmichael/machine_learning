import numpy as np

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
