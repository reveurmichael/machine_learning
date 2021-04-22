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
        _X = np.vstack(np.ones(n_samples), X)
        X = _X.T
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
            if(np.abs(np.dot(hessianInv, grad)) <= 0.00000001):
                break

    def predict(self, X):
        linear_model = np.dot(X, self.beta)
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)





