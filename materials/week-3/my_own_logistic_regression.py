import numpy as np


'''
https://gitee.com/lundechen/machine_learning/blob/2021/Week-5/linear_regression.py
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
    

'''
https://www.python-engineer.com/courses/mlfromscratch/03_logisticregression/
'''
class LogisticRegressionGD:

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
    

'''
https://www.askpython.com/python/examples/logistic-regression-from-scratch
'''
class LogisticRegressionLoss:
    def __init__(self,x,y):      
        self.intercept = np.ones((x.shape[0], 1))  
        self.x = np.concatenate((self.intercept, x), axis=1)
        self.weight = np.zeros(self.x.shape[1])
        self.y = y
         
    #Sigmoid method
    def sigmoid(self, x, weight):
        z = np.dot(x, weight)
        return 1 / (1 + np.exp(-z))
     
    #method to calculate the Loss
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
     
    #Method for calculating the gradients
    def gradient_descent(self, X, h, y):
        return np.dot(X.T, (h - y)) / y.shape[0]
 
     
    def fit(self, lr , iterations):
        for i in range(iterations):
            sigma = self.sigmoid(self.x, self.weight)
             
            loss = self.loss(sigma,self.y)
 
            dW = self.gradient_descent(self.x , sigma, self.y)
             
            #Updating the weights
            self.weight -= lr * dW
 
        return print('fitted successfully to data')
     
    #Method to predict the class label.
    def predict(self, x_new , treshold):
        x_new = np.concatenate((self.intercept, x_new), axis=1)
        result = self.sigmoid(x_new, self.weight)
        result = result >= treshold
        y_pred = np.zeros(result.shape[0])
        for i in range(len(y_pred)):
            if result[i] == True: 
                y_pred[i] = 1
            else:
                continue
                 
        return y_pred