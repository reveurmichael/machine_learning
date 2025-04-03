

## References that can be useful 

https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2

https://www.python-engineer.com/courses/mlfromscratch/03_logisticregression/

https://philippmuens.com/logistic-regression-from-scratch


## From Linear Regression to Logistic Regression


Priviously (week 2), we had Linear Regression from scratch :



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
a = regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

```


Now we need just add several lines of code, then we will have Logistic Regression from scratch :





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



And now, it's time to test it on our Social_Network_Ads.csv dataset , as well as a sklearn generated sample dataset: 


????????




Here is the explaination of the code as well as the math behind it:

????



As we can see, the code is very similar to the Linear Regression from scratch, but with a different loss function and a different gradient. The code added is very minimal.



## ALternative 1: Newton Raphson Method




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
        


Here is the explaination of the code as well as the math behind it:

????



## ALternative 2 : Cross Entropy Loss




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


Here is the explaination of the code as well as the math behind it:

????
