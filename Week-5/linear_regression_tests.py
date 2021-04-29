import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn
from linear_regression import *

heart = pd.read_csv("SAheart.data")
heart.famhist.replace(to_replace=['Present', 'Absent'], value=[1, 0], inplace=True)
heart.drop(['row.names'], axis=1, inplace=True)
X = heart.iloc[:, :-1]
y = heart.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# regressor = LogisticRegressionGradientDescent(learning_rate=0.0001, n_iters=1000)

regressor = LogisticRegressionNewtonRaphson(n_iters=1000)


regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
perf = sklearn.metrics.confusion_matrix(y_test, y_pred)
print("LR classification perf:\n", perf)

error_rate = np.mean(y_test != y_pred)
print("LR classification error rate:\n", error_rate)
