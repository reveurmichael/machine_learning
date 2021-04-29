import numpy as np
import pandas as pd
import sklearn
from linear_regression import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    prostate = pd.read_table("prostate.data")
    prostate.drop(prostate.columns[0], axis=1, inplace=True)
    
    X = prostate.drop(["lpsa", "train"], axis=1)
    y = prostate["lpsa"]

    regressor = LinearRegressionGradientDescent()

    regressor.fit(X, y)
    y_pred = regressor.predict(X)

    print(regressor.__dict__)
    print(y - y_pred)

    plt.scatter(y, y_pred)
    plt.plot([0, 5], [0, 5])
    plt.show()
