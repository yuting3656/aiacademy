import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

X = np.array([i * np.pi/180 for i in range(60, 300, 4)])
np.random.seed(100)
y = np.sin(X) + np.random.normal(0, 0.15, len(X))

X.reshape(60, -1)
y.reshape(60, -1)

data = pd.DataFrame(np.column_stack([X, y]), columns=['X', 'y'])
plt.plot(data['X'], data['y'], '.')
plt.show()


def example_svm_regression(X, y, plot_dict, kernel, C = 1):

    for params in plot_dict:
        kernel_dict = { 'linear': SVR(kernel='linear',C=params),
                       'ploy': SVR(kernel='ploy', C=C, degree=params),
                       'rbf': SVR(kernel='rbf', C =C, gamma=params)}

        if kernel in kernel_dict:
            model = kernel_dict[kernel]
            model.fit(X, y)
            y_pred = model.predict(X)
            mae = mean_absolute_error(y_pred, y)

