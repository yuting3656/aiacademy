import numpy as np

from sklearn.model_selection import KFold

x = np.array([[1, 2],
              [3, 4], [5, 6],
              [7, 8], [9, 10]])

y = np.array([1, 2, 3, 4, 5])

kf = KFold(n_splits=5,
           random_state=None,
           shuffle=False)

for train_index, val_index in kf.split(x):
    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]
    print("TRAIN index:", train_index, "Validation index:", val_index)
