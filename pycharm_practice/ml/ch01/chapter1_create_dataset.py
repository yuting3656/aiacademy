import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

"""
畫四個象限的分布圖


             |
    紅色      |    黃色   --- 5
             |
-----(-5)----0-----5-------
             |
    綠色      |    藍色   --- -5
             |         
"""

centers = [[5, 5], [-5, 5], [-5, -5], [5, -5]]

X, y =make_blobs(n_samples= 1000, random_state=50, centers= centers)
colors = 'yrgb'
color_plot = [colors[y[i]] for i in range(len(y))]
plt.scatter(X[:, 0], X[:, 1], color=color_plot, linewidths=0.1)
plt.grid(True)
plt.show()


