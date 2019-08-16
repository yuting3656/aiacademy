import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# maker 3 class dataset for classification
centers = [[0, 0], [3.5, 0]]
X, y = make_blobs(n_samples=100, centers=centers, random_state=42)

color = 'yg'
color = [ color[y[i]] for i in range(len(y))]
plt.scatter(X[:, 0], X[:, 1], c=color)


# plot_decision_regions
def plot_decision_regions(X, y, classifier, test_idx = None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_min, resolution),
                           np.arange(x2_min, x2_max, resolution))
    np.ravel()


def knn_example(plot_dict, X, y, k):
    # create model
    model = KNeighborsClassifier(n_neighbors=k)

    # training
    model.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    if k in plot_dict:
        plt.subplot(plot_dict[k])
        plt.tight_layout()


