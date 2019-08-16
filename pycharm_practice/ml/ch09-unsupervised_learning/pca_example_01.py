import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.decomposition import PCA

"""
PCA: Principal Component Analysis
"""
np.random.seed(1)
X = np.dot(np.random.random(size=(2, 2)), np.random.normal(size=(2, 200))).T
print(X.shape) # (200, 2)

# plt.plot(X[:, 0], X[:, 1], 'o')
# plt.axis('equal')
# plt.show()

# 把維度 降成2維度
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_)
print(pca.components_)

