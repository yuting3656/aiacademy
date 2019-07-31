from sklearn import datasets
from scipy import stats
import pandas as pd

iris = datasets.load_iris()
iris_data = iris.data
print(iris.feature_names)

# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = iris[:, 0]  # sepal length (cm)
y = iris[:, 2]  # petal length (cm)
alpha = 0.05


def f_test(x, y):
    F = x.var() / y.var()
    df1 = len(x) - 1
    df2 = len(y) - 1
    p_value = stats.f.cdf(F, df1, df2)
    return p_value


