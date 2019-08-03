from sklearn import metrics
from sklearn import datasets
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize']= 8, 4


def linear_prediction(plot_dict):
    for noise in plot_dict:
        X, y = datasets.make_regression(n_features=1, random_state=42, noise=noise)
        model = LinearRegression()
        model.fit(X, y)
        prediction = model.predict(X)
        mae = metrics.mean_absolute_error(prediction, y)
        mse = metrics.mean_squared_error(prediction, y)
        r2 = metrics.r2_score(prediction, y)

        plt.subplot(plot_dict[noise])
        plt.xlabel('prediction')
        plt.ylabel('actual')

        plt.plot(prediction, y, '.')
        plt.title('Plot for noise: %d'%noise + '\n' + 'mae:%.2f'%mae
                  + '\n' + 'mse: %.2f'%mse
                  + '\n' + 'r2:.2f'%r2)

    plt.show()


if __name__ == "__main__":
    x = {1:141, 9:142, 18:143, 1000:144}
    linear_prediction(x)
