import numpy as np
from sklearn.datasets import load_boston
#  metrics for Model evaluation: mae, mse , r2
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

"""
練習畫圖畫:
  
       普通 linear regression
---------------------------
       用 Lasso 的圖

"""

rcParams['figure.figsize'] = 15, 15
data = load_boston()
# print(data.keys())
# dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])
# print(data.DESCR)

fig = plt.figure(1, figsize=(12, 12))


# for i in range(13):
#     ax = fig.add_subplot(4, 4, i + 1)
#     ax.scatter(data.data[:, i], data.target)
# plt.show()
# 練習用 sklearn.model_selection 來練 linear regression

# 資料轉換
data_transform = [np.hstack([data.data[i],np.sqrt(data.data[i]),np.log(data.data[i]+1)]) for i in range(len(data.data))]
"""
=============== vstack ===================

>>> a = np.array([[1, 2], [3, 4]])
>>> a
array([[1, 2],
       [3, 4]])
>>> a.shape
(2, 2)
>>>
>>> b = np.array([[11, 22], [33, 44], [55, 66], [77, 88] ])
>>> b.shape
(4, 2)
>>> np.vstack((a, b))
array([[ 1,  2],
       [ 3,  4],
       [11, 22],
       [33, 44],
       [55, 66],
       [77, 88]])
===========================  hstack ============================== 
>>> c = np.array([[11, 22, 00], [33, 44, 00], [55, 66, 00], [77, 88, 00] ])
>>> np.hstack((b, c))
array([[11, 22, 11, 22,  0],
       [33, 44, 33, 44,  0],
       [55, 66, 55, 66,  0],
       [77, 88, 77, 88,  0]])
>>> np.hstack((c, b))
array([[11, 22,  0, 11, 22],
       [33, 44,  0, 33, 44],
       [55, 66,  0, 55, 66],
       [77, 88,  0, 77, 88]])
>>>

"""

X_train, X_test, y_train, y_test = train_test_split(data_transform, data.target, test_size= 0.33, random_state= 5, shuffle=True)

# 實體化 linearRegression model
model = LinearRegression()
# 丟剛剛 分好的資料 給 model 練
model.fit(X_train, y_train)
# 把測試資料餵給 練好的 model
y_pred = model.predict(X_test)
# plt.scatter(y_test, y_pred)

# 來計算 mse (mean squared error) and rmse (root mean squared error)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)
# 出來的數字都不好看
print('mse: {}, rmse: {}, r2: {}'.format(mse, rmse, r2))

# 畫圖: linear regression
plt.subplot(2, 1, 1)
plt.scatter(y_pred, y_test, c='r')
plt.plot([0, 50], [0, 50], '--b')
plt.title("Linear Regression\
           \n mse: {}, rmse: {}, r2: {}".format(mse, rmse, r2) )

model_2 = Lasso(alpha=0.001, max_iter= 99999999)
model_2.fit(X_train, y_train)
y_pred_2 = model_2.predict(X_test)

# 來計算 mse (mean squared error) and rmse (root mean squared error)
mse_2 = metrics.mean_squared_error(y_test, y_pred_2)
rmse_2 = np.sqrt(mse_2)
r2_2 = metrics.r2_score(y_test, y_pred_2)
print('[with Lasso] mse: {}, rmse: {}, r2: {}'.format( mse_2, rmse_2, r2_2 ))

# 畫圖 Linear Regression with Lasso
plt.subplot(2, 1, 2)
plt.scatter(y_pred_2, y_test, c='y')
plt.plot([0, 50], [0, 50], '--g')
plt.title("Linear Regression with Lasso\
          \n mse: {}, rmse: {}, r2: {}".format(mse_2, rmse_2, r2_2))
plt.show()


