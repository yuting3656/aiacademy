from sklearn import datasets
import matplotlib.pyplot as plt

"""
自己練一個:

長出 3 * 3 圖
 
 1 2 3 
 4 5 6 
 7 8 9
"""

digits = datasets.load_digits()

# 開一張圖畫紙
fig = plt.figure()

for i in range(10):
    if i != 0:
        fig.add_subplot(3, 3, i )
        # 讓顯示出來的 顏色樣式不一樣參考網站
        # https://chrisalbon.com/python/basics/set_the_color_of_a_matplotlib/
        # EX: cmap=plt.cm.Blues
        plt.imshow(digits.images[i], cmap=plt.cm.bone)
        plt.text(0, 7, digits.target[i], color='r')
# plt.imshow(digits.images[1], cmap=plt.cm.binary)
plt.show()